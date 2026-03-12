"""
ml_models/self_learning.py - AI Self-Learning and Adaptive Training Pipeline.

Learns from every real trade outcome:
  * Records feature vectors at signal time → saved to PostgreSQL (ml_training_samples)
  * Labels them win/loss after trade closes → DB updated
  * Retrains XGBoost incrementally every 50 new samples
  * Per-coin win rates derived from DB (replaces coin_stats.json)

Persistence strategy:
  - PRIMARY: PostgreSQL ml_training_samples table (survives VPS resets)
  - FALLBACK: .pkl/.json files used only for first-boot migration
"""
import asyncio
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from ml_models.prediction_model import build_features, MODEL_DIR

if TYPE_CHECKING:
    from data_engine.coin_database import CoinDatabase

# Legacy file paths — only used for one-time migration
_LEGACY_SAMPLES_FILE = MODEL_DIR / "training_samples.pkl"
_LEGACY_STATS_FILE   = MODEL_DIR / "coin_stats.json"

RETRAIN_EVERY = 50


class SelfLearningEngine:
    """Collects labelled trade outcomes and periodically retrains XGBoost.

    Call ``await initialize(db)`` once after bot startup to load prior
    training data from PostgreSQL and recover any pending predictions.
    """

    def __init__(self) -> None:
        self._db: Optional["CoinDatabase"] = None
        self._pending: Dict[str, dict] = {}       # trade_id → {symbol, features, ts}
        self._samples_X: List[np.ndarray] = []
        self._samples_y: List[int] = []
        self._coin_stats: Dict[str, dict] = {}    # symbol → {wins, losses, total, win_rate}
        self._new_since_retrain: int = 0
        # Load legacy files so the engine works even before initialize() is called
        self._load_legacy_files()

    # ── Startup ───────────────────────────────────────────────────────────────
    def _load_legacy_files(self) -> None:
        """Load from .pkl/.json files (migration fallback only)."""
        if _LEGACY_SAMPLES_FILE.exists():
            try:
                with open(_LEGACY_SAMPLES_FILE, "rb") as f:
                    data = pickle.load(f)
                    self._samples_X = data.get("X", [])
                    self._samples_y = data.get("y", [])
                logger.info("SelfLearning: loaded {} samples from legacy .pkl", len(self._samples_y))
            except Exception as exc:
                logger.warning("Could not load legacy samples: {}", exc)
        if _LEGACY_STATS_FILE.exists():
            try:
                with open(_LEGACY_STATS_FILE) as f:
                    self._coin_stats = json.load(f)
            except Exception:
                pass

    async def initialize(self, db: "CoinDatabase") -> None:
        """Load state from PostgreSQL. Call once after CoinDatabase.start().

        If DB has more labeled samples than the legacy files (including when
        files don't exist), DB wins. Any pending predictions from before the
        last restart are recovered so those trades can still be labeled.
        """
        self._db = db
        try:
            db_X, db_y, db_coin_stats = await db.load_ml_samples()
            if len(db_y) >= len(self._samples_y):
                self._samples_X = db_X
                self._samples_y = db_y
                self._coin_stats = db_coin_stats
                logger.info("SelfLearning: {} samples loaded from PostgreSQL ({} coins)",
                            len(db_y), len(db_coin_stats))
            else:
                # Legacy files have more data — keep them, they'll drip into DB going forward
                logger.info("SelfLearning: legacy files ({}) > DB ({}); using files, DB will catch up",
                            len(self._samples_y), len(db_y))

            # Recover pending predictions that were in-flight before last restart
            db_pending = await db.load_pending_ml_samples()
            for trade_id, entry in db_pending.items():
                if trade_id not in self._pending:
                    self._pending[trade_id] = entry
        except Exception as exc:
            logger.error("SelfLearning.initialize error (DB not available?): {}", exc)

    # ── Core learning methods ─────────────────────────────────────────────────
    def record_prediction(self, trade_id: str, symbol: str, df: pd.DataFrame) -> None:
        """Store feature vector when signal fires.
        Persists to DB in background (non-blocking).
        """
        try:
            features = build_features(df)
            fvec = features.iloc[-1].values.astype(np.float32)
            self._pending[trade_id] = {
                "symbol":   symbol,
                "features": fvec,
                "ts":       datetime.utcnow().isoformat(),
            }
            # Async background save to DB
            if self._db is not None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        self._db.save_ml_sample(trade_id, symbol, fvec.tolist()),
                        name=f"save_ml_{trade_id}",
                    )
                except RuntimeError:
                    pass  # no running loop (e.g. unit test) — skip DB save
        except Exception as exc:
            logger.debug("record_prediction error: {}", exc)

    def label_trade(self, trade_id: str, pnl: float) -> None:
        """Label trade as win(1) or loss(0) and schedule retraining.
        Persists to DB in background (non-blocking).
        """
        pending = self._pending.pop(trade_id, None)
        if pending is None:
            return

        label = 1 if pnl > 0 else 0
        self._samples_X.append(pending["features"])
        self._samples_y.append(label)
        self._new_since_retrain += 1

        # Update in-memory per-coin stats
        sym = pending["symbol"]
        stat = self._coin_stats.setdefault(sym, {"wins": 0, "losses": 0, "total": 0})
        stat["total"] += 1
        if label == 1:
            stat["wins"] += 1
        else:
            stat["losses"] += 1
        stat["win_rate"] = stat["wins"] / stat["total"]

        logger.info("Trade labelled: {} pnl={:.2f} label={}", sym, pnl, label)

        # Async background: update DB label
        if self._db is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._db.label_ml_sample(trade_id, label),
                    name=f"label_ml_{trade_id}",
                )
            except RuntimeError:
                pass

        if self._new_since_retrain >= RETRAIN_EVERY:
            try:
                asyncio.get_running_loop().create_task(self._retrain())
            except RuntimeError:
                pass

    # ── Retraining ───────────────────────────────────────────────────────────
    async def _retrain(self) -> None:
        if not XGB_AVAILABLE or len(self._samples_y) < 30:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._do_retrain)

    def _do_retrain(self) -> None:
        try:
            X = np.array(self._samples_X)
            y = np.array(self._samples_y)
            if len(X) > 5000:
                X, y = X[-5000:], y[-5000:]
            model = xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42,
            )
            model.fit(X, y, verbose=False)
            with open(MODEL_DIR / "xgb_model.pkl", "wb") as f:
                pickle.dump(model, f)
            self._new_since_retrain = 0
            logger.info("Model retrained on {} samples — win-rate: {:.1f}%",
                        len(y), sum(y) / len(y) * 100)
        except Exception as exc:
            logger.error("Retraining failed: {}", exc)

    # ── Query helpers ────────────────────────────────────────────────────────
    def get_coin_win_rate(self, symbol: str) -> float:
        return self._coin_stats.get(symbol, {}).get("win_rate", 0.5)

    def get_top_coins(self, n: int = 20) -> List[Tuple[str, float]]:
        eligible = [
            (sym, stat["win_rate"]) for sym, stat in self._coin_stats.items()
            if stat.get("total", 0) >= 5
        ]
        return sorted(eligible, key=lambda x: x[1], reverse=True)[:n]

    def get_stats(self) -> dict:
        total = len(self._samples_y)
        wins  = sum(self._samples_y) if self._samples_y else 0
        return {
            "total_samples":     total,
            "wins":              wins,
            "losses":            total - wins,
            "overall_win_rate":  round(wins / total * 100, 2) if total else 0,
            "pending_labels":    len(self._pending),
            "coins_tracked":     len(self._coin_stats),
            "new_since_retrain": self._new_since_retrain,
            "next_retrain_at":   RETRAIN_EVERY - self._new_since_retrain,
            "db_connected":      self._db is not None,
        }
