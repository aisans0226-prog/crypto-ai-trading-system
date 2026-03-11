"""
ml_models/self_learning.py - AI Self-Learning and Adaptive Training Pipeline.

Learns from every real trade outcome:
  * Records feature vectors at signal time
  * Labels them win/loss after trade closes
  * Retrains XGBoost incrementally every 50 new samples
  * Tracks per-coin win rates and adapts signal thresholds
"""
import asyncio
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from ml_models.prediction_model import build_features, MODEL_DIR

SAMPLES_FILE = MODEL_DIR / "training_samples.pkl"
STATS_FILE = MODEL_DIR / "coin_stats.json"
RETRAIN_EVERY = 50


class SelfLearningEngine:
    """Collects labelled trade outcomes and periodically retrains XGBoost."""

    def __init__(self) -> None:
        self._pending: Dict[str, dict] = {}
        self._samples_X: List[np.ndarray] = []
        self._samples_y: List[int] = []
        self._coin_stats: Dict[str, dict] = {}
        self._new_since_retrain: int = 0
        self._load_state()

    def _load_state(self) -> None:
        if SAMPLES_FILE.exists():
            try:
                with open(SAMPLES_FILE, "rb") as f:
                    data = pickle.load(f)
                    self._samples_X = data.get("X", [])
                    self._samples_y = data.get("y", [])
                logger.info("Self-learning: loaded {} samples", len(self._samples_y))
            except Exception as exc:
                logger.warning("Could not load samples: {}", exc)
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE) as f:
                    self._coin_stats = json.load(f)
            except Exception:
                pass

    def _save_state(self) -> None:
        try:
            with open(SAMPLES_FILE, "wb") as f:
                pickle.dump({"X": self._samples_X, "y": self._samples_y}, f)
            with open(STATS_FILE, "w") as f:
                json.dump(self._coin_stats, f)
        except Exception as exc:
            logger.warning("Could not save learning state: {}", exc)

    def record_prediction(self, trade_id: str, symbol: str, df: pd.DataFrame) -> None:
        """Store feature vector when signal fires."""
        try:
            features = build_features(df)
            fvec = features.iloc[-1].values.astype(np.float32)
            self._pending[trade_id] = {"symbol": symbol, "features": fvec, "ts": datetime.utcnow().isoformat()}
        except Exception as exc:
            logger.debug("record_prediction error: {}", exc)

    def label_trade(self, trade_id: str, pnl: float) -> None:
        """Label trade as win(1) or loss(0) and schedule retraining."""
        pending = self._pending.pop(trade_id, None)
        if pending is None:
            return
        label = 1 if pnl > 0 else 0
        self._samples_X.append(pending["features"])
        self._samples_y.append(label)
        self._new_since_retrain += 1
        sym = pending["symbol"]
        stat = self._coin_stats.setdefault(sym, {"wins": 0, "losses": 0, "total": 0})
        stat["total"] += 1
        if label == 1:
            stat["wins"] += 1
        else:
            stat["losses"] += 1
        stat["win_rate"] = stat["wins"] / stat["total"]
        logger.info("Trade labelled: {} pnl={:.2f} label={}", sym, pnl, label)
        self._save_state()
        if self._new_since_retrain >= RETRAIN_EVERY:
            asyncio.create_task(self._retrain())

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

    def get_coin_win_rate(self, symbol: str) -> float:
        return self._coin_stats.get(symbol, {}).get("win_rate", 0.5)

    def get_top_coins(self, n: int = 20) -> List[Tuple[str, float]]:
        eligible = [
            (sym, stat["win_rate"]) for sym, stat in self._coin_stats.items()
            if stat["total"] >= 5
        ]
        return sorted(eligible, key=lambda x: x[1], reverse=True)[:n]

    def get_stats(self) -> dict:
        total = len(self._samples_y)
        wins = sum(self._samples_y) if self._samples_y else 0
        return {
            "total_samples": total,
            "wins": wins,
            "losses": total - wins,
            "overall_win_rate": round(wins / total * 100, 2) if total else 0,
            "pending_labels": len(self._pending),
            "coins_tracked": len(self._coin_stats),
            "new_since_retrain": self._new_since_retrain,
            "next_retrain_at": RETRAIN_EVERY - self._new_since_retrain,
        }
