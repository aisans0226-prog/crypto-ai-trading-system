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

try:
    import tensorflow as _tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ml_models.prediction_model import build_features, MODEL_DIR

if TYPE_CHECKING:
    from data_engine.coin_database import CoinDatabase

# Legacy file paths — only used for one-time migration
_LEGACY_SAMPLES_FILE = MODEL_DIR / "training_samples.pkl"
_LEGACY_STATS_FILE   = MODEL_DIR / "coin_stats.json"

# Persisted retrain metadata — survives restarts
_META_FILE = MODEL_DIR / "retrain_meta.json"

RETRAIN_EVERY = 50

# Numeric ID for each strategy (used as a normalised context feature)
_STRATEGY_IDS: dict = {
    "FALLBACK": 0,
    "TREND_FOLLOW": 1, "TREND": 1,
    "BREAKOUT": 2,
    "LIQUIDITY_SWEEP": 3, "LIQUIDITY": 3,
    "MOMENTUM": 4,
    "MEAN_REVERSION": 5,
    "SCALP_BB": 6, "SCALP": 6,
}
_NUM_STRATEGIES = 6   # denominator for normalisation


class SelfLearningEngine:
    """Collects labelled trade outcomes and periodically retrains XGBoost + LSTM.

    Call ``await initialize(db)`` once after bot startup to load prior
    training data from PostgreSQL and recover any pending predictions.

    Call ``set_predictor(ensemble)`` to enable hot-reload of models into the
    live predictor immediately after retraining (no restart required).
    """

    def __init__(self) -> None:
        self._db: Optional["CoinDatabase"] = None
        self._predictor = None                         # EnsemblePredictor ref for hot-reload
        self._pending: Dict[str, dict] = {}            # trade_id → {symbol, features, seq, ts}
        self._samples_X: List[np.ndarray] = []         # flat feature vectors for XGBoost
        self._samples_seq: List[np.ndarray] = []       # 60-row sequences for LSTM (memory-only)
        self._samples_seq_y: List[int] = []            # labels aligned 1:1 with _samples_seq
        self._samples_y: List[int] = []
        self._coin_stats: Dict[str, dict] = {}         # symbol → {wins, losses, total, win_rate}
        self._new_since_retrain: int = 0
        # Retrain tracking
        self._retrain_count: int = 0
        self._last_retrain_ts: Optional[str] = None
        self._model_trained: bool = False
        # Timeline of last 100 labeled trades — used for learning curve chart
        self._labels_timeline: List[dict] = []
        # Load legacy files so the engine works even before initialize() is called
        self._load_legacy_files()
        # Restore retrain metadata so counts survive VPS restarts
        self._load_retrain_meta()

    def set_predictor(self, predictor) -> None:
        """Register the live EnsemblePredictor so hot-reload works after retrain."""
        self._predictor = predictor

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

    def _load_retrain_meta(self) -> None:
        """Restore _retrain_count / _last_retrain_ts from JSON so they survive restarts."""
        if _META_FILE.exists():
            try:
                with open(_META_FILE) as f:
                    meta = json.load(f)
                self._retrain_count   = meta.get("retrain_count",   self._retrain_count)
                self._last_retrain_ts = meta.get("last_retrain_ts", self._last_retrain_ts)
                self._model_trained   = meta.get("model_trained",   self._model_trained)
            except Exception as exc:
                logger.debug("Could not load retrain meta: {}", exc)

    def _save_retrain_meta(self) -> None:
        """Persist retrain metadata to JSON file after every retrain."""
        try:
            with open(_META_FILE, "w") as f:
                json.dump({
                    "retrain_count":   self._retrain_count,
                    "last_retrain_ts": self._last_retrain_ts,
                    "model_trained":   self._model_trained,
                }, f)
        except Exception as exc:
            logger.debug("Could not save retrain meta: {}", exc)

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

            # Remove stale unlabeled samples older than 24h (scan-loop zombies)
            await db.cleanup_stale_ml_samples(hours=24)

            # Rebuild labels_timeline from DB — it's in-memory only and lost on restart.
            # Restores the learning curve chart and trade timeline in the dashboard.
            try:
                tl = await db.load_labels_timeline(100)
                if tl:
                    self._labels_timeline = tl
                    logger.info("SelfLearning: {} timeline entries rebuilt from DB", len(tl))
            except Exception as exc:
                logger.warning("Could not rebuild labels_timeline from DB: {}", exc)

            # Mark model as trained if the model file exists on disk — prevents the dashboard
            # showing "Untrained" after a restart when the model was already trained.
            if (MODEL_DIR / "xgb_model.pkl").exists():
                self._model_trained = True

            # For old deployments that lack retrain_meta.json: estimate counts from the
            # model file's mtime and sample count so the dashboard is not misleading.
            model_path = MODEL_DIR / "xgb_model.pkl"
            if self._retrain_count == 0 and self._model_trained and len(self._samples_y) >= RETRAIN_EVERY:
                self._retrain_count = max(1, len(self._samples_y) // RETRAIN_EVERY)
                try:
                    mtime = model_path.stat().st_mtime
                    self._last_retrain_ts = datetime.utcfromtimestamp(mtime).isoformat()
                except Exception:
                    pass
                self._save_retrain_meta()
                logger.info("SelfLearning: retrain meta restored — estimated {} retrain(s) from {} samples",
                            self._retrain_count, len(self._samples_y))

            # Trigger initial retrain ONLY when model file is missing but we have enough data.
            # Skip if model already exists — avoids wasteful retrain on every VPS restart.
            model_exists = (MODEL_DIR / "xgb_model.pkl").exists()
            if self._retrain_count == 0 and len(self._samples_y) >= 30 and not model_exists:
                logger.info("SelfLearning: {} labeled samples found, scheduling initial retrain",
                            len(self._samples_y))
                asyncio.get_running_loop().create_task(self._retrain())
        except Exception as exc:
            logger.error("SelfLearning.initialize error (DB not available?): {}", exc)

    # ── Core learning methods ─────────────────────────────────────────────────
    def record_prediction(
        self,
        trade_id: str,
        symbol: str,
        df: pd.DataFrame,
        context: dict | None = None,
    ) -> None:
        """Store feature vector (for XGBoost) and sequence (for LSTM) when signal fires.

        ``context`` carries trade-quality metadata that price-only indicators can't
        capture.  Keys (all optional, sensible defaults applied):
          signal_score   int  0-18  — raw scanner total score
          research_score float 0-10 — deep-research normalised score
          mtf_alignment  float 0-1  — fraction of TFs agreeing
          direction      str  LONG/SHORT
          rr_ratio       float       — setup risk/reward
          strategy       str         — strategy name (see _STRATEGY_IDS)

        XGBoost flat vector is persisted to DB. LSTM sequence is memory-only.
        """
        try:
            features = build_features(df)
            fvec = features.iloc[-1].values.astype(np.float32)   # flat, for XGBoost
            seq  = features.values[-60:].astype(np.float32)       # 60-row, for LSTM

            # Append 8 context features so XGBoost/LGB can learn signal quality → outcome
            # Base-6: signal_score, research_score, mtf_alignment, direction, rr_ratio, strategy
            # New-2:  oi_delta_pct (OI % change, normalised), funding_rate (normalised)
            ctx = context or {}
            oi_delta_norm = min(max(ctx.get("oi_delta_pct", 0.0) / 5.0, -1.0), 1.0)
            funding_norm  = min(max(ctx.get("funding_rate", 0.0) / 0.01, -1.0), 1.0)
            ctx_vec = np.array([
                min(ctx.get("signal_score", 7), 18) / 18.0,
                ctx.get("research_score", 5.0) / 10.0,
                float(ctx.get("mtf_alignment", 0.5)),
                1.0 if ctx.get("direction", "LONG") == "LONG" else 0.0,
                min(float(ctx.get("rr_ratio", 2.0)), 4.0) / 4.0,
                _STRATEGY_IDS.get(str(ctx.get("strategy", "FALLBACK")).upper(), 0) / _NUM_STRATEGIES,
                oi_delta_norm,   # OI delta: 5% move normalised to ±1.0
                funding_norm,    # funding rate: 0.01% normalised to ±1.0
            ], dtype=np.float32)
            fvec_enriched = np.concatenate([fvec, ctx_vec])

            self._pending[trade_id] = {
                "symbol":   symbol,
                "features": fvec_enriched,
                "seq":      seq,   # stored in-memory; not persisted (too large for DB)
                "ts":       datetime.utcnow().isoformat(),
            }
            # Async background save to DB (enriched flat vector only)
            if self._db is not None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        self._db.save_ml_sample(trade_id, symbol, fvec_enriched.tolist()),
                        name=f"save_ml_{trade_id}",
                    )
                except RuntimeError:
                    pass
        except Exception as exc:
            logger.debug("record_prediction error: {}", exc)

    def label_trade(self, trade_id: str, pnl: float) -> None:
        """Label trade as win(1) or loss(0) and schedule retraining.
        Persists to DB in background (non-blocking).
        """
        pending = self._pending.pop(trade_id, None)
        if pending is None:
            return

        label = 1 if pnl >= 0.3 else 0   # require 0.3% net gain; breakeven / micro-wins count as loss
        self._samples_X.append(pending["features"])
        self._samples_y.append(label)
        # Register LSTM sequence only when a full 60-bar history was available.
        # Maintain _samples_seq_y in lock-step so LSTM labels are always aligned.
        if "seq" in pending and len(pending["seq"]) == 60:
            self._samples_seq.append(pending["seq"])
            self._samples_seq_y.append(label)
            # Trim to 6000 to cap memory usage (~36 MB at worst)
            if len(self._samples_seq) > 6000:
                self._samples_seq   = self._samples_seq[-5000:]
                self._samples_seq_y = self._samples_seq_y[-5000:]
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

        # Record in timeline for learning curve
        self._labels_timeline.append({
            "ts":     datetime.utcnow().isoformat(),
            "symbol": sym,
            "pnl":    round(pnl, 2),
            "label":  label,
        })
        if len(self._labels_timeline) > 100:
            self._labels_timeline = self._labels_timeline[-100:]

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
        # Hot-reload all models into live predictor without restarting bot
        if self._predictor is not None:
            self._predictor.reload_xgb()
            self._predictor.reload_lgb()   # reload LightGBM model after retrain
            if TF_AVAILABLE:
                self._predictor.reload_lstm()

    def _do_retrain(self) -> None:
        """Retrain XGBoost + LightGBM (always) + LSTM (when enough sequences are available).

        Delegates all training to the improved predictor classes in prediction_model.py:
          - XGBoostPredictor.train(): stratified 80/20 split, scale_pos_weight, AUC early-stop
          - LGBMPredictor.train():    same improvements, second tree model for diversity
          - LSTMPredictor.train():    StandardScaler normalisation added
        Both tree models log their validation AUC after training.
        """
        try:
            X = np.array(self._samples_X)
            y = np.array(self._samples_y)
            if len(X) > 5000:
                X, y = X[-5000:], y[-5000:]

            # Deferred imports avoid circular-import issues at module load time
            from ml_models.prediction_model import (
                XGBoostPredictor, LGBMPredictor, HAS_LGB as _HAS_LGB,
            )

            # ── XGBoost retrain ──────────────────────────────────────────
            # XGBoostPredictor() loads existing model; train() creates a fresh one
            # and overwrites the file — hot-reload picks it up after this returns.
            xgb_pred = XGBoostPredictor()
            xgb_pred.train(X, y)

            # ── LightGBM retrain (when lightgbm is installed) ────────────
            if _HAS_LGB:
                lgb_pred = LGBMPredictor()
                lgb_pred.train(X, y)

            self._new_since_retrain = 0
            self._retrain_count += 1
            self._last_retrain_ts = datetime.utcnow().isoformat()
            self._model_trained = True
            self._save_retrain_meta()   # persist so counts survive restarts
            logger.info(
                "Models retrained on {} samples — win-rate: {:.1f}%",
                len(y), sum(y) / len(y) * 100,
            )

            # ── LSTM retrain (requires TF and in-session sequences) ───────
            # Uses _samples_seq_y which is kept in lock-step with _samples_seq,
            # preventing the label-misalignment bug that arose from trades with
            # no 60-bar history interleaving with ones that do have sequences.
            if TF_AVAILABLE and len(self._samples_seq) >= 30:
                self._do_retrain_lstm(np.array(self._samples_seq_y[-5000:]))

        except Exception as exc:
            logger.error("Retraining failed: {}", exc)

    def _do_retrain_lstm(self, y_labels: np.ndarray) -> None:
        """Retrain LSTM on accumulated 60-bar sequences from the current session."""
        try:
            from ml_models.prediction_model import LSTMPredictor
            seqs = np.array(self._samples_seq[-5000:], dtype=np.float32)
            y = np.array(y_labels[-len(seqs):], dtype=np.float32)
            if len(seqs) != len(y) or len(seqs) < 30:
                return
            lstm = LSTMPredictor()
            lstm.train(seqs, y, epochs=10)   # fewer epochs for incremental retraining
            logger.info("LSTM retrained on {} sequences", len(seqs))
        except Exception as exc:
            logger.error("LSTM retraining failed: {}", exc)

    # ── Query helpers ────────────────────────────────────────────────────────
    def get_coin_win_rate(self, symbol: str) -> float:
        return self._coin_stats.get(symbol, {}).get("win_rate", 0.5)

    def get_top_coins(self, n: int = 20) -> List[Tuple[str, float]]:
        eligible = [
            (sym, stat["win_rate"]) for sym, stat in self._coin_stats.items()
            if stat.get("total", 0) >= 2
        ]
        return sorted(eligible, key=lambda x: x[1], reverse=True)[:n]

    def get_stats(self) -> dict:
        total = len(self._samples_y)
        wins  = sum(self._samples_y) if self._samples_y else 0
        overall_wr = round(wins / total * 100, 2) if total else 0

        # Recent win rates — use correct threshold to avoid partial-window mislabeling
        last10 = self._samples_y[-10:] if len(self._samples_y) >= 10 else []
        last20 = self._samples_y[-20:] if len(self._samples_y) >= 20 else []
        wr10 = round(sum(last10) / len(last10) * 100, 1) if last10 else None
        wr20 = round(sum(last20) / len(last20) * 100, 1) if last20 else None

        # Learning trend: compare recent vs overall
        if wr10 is not None and wr20 is not None:
            if wr10 > wr20 + 5:        trend = "improving"
            elif wr10 < wr20 - 5:     trend = "declining"
            else:                      trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "total_samples":              total,
            "wins":                       wins,
            "losses":                     total - wins,
            "overall_win_rate":           overall_wr,
            "win_rate_last_10":           wr10,
            "win_rate_last_20":           wr20,
            "learning_trend":             trend,
            "pending_labels":             len(self._pending),
            "coins_tracked":              len(self._coin_stats),
            "new_since_retrain":          self._new_since_retrain,
            "next_retrain_at":            RETRAIN_EVERY - self._new_since_retrain,
            "retrain_threshold":          RETRAIN_EVERY,
            "db_connected":               self._db is not None,
            "model_trained":              self._model_trained,
            "retrain_count":              self._retrain_count,
            "last_retrain_ts":            self._last_retrain_ts,
            "samples_needed_for_first":   max(0, 30 - total),
        }

    def get_labels_timeline(self, n: int = 50) -> List[dict]:
        """Return last N labeled trades with per-entry rolling 10-trade win rate."""
        entries = self._labels_timeline[-n:]
        result = []
        for i, e in enumerate(entries):
            window = [x["label"] for x in entries[max(0, i - 9): i + 1]]
            rolling_wr = round(sum(window) / len(window) * 100, 1) if window else 50.0
            seq = max(0, len(self._labels_timeline) - n) + i
            result.append({**e, "rolling_wr": rolling_wr, "seq": seq + 1})
        return result

    def get_feature_importance(self, top_n: int = 10) -> List[dict]:
        """Return top-N feature importances from the saved XGBoost model (if trained).
        Uses model file existence as gate — _model_trained flag is not required so this
        works correctly after a restart where the model was trained in a previous session."""
        model_path = MODEL_DIR / "xgb_model.pkl"
        if not XGB_AVAILABLE or not model_path.exists():
            return []
        try:
            import pickle as _pkl
            with open(model_path, "rb") as fh:
                model = _pkl.load(fh)
            importances = model.feature_importances_
            # Recover feature names: prefer model's stored names, else build from dummy df
            if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
                names = list(model.feature_names_in_)
            else:
                from ml_models.prediction_model import build_features as _bf
                dummy = pd.DataFrame({
                    "close":  np.ones(60), "high": np.ones(60),
                    "low":    np.ones(60), "volume": np.ones(60),
                })
                names = list(_bf(dummy).columns)
            pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:top_n]
            return [{"feature": n, "importance": round(float(v), 4)} for n, v in pairs]
        except Exception as exc:
            logger.debug("get_feature_importance error: {}", exc)
            return []

