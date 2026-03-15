"""
ml_models/prediction_model.py — XGBoost + LightGBM + LSTM ensemble for price direction prediction.

Feature engineering pipeline:
  - Technical indicators (RSI, MACD, Bollinger, ATR, EMA stack incl. EMA200)
  - Volatility features (20-bar return std)
  - OBV momentum (directional volume flow)
  - Taker buy ratio (when available from Binance klines)
  - Price rate-of-change (multiple timeframes)

Output: probability score 0–1 for bullish direction
AI contribution to signal score: +3 if prediction >= 0.7

Ensemble weights: XGB 45% | LGB 35% | LSTM 20%
  LSTM carries lowest weight — least reliable on small datasets.
"""
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ta
from loguru import logger
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed — ML prediction disabled")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not installed — LGB ensemble component disabled")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicator features from OHLCV dataframe.

    Base feature count: 26
      - 5 price returns (1/3/5/10/20-bar)
      - 2 RSI (7, 14)
      - 3 MACD (macd, signal, hist)
      - 2 Bollinger (pct_b, width)
      - 1 ATR
      - 5 EMA distances (9/20/50/100/200) — added EMA200 vs research_engine
      - 2 volume ratios (5, 20-bar)
      - 2 Stochastic (K, D)
      - 1 Williams %R
      - 1 CCI
      - 1 price_volatility (NEW)
      - 1 obv_momentum     (NEW)
      - 1 taker_buy_ratio  (NEW)
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Price returns
    for window in [1, 3, 5, 10, 20]:
        feat[f"ret_{window}"] = close.pct_change(window)

    # RSI
    feat["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    feat["rsi_7"] = ta.momentum.RSIIndicator(close=close, window=7).rsi()

    # MACD
    macd = ta.trend.MACD(close=close)
    feat["macd"] = macd.macd()
    feat["macd_signal"] = macd.macd_signal()
    feat["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=close, window=20)
    feat["bb_pct"] = bb.bollinger_pband()
    feat["bb_width"] = bb.bollinger_wband()

    # ATR
    feat["atr_14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    # EMA stack — now includes EMA200 for alignment with research_engine MTF logic
    for period in [9, 20, 50, 100, 200]:
        ema = ta.trend.EMAIndicator(close=close, window=period).ema_indicator()
        feat[f"ema_{period}_dist"] = (close - ema) / close

    # Volume features
    feat["vol_ratio_20"] = volume / volume.rolling(20).mean()
    feat["vol_ratio_5"] = volume / volume.rolling(5).mean()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close)
    feat["stoch_k"] = stoch.stoch()
    feat["stoch_d"] = stoch.stoch_signal()

    # Williams %R
    feat["williams_r"] = ta.momentum.WilliamsRIndicator(
        high=high, low=low, close=close, lbp=14
    ).williams_r()

    # CCI
    feat["cci_20"] = ta.trend.CCIIndicator(
        high=high, low=low, close=close, window=20
    ).cci()

    # ── New features ──────────────────────────────────────────────────────────

    # Volatility: 20-bar return std (captures current market regime)
    feat["price_volatility"] = close.pct_change().rolling(20).std() * 100

    # OBV momentum: directional volume flow normalised by 20-bar average volume
    obv_raw = (volume * np.sign(close.diff())).rolling(10).mean()
    feat["obv_momentum"] = obv_raw / (volume.rolling(20).mean() + 1e-10)

    # Taker buy ratio — available when Binance klines include taker_buy_base column
    if "taker_buy_base" in df.columns:
        feat["taker_buy_ratio"] = df["taker_buy_base"] / (volume + 1e-10)
        feat["taker_buy_ratio"] = feat["taker_buy_ratio"].fillna(0.5)
    else:
        feat["taker_buy_ratio"] = 0.5

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.fillna(0, inplace=True)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Predictor
# ─────────────────────────────────────────────────────────────────────────────
class XGBoostPredictor:
    MODEL_PATH = MODEL_DIR / "xgb_model.pkl"

    def __init__(self) -> None:
        self._model: Optional["xgb.XGBClassifier"] = None
        self._load_or_create()

    def _load_or_create(self) -> None:
        if self.MODEL_PATH.exists():
            with open(self.MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            logger.info("XGBoost model loaded from {}", self.MODEL_PATH)
        else:
            if XGB_AVAILABLE:
                self._model = xgb.XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="auc",
                    random_state=42,
                )
                logger.info("New XGBoost model created — needs training")

    def _build_ctx_vec(self, context: Optional[dict], ctx_size: int) -> np.ndarray:
        """Build context feature vector.

        Supports ctx_size=6 (legacy) and ctx_size=8 (current: adds OI delta + funding).
        Having both sizes lets old saved models remain usable until next retrain.
        """
        ctx = context or {}
        from ml_models.self_learning import _STRATEGY_IDS, _NUM_STRATEGIES
        ctx_list = [
            min(ctx.get("signal_score", 9), 18) / 18.0,
            ctx.get("research_score", 5.0) / 10.0,
            float(ctx.get("mtf_alignment", 0.5)),
            1.0 if ctx.get("direction", "LONG") == "LONG" else 0.0,
            min(float(ctx.get("rr_ratio", 2.0)), 4.0) / 4.0,
            _STRATEGY_IDS.get(str(ctx.get("strategy", "FALLBACK")).upper(), 0) / _NUM_STRATEGIES,
        ]
        if ctx_size == 8:
            # 5% OI change normalised to ±1.0; 0.01% funding rate to ±1.0
            oi_delta_norm = min(max(ctx.get("oi_delta_pct", 0.0) / 5.0, -1.0), 1.0)
            funding_norm  = min(max(ctx.get("funding_rate", 0.0) / 0.01, -1.0), 1.0)
            ctx_list.extend([oi_delta_norm, funding_norm])
        return np.array([ctx_list], dtype=np.float32)

    def predict_proba(self, features: pd.DataFrame, context: Optional[dict] = None) -> float:
        """Return probability of upward price move (0–1).

        Handles models trained with 6-feature (legacy) or 8-feature (current) context.
        Falls back to 0.5 on any feature count mismatch — safe until next retrain.
        """
        if self._model is None or not XGB_AVAILABLE:
            return 0.5
        try:
            last_row = features.iloc[[-1]].values  # shape (1, N_base)
            expected_n = getattr(self._model, "n_features_in_", last_row.shape[1])
            base_n = last_row.shape[1]

            if expected_n > base_n:
                ctx_size = expected_n - base_n
                if ctx_size not in (6, 8):
                    # Old model trained on a different base feature set — skip safely
                    logger.debug(
                        "XGB feature mismatch: model expects {} features, "
                        "got base={} ctx_size={} (not 6 or 8). "
                        "Returning 0.5 until retrained with new features.",
                        expected_n, base_n, ctx_size,
                    )
                    return 0.5
                last_row = np.concatenate(
                    [last_row, self._build_ctx_vec(context, ctx_size)], axis=1
                )

            # Final shape guard — should never trigger but keeps inference safe
            if last_row.shape[1] != expected_n:
                logger.warning("XGB shape mismatch after ctx: {} vs {}", last_row.shape[1], expected_n)
                return 0.5

            proba = self._model.predict_proba(last_row)[0][1]
            return float(proba)
        except Exception as exc:
            logger.debug("XGB predict error: {}", exc)
            return 0.5

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train with proper 80/20 validation split, class-imbalance weighting, and early stopping.

        Improvements over original inline retrain code:
          - Stratified 80/20 split (was sequential 90/10, prone to temporal leakage)
          - scale_pos_weight accounts for ~44% win-rate imbalance
          - early_stopping_rounds=50 with AUC metric (not logloss) prevents overfitting
          - max_depth reduced from 6 to 5 for better generalisation
        """
        if not XGB_AVAILABLE:
            return
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        # Stratified split keeps class ratios consistent in both sets
        stratify = y if len(set(y)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        # Compensate for class imbalance (~56% losses vs ~44% wins)
        neg_count = int(sum(1 for lbl in y_train if lbl == 0))
        pos_count = int(sum(1 for lbl in y_train if lbl == 1))
        scale_pos_weight = neg_count / max(pos_count, 1)

        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,            # reduced from 6 — fewer splits means less overfitting
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=50,
            eval_metric="auc",      # AUC is more informative than logloss for imbalanced data
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Log validation AUC to gauge real-world discriminative power
        try:
            val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            logger.info(
                "XGBoost trained on {} samples — val_AUC: {:.4f}  scale_pos_weight: {:.2f}",
                len(y_train), val_auc, scale_pos_weight,
            )
        except Exception:
            logger.info("XGBoost trained — scale_pos_weight: {:.2f}", scale_pos_weight)

        self._model = model
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info("XGBoost model saved to {}", self.MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM Predictor
# ─────────────────────────────────────────────────────────────────────────────
class LGBMPredictor:
    MODEL_PATH = MODEL_DIR / "lgb_model.pkl"

    def __init__(self) -> None:
        self._model = None
        if HAS_LGB:
            self._load_or_create()

    def _load_or_create(self) -> None:
        if self.MODEL_PATH.exists():
            try:
                with open(self.MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("LightGBM model loaded from {}", self.MODEL_PATH)
            except Exception as exc:
                logger.warning("Could not load LightGBM model: {}", exc)
        else:
            logger.info("LightGBM model not found — will train on first retrain cycle")

    def predict_proba(self, features: pd.DataFrame, context: Optional[dict] = None) -> float:
        """Return probability of upward price move (0–1).

        Mirrors XGBoostPredictor.predict_proba() feature-count mismatch handling.
        """
        if self._model is None or not HAS_LGB:
            return 0.5
        try:
            last_row = features.iloc[[-1]].values
            expected_n = getattr(self._model, "n_features_in_", last_row.shape[1])
            base_n = last_row.shape[1]

            if expected_n > base_n:
                ctx_size = expected_n - base_n
                if ctx_size not in (6, 8):
                    logger.debug("LGB feature mismatch (ctx_size={}), returning 0.5", ctx_size)
                    return 0.5
                ctx = context or {}
                from ml_models.self_learning import _STRATEGY_IDS, _NUM_STRATEGIES
                ctx_list = [
                    min(ctx.get("signal_score", 9), 18) / 18.0,
                    ctx.get("research_score", 5.0) / 10.0,
                    float(ctx.get("mtf_alignment", 0.5)),
                    1.0 if ctx.get("direction", "LONG") == "LONG" else 0.0,
                    min(float(ctx.get("rr_ratio", 2.0)), 4.0) / 4.0,
                    _STRATEGY_IDS.get(str(ctx.get("strategy", "FALLBACK")).upper(), 0) / _NUM_STRATEGIES,
                ]
                if ctx_size == 8:
                    ctx_list.extend([
                        min(max(ctx.get("oi_delta_pct", 0.0) / 5.0, -1.0), 1.0),
                        min(max(ctx.get("funding_rate", 0.0) / 0.01, -1.0), 1.0),
                    ])
                ctx_vec = np.array([ctx_list], dtype=np.float32)
                last_row = np.concatenate([last_row, ctx_vec], axis=1)

            if last_row.shape[1] != expected_n:
                return 0.5

            proba = self._model.predict_proba(last_row)[0][1]
            return float(proba)
        except Exception as exc:
            logger.debug("LGB predict error: {}", exc)
            return 0.5

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train with proper 80/20 validation split, class-imbalance weighting, and early stopping."""
        if not HAS_LGB:
            return
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        stratify = y if len(set(y)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        neg_count = int(sum(1 for lbl in y_train if lbl == 0))
        pos_count = int(sum(1 for lbl in y_train if lbl == 1))
        scale_pos_weight = neg_count / max(pos_count, 1)

        model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        try:
            val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            logger.info(
                "LightGBM trained on {} samples — val_AUC: {:.4f}  scale_pos_weight: {:.2f}",
                len(y_train), val_auc, scale_pos_weight,
            )
        except Exception:
            logger.info("LightGBM trained — scale_pos_weight: {:.2f}", scale_pos_weight)

        self._model = model
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info("LightGBM model saved to {}", self.MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Predictor (TensorFlow)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMPredictor:
    MODEL_PATH  = MODEL_DIR / "lstm_model.keras"
    SCALER_PATH = MODEL_DIR / "lstm_scaler.pkl"
    SEQ_LEN = 60

    def __init__(self) -> None:
        self._model = None
        self._lstm_scaler: Optional[StandardScaler] = None
        if TF_AVAILABLE:
            self._load_or_create()

    def _load_or_create(self) -> None:
        if self.MODEL_PATH.exists():
            self._model = tf.keras.models.load_model(str(self.MODEL_PATH))
            logger.info("LSTM model loaded")
        else:
            logger.info("LSTM model not found — will create on first training")

        # Load paired scaler — must match the model or predictions are garbage
        if self.SCALER_PATH.exists():
            try:
                with open(self.SCALER_PATH, "rb") as f:
                    self._lstm_scaler = pickle.load(f)
                logger.info("LSTM scaler loaded")
            except Exception as exc:
                logger.warning("Could not load LSTM scaler: {}", exc)

    def _build_model(self, n_features: int) -> "tf.keras.Model":
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True,
                                  input_shape=(self.SEQ_LEN, n_features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def predict_proba(self, features: pd.DataFrame) -> float:
        if self._model is None or not TF_AVAILABLE:
            return 0.5
        try:
            seq = features.values[-self.SEQ_LEN:].astype(np.float32)
            if len(seq) < self.SEQ_LEN:
                return 0.5
            n_feat = seq.shape[1]
            # Apply the same StandardScaler fitted during training
            if self._lstm_scaler is not None:
                seq = self._lstm_scaler.transform(
                    seq.reshape(-1, n_feat)
                ).reshape(self.SEQ_LEN, n_feat)
            x = seq.reshape(1, self.SEQ_LEN, n_feat)
            proba = float(self._model.predict(x, verbose=0)[0][0])
            return proba
        except Exception as exc:
            logger.debug("LSTM predict error: {}", exc)
            return 0.5

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20) -> None:
        """Train LSTM on (n_samples, seq_len, n_features) sequences.

        Applies StandardScaler across the feature dimension before fitting so
        that gradient magnitudes are comparable regardless of feature scale.
        The scaler is saved alongside the model so predict_proba uses the same
        normalisation at inference time.
        """
        if not TF_AVAILABLE:
            return
        n_samples, seq_len, n_features = X.shape

        # Flatten to (n_samples * seq_len, n_features) to fit scaler, then reshape back
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = scaler.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)

        if self._model is None:
            self._model = self._build_model(n_features)
        self._model.fit(X_scaled, y, epochs=epochs, batch_size=32,
                        validation_split=0.1, verbose=1)
        self._model.save(str(self.MODEL_PATH))

        # Persist scaler so prediction path applies the identical transform
        self._lstm_scaler = scaler
        with open(self.SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(
            "LSTM model and scaler trained & saved ({} sequences, {} features)",
            n_samples, n_features,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Predictor
# ─────────────────────────────────────────────────────────────────────────────
class EnsemblePredictor:
    """
    Combines XGBoost + LightGBM + LSTM predictions.

    Weights: XGB 45% | LGB 35% | LSTM 20%
      - XGB and LGB carry most weight — tree models generalise well on tabular data
      - LSTM gets lowest weight — most sensitive to data sparsity on small datasets
      - When LGB or LSTM are untrained, their share is redistributed to XGB

    Returns (score, confidence):
      score: int  (+3 if bullish >= 0.7, +1 if >= 0.55, +0 otherwise)
      confidence: float  0–1
    """

    def __init__(self) -> None:
        self._xgb = XGBoostPredictor()
        self._lgb = LGBMPredictor()
        self._lstm = LSTMPredictor()

    # ── Hot-reload helpers — called by SelfLearningEngine after every retrain ──

    def reload_xgb(self) -> None:
        """Hot-reload XGBoost model from disk — no restart needed."""
        self._xgb._load_or_create()
        logger.info("EnsemblePredictor: XGBoost hot-reloaded from disk")

    def reload_lgb(self) -> None:
        """Hot-reload LightGBM model from disk — no restart needed."""
        if HAS_LGB:
            self._lgb._load_or_create()
            logger.info("EnsemblePredictor: LightGBM hot-reloaded from disk")

    def reload_lstm(self) -> None:
        """Hot-reload LSTM model and scaler from disk — no restart needed."""
        if TF_AVAILABLE:
            self._lstm._load_or_create()
            logger.info("EnsemblePredictor: LSTM hot-reloaded from disk")

    def predict(self, df: pd.DataFrame, context: Optional[dict] = None) -> Tuple[int, float]:
        features = build_features(df)
        xgb_prob = self._xgb.predict_proba(features, context=context)

        # Fall back to xgb_prob when LGB/LSTM are not yet trained so the
        # ensemble never collapses to a neutral 0.5 bias on first boot
        lgb_prob = (
            self._lgb.predict_proba(features, context=context)
            if self._lgb._model is not None
            else xgb_prob
        )
        lstm_prob = (
            self._lstm.predict_proba(features)
            if self._lstm._model is not None
            else xgb_prob
        )

        # Weighted ensemble: XGB 45% | LGB 35% | LSTM 20%
        combined = xgb_prob * 0.45 + lgb_prob * 0.35 + lstm_prob * 0.20

        score = 0
        if combined >= 0.7:
            score = 3      # AI bullish bonus (+3 as per spec)
        elif combined >= 0.55:
            score = 1

        return score, round(combined, 4)
