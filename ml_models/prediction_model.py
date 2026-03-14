"""
ml_models/prediction_model.py — XGBoost + LSTM ensemble for price direction prediction.

Feature engineering pipeline:
  - Technical indicators (RSI, MACD, Bollinger, ATR, EMA stack)
  - Volume delta, OI momentum
  - Price rate-of-change (multiple timeframes)

Output: probability score 0–1 for bullish direction
AI contribution to signal score: +3 if prediction ≥ 0.7
"""
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import ta
from loguru import logger

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed — ML prediction disabled")

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
    """Compute technical indicator features from OHLCV dataframe."""
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

    # EMA stack
    for period in [9, 20, 50, 100]:
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

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.fillna(0, inplace=True)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Predictor
# ─────────────────────────────────────────────────────────────────────────────
class XGBoostPredictor:
    MODEL_PATH = MODEL_DIR / "xgb_model.pkl"

    def __init__(self) -> None:
        self._model: Optional[xgb.XGBClassifier] = None
        self._load_or_create()

    def _load_or_create(self) -> None:
        if self.MODEL_PATH.exists():
            with open(self.MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            logger.info("XGBoost model loaded from {}", self.MODEL_PATH)
        else:
            if XGB_AVAILABLE:
                self._model = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                )
                logger.info("New XGBoost model created — needs training")

    def predict_proba(self, features: pd.DataFrame, context: Optional[dict] = None) -> float:
        """Return probability of upward price move (0–1).

        If the model was trained with enriched features (N + 6 context features),
        the caller should pass ``context`` so the same feature vector shape is used.
        Falls back to neutral defaults when context is not provided (pre-training or
        prediction-time calls that don't have research data yet).
        """
        if self._model is None or not XGB_AVAILABLE:
            return 0.5
        try:
            last_row = features.iloc[[-1]].values  # shape (1, N)
            # Detect whether the trained model expects more features than the base set
            expected_n = getattr(self._model, "n_features_in_", last_row.shape[1])
            base_n = last_row.shape[1]
            if expected_n > base_n:
                # Reconstruct context extension with provided values or sensible defaults
                ctx = context or {}
                from ml_models.self_learning import _STRATEGY_IDS, _NUM_STRATEGIES
                ctx_vec = np.array([[
                    min(ctx.get("signal_score", 9), 18) / 18.0,
                    ctx.get("research_score", 5.0) / 10.0,
                    float(ctx.get("mtf_alignment", 0.5)),
                    1.0 if ctx.get("direction", "LONG") == "LONG" else 0.0,
                    min(float(ctx.get("rr_ratio", 2.0)), 4.0) / 4.0,
                    _STRATEGY_IDS.get(str(ctx.get("strategy", "FALLBACK")).upper(), 0) / _NUM_STRATEGIES,
                ]], dtype=np.float32)
                last_row = np.concatenate([last_row, ctx_vec], axis=1)
            proba = self._model.predict_proba(last_row)[0][1]
            return float(proba)
        except Exception as exc:
            logger.debug("XGB predict error: {}", exc)
            return 0.5

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not XGB_AVAILABLE or self._model is None:
            return
        self._model.fit(X, y, eval_set=[(X, y)], verbose=False)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("XGBoost model trained & saved")


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Predictor (TensorFlow)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMPredictor:
    MODEL_PATH = MODEL_DIR / "lstm_model.keras"
    SEQ_LEN = 60

    def __init__(self) -> None:
        self._model = None
        if TF_AVAILABLE:
            self._load_or_create()

    def _load_or_create(self) -> None:
        if self.MODEL_PATH.exists():
            self._model = tf.keras.models.load_model(str(self.MODEL_PATH))
            logger.info("LSTM model loaded")
        else:
            logger.info("LSTM model not found — will create on first training")

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
            x = seq.reshape(1, self.SEQ_LEN, -1)
            proba = float(self._model.predict(x, verbose=0)[0][0])
            return proba
        except Exception as exc:
            logger.debug("LSTM predict error: {}", exc)
            return 0.5

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20) -> None:
        if not TF_AVAILABLE:
            return
        if self._model is None:
            self._model = self._build_model(X.shape[2])
        self._model.fit(X, y, epochs=epochs, batch_size=32,
                        validation_split=0.1, verbose=1)
        self._model.save(str(self.MODEL_PATH))
        logger.info("LSTM model trained & saved")


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Predictor
# ─────────────────────────────────────────────────────────────────────────────
class EnsemblePredictor:
    """
    Combines XGBoost + LSTM predictions.
    Returns (score, confidence):
      score: int  (+3 if bullish, +0 if neutral, +0 if bearish)
      confidence: float  0–1
    """

    def __init__(self) -> None:
        self._xgb = XGBoostPredictor()
        self._lstm = LSTMPredictor()

    def reload_xgb(self) -> None:
        """Hot-reload XGBoost model from disk after self-learning retrain.
        Called by SelfLearningEngine immediately after _do_retrain() saves a new .pkl.
        Ensures live predictions use the freshly trained model without restarting.
        """
        self._xgb._load_or_create()
        logger.info("EnsemblePredictor: XGBoost hot-reloaded from disk")

    def reload_lstm(self) -> None:
        """Hot-reload LSTM model from disk after self-learning retrain."""
        if TF_AVAILABLE:
            self._lstm._load_or_create()
            logger.info("EnsemblePredictor: LSTM hot-reloaded from disk")

    def predict(self, df: pd.DataFrame, context: Optional[dict] = None) -> Tuple[int, float]:
        features = build_features(df)
        xgb_prob = self._xgb.predict_proba(features, context=context)
        lstm_prob = self._lstm.predict_proba(features)

        # Weighted ensemble
        combined = xgb_prob * 0.6 + lstm_prob * 0.4

        score = 0
        if combined >= 0.7:
            score = 3      # AI bullish bonus (+3 as per spec)
        elif combined >= 0.55:
            score = 1

        return score, round(combined, 4)
