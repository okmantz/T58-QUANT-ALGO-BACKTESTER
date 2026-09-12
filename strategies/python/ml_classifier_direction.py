"""
ML Classifier Direction Strategy.

The one strategy family in this repo that LEARNS a rule from data rather
than encoding a hand-designed hypothesis -- every other strategy (Manual
families, every other Python/PineScript/MQL5 file) is a fixed, named rule
whose PARAMETERS get tuned; this one fits a classifier's own DECISION
BOUNDARY to historical features.

WALK-FORWARD SAFETY (read this before changing TRAIN_FRAC or LABEL_HORIZON)
----------------------------------------------------------------------------
generate_signals(df) is called ONCE over the whole dataset (see
app/strategy/python.py's own module docstring). A naive "fit on all the
data, predict on all the data" classifier would be trading on its own
training set for every single bar -- the single most common way an ML
strategy silently cheats. This file avoids that with a plain, auditable
chronological split:

  1. The model is fit ONCE on a strict PREFIX of the data (bars
     [0, train_end)), using labels built from FORWARD returns -- but only
     for training rows whose forward-return window is itself entirely
     inside the training prefix (`i + LABEL_HORIZON < train_end`). A
     label straddling the train/test boundary would leak test-period
     price action into training.
  2. Every FEATURE (for both training and prediction) at bar i is built
     only from data at or before bar i -- rolling windows, no forward
     shifts.
  3. Signals are emitted ONLY for bars at or after train_end. The
     training segment itself is always flat (signal = 0), so this
     strategy can never appear to trade profitably on data it fit itself
     to -- exactly the kind of result the falsification-kit methodology
     this repo already leans on is built to catch.

This is intentionally a SIMPLE, INSPECTABLE model (scikit-learn logistic
regression on a handful of engineered features) rather than a deep or
ensemble model with a much larger hyperparameter surface to overfit --
consistent with this repo's existing "distrust of raw brute-force
combinatorics over arbitrary indicators" (see app/search/strategy_space.py
module docstring); more capacity would need proportionally more
out-of-sample bars to trust, not more train/test tricks.

Run this exactly like any other uploaded Python strategy: through Run &
Report, Search Lab (mode=single wraps it for CPCV/WFO/sensitivity/etc.),
or Full Pipeline. TRAIN_FRAC/LABEL_HORIZON/PREDICT_PROB_THRESHOLD below are
this strategy's OWN tunable parameters (discovered like any Python
strategy's numeric constants -- see app.optimize.code_parameter_space).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "ML Classifier Direction (logistic regression)"

# --- tunable parameters -----------------------------------------------
TRAIN_FRAC = 0.6          # fraction of the dataset used to fit the model; rest is out-of-sample
LABEL_HORIZON = 5         # bars ahead the model predicts the direction of
LABEL_DEADZONE_ATR = 0.15 # forward moves smaller than this many ATRs are excluded from training
                          # (a genuine "no edge either way" zone the classifier shouldn't be forced to call)
PROB_THRESHOLD = 0.58     # only trade when predicted P(up)/P(down) clears this -- above 0.5 by design,
                          # so the model has to show real conviction, not just a coin-flip edge
RSI_PERIOD = 14
ATR_PERIOD = 14
STOP_ATR_MULT = 1.5
TARGET_ATR_MULT = 2.5


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.where(avg_loss.ne(0), 100).fillna(50)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every column here is computed causally -- rolling/ewm windows and
    .diff()/.pct_change() only ever look backward from each row."""
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    macd_line = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    atr = _atr(df, ATR_PERIOD)
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    avg_volume = volume.rolling(20, min_periods=20).mean()

    feats = pd.DataFrame(index=df.index)
    feats["ret_1"] = close.pct_change(1)
    feats["ret_5"] = close.pct_change(5)
    feats["ret_10"] = close.pct_change(10)
    feats["rsi"] = _rsi(close, RSI_PERIOD)
    feats["macd_hist"] = macd_line - macd_signal
    feats["atr_norm"] = atr / close
    feats["dist_ema20"] = (close - ema20) / ema20
    feats["dist_ema50"] = (close - ema50) / ema50
    feats["vol_rel"] = (volume / avg_volume.replace(0, np.nan)).fillna(1.0)
    feats["_atr"] = atr  # kept only to size the label deadzone/risk mgmt below, dropped before fitting
    return feats


def generate_signals(df: pd.DataFrame) -> pd.Series:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "ml_classifier_direction.py requires scikit-learn (pip install scikit-learn) -- "
            "it isn't a hard dependency of the rest of this app, only of this one strategy file."
        ) from exc

    n = len(df)
    signals = pd.Series(0, index=df.index, dtype=int)
    if n < 200:
        # Not enough bars for a meaningful train/test split -- stand down
        # rather than fit a classifier on a handful of rows.
        return signals

    feats = _build_features(df)
    feature_cols = [c for c in feats.columns if c != "_atr"]
    atr = feats["_atr"]

    train_end = int(n * TRAIN_FRAC)
    train_end = max(min(train_end, n - LABEL_HORIZON - 1), RSI_PERIOD + 50)

    # Forward return over LABEL_HORIZON bars, in ATR units so the deadzone
    # threshold is scale-free across instruments.
    forward_return = df["close"].shift(-LABEL_HORIZON) / df["close"] - 1.0
    forward_return_atr = forward_return * df["close"] / atr.replace(0, np.nan)

    valid_features = feats[feature_cols].notna().all(axis=1)
    # Only rows whose OWN forward window is fully inside the training
    # prefix may be used as training labels -- see module docstring.
    trainable = valid_features & (forward_return_atr.notna()) & (pd.Series(np.arange(n), index=df.index) + LABEL_HORIZON < train_end)
    trainable &= forward_return_atr.abs() >= LABEL_DEADZONE_ATR

    train_idx = df.index[trainable]
    if len(train_idx) < 100:
        # Too few clean training examples (e.g. a very quiet instrument
        # where almost nothing clears the deadzone) -- stand down rather
        # than fit on a handful of rows.
        return signals

    X_train = feats.loc[train_idx, feature_cols].to_numpy()
    y_train = (forward_return_atr.loc[train_idx] > 0).astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        # Every training label came out the same direction -- nothing for
        # a binary classifier to actually discriminate.
        return signals

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    model.fit(X_train, y_train)

    # Predict for every bar from train_end onward with a complete feature
    # row -- this is the strategy's genuine out-of-sample segment; nothing
    # before train_end is ever given a non-zero signal.
    predict_mask = valid_features.copy()
    predict_mask.iloc[:train_end] = False
    predict_idx = df.index[predict_mask]
    if len(predict_idx) == 0:
        return signals

    X_predict = feats.loc[predict_idx, feature_cols].to_numpy()
    proba_up = model.predict_proba(X_predict)[:, 1]

    raw_signal = np.zeros(len(predict_idx), dtype=int)
    raw_signal[proba_up >= PROB_THRESHOLD] = 1
    raw_signal[proba_up <= (1 - PROB_THRESHOLD)] = -1
    signals.loc[predict_idx] = raw_signal

    stop_distance = (atr * STOP_ATR_MULT).reindex(df.index)
    target_distance = (atr * TARGET_ATR_MULT).reindex(df.index)
    signals.attrs["stop_loss_distance"] = stop_distance
    signals.attrs["take_profit_distance"] = target_distance
    return signals
