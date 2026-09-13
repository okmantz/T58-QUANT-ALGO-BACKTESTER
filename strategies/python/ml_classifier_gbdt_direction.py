"""
ML Classifier Direction (Gradient-Boosted Trees) Strategy.

The second ML family in this repo, added alongside
ml_classifier_direction.py's LogisticRegression baseline once that file's
own walk-forward retraining was verified and fixed (see its WALK-FORWARD
RETRAINING section and app/strategy/python.py's WALK-FORWARD RETRAIN
CONTEXT protocol) -- adding a second, higher-capacity ML family before
that fix existed would have compounded the same silent-waste bug instead
of just inheriting a working pattern.

Uses LightGBM (LGBMClassifier) rather than XGBoost: equivalent gradient-
boosted-tree capability, a materially lighter/faster install (pure
prebuilt wheels, no separate native-library step), and it's the one this
environment could actually verify installs and trains cleanly. Optional
in spirit exactly like scikit-learn is for ml_classifier_direction.py --
importing this file without lightgbm installed raises one clear
"pip install lightgbm" error instead of a bare ImportError traceback.

WHY A SEPARATE FILE, NOT A "MODEL" PARAMETER ON THE EXISTING ONE
------------------------------------------------------------------
A gradient-boosted-tree model has a genuinely different hyperparameter
surface (n_estimators, max_depth, learning_rate, num_leaves) than a
logistic regression's (mostly just C), a materially larger capacity to
overfit a small dataset, and a different feature-importance story worth
inspecting on its own -- treating it as this repo's OTHER strategy family
(a fixed, named rule with tunable numeric constants -- see
app.optimize.code_parameter_space) rather than bolting a model-choice
switch onto the existing file keeps both files simple, inspectable, and
independently falsifiable, consistent with every other family in this
app being one hypothesis, not a bundle of them behind a flag.

SAME NO-LOOKAHEAD DISCIPLINE AS ml_classifier_direction.py
-------------------------------------------------------------
Identical chronological-split contract: fit once on a strict prefix using
only labels whose forward-return window is itself inside that prefix,
every feature causal (backward-looking only), signals emitted only from
train_end onward. See that file's own WALK-FORWARD SAFETY section for the
full reasoning -- not repeated here verbatim to avoid the two files
silently drifting out of sync in their prose while their actual split
logic (deliberately) stays identical.

WALK-FORWARD RETRAINING
--------------------------
RETRAIN_PER_FOLD = True below, honoring `df.attrs["wf_train_end_index"]`
exactly like ml_classifier_direction.py -- see app/strategy/python.py's
WALK-FORWARD RETRAIN CONTEXT section for the protocol itself. This family
gets real per-fold retraining from day one rather than needing its own
follow-up fix later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "ML Classifier Direction (LightGBM gradient-boosted trees)"

RETRAIN_PER_FOLD = True   # see app/strategy/python.py's WALK-FORWARD RETRAIN CONTEXT section

# --- tunable parameters -----------------------------------------------
TRAIN_FRAC = 0.6            # fraction of the dataset used to fit the model; rest is out-of-sample
LABEL_HORIZON = 5           # bars ahead the model predicts the direction of
LABEL_DEADZONE_ATR = 0.15   # forward moves smaller than this many ATRs are excluded from training
PROB_THRESHOLD = 0.58       # only trade when predicted P(up)/P(down) clears this
N_ESTIMATORS = 150          # number of boosting rounds (trees)
MAX_DEPTH = 4               # per-tree depth cap -- kept shallow on purpose; see module docstring
LEARNING_RATE = 0.05
NUM_LEAVES = 15
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
    """Same feature set as ml_classifier_direction.py, plus two extra
    causal features (candle-body ratio and a longer lookback return) a
    tree model can exploit via splits/interactions that a linear model
    can't -- every column here is still computed causally (rolling/ewm/
    diff/pct_change only ever look backward from each row)."""
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    macd_line = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    atr = _atr(df, ATR_PERIOD)
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    avg_volume = volume.rolling(20, min_periods=20).mean()
    body = (close - df["open"]).abs()
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)

    feats = pd.DataFrame(index=df.index)
    feats["ret_1"] = close.pct_change(1)
    feats["ret_5"] = close.pct_change(5)
    feats["ret_10"] = close.pct_change(10)
    feats["ret_20"] = close.pct_change(20)
    feats["rsi"] = _rsi(close, RSI_PERIOD)
    feats["macd_hist"] = macd_line - macd_signal
    feats["atr_norm"] = atr / close
    feats["dist_ema20"] = (close - ema20) / ema20
    feats["dist_ema50"] = (close - ema50) / ema50
    feats["vol_rel"] = (volume / avg_volume.replace(0, np.nan)).fillna(1.0)
    feats["body_ratio"] = (body / bar_range).fillna(0.5)
    feats["_atr"] = atr  # kept only to size the label deadzone/risk mgmt below, dropped before fitting
    return feats


def generate_signals(df: pd.DataFrame) -> pd.Series:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "ml_classifier_gbdt_direction.py requires lightgbm (pip install lightgbm) -- "
            "it isn't a hard dependency of the rest of this app, only of this one strategy file."
        ) from exc

    n = len(df)
    signals = pd.Series(0, index=df.index, dtype=int)
    if n < 200:
        return signals

    feats = _build_features(df)
    feature_cols = [c for c in feats.columns if c != "_atr"]
    atr = feats["_atr"]

    train_end = int(n * TRAIN_FRAC)
    train_end = max(min(train_end, n - LABEL_HORIZON - 1), RSI_PERIOD + 50)
    explicit_train_end = df.attrs.get("wf_train_end_index")
    if explicit_train_end is not None:
        # Real walk-forward retrain context -- see module docstring and
        # app/strategy/python.py's WALK-FORWARD RETRAIN CONTEXT protocol.
        train_end = max(min(int(explicit_train_end), n - LABEL_HORIZON - 1), RSI_PERIOD + 50)

    forward_return = df["close"].shift(-LABEL_HORIZON) / df["close"] - 1.0
    forward_return_atr = forward_return * df["close"] / atr.replace(0, np.nan)

    valid_features = feats[feature_cols].notna().all(axis=1)
    trainable = valid_features & (forward_return_atr.notna()) & (pd.Series(np.arange(n), index=df.index) + LABEL_HORIZON < train_end)
    trainable &= forward_return_atr.abs() >= LABEL_DEADZONE_ATR

    train_idx = df.index[trainable]
    if len(train_idx) < 100:
        return signals

    X_train = feats.loc[train_idx, feature_cols].to_numpy()
    y_train = (forward_return_atr.loc[train_idx] > 0).astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        return signals

    model = LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        random_state=42,
        verbosity=-1,
        min_child_samples=max(5, len(train_idx) // 100),
    )
    model.fit(X_train, y_train)

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
