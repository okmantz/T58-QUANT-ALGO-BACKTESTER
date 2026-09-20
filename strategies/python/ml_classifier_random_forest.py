"""
ML Classifier Direction (Random Forest) Strategy -- "Forest".

The third ML family in this repo, alongside ml_classifier_direction.py's
LogisticRegression baseline and ml_classifier_gbdt_direction.py's LightGBM
boosted trees. Added on request after watching a promo reel for a
consumer trading-bot product that markets its model as "N decision trees
voting" -- an ensemble of trees, each seeing a slightly different resample
of the data and a random subset of features at each split, whose majority
vote (for classification) is more stable than any single tree's opinion.
That reel showed the CONCEPT (a bagged-tree ensemble) and a "100 trees"
/ "95 trees" headline number, nothing about actual features, labels, or
training protocol -- everything below is this repo's own design, not a
reproduction of anyone else's code.

WHY A SEPARATE FILE, NOT A THIRD BRANCH ON THE EXISTING ONES
------------------------------------------------------------------
Same reasoning as ml_classifier_gbdt_direction.py's own module docstring:
a random forest has a genuinely different hyperparameter surface
(n_estimators, max_depth, min_samples_leaf, max_features) and a different
overfitting/variance profile than either logistic regression or gradient
boosting, and keeping it its own file/hypothesis keeps every family
independently falsifiable rather than bundling three models behind a
switch.

WHAT MAKES A FOREST DIFFERENT FROM THE GBDT FILE (WORTH INSPECTING, NOT
JUST A DROP-IN MODEL SWAP)
------------------------------------------------------------------
Bagging (random forest) and boosting (LightGBM) fail in different ways:
a forest's trees are independent and de-correlated by construction
(bootstrap-resampled rows + a random feature subset at every split), so
it tends to be more stable / less prone to memorizing a handful of noisy
rows than a boosted ensemble, at the cost of being slower to pick up a
genuinely sharp, narrow signal that boosting's sequential error-correction
would find faster. This file also exposes each feature's Gini importance
(`signals.attrs["feature_importances"]`) -- a random forest's importances
are usually the more stable/trustworthy of the two ensembles' importance
outputs (LightGBM's split/gain importances can be dominated by whichever
feature happened to get picked first in a correlated group), which is
useful for the same "don't trust a black box" instinct this repo already
applies everywhere else (see app.search.strategy_space's module docstring
on distrusting brute-force combinatorics over arbitrary indicators).

SAME NO-LOOKAHEAD DISCIPLINE AS THE OTHER TWO ML FILES
-------------------------------------------------------------
Identical chronological-split contract: fit once on a strict prefix,
using only labels whose forward-return window is itself inside that
prefix; every feature causal (backward-looking rolling/ewm windows only,
never a forward shift); signals emitted only from train_end onward, so
the training segment is always flat and this strategy can never appear to
trade profitably on data it fit itself to. See ml_classifier_direction.py's
own WALK-FORWARD SAFETY section for the full reasoning (not repeated
verbatim here, to avoid the three files' prose silently drifting out of
sync while their actual split logic stays identical by design).

WALK-FORWARD RETRAINING
--------------------------
RETRAIN_PER_FOLD = True below, honoring `df.attrs["wf_train_end_index"]`
exactly like the other two ML files -- see app/strategy/python.py's
WALK-FORWARD RETRAIN CONTEXT section for the protocol itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "ML Classifier Direction (Random Forest, \"Forest\")"

RETRAIN_PER_FOLD = True   # see app/strategy/python.py's WALK-FORWARD RETRAIN CONTEXT section

# --- tunable parameters -----------------------------------------------
TRAIN_FRAC = 0.6            # fraction of the dataset used to fit the model; rest is out-of-sample
LABEL_HORIZON = 5           # bars ahead the model predicts the direction of
LABEL_DEADZONE_ATR = 0.15   # forward moves smaller than this many ATRs are excluded from training
PROB_THRESHOLD = 0.58       # only trade when predicted P(up)/P(down) clears this (>0.5 by design)
RSI_PERIOD = 14
ATR_PERIOD = 14
STOP_ATR_MULT = 1.5
TARGET_ATR_MULT = 2.5

# Random-forest-specific hyperparameters. Kept modest relative to
# scikit-learn's defaults (max_depth capped, min_samples_leaf raised well
# above 1) because an unconstrained forest can memorize noise just as
# easily as any other high-capacity model -- more trees buys stability
# from averaging, it does not buy license to let each individual tree
# overfit. N_ESTIMATORS=100 deliberately echoes the "100 trees" style of
# ensemble size shown in the reel this file was inspired by.
N_ESTIMATORS = 100
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 30
MAX_FEATURES = "sqrt"       # de-correlates trees by only offering each split a random feature subset


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
    .diff()/.pct_change() only ever look backward from each row. A few
    extra REGIME features beyond ml_classifier_direction.py's baseline set
    (trend strength, a volatility-regime flag, and where price sits in its
    recent range) are included deliberately: a bagged-tree ensemble's real
    advantage over a single linear model is being able to carve up
    genuinely different market regimes into different leaves/trees rather
    than fitting one global rule, so it needs regime-describing inputs to
    actually exploit that, not just more of the same trend/momentum
    features a logistic regression already gets."""
    close = df["close"]
    high, low = df["high"], df["low"]
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    macd_line = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    atr = _atr(df, ATR_PERIOD)
    atr_baseline = atr.rolling(60, min_periods=30).mean()
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    avg_volume = volume.rolling(20, min_periods=20).mean()
    range_high20 = high.rolling(20, min_periods=20).max()
    range_low20 = low.rolling(20, min_periods=20).min()

    feats = pd.DataFrame(index=df.index)
    feats["ret_1"] = close.pct_change(1)
    feats["ret_5"] = close.pct_change(5)
    feats["ret_10"] = close.pct_change(10)
    feats["rsi"] = _rsi(close, RSI_PERIOD)
    feats["macd_hist"] = macd_line - macd_signal
    feats["atr_norm"] = atr / close
    feats["dist_ema20"] = (close - ema20) / ema20
    feats["dist_ema50"] = (close - ema50) / ema50
    feats["dist_ema200"] = (close - ema200) / ema200
    feats["trend_strength"] = (ema20 - ema50) / atr.replace(0, np.nan)   # regime feature: trend, ATR-normalized
    feats["vol_regime"] = (atr / atr_baseline.replace(0, np.nan)) - 1.0   # regime feature: expansion/contraction
    feats["range_position"] = ((close - range_low20) / (range_high20 - range_low20).replace(0, np.nan))  # 0..1
    feats["vol_rel"] = (volume / avg_volume.replace(0, np.nan)).fillna(1.0)
    feats["_atr"] = atr  # kept only to size the label deadzone/risk mgmt below, dropped before fitting
    return feats


def generate_signals(df: pd.DataFrame) -> pd.Series:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError(
            "ml_classifier_random_forest.py requires scikit-learn (pip install scikit-learn) -- "
            "it isn't a hard dependency of the rest of this app, only of this one strategy file."
        ) from exc

    n = len(df)
    signals = pd.Series(0, index=df.index, dtype=int)
    if n < 200:
        # Not enough bars for a meaningful train/test split -- stand down
        # rather than fit a forest on a handful of rows.
        return signals

    feats = _build_features(df)
    feature_cols = [c for c in feats.columns if c != "_atr"]
    atr = feats["_atr"]

    train_end = int(n * TRAIN_FRAC)
    train_end = max(min(train_end, n - LABEL_HORIZON - 1), RSI_PERIOD + 200)  # +200 for the ema200 warm-up
    explicit_train_end = df.attrs.get("wf_train_end_index")
    if explicit_train_end is not None:
        # Real walk-forward retrain context -- see module docstring and
        # app/strategy/python.py's WALK-FORWARD RETRAIN CONTEXT protocol.
        train_end = max(min(int(explicit_train_end), n - LABEL_HORIZON - 1), RSI_PERIOD + 200)

    forward_return = df["close"].shift(-LABEL_HORIZON) / df["close"] - 1.0
    forward_return_atr = forward_return * df["close"] / atr.replace(0, np.nan)

    valid_features = feats[feature_cols].notna().all(axis=1)
    # Only rows whose OWN forward window is fully inside the training
    # prefix may be used as training labels -- see module docstring.
    trainable = valid_features & (forward_return_atr.notna()) & (pd.Series(np.arange(n), index=df.index) + LABEL_HORIZON < train_end)
    trainable &= forward_return_atr.abs() >= LABEL_DEADZONE_ATR

    train_idx = df.index[trainable]
    if len(train_idx) < 150:
        # A forest with MIN_SAMPLES_LEAF=30 needs more clean examples than
        # the logistic-regression baseline to have any hope of forming
        # more than a couple of leaves per tree -- stand down rather than
        # fit on too few rows to be anything but noise.
        return signals

    X_train = feats.loc[train_idx, feature_cols].to_numpy()
    y_train = (forward_return_atr.loc[train_idx] > 0).astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        # Every training label came out the same direction -- nothing for
        # a binary classifier to actually discriminate.
        return signals

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        random_state=42,
        n_jobs=-1,
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
    # Not read by the backtest engine -- a bonus diagnostic for anyone
    # inspecting this strategy's report/logs, since a forest's Gini
    # importances are one of the more trustworthy "what is this model
    # actually using" signals available without a full SHAP pass.
    signals.attrs["feature_importances"] = dict(zip(feature_cols, model.feature_importances_.tolist()))
    return signals
