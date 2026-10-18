"""Lab 5 -- Statistical Arbitrage & Spread Lab.

Tests spreads between two markets (related instruments, or the two legs of a futures calendar spread) for
cointegration and mean reversion, estimates a rolling hedge ratio, and runs an honest spread mean-reversion backtest
with an in-sample / out-of-sample split.

The piece that matters most in practice is the **spread-stability monitor**. A spread that is merely *stretched* (it
will probably snap back) is a very different animal from one that has *structurally broken* (the relationship has
changed and "cheap" keeps getting cheaper). The monitor tells them apart by tracking, in rolling windows:

* the Engle-Granger / ADF statistic of the window's own residual (does mean reversion still exist?),
* the hedge-ratio drift versus its own history (is the relationship's slope moving?),
* the half-life of mean reversion versus the full-sample baseline (is reversion slowing to a crawl?),
* how long |z| has stayed extreme (a stretched spread comes back; a broken one stays).

Critical values for the cointegration test are the asymptotic MacKinnon values for a two-variable regression with a
constant (1% -3.90, 5% -3.34, 10% -3.04) -- an approximation, stated as such.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.research_labs import common as C

LAB = "stat_arb"
TITLE = "Statistical Arbitrage & Spread Lab"
ASSUMPTIONS = [
    "Spread (pair mode) = log(A) - beta*log(B); beta from OLS in-sample for the static test, and from a ROLLING window "
    "(past data only) for the trading backtest. Calendar mode uses the raw price difference front - back, hedge ratio 1.",
    "Cointegration = Engle-Granger two-step (OLS residual + ADF with 1 augmentation lag). Critical values are the asymptotic "
    "MacKinnon values (-3.90 / -3.34 / -3.04 at 1% / 5% / 10%) -- approximate for short samples.",
    "Half-life from the AR(1) regression of the spread change on the lagged spread level.",
    "Backtest: enter at |z| >= entry, exit at |z| <= exit, hard stop at |z| >= stop; positions use information up to the "
    "bar's close and earn the NEXT bar's spread change; a round-trip cost in bps of notional is charged on every change.",
    "Stability monitor windows use past data only; it classifies the latest state as normal, stretched (relationship "
    "intact) or structurally broken (mean reversion evidence gone).",
]
EG_CRIT = {"1%": -3.90, "5%": -3.34, "10%": -3.04}


def adf_stat(x: np.ndarray, lags: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30 + lags:
        return float("nan")
    dx = np.diff(x)
    y = dx[lags:]
    cols = [x[lags:-1] if lags else x[:-1], np.ones(len(y))]
    for i in range(1, lags + 1):
        cols.append(dx[lags - i:len(dx) - i])
    X = np.column_stack(cols)
    n = min(len(y), len(X))
    y, X = y[:n], X[:n]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return float("nan")
    se = math.sqrt(max(cov[0, 0], 1e-300))
    return float(beta[0] / se)


def half_life(spread: np.ndarray) -> float:
    s = np.asarray(spread, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 30:
        return float("nan")
    lag, d = s[:-1], np.diff(s)
    X = np.column_stack([lag - lag.mean(), np.ones(len(lag))])
    b = np.linalg.lstsq(X, d, rcond=None)[0][0]
    if b >= 0:
        return float("inf")
    return float(-math.log(2) / math.log(1 + b)) if b > -1 else 1.0


def ols_hedge(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    beta, alpha = np.polyfit(x, y, 1)
    return float(beta), float(alpha)


def cointegration(y: np.ndarray, x: np.ndarray) -> dict:
    beta, alpha = ols_hedge(y, x)
    resid = y - beta * x - alpha
    stat = adf_stat(resid, lags=1)
    verdict = ("cointegrated at 1%" if stat < EG_CRIT["1%"] else "cointegrated at 5%" if stat < EG_CRIT["5%"]
               else "weak evidence (10%)" if stat < EG_CRIT["10%"] else "no evidence of cointegration")
    hl = half_life(resid)
    return {"beta": beta, "alpha": alpha, "adf_stat": stat, "critical_values": EG_CRIT, "verdict": verdict,
            "half_life_bars": hl if math.isfinite(hl) else None, "correlation": float(np.corrcoef(y, x)[0, 1])}


def build_spread(a: pd.Series, b: pd.Series, mode: str = "pair", window: int = 250) -> pd.DataFrame:
    """Causal spread table: ``beta`` (rolling, known at the bar's close) and ``spread``/``z``."""
    out = pd.DataFrame({"a": a, "b": b})
    if mode == "calendar":
        out["beta"] = 1.0
        out["spread"] = out["a"] - out["b"]
    else:
        la, lb = np.log(out["a"]), np.log(out["b"])
        cov = la.rolling(window, min_periods=window // 2).cov(lb)
        var = lb.rolling(window, min_periods=window // 2).var()
        out["beta"] = (cov / var.replace(0, np.nan))
        out["spread"] = la - out["beta"] * lb
    out["z"] = np.nan
    return out


def _z(spread: pd.Series, win: int) -> pd.Series:
    m = spread.rolling(win, min_periods=max(10, win // 2)).mean()
    s = spread.rolling(win, min_periods=max(10, win // 2)).std()
    return (spread - m) / s.replace(0, np.nan)


def backtest_spread(tab: pd.DataFrame, mode: str, z_win: int, entry: float = 2.0, exit_: float = 0.5, stop: float = 4.0,
                    cost_bps: float = 2.0) -> pd.DataFrame:
    z = _z(tab["spread"], z_win).to_numpy()
    n = len(tab)
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        zi = z[i]
        if not math.isfinite(zi):
            pos[i] = 0.0
            cur = 0.0
            continue
        if cur == 0.0:
            if zi >= entry:
                cur = -1.0                      # spread rich -> short it
            elif zi <= -entry:
                cur = 1.0
        else:
            if abs(zi) <= exit_ or abs(zi) >= stop or (cur > 0 and zi >= 0) or (cur < 0 and zi <= 0):
                cur = 0.0
        pos[i] = cur
    if mode == "calendar":
        d_spread = tab["a"].diff() - tab["b"].diff()
        notional = (tab["a"].abs() + tab["b"].abs())
    else:
        beta = tab["beta"].shift(1)
        d_spread = np.log(tab["a"]).diff() - beta * np.log(tab["b"]).diff()
        notional = 1.0 + beta.abs()
    held = pd.Series(pos, index=tab.index).shift(1).fillna(0.0)
    gross = held * d_spread
    chg = held.diff().abs().fillna(0.0)
    cost = chg * cost_bps * 1e-4 * notional
    net = (gross - cost).fillna(0.0)
    return pd.DataFrame({"pos": held, "net": net, "gross": gross.fillna(0.0), "z": z}, index=tab.index)


def _perf(net: pd.Series, pos: pd.Series) -> dict:
    n = int(len(net))
    if n < 20:
        return {"bars": n}
    t = C.one_sample_t(net.to_numpy())
    eq = net.cumsum()
    dd = float((eq.cummax() - eq).max())
    trades = int(((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum())
    active = net[pos != 0]
    return {"bars": n, "trades": trades, "total_return": float(net.sum()), "mean_bps_per_bar": float(net.mean() * 1e4),
            "t_stat": t["t"], "p": t["p"], "max_drawdown": dd, "hit_rate_active_bars": float((active > 0).mean()) if len(active) else None,
            "time_in_market": float((pos != 0).mean())}


def stability_monitor(tab: pd.DataFrame, mode: str, win: int = 250, step: int = 25) -> dict:
    """Rolling classification of whether a stretched spread is still a mean-reverting relationship."""
    la, lb = (np.log(tab["a"]).to_numpy(), np.log(tab["b"]).to_numpy()) if mode != "calendar" else (tab["a"].to_numpy(), tab["b"].to_numpy())
    rows = []
    for end in range(win, len(tab) + 1, step):
        sl = slice(end - win, end)
        if mode == "calendar":
            resid = la[sl] - lb[sl]
            beta = 1.0
        else:
            beta, alpha = ols_hedge(la[sl], lb[sl])
            resid = la[sl] - beta * lb[sl] - alpha
        rows.append({"end": end - 1, "adf": adf_stat(resid, 1), "beta": beta, "half_life": half_life(resid),
                     "z_last": float((resid[-1] - resid.mean()) / (resid.std() or np.nan))})
    if not rows:
        return {"status": "insufficient history", "timeline": []}
    t = pd.DataFrame(rows)
    base_hl = float(t["half_life"].replace(np.inf, np.nan).median())
    beta_mu, beta_sd = float(t["beta"].mean()), float(t["beta"].std() or 1e-9)
    t["beta_drift_sigma"] = (t["beta"] - beta_mu).abs() / max(beta_sd, 1e-9) if mode != "calendar" else 0.0
    t["hl_ratio"] = t["half_life"] / base_hl if base_hl and math.isfinite(base_hl) else np.nan

    def classify(r) -> str:
        broken_adf = not (r["adf"] < -2.0) if math.isfinite(r["adf"]) else True
        drift = (r["beta_drift_sigma"] > 3.0) if mode != "calendar" else False
        slow = (r["hl_ratio"] > 3.0) if math.isfinite(r["hl_ratio"]) else False
        extreme = abs(r["z_last"]) >= 2.0 if math.isfinite(r["z_last"]) else False
        if (broken_adf and (drift or slow)) or (drift and slow):
            return "structurally broken"
        if extreme and r["adf"] < -3.0 and not drift and not slow:
            return "stretched (relationship intact)"
        return "normal" if not broken_adf else "weak mean reversion"

    t["status"] = t.apply(classify, axis=1)
    latest = t.iloc[-1]
    recent = t.tail(3)["status"].tolist()
    status = "structurally broken" if recent.count("structurally broken") >= 2 else latest["status"]
    timeline = t.assign(end=t["end"].astype(int))[["end", "adf", "beta", "half_life", "z_last", "status"]]
    timeline = timeline.replace([np.inf, -np.inf], np.nan)
    if len(timeline) > 200:
        timeline = timeline.iloc[:: int(math.ceil(len(timeline) / 200))]
    advice = {"stretched (relationship intact)": "Stretched but the statistical relationship still holds: fading it is consistent with the evidence.",
              "structurally broken": "The mean-reversion evidence has gone (ADF/hedge-ratio/half-life): do NOT keep averaging into this spread.",
              "weak mean reversion": "Mean reversion is weak right now; reduce size or wait for the ADF to recover.",
              "normal": "Spread is inside its usual range with a healthy relationship."}.get(status, "")
    return {"status": status, "advice": advice, "latest": {k: (None if (isinstance(v, float) and not math.isfinite(v)) else v)
                                                           for k, v in latest.to_dict().items()},
            "baseline_half_life": base_hl if math.isfinite(base_hl) else None, "timeline": timeline.to_dict(orient="records")}


def run(series: dict[str, pd.DataFrame], *, mode: str = "pair", freq: str = "1h", hedge_window: int = 250, z_entry: float = 2.0,
        z_exit: float = 0.5, z_stop: float = 4.0, cost_bps: float = 2.0, oos_fraction: float = 0.3, seed: int = C.DEFAULT_SEED,
        save: bool = True) -> C.ExperimentRecord:
    if len(series) != 2:
        raise C.ResearchLabError("Pick exactly two markets: A (first) and B (second). For a calendar spread, A = front contract, B = back contract.")
    names = list(series)
    cols = {}
    for nm, df in series.items():
        d = C.prepare_ohlcv(df)
        cols[nm] = d.set_index("timestamp")["close"].resample(freq).last().dropna()
    j = pd.concat(cols, axis=1, join="inner").dropna()
    if len(j) < max(300, hedge_window + 100):
        raise C.ResearchLabError(f"Only {len(j)} overlapping {freq} bars -- need at least {max(300, hedge_window + 100)}. Use a finer bucket or longer data.")
    j.columns = ["a", "b"]
    if (j <= 0).any().any() and mode != "calendar":
        raise C.ResearchLabError("Pair mode uses log prices and needs positive prices; use calendar mode for spreads that can be <= 0.")
    cut = int(len(j) * (1 - min(max(oos_fraction, 0.1), 0.6)))
    ya, xb = ((np.log(j["a"]), np.log(j["b"])) if mode != "calendar" else (j["a"], j["b"]))
    coint_is = cointegration(ya.to_numpy()[:cut], xb.to_numpy()[:cut]) if mode != "calendar" else {
        "beta": 1.0, "alpha": 0.0, "adf_stat": adf_stat((j["a"] - j["b"]).to_numpy()[:cut], 1), "critical_values": EG_CRIT,
        "half_life_bars": half_life((j["a"] - j["b"]).to_numpy()[:cut]), "correlation": float(np.corrcoef(j["a"][:cut], j["b"][:cut])[0, 1])}
    if mode == "calendar":
        s = coint_is["adf_stat"]
        coint_is["verdict"] = ("stationary at 1%" if s < -3.43 else "stationary at 5%" if s < -2.86 else "no evidence the spread is stationary")
    coint_oos = None
    if mode != "calendar":
        res_oos = ya.to_numpy()[cut:] - coint_is["beta"] * xb.to_numpy()[cut:] - coint_is["alpha"]
        coint_oos = {"adf_stat_with_is_hedge": adf_stat(res_oos, 1), "half_life_bars": half_life(res_oos)}
    tab = build_spread(j["a"], j["b"], mode, hedge_window)
    hl = coint_is.get("half_life_bars")
    z_win = int(min(max(20, 3 * hl if hl and math.isfinite(hl) else 60), 400))
    bt = backtest_spread(tab, mode, z_win, z_entry, z_exit, z_stop, cost_bps)
    perf_is = _perf(bt["net"].iloc[:cut], bt["pos"].iloc[:cut])
    perf_oos = _perf(bt["net"].iloc[cut:], bt["pos"].iloc[cut:])
    mon = stability_monitor(tab.assign(a=j["a"], b=j["b"]), mode, win=min(hedge_window, max(100, len(j) // 6)))
    ok_oos = (perf_oos.get("total_return") or 0) > 0 and (perf_oos.get("trades") or 0) >= 5
    ok_is = (perf_is.get("total_return") or 0) > 0
    warnings = []
    if perf_oos.get("trades", 0) < 10:
        warnings.append(f"Only {perf_oos.get('trades', 0)} out-of-sample trades -- the out-of-sample verdict is weak.")
    if mon["status"] == "structurally broken":
        warnings.append("The stability monitor flags this spread as structurally broken RIGHT NOW.")
    headline = (f"{coint_is['verdict']}; half-life {hl:.1f} bars; " if hl and math.isfinite(hl) else f"{coint_is['verdict']}; ") + \
               f"OOS {'profitable' if ok_oos else 'not profitable'} (IS {'profitable' if ok_is else 'not profitable'}); state: {mon['status']}"
    results = {"headline": headline, "markets": names, "mode": mode, "freq": freq, "bars": int(len(j)), "cointegration_in_sample": coint_is,
               "cointegration_out_of_sample": coint_oos, "backtest": {"in_sample": perf_is, "out_of_sample": perf_oos,
                                                                       "z_window": z_win, "entry": z_entry, "exit": z_exit, "stop": z_stop, "cost_bps": cost_bps},
               "stability_monitor": mon,
               "spread_tail": [{"i": int(i), "spread": None if not math.isfinite(v) else float(v)} for i, v in
                               zip(range(max(0, len(tab) - 300), len(tab)), tab["spread"].iloc[-300:].to_numpy())]}
    primary = next(iter(series.values()))
    return C.record_from(LAB, TITLE, {"mode": mode, "freq": freq, "hedge_window": hedge_window, "z_entry": z_entry, "z_exit": z_exit,
                                      "z_stop": z_stop, "cost_bps": cost_bps, "oos_fraction": oos_fraction}, ASSUMPTIONS, primary,
                         results, warnings, seed, " / ".join(names), save=save)


def spread_zscore_feature(a: pd.DataFrame, b: pd.DataFrame, freq: str = "1h", window: int = 250, mode: str = "pair",
                          z_window: int = 60) -> pd.Series:
    """Causal z-score of the spread aligned to A's bars (for strategies / indicators_v10)."""
    ca = C.prepare_ohlcv(a).set_index("timestamp")["close"].resample(freq).last().dropna()
    cb = C.prepare_ohlcv(b).set_index("timestamp")["close"].resample(freq).last().dropna()
    j = pd.concat([ca, cb], axis=1, join="inner").dropna()
    j.columns = ["a", "b"]
    tab = build_spread(j["a"], j["b"], mode, window)
    z = _z(tab["spread"], z_window)
    z.index = z.index + pd.Timedelta(freq)
    base = C.prepare_ohlcv(a)
    return pd.Series(z.reindex(base["timestamp"], method="ffill").to_numpy(), index=base.index)
