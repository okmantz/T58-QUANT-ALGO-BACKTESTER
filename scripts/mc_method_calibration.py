"""Which Monte Carlo resampling recovers the TRUE per-attempt pass odds best?
Ground truth is known: trades come from a process we control; truth = pass rate of fresh long histories of that same process."""
# Run from the repo root: python scripts/mc_method_calibration.py 30 out.csv
import sys, time, warnings, json
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules, simulate_account

RULES = PropRules(account_size=100_000.0, evaluation_profit_target_pct=8.0, daily_loss_limit_pct=3.0, max_drawdown_pct=6.0, drawdown_type="trailing",
                  consistency_rule_pct=None, min_trading_days=5)
R = 500.0                    # $ risk per trade (0.5%)

def process(kind, n_days, rng):
    """Return (pnls, dates). 2 trades/day on average."""
    days = pd.bdate_range("2022-01-03", periods=n_days)
    per_day = rng.poisson(2.0, n_days).clip(0, 6)
    dates = np.repeat(days.values, per_day)
    n = len(dates)
    if kind == "iid":
        win = rng.random(n) < 0.46
    elif kind == "regime":                         # good/bad regimes lasting ~12 trading days
        reg = np.zeros(n_days, bool); s = rng.random() < .5
        for i in range(n_days):
            if rng.random() < 1/12: s = not s
            reg[i] = s
        p = np.where(np.repeat(reg, per_day), 0.56, 0.35)
        win = rng.random(n) < p
    elif kind == "daily_shock":                    # a shared day-level shock moves every trade of the day
        shock = np.repeat(rng.normal(0, 0.12, n_days), per_day)
        win = rng.random(n) < np.clip(0.45 + shock, 0.05, 0.95)
    elif kind == "streaky":                         # outcome persistence (AR on sign)
        win = np.zeros(n, bool); prev = False
        for i in range(n):
            p = 0.52 if prev else 0.40
            prev = rng.random() < p; win[i] = prev
    pn = np.where(win, 1.6 * R, -R) * (1 + rng.normal(0, 0.05, n))
    return pn, dates

def mk_trades(pn, dates):
    out = []
    eq = 100_000.0
    for p, d in zip(pn, dates):
        t0 = pd.Timestamp(d) + pd.Timedelta(hours=10); eq += p
        out.append(Trade(entry_time=t0, exit_time=t0 + pd.Timedelta(minutes=45), direction=1, entry_price=5000.0, exit_price=5000.0 + p / 50,
                         size=1.0, pnl=float(p), pnl_pct=float(p) / 1000.0, exit_reason="stop_loss" if p < 0 else "take_profit",
                         commission=0.0, equity_after=eq, initial_risk=10.0))
    return out

def truth(kind, rng, n_hist=60, n_days=500):
    ok = tot = 0
    for _ in range(n_hist):
        pn, dates = process(kind, n_days, rng)
        res = simulate_account(list(pn), list(dates), RULES, reset_on_breach=True)
        ok += res.attempts_passed; tot += res.total_attempts
    return 100.0 * ok / max(tot, 1)

def mc(trades, method, bdays=3, bsize=None, sims=3000):
    cfg = MonteCarloConfig(method=method, n_simulations=sims, random_seed=11, reset_on_breach=True, day_block_days=bdays, block_size=bsize)
    return float(run_monte_carlo(trades, RULES, cfg).per_attempt_pass_probability)

if __name__ == "__main__":
    kinds = ["iid", "regime", "daily_shock", "streaky"]
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    variants = {"trade blocks (auto)": ("block_bootstrap", 3, None), "day blocks 1d": ("day_block_bootstrap", 1, None),
                "day blocks 3d": ("day_block_bootstrap", 3, None), "day blocks 5d": ("day_block_bootstrap", 5, None),
                "iid bootstrap": ("bootstrap", 3, None)}
    rows = []
    for ki, kind in enumerate(kinds):
        rng = np.random.default_rng(100 + ki)
        t0 = time.time(); tr = truth(kind, rng)
        errs = {k: [] for k in variants}; ests = {k: [] for k in variants}
        for r in range(reps):
            pn, dates = process(kind, 220, rng)               # ~9-10 months of history: a typical sample
            trades = mk_trades(pn, dates)
            for name, (m, bd, bs) in variants.items():
                e = mc(trades, m, bd, bs); ests[name].append(e); errs[name].append(e - tr)
        for name in variants:
            a = np.array(errs[name])
            rows.append({"process": kind, "truth": round(tr, 1), "method": name, "mean_est": round(float(np.mean(ests[name])), 1), "bias": round(float(a.mean()), 1),
                         "mae": round(float(np.abs(a).mean()), 1), "rmse": round(float(np.sqrt((a ** 2).mean())), 1)})
        print(kind, "truth", round(tr, 1), f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nOVERALL (mean over processes)")
    print(df.groupby("method")[["bias", "mae", "rmse"]].agg(lambda s: s.abs().mean() if s.name == "bias" else s.mean()).round(2).sort_values("mae").to_string())
    df.to_csv(sys.argv[2] if len(sys.argv) > 2 else "mc_calib.csv", index=False)
