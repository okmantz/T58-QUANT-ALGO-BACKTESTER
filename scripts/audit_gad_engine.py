#!/usr/bin/env python3
"""Reproduce the NQ Gap-Aligned Opening Drive audit numbers: skip units, adaptive-risk throttle, intended-vs-actual risk, and the
fixed-horizon 30/60-calendar-day pass probability from purchase. Usage: python scripts/audit_gad_engine.py [--out reports/prop30]"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.backtest.adaptive_risk import adaptive_from_settings  # noqa: E402
from app.backtest.engine import run_backtest  # noqa: E402
from app.backtest.risk import RiskConfig  # noqa: E402
from app.data.importer import import_csv  # noqa: E402
from app.data.instrument_specs import apply_instrument_spec  # noqa: E402
from app.prop.presets import get_preset  # noqa: E402
from app.prop.thirty_day import TradeLedger, evaluate_thirty_day  # noqa: E402
from app.strategy.python import PythonStrategy  # noqa: E402

STRATEGY = ROOT / "strategies/python/MNQ1! Gap Aligned Drive RW Tight Stop 1m.py"
DATA = ROOT / "data/raw/NQ1!/futures_NQ.F_1m.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/prop30")
    ap.add_argument("--risk", type=float, default=126.45)
    a = ap.parse_args()
    df = import_csv(str(DATA)).dataframe.reset_index(drop=True)
    rules = get_preset("lucid_50k").to_prop_rules()
    base = apply_instrument_spec(RiskConfig(), "MNQ")
    risk = RiskConfig(**{**base.__dict__, "initial_balance": rules.account_size, "risk_mode": "fixed", "risk_value": a.risk})
    strat = PythonStrategy(str(STRATEGY))
    out: dict = {}
    for label, ad in (("adaptive_balanced", adaptive_from_settings(rules)), ("no_adaptive", None)):
        bt = run_backtest(df, strat, risk, adaptive_risk=ad, lookahead_check="skip")
        tr = bt.trades
        ss = bt.equity_curve.attrs.get("sizing_summary", {})
        mult = np.array([t.adaptive_risk_multiplier for t in tr], float)
        act = np.array([bool(t.adaptive_risk_rules_active) for t in tr])
        intended = np.array([t.intended_risk_dollars or np.nan for t in tr], float)
        actual = np.array([(t.initial_risk or np.nan) * t.size for t in tr], float)
        fired = Counter(r for t in tr for r in t.adaptive_risk_rules_active)
        led = TradeLedger.from_trades(tr, risk, "GAD", "MNQ")
        res = evaluate_thirty_day(led, rules, data_start=df["timestamp"].iloc[0], data_end=df["timestamp"].iloc[-1])
        out[label] = {
            "trades": len(tr), "net_pnl": float(sum(t.pnl for t in tr)),
            "sizing_summary": {k: v for k, v in ss.items() if k != "skip_reasons"},
            "adaptive_mean_multiplier": float(np.nanmean(mult)), "adaptive_share_active": float(act.mean()), "rules_fired": dict(fired),
            "intended_risk_mean": float(np.nanmean(intended)), "actual_stop_risk_mean": float(np.nanmean(actual)),
            "exit_reasons": pd.Series([t.exit_reason for t in tr]).value_counts().to_dict(),
            "thirty_day": res.to_dict(),
        }
        print(f"== {label}: {len(tr)} trades, mean multiplier {np.nanmean(mult):.2f}, active {act.mean() * 100:.0f}%, "
              f"skip_rate {ss.get('skip_rate')}, intended ${np.nanmean(intended):.2f} vs actual ${np.nanmean(actual):.2f}")
        print(res.render())
    p = ROOT / a.out
    p.mkdir(parents=True, exist_ok=True)
    (p / "gad_audit.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
