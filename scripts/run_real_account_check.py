"""Run the real-account check (2026-10 audit): same strategy, same sessions as a REAL account, then compare.

    python scripts/run_real_account_check.py --real data/real_account/real_trades.csv \
        --data data/raw/NQ1!/futures_NQ.F_1m.parquet --strategy "strategies/python/NQ1! Opening Drive 5m.py" \
        --instrument NQ --preset lucid_50k

Writes data/real_account/last_check.json and last_check.md. Exit code 0 = agrees, 1 = does not agree, 2 = could not run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.validation.real_account_check import compare_trade_lists  # noqa: E402


def _trades_to_frame(trades) -> pd.DataFrame:
    return pd.DataFrame([{
        "entry_time": t.entry_time, "exit_time": t.exit_time, "direction": t.direction,
        "entry_price": t.entry_price, "exit_price": t.exit_price, "pnl": t.pnl, "size": t.size,
    } for t in trades])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", required=True)
    ap.add_argument("--data", required=True, help="OHLCV csv/parquet covering the real trades")
    ap.add_argument("--strategy", required=True, help=".py strategy file")
    ap.add_argument("--instrument", default="", help="symbol for spec/costs, e.g. ES, NQ, GC")
    ap.add_argument("--preset", default=None, help="prop preset key, e.g. lucid_50k")
    ap.add_argument("--pad-days", type=int, default=3)
    ap.add_argument("--cost-tolerance", type=float, default=60.0)
    ap.add_argument("--time-tolerance", default="10min")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "real_account"))
    a = ap.parse_args()

    from app.backtest.engine import run_backtest
    from app.backtest.risk import RiskConfig
    from app.data.importer import import_csv
    from app.data.instrument_specs import apply_instrument_spec
    from app.strategy.python import PythonStrategy

    real = pd.read_csv(a.real)
    real.columns = [c.strip().lower().replace(" ", "_") for c in real.columns]
    t0 = pd.to_datetime(real["entry_time"], utc=True).min().tz_convert(None) - pd.Timedelta(days=a.pad_days)
    t1 = pd.to_datetime(real["entry_time"], utc=True).max().tz_convert(None) + pd.Timedelta(days=a.pad_days)

    res = import_csv(a.data)
    if res.dataframe is None:
        print("could not load OHLCV:", [i.message for i in res.issues if i.level == "error"], file=sys.stderr)
        return 2
    df = res.dataframe
    ts = pd.to_datetime(df["timestamp"])
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None) if ts.dt.tz is not None else ts
    warm = t0 - pd.Timedelta(days=10)   # indicator warm-up
    df = df[(ts >= warm) & (ts <= t1)].reset_index(drop=True)
    if len(df) < 100:
        print(f"only {len(df)} bars in the real-trade window; wrong data file?", file=sys.stderr)
        return 2

    risk = RiskConfig()
    if a.instrument:
        risk = apply_instrument_spec(risk, a.instrument)
    if a.preset:
        from app.prop.presets import get_preset
        rules = get_preset(a.preset).to_prop_rules()
        risk = RiskConfig(**{**risk.__dict__, "initial_balance": rules.account_size})
    bt = run_backtest(df, PythonStrategy(a.strategy), risk)
    sim = _trades_to_frame([t for t in bt.trades if t0 <= pd.Timestamp(t.entry_time) <= t1])
    if sim.empty:
        print("the strategy took no trades in that window", file=sys.stderr)
        return 2
    sim["entry_time"] = pd.to_datetime(sim["entry_time"]).dt.tz_localize("UTC") if pd.to_datetime(sim["entry_time"]).dt.tz is None else sim["entry_time"]
    sim["exit_time"] = pd.to_datetime(sim["exit_time"]).dt.tz_localize("UTC") if pd.to_datetime(sim["exit_time"]).dt.tz is None else sim["exit_time"]
    rep = compare_trade_lists(real, sim, time_tolerance=a.time_tolerance, cost_tolerance_dollars=a.cost_tolerance)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "agrees": rep.agrees, "n_real": rep.n_real, "n_sim": rep.n_sim, "n_matched": rep.n_matched, "match_rate": rep.match_rate,
        "median_abs_pnl_diff": rep.median_abs_pnl_diff, "daily_agreement": rep.daily_agreement, "notes": rep.notes,
        "strategy": a.strategy, "data": a.data, "preset": a.preset, "checked_at": pd.Timestamp.utcnow().isoformat(),
    }
    (out / "last_check.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    (out / "last_check.md").write_text("```\n" + rep.render() + "\n```\n", encoding="utf-8")
    print(rep.render())
    return 0 if rep.agrees else 1


if __name__ == "__main__":
    raise SystemExit(main())
