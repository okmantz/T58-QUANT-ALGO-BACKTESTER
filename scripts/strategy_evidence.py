"""Evidence report for one strategy x instrument (2026-10 audit, build item 4).

For a strategy with FIXED, pre-registered parameters (nothing is optimised here, so there is no selection bias to
deflate -- but every strategy you try is still a trial; the report prints how many are in this batch):

  1. backtest over the full history and over a clean, never-tuned HOLDOUT (the last --holdout-frac of the bars)
  2. Monte Carlo prop-pass odds on the HOLDOUT trades (MAE/MFE bundled, reset_on_breach False) for a preset
  3. null-edge gate on the HOLDOUT: does the entry timing beat random timing with the same exits?
  4. EV verdict (p*q*V - c) using the holdout pass rate and the preset's fee-table entry
  5. pre-purchase checklist state (real-account check is NOT done by this script -- it needs your real trades)

    python scripts/strategy_evidence.py --strategy "strategies/python/ES1! Opening Range Breakout 5m.py" \
        --data data/raw/ES1!/futures_ES.F_1m.parquet --instrument ES --preset lucid_50k --out strategies/evidence

Writes <out>/<strategy stem>.md and .json. Losses are reported as losses.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def _stats_row(bt) -> dict:
    s = bt.statistics.to_dict() if bt.statistics is not None else {}
    keep = ("total_trades", "net_profit", "win_rate", "profit_factor", "expectancy", "max_drawdown", "sharpe_ratio")
    out = {k: s.get(k) for k in keep}
    out["n_trades_list"] = len(bt.trades)
    out["lookahead_check_failed"] = bool(getattr(bt, "lookahead_check_failed", False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--preset", default="lucid_50k")
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--mc-sims", type=int, default=2000)
    ap.add_argument("--null-seeds", type=int, default=100)
    ap.add_argument("--n-trials", type=int, default=6, help="strategies tried in this batch (for the deflation note)")
    ap.add_argument("--out", default=str(ROOT / "strategies" / "evidence"))
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    from app.backtest.engine import run_backtest
    from app.backtest.risk import RiskConfig
    from app.data.importer import import_csv
    from app.data.instrument_specs import apply_instrument_spec
    from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
    from app.orchestration.ev_verdict import evaluate_ev  # noqa: E402
    from app.prop.presets import get_fee_info, get_preset
    from app.strategy.python import PythonStrategy
    from app.validation.null_edge_gate import run_null_edge_gate

    res = import_csv(a.data)
    if res.dataframe is None:
        print("could not load data", file=sys.stderr)
        return 2
    df = res.dataframe.reset_index(drop=True)
    rules = get_preset(a.preset).to_prop_rules()
    risk = apply_instrument_spec(RiskConfig(), a.instrument)
    risk = RiskConfig(**{**risk.__dict__, "initial_balance": rules.account_size})
    strat = PythonStrategy(a.strategy)

    cut = int(len(df) * (1 - a.holdout_frac))
    warm = 3000
    hold_df = df.iloc[max(0, cut - warm):].reset_index(drop=True)
    hold_start = pd.Timestamp(df["timestamp"].iloc[cut])

    full = run_backtest(df, strat, risk)
    hold = run_backtest(hold_df, strat, risk)
    htr = [t for t in hold.trades if pd.Timestamp(t.entry_time).tz_localize(None) >= hold_start.tz_localize(None)] if hold.trades else []
    rep = {"strategy": Path(a.strategy).name, "instrument": a.instrument, "preset": a.preset,
           "data_start": str(df["timestamp"].iloc[0]), "data_end": str(df["timestamp"].iloc[-1]),
           "holdout_start": str(hold_start), "full": _stats_row(full), "holdout": _stats_row(hold),
           "holdout_trades_n": len(htr)}

    from app.orchestration.audit_layers import holdout_pass_rate_mc
    p, mc = holdout_pass_rate_mc(htr, rules, sims=a.mc_sims, seed=7)
    if mc is None:
        rep["mc_holdout"] = {"skipped": f"only {len(htr)} holdout trades (too few for a verdict)"}
    else:
        rep["mc_holdout"] = {"pass_rate_pct": round(p * 100, 2), "first_payout_pct": mc.first_payout_probability,
                             "risk_of_ruin_pct": mc.risk_of_ruin_pct, "expected_payout": mc.expected_payout, "n_simulations": a.mc_sims}

    null = run_null_edge_gate(hold_df, strat, risk, n_seeds=a.null_seeds, seed=3)
    rep["null_edge"] = null.to_dict()

    fee = get_fee_info(a.preset)
    if fee is not None and mc is not None:
        target_usd = float(rules.account_size) * float(rules.evaluation_profit_target_pct) / 100.0
        fp = mc.first_payout_probability / 100.0
        gross = (mc.expected_payout / fp) if fp > 0 else None
        ev = evaluate_ev(
            legacy_verdict="n/a", holdout_pass_rate=p, window_payout_rate=fp,
            payout_value=min(gross, target_usd) if gross else target_usd, challenge_price=fee.challenge_price,
            reset_price=fee.reset_price, profit_split_pct=fee.profit_split_pct, n_trials=a.n_trials,
            n_effective=float(len(htr)), rules_verified=bool(getattr(rules, "rules_verified", False)),
            lookahead_ok=not rep["holdout"]["lookahead_check_failed"], holdout_clean=True,
            automation_confirmed=fee.written_confirmation)
        rep["ev"] = {"label": ev.label, "reasons": ev.reasons}
    else:
        rep["ev"] = "EV not computed (no fee-table entry or too few holdout trades)"

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.strategy).stem
    (out / f"{stem}.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    def _r(d):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}
    f, h = _r(rep["full"]), _r(rep["holdout"])
    md = [f"# Evidence: {stem}", "",
          f"Data {rep['data_start'][:10]} to {rep['data_end'][:10]}; clean holdout from {rep['holdout_start'][:10]} (parameters fixed in advance, never tuned).", "",
          "| Slice | Trades | Net profit | Win rate | Profit factor | Max DD |", "|---|---|---|---|---|---|",
          f"| Full | {f['total_trades']} | {f['net_profit']} | {f['win_rate']} | {f['profit_factor']} | {f['max_drawdown']} |",
          f"| Holdout | {h['total_trades']} | {h['net_profit']} | {h['win_rate']} | {h['profit_factor']} | {h['max_drawdown']} |", "",
          f"Monte Carlo ({a.preset}, holdout trades): {rep['mc_holdout']}", "", "```", null.render(), "```", "",
          f"EV: {rep['ev']}", "",
          "Real-account check: NOT DONE (needs your real trades; see data/real_account/README.md).",
          "No strategy is cleared for a fee until null-edge passes, the real-account check agrees and the firm rules are confirmed in writing."]
    (out / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
