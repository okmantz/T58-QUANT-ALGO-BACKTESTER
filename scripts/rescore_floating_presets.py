#!/usr/bin/env python3
"""Re-score the 30-day leaderboard for every preset whose daily loss is measured on FLOATING P&L.

Why (2026-11-11): before the ``lock_day`` booking fix in ``app/prop/account.py`` / ``vector_account.py`` a floating daily-loss breach returned
DAY_LOCKED before ``balance += pnl``, so a stop-out larger than the daily limit cost $0. Every result produced under such a preset
(daily_loss_basis="floating"; e.g. Lucid, Apex Intraday, FTMO, FundedNext, Futures Desk) is suspect. This script re-runs the reproducible
prop30 pipeline (shared INST costs, order-type slippage, stress mode, roll exclusion, honest evaluation) for each of them.

    python scripts/rescore_floating_presets.py                       # every floating-daily-loss preset
    python scripts/rescore_floating_presets.py --presets lucid_50k --no-time-limit
    python scripts/rescore_floating_presets.py --list

Outputs ``<out>/rescore_<preset>.json`` per preset and ``<out>/rescore_floating_presets.md`` (one summary table).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from app.data.importer import import_csv  # noqa: E402
from app.prop.presets import get_preset, list_presets  # noqa: E402
from app.research.prop30_pipeline import Prop30Config, run  # noqa: E402

FILES = {"NQ": "data/raw/NQ1!/futures_NQ.F_1m.parquet", "ES": "data/raw/ES1!/futures_ES.F_1m.parquet",
         "GC": "data/raw/GC1!/futures_GC.F_1m.parquet", "CL": "data/raw/CL1!/futures_CL.F_1m.parquet"}


def floating_daily_loss_presets() -> list:
    """Presets whose daily loss is measured on floating P&L AND that actually have a daily loss limit (< 100% of the account)."""
    return [p for p in list_presets() if getattr(p, "daily_loss_basis", "realized") == "floating" and p.daily_loss_limit_pct < 100.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/prop30")
    ap.add_argument("--presets", nargs="*", default=None, help="preset keys (default: every floating-daily-loss preset)")
    ap.add_argument("--instruments", nargs="*", default=["NQ", "ES", "GC"])
    ap.add_argument("--list", action="store_true", help="print the affected presets and exit")
    ap.add_argument("--risk-usd", type=float, default=None, help="per-trade risk (default: 1% of the account)")
    a = ap.parse_args()
    keys = a.presets or [p.key for p in floating_daily_loss_presets()]
    if a.list:
        for k in keys:
            print(k)
        return 0
    frames = {}
    for k in a.instruments:
        pth = ROOT / FILES[k]
        if pth.exists():
            frames[k] = import_csv(str(pth)).dataframe.reset_index(drop=True)
        else:
            print(f"[skip] {k}: {pth} not found", file=sys.stderr)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in keys:
        pre = get_preset(key)
        risk = a.risk_usd if a.risk_usd is not None else round(0.01 * pre.account_size, 2)
        maxc = int(pre.max_contracts or 4)
        res = run(frames, Prop30Config(preset=key, risk_usd=risk, max_contracts=maxc))
        (out / f"rescore_{key}.json").write_text(json.dumps(res, indent=2, default=str))
        top = (res.get("scan", {}).get("top") or [None])[0]
        rows.append((key, risk, res.get("verdict"), top))
        print(f"{key}: {res.get('verdict')}")
    md = ["# Floating-daily-loss presets re-scored after the lock_day booking fix", "",
          "| Preset | Risk $ | Verdict | Best scanned cell | P(pass 30d) | 95% lower | Trusted at 1+2 ticks |", "|---|---|---|---|---|---|---|"]
    for key, risk, verdict, top in rows:
        if top:
            p = "n/a" if top["p_pass_30"] is None else f"{top['p_pass_30'] * 100:.1f}%"
            lb = "n/a" if top["lower_bound"] is None else f"{top['lower_bound'] * 100:.1f}%"
            md.append(f"| {key} | {risk:,.0f} | {verdict} | {top['cell']} | {p} | {lb} | {top.get('trusted')} |")
        else:
            md.append(f"| {key} | {risk:,.0f} | {verdict} | (none) | n/a | n/a | n/a |")
    (out / "rescore_floating_presets.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
