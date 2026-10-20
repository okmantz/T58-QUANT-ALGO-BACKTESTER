"""Register and run the report's pre-registered family tests on YOUR data.

    python -m scripts.run_preregistered_families NQ=data/nq_15m.csv ES=data/es_15m.csv [GC=data/gc_15m.csv ...]

The first instrument is each family's PRIMARY, the second its REPLICATION instrument. Every family is registered (fingerprinted,
timestamped, one trial booked) BEFORE anything runs, then the batch is run once at 2x costs with Benjamini-Hochberg correction.
Results print as a table and are saved to reports/preregistered_results.json. Nothing is re-tuned after a fail.
"""
import json
import sys
from pathlib import Path

from app.backtest.risk import RiskConfig
from app.data.importer import import_csv
from app.data.instrument_specs import apply_any_instrument_spec, micro_equivalent
from app.research.preregistered_tests import preregister, run_preregistered
from app.search import strategy_space as sps  # noqa: F401  (registers the v11 families)
from app.search.families_v11 import V11_FAMILY_NAMES, V11_REQUIRES_AUX


def main(argv: list[str]) -> int:
    pairs = [a.split("=", 1) for a in argv if "=" in a]
    if len(pairs) < 2:
        print(__doc__)
        return 2
    frames, risks = {}, {}
    for sym, path in pairs:
        res = import_csv(path)
        if not res.is_valid:
            print(f"{sym}: could not import {path}: {[i.message for i in res.issues][:2]}")
            return 1
        frames[sym.upper()] = res.dataframe
        # Edge tests are measured in R: size on the MICRO contract (point-for-point identical prices) so whole-contract rounding never skips signals.
        _m = micro_equivalent(sym.upper())
        risks[sym.upper()] = apply_any_instrument_spec(RiskConfig(initial_balance=100000, risk_mode="percent", risk_value=0.5), (_m.symbol if _m else sym.upper()))
    primary, replication = pairs[0][0].upper(), pairs[1][0].upper()
    regs = [preregister(f, primary, replication, note="v11 batch") for f in V11_FAMILY_NAMES if f not in V11_REQUIRES_AUX]
    print(f"Registered {len(regs)} test(s) on {primary} (replication {replication}); running once at 2x costs...")
    results = run_preregistered(regs, frames, risks, log=print)
    for r in results:
        print(f"{r.family:30s} n={r.n_trades:5d} meanR={r.mean_r if r.mean_r is None else round(r.mean_r, 3)} "
              f"p={r.p_value if r.p_value is None else round(r.p_value, 4)} -> {r.verdict}  {'; '.join(r.reasons)}")
    out = Path("reports") / "preregistered_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([r.to_dict() for r in results], indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
