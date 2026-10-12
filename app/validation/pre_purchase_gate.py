"""Pre-purchase gate: the three things that must be true before a challenge fee is spent (2026-10 audit).

    1. null-edge gate passed          (entry timing beats random timing)          -- app.validation.null_edge_gate
    2. real-account check agrees      (the model reproduces a real account)       -- data/real_account/last_check.json
    3. EV verdict is FORWARD-TEST CANDIDATE or HIGH CONFIDENCE                    -- app.orchestration.ev_verdict
       and the firm confirmed in writing that the automation is allowed          -- FIRM_FEES[...].written_confirmation

``pre_purchase_report`` returns every check with a reason; ``ready`` is True only if all pass. Missing evidence fails.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.orchestration.ev_verdict import FORWARD_TEST_CANDIDATE, HIGH_CONFIDENCE

REAL_CHECK_PATH = Path(__file__).resolve().parents[2] / "data" / "real_account" / "last_check.json"
MAX_REAL_CHECK_AGE_DAYS = 90


@dataclass
class PrePurchaseReport:
    ready: bool
    checks: list = field(default_factory=list)   # [(name, ok, detail)]

    def render(self) -> str:
        lines = [f"PRE-PURCHASE GATE: {'CLEARED' if self.ready else 'NOT CLEARED'}"]
        lines += [f"  [{'x' if ok else ' '}] {n}: {d}" for n, ok, d in self.checks]
        return "\n".join(lines)


def load_real_check(path: Path | None = None) -> dict | None:
    try:
        return json.loads((path or REAL_CHECK_PATH).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def pre_purchase_report(*, ev_verdict: dict | None, null_result: dict | None, preset_key: str | None, strategy_path: str | None = None,
                        real_check_path: Path | None = None) -> PrePurchaseReport:
    checks = []
    # 1 null-edge
    if null_result and null_result.get("p_value") is not None:
        ok = bool(null_result.get("passed"))
        checks.append(("null-edge gate", ok, f"p = {null_result['p_value']:.3f} (limit {null_result.get('p_max', 0.10):.2f}) over {null_result.get('n_seeds')} random-timing runs"))
    else:
        checks.append(("null-edge gate", False, "not run (run app.validation.null_edge_gate.run_null_edge_gate)"))
    # 2 real-account check
    rc = load_real_check(real_check_path)
    if rc is None:
        checks.append(("real-account check", False, "no data/real_account/last_check.json (run scripts/run_real_account_check.py on ~20 real sessions)"))
    else:
        ok = bool(rc.get("agrees"))
        detail = f"{rc.get('n_matched')}/{rc.get('n_real')} real trades matched ({(rc.get('match_rate') or 0) * 100:.0f}%), checked {str(rc.get('checked_at'))[:10]}"
        if strategy_path and rc.get("strategy") and Path(rc["strategy"]).name != Path(strategy_path).name:
            ok, detail = False, detail + f"; the check was for a DIFFERENT strategy ({Path(rc['strategy']).name})"
        if (rc.get("n_real") or 0) < 20:
            ok, detail = False, detail + "; fewer than 20 real trades is weak evidence"
        checks.append(("real-account check", ok, detail))
    # 3 EV verdict
    lab = (ev_verdict or {}).get("label")
    checks.append(("EV verdict", lab in (FORWARD_TEST_CANDIDATE, HIGH_CONFIDENCE), f"{lab or 'not computed'}"))
    # 4 firm confirmation
    try:
        from app.prop.presets import get_fee_info
        fee = get_fee_info(preset_key or "")
    except Exception:  # noqa: BLE001
        fee = None
    if fee is None:
        checks.append(("firm automation confirmed in writing", False, "no fee row for this preset"))
    else:
        checks.append(("firm automation confirmed in writing", bool(fee.written_confirmation),
                       "written confirmation on file" if fee.written_confirmation else "ask the firm's support in writing, then set written_confirmation=True in app/prop/presets.py"))
    return PrePurchaseReport(all(ok for _, ok, _ in checks), checks)
