"""v9.4: prop-firm percentages must follow Owen's accounting.

His example, on a 50k account with a $3,000 eval target: +$300, -$150,
+$3,000 passes the eval; funded +$100 then -$2,000 blows it; reset to a
fresh 50k; pass a second eval; funded +$3,000 pays out. At that point
eval pass = 100% (2 of 2 attempted) and first payout = 50% (1 of the 2
FUNDED accounts) -- attempts that blew their eval never had a payout
chance and must not dilute the payout rate.

Pre-fix, per_attempt_payout_probability divided by ALL attempts
(including eval busts), and under reset_on_breach the report headlined
the chain-level "did >=1 attempt in a long rebuy chain ever pass"
numbers instead of the per-attempt ones.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRISTINE = Path.home() / "workspace" / "tmp-pristine"

from app.prop.simulator import AccountSimResult, PropRules, simulate_account  # noqa: E402


def _owen_rules() -> PropRules:
    return PropRules(
        account_size=50_000.0,
        evaluation_profit_target_pct=6.0,   # exactly +$3,000
        daily_loss_limit_pct=3.0,           # $1,500/day -- the -$2,000 day blows the funded account
        consistency_rule_pct=None,
        min_trading_days=1,
        # Alpha-style payout gate: 5 winning days of $200+ before a payout.
        winning_days_for_payout=5,
        min_winning_day_profit=200.0,
        payout_frequency_days=1,
    )


# Eval 1: +300, -150, +3000 (pass). Funded 1: +100, -2000 (daily-loss
# blow, never 5 winning days -> no payout). Reset. Eval 2: +3000
# (pass). Funded 2: five +600 winning days -> first payout.
_OWEN_PNLS = [300.0, -150.0, 3000.0, 100.0, -2000.0, 3000.0,
              600.0, 600.0, 600.0, 600.0, 600.0]


def _dates(n: int):
    import pandas as pd
    start = pd.Timestamp("2024-01-01")
    return [start + pd.Timedelta(days=i) for i in range(n)]


def test_owen_example_attempt_ledger():
    result = simulate_account(_OWEN_PNLS, _dates(len(_OWEN_PNLS)), _owen_rules(),
                              reset_on_breach=True)
    assert result.total_attempts == 2
    assert result.attempts_passed == 2
    assert result.attempts_reached_payout == 1
    first, second = result.attempts
    assert first.passed_evaluation and not first.reached_first_payout
    assert second.passed_evaluation and second.reached_first_payout


def test_report_ledger_shows_owens_running_percentages():
    from app.reports.generator import _account_attempts_section

    result = simulate_account(_OWEN_PNLS, _dates(len(_OWEN_PNLS)), _owen_rules(),
                              reset_on_breach=True)
    from dataclasses import asdict
    html = _account_attempts_section([asdict(a) for a in result.attempts])
    assert "Account 1" in html and "Account 2" in html
    # After account 1: eval 100%, first payout 0%. After account 2:
    # eval still 100%, first payout 50% (1 of 2 funded).
    assert "<td>0%</td>" in html
    assert "<td>50%</td>" in html
    assert html.index("<td>0%</td>") < html.index("<td>50%</td>")


def test_mc_pooled_payout_uses_funded_denominator(monkeypatch):
    """Every simulated path: 4 eval attempts, 2 funded, 1 payout.
    Owen's accounting -> pass 50%, payout 50%. The pre-fix formula
    (payout / all attempts) reported 25%."""
    import app.monte_carlo.engine as engine
    from app.backtest.execution import Trade
    from app.prop.simulator import AttemptRecord
    import pandas as pd

    def fake_sim(pnls, dates, rules, **kwargs):
        return AccountSimResult(
            passed_evaluation=True, failed=False, failure_reason=None,
            failure_day_index=None, days_to_pass=3, first_payout_day_index=5,
            first_payout_amount=1500.0, final_balance=rules.account_size + 3000.0,
            attempts=[
                AttemptRecord(0, 0, 1, 2, True, False, None, True, 1500.0, 53000.0),
                AttemptRecord(1, 2, 3, 2, True, False, None, False, 0.0, 51000.0),
                AttemptRecord(2, 4, 5, 2, False, True, "daily_loss_limit", False, 0.0, 48000.0),
                AttemptRecord(3, 6, 7, 2, False, True, "daily_loss_limit", False, 0.0, 48000.0),
            ],
            total_attempts=4, attempts_passed=2, attempts_reached_payout=1,
        )

    monkeypatch.setattr(engine, "simulate_account", fake_sim)
    ts = pd.Timestamp("2024-01-01")
    trades = [
        Trade(entry_time=ts, exit_time=ts, direction=1, entry_price=100.0,
              exit_price=101.0, size=1.0, pnl=100.0, pnl_pct=1.0,
              exit_reason="signal", commission=0.0, equity_after=10000.0)
        for _ in range(8)
    ]
    result = engine.run_monte_carlo(
        trades, _owen_rules(),
        engine.MonteCarloConfig(n_simulations=10, random_seed=7, reset_on_breach=True),
    )
    assert result.per_attempt_pass_probability == pytest.approx(50.0)
    assert result.per_attempt_payout_probability == pytest.approx(50.0)
    assert result.per_attempt_failure_before_payout_probability == pytest.approx(50.0)


def _run_against(tree: Path):
    probe = "tests/test_v94_prop_owen_example.py"
    dest = tree / probe
    shutil.copy2(ROOT / probe, dest)
    try:
        return subprocess.run(
            [sys.executable, "-m", "pytest", probe, "-x", "-q",
             "-k", "not fails_on_upstream and not _run_against"],
            cwd=tree, capture_output=True, text=True, timeout=900)
    finally:
        dest.unlink(missing_ok=True)


@pytest.mark.skipif(not PRISTINE.exists(), reason="pristine upstream copy not available")
def test_fails_on_upstream_v92():
    r = _run_against(PRISTINE)
    assert r.returncode != 0, r.stdout[-1200:]
