"""
Regression test for the "Evolution Lab ignores my real prop account / risk
settings" bug: /evolution/start and /evolution/multi-instrument/start used
to build RiskConfig/PropRules from ONLY the `initial_balance` form field --
every other setting (profit target, daily loss limit, max drawdown, risk
mode/value, pip size, spread, slippage, commission, max trades/day) silently
fell back to RiskConfig/PropRules' own dataclass defaults no matter what was
posted. That meant Evolution Lab could search and score candidates under a
completely different (and invisible) rule set than Full Pipeline actually
checks them against afterward -- looking like a Full Pipeline bug when it
was really a config mismatch between the two pages.

These tests monkeypatch EvolutionRunner / MultiInstrumentEvolutionGroup to
record the RiskConfig/PropRules they were constructed with (never actually
starting a background run), then POST a full set of non-default prop/risk
fields and assert every one of them made it through -- not just
initial_balance/account_size.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import app.web.server as server_module
from app.web.server import app
from app.orchestration.resource_guard import (
    HEAVY_JOB_GUARD, JOB_EVOLUTION_LAB, JOB_MULTI_INSTRUMENT_EVOLUTION,
)

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

# A full set of intentionally non-default prop/risk values -- close to the
# real numbers reported in the field (a $50k account, 6% target, 2%/4%
# drawdown, 0.5% risk per trade, pip size 1.0 for an ES1!-style futures
# contract) so this test fails loudly if any one of them silently reverts
# to RiskConfig/PropRules' own defaults instead of the posted value.
_CUSTOM_RISK_FIELDS = {
    "account_size": "50000",
    "initial_balance": "50000",
    "profit_target": "6",
    "daily_loss": "2",
    "max_dd": "4",
    "max_trades_day": "6",
    "risk_mode": "percent",
    "risk_value": "0.5",
    "commission": "2.5",
    "slippage_pips": "0.25",
    "spread_pips": "0.5",
    "pip_size": "1.0",
}

_FAST_EVO_FIELDS = {
    "population_size": "4", "elite_keep": "1", "mc_sims": "10", "max_generations": "1",
}


class _FakeRunner:
    """Stands in for app.evolution.engine.EvolutionRunner: records the
    exact RiskConfig/PropRules/EvolutionConfig it was built with and never
    actually starts a background thread, so the route under test can be
    checked in milliseconds instead of running a real (even tiny)
    generation."""
    last_instance = None

    def __init__(self, df, risk, rules, cfg, progress_cb=None):
        self.df = df
        self.risk = risk
        self.rules = rules
        self.cfg = cfg
        self.is_running = False
        self.leaderboard = []
        _FakeRunner.last_instance = self

    def start(self):
        self.is_running = False  # "finishes" immediately

    def stop_and_wait(self, timeout=5.0):
        return True


class _FakeMultiGroup:
    """Same idea as _FakeRunner but for MultiInstrumentEvolutionGroup."""
    last_instance = None

    def __init__(self, group_id, jobs, risk, rules, base_cfg):
        self.group_id = group_id
        self.jobs = jobs
        self.risk = risk
        self.rules = rules
        self.base_cfg = base_cfg
        _FakeMultiGroup.last_instance = self

    def start_all(self):
        pass

    def is_running(self):
        return False

    def status(self):
        return {"running": False, "labels": [j.instrument + "/" + j.timeframe for j in self.jobs]}


@pytest.fixture(autouse=True)
def _isolated_evolution_state(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "_EVOLUTION_RUNNER", None, raising=False)
    monkeypatch.setattr(server_module, "EvolutionRunner", _FakeRunner)
    monkeypatch.setattr(server_module, "MultiInstrumentEvolutionGroup", _FakeMultiGroup)
    yield
    HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
    HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_EVOLUTION)
    _FakeRunner.last_instance = None
    _FakeMultiGroup.last_instance = None


def _assert_custom_risk_and_rules(risk, rules):
    assert rules.account_size == 50000
    assert rules.evaluation_profit_target_pct == 6
    assert rules.daily_loss_limit_pct == 2
    assert rules.max_drawdown_pct == 4
    assert risk.initial_balance == 50000
    assert risk.risk_mode == "percent"
    assert risk.risk_value == 0.5
    assert risk.max_trades_per_day == 6
    assert risk.commission_per_trade == 2.5
    assert risk.slippage_pips == 0.25
    assert risk.spread_pips == 0.5
    assert risk.pip_size == 1.0


def test_evolution_start_uses_full_posted_prop_and_risk_fields():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            **_CUSTOM_RISK_FIELDS,
            **_FAST_EVO_FIELDS,
        }
        r = client.post("/evolution/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302

    runner = _FakeRunner.last_instance
    assert runner is not None, "EvolutionRunner was never constructed"
    _assert_custom_risk_and_rules(runner.risk, runner.rules)


def test_evolution_start_still_defaults_when_fields_omitted():
    """Guards the other direction too: omitting the new fields entirely
    (an old bookmark, a script posting only the historical minimum) must
    not crash -- it should fall back to the same defaults as before."""
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {"csv_file": (f, "EURUSD_5M_sample.csv"), **_FAST_EVO_FIELDS}
        r = client.post("/evolution/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302

    runner = _FakeRunner.last_instance
    assert runner.risk.pip_size == 0.0001
    assert runner.rules.evaluation_profit_target_pct == 8
    assert runner.rules.max_drawdown_pct == 10


def test_multi_instrument_evolution_start_uses_full_posted_prop_and_risk_fields(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(server_module, "get_raw_data_dir", lambda: raw_dir)
    for name in ("a.csv", "b.csv"):
        (raw_dir / name).write_bytes(SAMPLE_CSV.read_bytes())

    client = app.test_client()
    r = client.post(
        "/evolution/multi-instrument/start",
        data={"datasets": ["a.csv", "b.csv"], **_CUSTOM_RISK_FIELDS, **_FAST_EVO_FIELDS},
    )
    assert r.status_code == 302

    group = _FakeMultiGroup.last_instance
    assert group is not None, "MultiInstrumentEvolutionGroup was never constructed"
    _assert_custom_risk_and_rules(group.risk, group.rules)
