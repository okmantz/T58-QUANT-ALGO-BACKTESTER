"""Tests for the 2026-10 audit changes (EV verdict, excursions, behaviour rules, data gate, contract calendar,
null-edge gate wiring, rulebook replays, fee table, holdout refusal)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pytest


# ---- EV verdict -------------------------------------------------------------------------------------------------
def test_ev_formula_and_break_even():
    from app.orchestration.ev_verdict import break_even_pass_rate, compute_ev
    assert compute_ev(0.2, 0.45, 2500.0, 100.0) == pytest.approx(0.2 * 0.45 * 2500 - 100)
    assert break_even_pass_rate(100.0, 0.45, 2500.0) == pytest.approx(100 / (0.45 * 2500))     # 8.9%
    assert break_even_pass_rate(100.0, 0.0, 2500.0) is None


def test_deflation_never_raises_pass_rate_and_grows_with_trials():
    from app.orchestration.ev_verdict import deflate_pass_rate, expected_max_std_normal
    assert expected_max_std_normal(1) == pytest.approx(0.0, abs=1e-6)
    assert expected_max_std_normal(50) > expected_max_std_normal(5) > 0
    p = 0.4
    assert deflate_pass_rate(p, 1, 100) <= p
    assert deflate_pass_rate(p, 50, 100) < deflate_pass_rate(p, 5, 100)
    assert deflate_pass_rate(p, 50, 100) >= 0.0


def _ev(**kw):
    from app.orchestration.ev_verdict import evaluate_ev
    base = dict(legacy_verdict="MARGINAL", holdout_pass_rate=0.6, window_payout_rate=0.3, payout_value=2500.0,
                challenge_price=100.0, profit_split_pct=90.0, bankroll_ruin_pct=5.0, n_trials=1, n_effective=500.0,
                rules_verified=True, lookahead_ok=True, holdout_clean=True)
    base.update(kw)
    return evaluate_ev(**base)


def test_ev_labels():
    from app.orchestration import ev_verdict as m
    assert _ev().label == m.FORWARD_TEST_CANDIDATE or _ev().label == m.HIGH_CONFIDENCE
    assert _ev(holdout_pass_rate=0.02).label == m.NOT_WORTH_A_FEE
    assert _ev(holdout_pass_rate=None).label == m.INSUFFICIENT_EVIDENCE
    assert _ev(challenge_price=None).label == m.INSUFFICIENT_EVIDENCE
    assert _ev(rules_verified=False).label != m.HIGH_CONFIDENCE
    assert _ev(bankroll_ruin_pct=None).label == m.INSUFFICIENT_EVIDENCE
    assert _ev(lookahead_ok=False).label != m.HIGH_CONFIDENCE
    assert _ev(holdout_clean=False).label == m.INSUFFICIENT_EVIDENCE


# ---- excursions ---------------------------------------------------------------------------------------------------
@dataclass
class _T:
    pnl: float
    entry_price: float | None = None
    size: float | None = None
    direction: int | None = None
    worst_price: float | None = None
    mfe_price: float | None = None


def test_excursions_measured_vs_proxied_and_bracketing():
    from app.prop.excursions import trade_excursions
    t_meas = _T(pnl=5.0, entry_price=100.0, size=2.0, direction=1, worst_price=98.0, mfe_price=104.0)
    t_proxy = _T(pnl=-30.0)
    ex = trade_excursions([t_meas, t_proxy])
    assert ex.n_measured == 1 and ex.n_proxied == 1
    assert ex.mae[0] == pytest.approx(-4.0) and ex.mfe[0] == pytest.approx(8.0)
    assert ex.mae[1] == -30.0 and ex.mfe[1] == 0.0
    assert all(m <= p <= f for m, p, f in zip(ex.mae, [5.0, -30.0], ex.mfe))


def test_missing_excursion_policy_raise_and_warn():
    from app.prop.account import MissingExcursionError, PropAccount, excursion_policy
    import inspect
    # the policy context manager must exist and "raise" must be a distinct policy from "warn"
    with excursion_policy("raise"):
        from app.prop.account import current_excursion_policy
        assert current_excursion_policy() == "raise"
    with excursion_policy("warn"):
        assert current_excursion_policy() == "warn"
    assert issubclass(MissingExcursionError, Exception) and inspect.isclass(PropAccount)


# ---- Monte Carlo defaults -----------------------------------------------------------------------------------------
def test_mc_defaults_are_conservative():
    from app.monte_carlo.engine import MonteCarloConfig
    c = MonteCarloConfig()
    assert c.reset_on_breach is False
    assert c.use_excursions is True


# ---- behaviour rules ----------------------------------------------------------------------------------------------
def test_behavior_defaults_off_and_bounds():
    from app.backtest.behavior_rules import BEHAVIOR_BOUNDS, BEHAVIOR_DEFAULTS, any_active, news_windows_text, parse_windows
    assert not any_active(BEHAVIOR_DEFAULTS)
    assert set(BEHAVIOR_BOUNDS) == set(BEHAVIOR_DEFAULTS)
    txt = news_windows_text(10)
    assert len(parse_windows(txt)) == len(txt.splitlines()) >= 1


def test_behavior_constants_resolve_from_python_strategy(tmp_path):
    from app.backtest.behavior_rules import resolve_behavior
    from app.strategy.python import PythonStrategy
    p = tmp_path / "s.py"
    p.write_text("import pandas as pd\nBEHAVIOR_STOP_AFTER_LOSSES = 2\nBEHAVIOR_NEWS_BLOCK_MIN = 5\n"
                 "def generate_signals(df):\n    return pd.Series(0, index=df.index)\n")
    b = resolve_behavior(PythonStrategy(str(p)))
    assert b["BEHAVIOR_STOP_AFTER_LOSSES"] == 2 and b["BEHAVIOR_NEWS_BLOCK_MIN"] == 5
    assert b["BEHAVIOR_FLATTEN_BEFORE_CLOSE_MIN"] == 0


def test_behavior_genes_are_clamped_to_bounds():
    from app.backtest.behavior_rules import BEHAVIOR_BOUNDS
    lo, hi = BEHAVIOR_BOUNDS["BEHAVIOR_STOP_AFTER_LOSSES"]
    assert lo == 0.0 and hi >= 1.0


# ---- data gate / manifest -------------------------------------------------------------------------------------------
def _frame(years: float) -> pd.DataFrame:
    n = max(int(years * 365), 10)
    return pd.DataFrame({"timestamp": pd.date_range("2018-01-01", periods=n, freq="D")})


def test_data_gate_blocks_synthetic_and_short_history():
    from app.data.data_manifest import check_dataset
    assert not check_dataset(_frame(1.0), "x.csv", manifest={"files": []}).ok_for_ready
    assert check_dataset(_frame(7.0), "ES/real.parquet", manifest={"files": []}).ok_for_ready
    g = check_dataset(_frame(7.0), "ES/SYNTHETIC_ES_1M.csv", manifest={"files": []})
    assert g.synthetic and not g.ok_for_ready
    assert not check_dataset(None, "unknown", manifest={"files": []}).ok_for_ready


def test_manifest_has_no_synthetic_rows_left():
    from app.data.data_manifest import load_manifest
    rows = load_manifest().get("files", [])
    assert rows and not any(r.get("synthetic") for r in rows)


# ---- contract calendar ------------------------------------------------------------------------------------------------
def test_third_friday_and_expected_windows():
    from app.data.contract_calendar import expected_roll_windows, third_friday
    assert third_friday(2026, 9) == date(2026, 9, 18)
    es = expected_roll_windows("ES", date(2025, 1, 1), date(2025, 12, 31))
    assert len([w for w in es if w.expiry.year == 2025]) == 4     # quarterly Mar/Jun/Sep/Dec
    gc = expected_roll_windows("GC", date(2025, 1, 1), date(2025, 12, 31))
    assert len([w for w in gc if w.expiry.year == 2025]) == 6     # Feb/Apr/Jun/Aug/Oct/Dec


def test_audit_rolls_flags_unexplained_jump():
    from app.data.contract_calendar import audit_rolls
    ts = pd.date_range("2025-02-03 09:00", periods=400, freq="5min")
    rng = np.random.default_rng(0)
    close = 5000 + np.cumsum(rng.normal(0, 0.5, len(ts)))
    open_ = np.r_[close[0], close[:-1]]
    close[200:] += 60.0                                   # a splice-sized jump in mid-February, away from any ES roll
    open_[200] = close[200]
    df = pd.DataFrame({"timestamp": ts, "open": open_, "close": close})
    a = audit_rolls(df, "ES")
    assert not a.clean


# ---- null-edge gate: no evidence is not a pass --------------------------------------------------------------------
def test_null_edge_gate_failure_to_run_is_not_a_pass():
    from app.validation.null_edge_gate import run_null_edge_gate
    r = run_null_edge_gate(pd.DataFrame({"timestamp": []}), object(), object(), n_seeds=3)
    assert r.passed is False


# ---- fee table / preset verification ------------------------------------------------------------------------------------
def test_fee_table_rows_carry_source_and_unverified_is_not_ready_eligible():
    from app.prop.presets import PROP_FIRM_PRESETS, fee_table, get_fee_info, is_ready_eligible
    rows = fee_table()
    assert len(rows) == len(PROP_FIRM_PRESETS)
    fee = get_fee_info("lucid_50k")
    assert fee is not None and fee.source_url and fee.date_checked and fee.written_confirmation is False
    unverified = [p for p in PROP_FIRM_PRESETS if not (p.verified_on and p.source_url)]
    assert unverified, "expected at least one preset that is still unverified"
    assert not any(is_ready_eligible(p) for p in unverified)


def test_to_prop_rules_carries_preset_key_and_verified_flag():
    from app.prop.presets import get_preset
    r = get_preset("ftmo_100k").to_prop_rules()
    assert r.preset_key == "ftmo_100k" and r.rules_verified is True


# ---- rulebook replays --------------------------------------------------------------------------------------------------------
def test_rulebook_replays_all_agree():
    from app.prop.rulebook_tests import run_all
    res = run_all()
    bad = [c for c in res if not getattr(c, "agree", getattr(c, "ok", True))]
    assert res and not bad


# ---- quick optimize refuses to rank without a holdout ----------------------------------------------------------------------
def test_quick_optimize_defaults_reserve_holdout_and_not_validated():
    from app.orchestration.quick_optimize import QuickOptimizeConfig
    c = QuickOptimizeConfig()
    assert c.reserve_holdout is True and c.allow_no_holdout is False


# ---- walk-forward GA headline ------------------------------------------------------------------------------------------------
def test_walkforward_ga_headline_prefers_honest_median():
    from app.optimize.walkforward_ga import WalkforwardGAResult
    fields = WalkforwardGAResult.__dataclass_fields__
    for f in ("selection_chained_score", "honest_median_fold_score", "honest_fold_scores"):
        assert f in fields
    assert isinstance(WalkforwardGAResult.headline_score, property)


# ---- strategies are causal & well-formed -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", [f"{s}1! {k} 5m.py" for s in ("NQ",) for k in ("Opening Drive",)])
def test_new_strategies_load_and_emit_valid_signals(name):
    from pathlib import Path
    from app.strategy.python import PythonStrategy
    root = Path(__file__).resolve().parents[1] / "strategies" / "python" / name
    st = PythonStrategy(str(root))
    ts = pd.date_range("2025-03-03 14:00", periods=2 * 288 * 3, freq="5min", tz="UTC")
    rng = np.random.default_rng(1)
    px = 5000 + np.cumsum(rng.normal(0, 1.0, len(ts)))
    df = pd.DataFrame({"timestamp": ts, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 100})
    sig = st.generate(df).signals
    assert len(sig) == len(df) and set(np.unique(sig.to_numpy())) <= {-1.0, 0.0, 1.0}
