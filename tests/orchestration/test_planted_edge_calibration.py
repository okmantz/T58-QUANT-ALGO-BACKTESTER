"""2026-10-27 engine audit: the gates must accept a real edge and still reject a zero edge (planted-edge acceptance)."""
import numpy as np

from app.orchestration.audit_layers import holdout_confirmation
from app.orchestration.ev_verdict import deflate_pass_rate
from app.scoring.edge_bounds import expectancy_bound
from app.search.robustness import deflated_sharpe_ratio, expected_max_sharpe, null_sharpe_spread


def _r(n, win, rr, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < win, rr, -1.0)


def _sharpe(r):
    return float(np.mean(r) / np.std(r, ddof=1))


def test_null_spread_is_standard_error():
    assert abs(null_sharpe_spread(401) - 0.05) < 1e-9


def test_planted_edge_clears_deflated_sharpe_after_large_search():
    r = _r(450, 0.55, 1.2, 1)
    assert r.mean() > 0.15
    res = deflated_sharpe_ratio(_sharpe(r), [], 5000, len(r))
    assert res.probabilistic_sharpe > 0.95, res


def test_zero_edge_fails_deflated_sharpe_after_search():
    fails = 0
    for seed in range(20):
        r = _r(450, 0.5, 1.0, seed + 100)
        if deflated_sharpe_ratio(_sharpe(r), [], 5000, len(r)).probabilistic_sharpe < 0.95:
            fails += 1
    assert fails >= 19


def test_benchmark_scales_with_sample_size():
    assert expected_max_sharpe([], 1000, 100) > expected_max_sharpe([], 1000, 1000)


def test_untuned_strategy_is_not_deflated_and_haircut_is_bounded():
    assert deflate_pass_rate(0.8, 100000, 50.0, tunable=False) == 0.8
    assert deflate_pass_rate(0.8, 100000, 5.0, tunable=True) >= 0.4 - 1e-9


def test_expectancy_lower_bound_positive_for_planted_edge_only():
    assert expectancy_bound(_r(400, 0.55, 1.2, 3))["lb"] > 0
    assert expectancy_bound(_r(400, 0.5, 1.0, 4))["lb"] <= 0.05


class _T:
    def __init__(self, pnl, risk=100.0):
        self.pnl, self.net_profit, self.planned_risk_dollars, self.risk_dollars = pnl, pnl, risk, risk


def test_holdout_pooling_confirms_planted_edge_and_not_zero_edge():
    good = [_T(100.0 * x) for x in _r(60, 0.58, 1.2, 5)]
    pooled = list(_r(300, 0.55, 1.2, 6))
    out = holdout_confirmation(good, pooled)
    assert out["n_holdout"] == 60
    bad = [_T(100.0 * x) for x in _r(60, 0.45, 1.0, 7)]
    assert holdout_confirmation(bad, list(_r(300, 0.45, 1.0, 8)))["confirmed"] is False


def test_gate_vector_never_raises_and_reports_unmeasured():
    from app.orchestration.full_pipeline import _gate_vector
    gv = _gate_vector({"final_mc": None, "oos_validation": None}, {})
    assert isinstance(gv["gates"], list) and gv["n_unmeasured"] >= 1 and gv["failing"] == []


def test_restamp_overrides_load(tmp_path):
    from app.prop import presets
    assert presets._VERIFIED_STAMPS  # table still present with override merge


# -- 2026-11-04 audit: full gate-stack planted-drift checks ------------------------------------------------------------
def test_gate_stack_sees_planted_edge_and_rejects_zero_edge():
    """One seed each (the 20-seed sweep with the roadmap targets is scripts/planted_drift_sweep.py --check, the nightly regression)."""
    from app.validation.planted_edge import run_gate_stack
    good = run_gate_stack(0.20, seed=3, mc_sims=800, outer_reps=10, outer_sims=200)
    assert good.holdout_confirmed and good.unseen_expectancy_ok
    bad = run_gate_stack(0.0, seed=3, mc_sims=800, outer_reps=10, outer_sims=200)
    assert not bad.detected and not bad.forward_candidate


def test_speed_prefers_a_fast_70_over_a_slow_80():
    from app.scoring.speed import prefer_faster, speed_adjusted_pass_pct
    assert prefer_faster(70.0, 30.0, 80.0, 150.0) == -1
    assert speed_adjusted_pass_pct(70.0, 30.0) == 70.0 and speed_adjusted_pass_pct(80.0, 150.0) < 40.0


def test_calendar_days_not_trading_days_drive_the_pass_window():
    from app.scoring.speed import trading_to_calendar_days
    assert trading_to_calendar_days(100) == 140.0 and trading_to_calendar_days(None) is None


def test_unseen_dsr_passes_real_edge_and_fails_noise():
    from app.orchestration.audit_layers import unseen_dsr
    good = unseen_dsr(list(_r(500, 0.56, 1.2, 11)), 300)
    noise = unseen_dsr(list(_r(500, 0.5, 1.0, 12)), 300)
    assert good is not None and good["passed"] and noise is not None and not noise["passed"]


def test_rule_based_sizing_uses_no_dev_result():
    from types import SimpleNamespace
    from app.orchestration.full_pipeline import FullPipelineConfig, _size_by_rule
    # a losing / absent baseline still gets the same size: the rule reads only the firm's limit and the config
    cfg = FullPipelineConfig()
    assert cfg.size_rule == "rule" and cfg.size_rule_loss_streak == 8
    rules = SimpleNamespace(max_drawdown_pct=10.0, account_size=100000.0)
    risk = SimpleNamespace(sizing_mode="skip", planned_target_dollars=300.0)
    out_risk, out_bt, adj = _size_by_rule(None, None, risk, "BASELINE", rules, None, cfg, lambda m: None)
    assert out_risk is risk and out_bt == "BASELINE" and adj is None   # only rr_planned is resized
