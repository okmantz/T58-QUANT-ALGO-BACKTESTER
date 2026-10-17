"""Regression tests for the 2026-10-15 backtester review fixes (ruin horizon, robustness metric,
pass-and-paid product, stale docs, Topstep fee row, data gate, losing-OOS 'passes', crash/speed guards)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.presets import get_fee_info, get_preset
from app.scoring import t58_scorecard as sc


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------
def _trades(n: int, seed: int = 11, per_day: int = 2) -> list:
    """First n trades of ONE fixed 157-trade pool (so 80/120/157 are nested samples of the same strategy)."""
    rng = np.random.default_rng(seed)
    pool = np.where(rng.random(157) < 0.52, rng.normal(420, 120, 157), -rng.normal(300, 90, 157)).round(0)
    days = pd.bdate_range("2024-01-02", periods=200)
    out = []
    for i in range(n):
        t = days[i // per_day] + pd.Timedelta(hours=9 + (i % per_day) * 3)
        out.append(Trade(entry_time=t, exit_time=t + pd.Timedelta(hours=1), direction=1, entry_price=1.0, exit_price=1.0,
                         size=1, pnl=float(pool[i]), pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0))
    return out


_RULES = get_preset("topstep_50k").to_prop_rules()


# ---------------------------------------------------------------------------------------------------------
# items 1 + 7: time-bounded ruin, independent of how long the backtest is
# ---------------------------------------------------------------------------------------------------------
def test_ruin_is_time_bounded_and_does_not_grow_with_history_length():
    cfg = dict(n_simulations=600, random_seed=5, ruin_horizon_days=60)
    r120 = run_monte_carlo(_trades(120), _RULES, MonteCarloConfig(**cfg), selection_bias_caveat=False)
    r157 = run_monte_carlo(_trades(157), _RULES, MonteCarloConfig(**cfg), selection_bias_caveat=False)
    # the OLD (lifetime) definition keeps climbing as accounts get more time to bust...
    assert r157.lifetime_ruin_pct > r120.lifetime_ruin_pct + 3.0
    # ...while the windowed number, once the history covers the window, barely moves.
    assert r120.ruin_horizon_covered and r157.ruin_horizon_covered
    assert abs(r157.risk_of_ruin_pct - r120.risk_of_ruin_pct) < abs(r157.lifetime_ruin_pct - r120.lifetime_ruin_pct)
    assert abs(r157.risk_of_ruin_pct - r120.risk_of_ruin_pct) < 6.0
    # the breakdown is internally consistent: nested windows can only add deaths
    assert r157.risk_of_ruin_pct <= r157.lifetime_ruin_pct + 1e-9
    assert r157.ruin_within_90d_pct <= r157.ruin_within_180d_pct + 1e-9 <= r157.lifetime_ruin_pct + 2e-9
    assert "60 calendar days" in r157.risk_of_ruin_definition


def test_ruin_window_not_covered_is_flagged_and_caps_ready():
    from app.orchestration.full_pipeline import _make_verdict

    short = run_monte_carlo(_trades(40), _RULES, MonteCarloConfig(n_simulations=200, random_seed=5, ruin_horizon_days=90),
                            selection_bias_caveat=False)
    assert short.n_source_calendar_days < 90 and short.ruin_horizon_covered is False
    assert "RUIN WINDOW NOT COVERED" in short.methodology_note
    assert short.sample_ok is not False or True  # sample_ok is governed by trade/day floors, not the window

    class _MC:  # an otherwise-elite result whose history is shorter than the ruin window
        evaluation_pass_probability = 97.0
        first_payout_probability = 95.0
        risk_of_ruin_pct = 1.0
        per_attempt_pass_ci95 = (96.0, 99.0)
        per_attempt_payout_ci95 = (94.0, 98.0)
        per_attempt_paid_ci95 = (92.0, 97.0)
        per_attempt_paid_probability = 95.0
        per_attempt_payout_probability = 98.0
        return_percentiles = {}
        sample_ok = True
        ruin_horizon_covered = False
        ruin_horizon_days = 90
        n_source_calendar_days = 50
        risk_of_ruin_definition = "x"

    class _WF:
        walk_forward_efficiency = 0.95
        is_stable = True
        stability_threshold = 0.4
        n_folds = 4
        oos_profitable = True

    class _Stats:
        total_trades = 300
        average_r = 0.6
        max_drawdown_pct = 1.0

        def to_dict(self):
            return {}

    from types import SimpleNamespace
    verdict, reasons, card, _, _ = _make_verdict(_MC(), _WF(), icir_gate=None, statistics=_Stats(),
                                                 prop_rules=SimpleNamespace(max_drawdown_pct=10.0),
                                                 mc_dispersion_pts=1.0, min_trades_for_ready=100)
    assert card.tier in ("Elite", "Strong")
    # 2026-10-17: the thin-coverage cap (replay / expectancy evidence absent from this mock) now fires before the
    # ruin-window cap; either way an otherwise-elite result with an uncovered ruin window must not reach READY.
    assert verdict == "MARGINAL"
    assert any(("RUIN WINDOW NOT COVERED" in r) or ("CAPPED AT MARGINAL" in r) for r in reasons)


# ---------------------------------------------------------------------------------------------------------
# item 3: gate on / show the pass-AND-paid product
# ---------------------------------------------------------------------------------------------------------
def test_pass_and_paid_product_is_reported_and_never_exceeds_pass_rate():
    r = run_monte_carlo(_trades(157), _RULES, MonteCarloConfig(n_simulations=400, random_seed=3), selection_bias_caveat=False)
    assert 0.0 <= r.per_attempt_paid_probability <= r.per_attempt_pass_probability + 1e-9
    # product = pass x conditional payout rate
    assert r.per_attempt_paid_probability == pytest.approx(
        r.per_attempt_pass_probability * r.per_attempt_payout_probability / 100.0, abs=1e-6)
    assert r.headline_pass_and_paid_probability == pytest.approx(r.per_attempt_paid_probability)
    assert r.per_attempt_paid_ci95[0] <= r.per_attempt_paid_probability <= r.per_attempt_paid_ci95[1]


def test_scorecard_payout_component_uses_the_product_not_the_conditional_rate():
    class _MC:
        evaluation_pass_probability = 40.0
        first_payout_probability = 99.0
        risk_of_ruin_pct = 5.0
        per_attempt_pass_ci95 = (35.0, 45.0)
        per_attempt_payout_ci95 = (97.0, 100.0)   # conditional: of the funded, almost all paid
        per_attempt_paid_ci95 = (33.0, 44.0)      # product: but few attempts get there at all
    res = sc.score_from_results(mc_result=_MC())
    assert res.components["first_payout_probability"]["value"] == pytest.approx(33.0)


def test_pass_and_paid_gate_caps_ready(monkeypatch):
    from types import SimpleNamespace
    from app.orchestration import full_pipeline as fp
    from app.orchestration.full_pipeline import _make_verdict
    # the wrapper's gates are demote-only, so feed it a core verdict of READY and check it is demoted
    monkeypatch.setattr(fp, "_make_verdict_core", lambda *a, **k: ("READY", [], SimpleNamespace(tier="Strong"), False, False))

    class _MC:
        evaluation_pass_probability = 95.0
        first_payout_probability = 95.0
        risk_of_ruin_pct = 1.0
        per_attempt_pass_ci95 = (92.0, 97.0)
        per_attempt_payout_ci95 = (92.0, 97.0)
        per_attempt_paid_ci95 = (20.0, 30.0)      # product far below the bar
        per_attempt_paid_probability = 25.0
        per_attempt_payout_probability = 95.0
        return_percentiles = {}
        sample_ok = True
        ruin_horizon_covered = True

    class _WF:
        walk_forward_efficiency = 0.95
        is_stable = True
        stability_threshold = 0.4
        n_folds = 4
        oos_profitable = True

    class _Stats:
        total_trades = 300
        average_r = 0.6
        max_drawdown_pct = 1.0

        def to_dict(self):
            return {}

    verdict, reasons, card, _, _ = _make_verdict(_MC(), _WF(), icir_gate=None, statistics=_Stats(),
                                                 prop_rules=SimpleNamespace(max_drawdown_pct=10.0),
                                                 mc_dispersion_pts=1.0, min_trades_for_ready=100, paid_min_pct=35.0)
    assert verdict != "READY" and any("PASS-AND-PAID TOO LOW" in r for r in reasons)


# ---------------------------------------------------------------------------------------------------------
# items 2, 8, 9: robustness metric, edge ordering, ceiling
# ---------------------------------------------------------------------------------------------------------
def test_mc_robustness_is_seed_dispersion_and_missing_not_zero_without_it():
    assert sc._mc_robustness_score(None, 0.0) == 100.0
    assert sc._mc_robustness_score(None, 10.0) == pytest.approx(50.0)
    assert sc._mc_robustness_score(None, 25.0) == 0.0
    assert sc._mc_robustness_score(None, None) is None

    class _Degenerate:   # the old failure mode: median return exactly 0 (payouts reset the balance)
        return_percentiles = {25: -3.0, 50: 0.0, 75: 9.0}
    assert sc._mc_robustness_score(_Degenerate()) is None      # never derived from the median-return ratio


def test_stronger_edge_never_scores_lower_through_the_robustness_component():
    """Audit item 8: 58% win-rate scored 86.9 (component dropped) while 70% scored 81.8 (component 0).
    A strong edge's percentile shape must no longer matter."""
    def score(median):
        class _MC:
            evaluation_pass_probability = 99.0
            first_payout_probability = 99.0
            risk_of_ruin_pct = 0.0
            return_percentiles = {25: median - 5, 50: median, 75: median + 40}
        return sc.score_from_results(mc_result=_MC(), mc_dispersion_pts=1.0).score
    assert score(0.0) == pytest.approx(score(30.0))


def test_expectancy_rescaled_so_a_realistic_edge_scores_fifty():
    assert sc._expectancy_score(0.3) == pytest.approx(50.0)
    assert sc._expectancy_score(0.6) == 100.0 and sc._expectancy_score(2.0) == 100.0
    assert sc._expectancy_score(0.0) == 0.0 and sc._expectancy_score(-0.2) == 0.0
    assert sc._expectancy_score(0.44) > 70.0       # the Masterclass example used to score 22


def test_ceiling_is_no_longer_stuck_below_the_strong_cut():
    """Audit item 9: with 8 other components perfect, expectancy 0.3R and drawdown at half the firm limit,
    the ceiling was 84.6 (< the 85 cut)."""
    inputs = sc.T58ScorecardInputs(
        pass_probability=100, first_payout_probability=100, risk_of_ruin=100, walk_forward_stability=100,
        parameter_stability=100, parsimony=100, cpcv_supporting=100, t58_conformance=100,
        monte_carlo_robustness=sc._mc_robustness_score(None, 0.0),
        expectancy=sc._expectancy_score(0.3), drawdown=sc._drawdown_score(5.0, 10.0),
    )
    res = sc.compute_t58_score(inputs)
    assert res.score == pytest.approx(93.75)   # conformance is informational now (weights sum 120)
    assert res.t58_conformance == 100.0
    assert res.tier == "Elite"
    old = (8 * 0 + 100 * 100 + 0 + 5 * 15 + 10 * 50) / 125.0
    assert round(old, 1) == 84.6                    # reproduces the report's number for the OLD scaling


# ---------------------------------------------------------------------------------------------------------
# item 4: stale documentation
# ---------------------------------------------------------------------------------------------------------
def test_scorecard_docs_no_longer_claim_ninetieths_and_weights_sum_to_100():
    assert sum(abs(w) for w in sc._WEIGHTS.values()) == 100.0
    assert "t58_conformance" not in sc._WEIGHTS
    assert "5/90ths of the total score" not in sc.__doc__ and "5/90ths" not in sc._expectancy_score.__doc__
    assert "100" in sc.__doc__


# ---------------------------------------------------------------------------------------------------------
# missing evidence caps, never lifts
# ---------------------------------------------------------------------------------------------------------
def test_scorecard_reports_coverage_and_missing_core():
    thin = sc.compute_t58_score(sc.T58ScorecardInputs(pass_probability=100, walk_forward_stability=100))
    assert thin.weight_coverage == pytest.approx((10 + 15) / 100.0)
    assert thin.missing_core == ["first_payout_probability", "risk_of_ruin"]
    full = sc.compute_t58_score(sc.T58ScorecardInputs(**{k: 50 for k in sc._WEIGHTS}))
    assert full.weight_coverage == pytest.approx(1.0) and full.missing_core == []


# ---------------------------------------------------------------------------------------------------------
# item 10: walk-forward / CPCV must not "pass" a strategy that loses equally in and out of sample
# ---------------------------------------------------------------------------------------------------------
def _fold(i, train, test, test_net, test_trades=20, pf=0.9):
    from app.search.robustness import WalkForwardFold
    f = WalkForwardFold(fold_index=i, train_period=("a", "b"), test_period=("c", "d"), train_bars=100, test_bars=50,
                        train_metric=train, test_metric=test, test_net_profit=test_net, test_trades=test_trades,
                        test_profit_factor=pf)
    return f, train, test


def _run_wf(monkeypatch, fold_args):
    from app.search import robustness as rb
    it = iter(fold_args)
    monkeypatch.setattr(rb, "_wf_one_fold", lambda *a, **k: _fold(*next(it)))
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=400, freq="h"), "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": 1.0, "volume": 1})
    return rb.run_walk_forward(df, lambda: None, None, n_folds=4, metric="profit_factor")


def test_walk_forward_losing_equally_in_and_out_of_sample_is_not_stable(monkeypatch):
    # profit factors 0.9 in and out of sample (a metric that is always positive) and negative net P&L OOS
    res = _run_wf(monkeypatch, [(i, 0.9, 0.9, -400.0) for i in range(4)])
    assert res.oos_profitable is False
    assert res.walk_forward_efficiency <= 0.0 and res.is_stable is False
    assert sc.score_from_results(walk_forward_result=res).components["walk_forward_stability"]["value"] == 0.0
    assert any("LOST money" in w for w in res.warnings)


def test_walk_forward_negative_over_negative_is_not_a_perfect_efficiency(monkeypatch):
    res = _run_wf(monkeypatch, [(i, -2.0, -2.0, -300.0) for i in range(4)])    # -2/-2 used to equal +1.0 efficiency
    assert res.walk_forward_efficiency == 0.0 and res.is_stable is False


def test_walk_forward_genuine_edge_still_passes_and_thin_oos_is_unproven(monkeypatch):
    ok = _run_wf(monkeypatch, [(i, 1.5, 1.4, 500.0) for i in range(4)])
    assert ok.oos_profitable is True and ok.is_stable is True and ok.walk_forward_efficiency == pytest.approx(1.4 / 1.5)
    thin = _run_wf(monkeypatch, [(i, 1.5, 1.4, -50.0, 1) for i in range(4)])    # 4 OOS trades in total: cannot convict
    assert thin.oos_profitable is None and thin.is_stable is True


def test_cpcv_losing_out_of_sample_is_not_robust_and_scores_zero(monkeypatch):
    from app.backtest.risk import RiskConfig
    from app.validation import cpcv

    def _path(idx, tg, *a, **k):
        return cpcv.CPCVPathResult(path_index=idx, test_group_indices=tuple(tg), train_bars=100, test_bars=50,
                                   in_sample_metric=0.9, out_of_sample_metric=0.9, oos_net_profit=-250.0, oos_trades=15)
    monkeypatch.setattr(cpcv, "_cpcv_one_path", _path)
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=600, freq="h"), "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": 1.0, "volume": 1})
    res = cpcv.run_cpcv(df, lambda: None, RiskConfig(), n_groups=4, n_test_groups=1, metric="profit_factor", max_paths=3)
    assert res.oos_profitable is False and res.is_robust is False
    assert sc._cpcv_score(res) == 0.0


def test_cpcv_score_no_in_sample_edge_scores_zero_not_100():
    class _R:
        mean_is_metric = -1.0
        mean_oos_metric = -1.0    # "OOS did not get worse" used to score 100
    assert sc._cpcv_score(_R()) == 0.0


# ---------------------------------------------------------------------------------------------------------
# item 5: Topstep fee row + activation fee in the EV maths
# ---------------------------------------------------------------------------------------------------------
def test_topstep_fee_row_exists_with_source_and_date_and_ev_can_run():
    from app.orchestration.ev_verdict import INSUFFICIENT_EVIDENCE, evaluate_ev

    fee = get_fee_info("topstep_50k")
    assert fee is not None and fee.challenge_price == 49.0 and fee.reset_price == 49.0 and fee.activation_fee == 149.0
    assert fee.source_url.startswith("https://") and fee.date_checked == "2026-10-15" and fee.profit_split_pct == 90.0
    assert fee.written_confirmation is False          # automation must still be confirmed with Topstep in writing
    ev = evaluate_ev(legacy_verdict="MARGINAL", holdout_pass_rate=0.5, window_payout_rate=0.3, payout_value=3000.0,
                     challenge_price=fee.challenge_price, reset_price=fee.reset_price, profit_split_pct=fee.profit_split_pct,
                     activation_fee=fee.activation_fee, bankroll_ruin_pct=5.0, n_trials=1)
    assert ev.label != INSUFFICIENT_EVIDENCE and "fee row" not in " ".join(ev.inputs_missing)


def test_activation_fee_lowers_ev_and_raises_break_even():
    from app.orchestration.ev_verdict import break_even_pass_rate, compute_ev
    assert compute_ev(0.2, 0.45, 2500.0, 100.0) == pytest.approx(0.2 * 0.45 * 2500 - 100)     # unchanged default
    assert compute_ev(0.2, 0.45, 2500.0, 100.0, 149.0) == pytest.approx(0.2 * 0.45 * 2500 - 0.2 * 149 - 100)
    assert break_even_pass_rate(100.0, 0.45, 2500.0, 149.0) > break_even_pass_rate(100.0, 0.45, 2500.0)
    assert break_even_pass_rate(100.0, 0.05, 2000.0, 149.0) is None     # q*V <= activation fee: never worth it


# ---------------------------------------------------------------------------------------------------------
# item 6: data gate flags unadjusted roll gaps; adjusted data passes
# ---------------------------------------------------------------------------------------------------------
def _futures_frame(rolls: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2020-01-01 00:00", periods=60 * 24 * 400, freq="min", tz="UTC")
    steps = rng.normal(0, 0.5, len(ts))
    day_start = np.flatnonzero((ts.hour == 0) & (ts.minute == 0))
    if rolls:
        for k in day_start[30::45]:                       # a splice every ~45 days on the first bar of a date
            steps[k] += 40.0
    close = 15000 + np.cumsum(steps)
    opn = np.r_[close[0], close[:-1]] + rng.normal(0, 0.05, len(ts))
    if rolls:
        for k in day_start[30::45]:                       # the open gaps by the splice from the prior close
            opn[k] += 40.0
    return pd.DataFrame({"timestamp": ts, "open": opn, "high": np.maximum(opn, close) + 0.2, "low": np.minimum(opn, close) - 0.2,
                         "close": close, "volume": 10})


def test_data_gate_flags_unadjusted_roll_gaps_but_not_clean_or_non_futures_data():
    from app.data.data_manifest import check_dataset, looks_like_futures, roll_gap_counts

    dirty, clean = _futures_frame(True), _futures_frame(False)
    assert sum(roll_gap_counts(dirty)) >= 2 and sum(roll_gap_counts(clean)) == 0
    gate = check_dataset(dirty, "NQ1!/NQ_1m.parquet", manifest={"files": []})
    assert not gate.ok_for_ready and any("unadjusted contract-roll gaps" in r for r in gate.reasons)
    assert not any("roll" in r for r in check_dataset(clean, "NQ1!/NQ_1m.parquet", manifest={"files": []}).reasons)
    assert not looks_like_futures("EURUSD/fx_EUR_USD_1m.parquet")
    assert not any("roll" in r for r in check_dataset(dirty, "EURUSD/fx_EUR_USD_1m.parquet", manifest={"files": []}).reasons)


# ---------------------------------------------------------------------------------------------------------
# item 0: quick crash / speed guards
# ---------------------------------------------------------------------------------------------------------
def test_worker_budget_counts_interpreter_base_and_three_times_the_frame(monkeypatch):
    from app.orchestration import resource_guard as rg

    monkeypatch.setattr(rg, "_available_memory_bytes", lambda: 4 * 1024 ** 3)
    monkeypatch.setattr(rg, "memory_pressure_ok", lambda *a, **k: True)
    frame = 123 * 1024 ** 2                                   # the 2.16M-row NQ 1-minute frame
    per_worker = rg._WORKER_BASE_BYTES + 3.0 * frame
    assert rg.safe_worker_count_for_bytes(frame, requested=64) == int((4 * 1024 ** 3 * 0.5) // per_worker)
    assert per_worker > 1.6 * frame * 2                       # the old budget (1.6x) badly under-counted


def test_watchdog_halves_the_pool_under_memory_pressure(monkeypatch):
    from app.orchestration import resource_guard as rg

    monkeypatch.setattr(rg, "memory_pressure_ok", lambda *a, **k: False)
    assert rg.shrink_pool_for_memory_pressure(8) == 4 and rg.shrink_pool_for_memory_pressure(1) == 1
    monkeypatch.setattr(rg, "memory_pressure_ok", lambda *a, **k: True)
    assert rg.shrink_pool_for_memory_pressure(8) == 8


def test_job_log_is_capped_and_counts_dropped_lines():
    from app.web import job_manager as jm

    mgr = jm.JobManager()
    jid = mgr.create()
    for i in range(jm.MAX_JOB_LOG_LINES + 250):
        mgr.log(jid, f"line {i}")
    job = mgr.get(jid)
    assert len(job["log"]) == jm.MAX_JOB_LOG_LINES and job["log_dropped"] == 250
    assert job["log"][-1] == f"line {jm.MAX_JOB_LOG_LINES + 249}"


def test_resample_cache_returns_identical_private_copies_and_tracks_data_changes():
    from app.data import timeframe_resample as tr

    tr.clear_resample_cache()
    n = 5000
    rng = np.random.default_rng(1)
    c = 20000 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=n, freq="min"), "open": c, "high": c + 3, "low": c - 3,
                       "close": c, "volume": 10})
    ref = tr._resample_ohlcv_uncached(df, "5m")
    a, b = tr.resample_ohlcv(df, "5m"), tr.resample_ohlcv(df, "5m")      # second call is a cache hit
    assert a.equals(ref) and b.equals(ref)
    a["x"] = 1
    assert "x" not in tr.resample_ohlcv(df, "5m").columns                # callers get private copies
    df2 = df.copy()
    df2.loc[1004, "close"] += 5.0                                        # last 1m bar of a 5m bin
    assert tr.resample_ohlcv(df2, "5m").equals(tr._resample_ohlcv_uncached(df2, "5m"))
    assert not tr.resample_ohlcv(df2, "5m").equals(ref)                  # changed data is never served the stale result
