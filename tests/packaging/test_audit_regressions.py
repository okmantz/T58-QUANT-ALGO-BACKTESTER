"""Tests for the 2026-10 audit changes (EV verdict, excursions, behaviour rules, data gate, contract calendar, / Audit 2026-10-12 independent tests. / Regression tests for the 2026-09 codebase audit findings: / Regression tests for bugs found while reviewing the telemetry work. Each"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd
import pytest
from app.search.robustness import deflated_sharpe_gate
# --------------------------------------------------------------------------
# F-FUNDED: funded stage must start fresh on Topstep presets
# --------------------------------------------------------------------------
from app.prop.account import PropAccount
from app.prop.presets import get_preset
# --------------------------------------------------------------------------
# F-ROLL: root-symbol filenames are continuous futures
# --------------------------------------------------------------------------
import os
from app.data.importer import _looks_like_continuous_future, import_csv
from app.backtest.engine import run_backtest
from app.backtest.execution import run_execution
from app.backtest.risk import (
    RiskConfig,
    account_size_mismatch_message,
    with_prop_safety_defaults,
)
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.walkforward_ga import run_walkforward_aware_refinement
from app.optimize.refinement import RefinementConfig
from app.prop.simulator import PropRules, simulate_account
from app.search.robustness import run_walk_forward
from app.strategy.base import StrategyResult
from app.strategy.manual import ManualStrategy
from app.validation.regime_matrix import build_regime_matrix
from app.validation.walk_forward_opt import build_folds
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from app.orchestration import universe as un
from app.web.job_manager import JobManager


# ---- from test_audit_2026_10.py ----

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
def _shipped_python_strategies():
    """Every strategy file actually present in strategies/python. This used to hard-code
    'NQ1! Opening Drive 5m.py', which is not in the repository, so the test failed with
    'file not found' instead of checking anything. Discovering the folder keeps the check
    (loads, signals are causal and in {-1, 0, 1}) on whatever strategies are really shipped."""
    from pathlib import Path
    folder = Path(__file__).resolve().parents[2] / "strategies" / "python"
    return sorted(p.name for p in folder.glob("*.py") if not p.name.startswith("_")) if folder.is_dir() else []


@pytest.mark.parametrize("name", _shipped_python_strategies() or [pytest.param("<no python strategies shipped>", marks=pytest.mark.skip(reason="strategies/python is empty"))])
def test_new_strategies_load_and_emit_valid_signals(name):
    from pathlib import Path
    from app.strategy.python import PythonStrategy
    root = Path(__file__).resolve().parents[2] / "strategies" / "python" / name
    st = PythonStrategy(str(root))
    ts = pd.date_range("2025-03-03 14:00", periods=2 * 288 * 3, freq="5min", tz="UTC")
    rng = np.random.default_rng(1)
    px = 5000 + np.cumsum(rng.normal(0, 1.0, len(ts)))
    df = pd.DataFrame({"timestamp": ts, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 100})
    sig = st.generate(df).signals
    assert len(sig) == len(df) and set(np.unique(sig.to_numpy())) <= {-1.0, 0.0, 1.0}


# ---- from test_audit_2026_10_12.py ----

def _null_false_positive_rate(units: str, n_sims=300, n_trades=150, trades_per_year=600, n_trials=50, seed=1):
    """Best of n_trials null strategies (zero-mean trade returns). Fraction of
    sims where the best one passes the 0.95 DSR gate."""
    rng = np.random.default_rng(seed)
    passes = 0
    for _ in range(n_sims):
        r = rng.normal(0, 1, size=(n_trials, n_trades))
        sr = r.mean(1) / r.std(1, ddof=1)               # per-trade Sharpe
        if units == "annualized":
            sr = sr * np.sqrt(trades_per_year)          # legacy (buggy) units
        best = float(sr.max())
        g = deflated_sharpe_gate(best, [float(x) for x in sr], n_trials, n_trades)
        passes += bool(getattr(g, "passed", getattr(g, "is_significant", False)))
    return passes / n_sims


def test_dsr_per_trade_units_control_false_positives():
    fixed = _null_false_positive_rate("per_trade")
    legacy = _null_false_positive_rate("annualized")
    print("DSR null false-positive rate  fixed=%.3f legacy=%.3f" % (fixed, legacy))
    assert fixed <= 0.10
    assert legacy > fixed


def _run(rules, pnls):
    acct = PropAccount(rules)
    out = []
    for day, pnl in enumerate(pnls):
        res = acct.on_trade_close(pnl, day, mae=min(pnl, 0.0), mfe=max(pnl, 0.0))
        out.append(res)
    return acct, out


def test_topstep_funded_account_starts_fresh_and_floor_is_2000_below_start():
    rules = get_preset("topstep_50k").to_prop_rules()
    # +1400/day x3 days, each <55% of total profit -> passes the 6% ($3,000) target
    acct, _ = _run(rules, [1100, 1100, 1100])
    assert acct.passed_evaluation and acct.stage == "funded"
    assert acct.balance == 50_000          # fresh XFA, eval profit not carried
    assert acct.floor() == 48_000          # full $2,000 MLL below the start

    # -1,950 stays alive ; a further -100 breaches 48,000
    r1 = acct.on_trade_close(-1950, 3, mae=-1950, mfe=0)
    assert not acct.failed
    r2 = acct.on_trade_close(-100, 4, mae=-100, mfe=0)
    assert acct.failed


def test_legacy_rules_carry_eval_profit_into_funded():
    rules = get_preset("topstep_50k").to_prop_rules()
    rules.funded_starts_fresh = False
    acct, _ = _run(rules, [1100, 1100, 1100])
    assert acct.balance == 53_300


def test_topstep_floor_is_fixed_dollar_and_locks_at_start():
    rules = get_preset("topstep_50k").to_prop_rules()
    acct = PropAccount(rules)
    acct.trailing_peak = 51_000
    assert acct.floor() == 49_000          # peak - $2,000 (not 4% of peak = 48,960)
    acct.trailing_peak = 60_000
    assert acct.floor() == 50_000          # locked at starting balance


@pytest.mark.parametrize("name,expected", [
    ("MNQ_1min_20260120_20260415.csv", True), ("ES_5min.csv", True), ("gc_1min.csv", True),
    ("ES1!_5.csv", True), ("ESH6_1min.csv", False), ("NQ_H26.csv", False),
    ("EURUSD_1min.csv", False), ("BTCUSD.csv", False), ("esport_data.csv", False),
])
def test_continuous_future_filename_detection(name, expected):
    assert _looks_like_continuous_future(name) is expected


_MNQ = "data/raw/MNQ1!/MNQ_1min_20260120_20260415.csv"


@pytest.mark.skipif(not os.path.exists(_MNQ), reason="MNQ file not present")
def test_mnq_contract_flip_is_repaired_on_import():
    df = import_csv(_MNQ).dataframe
    assert df.attrs.get("roll_adjusted") is True
    # raw file jumps -215/+224 around 2026-03-18 21:00-22:00 UTC (contract flip)
    import pandas as pd
    ts = pd.to_datetime(df["timestamp"])
    w = df[(ts >= "2026-03-18 20:30") & (ts <= "2026-03-18 22:30")]
    assert w["close"].diff().abs().max() < 60


def test_timezone_warning_flags_shifted_futures_data():
    import pandas as pd
    from app.data.importer import _futures_timezone_warning

    def mk(shift):
        t = pd.date_range("2026-01-05", "2026-02-20", freq="1min")
        t = t[~(((t.hour - shift) % 24) == 22)]
        return pd.DataFrame({"timestamp": t})

    ok, bad = [], []
    _futures_timezone_warning(mk(0), "ES_1min.csv", ok)
    _futures_timezone_warning(mk(-5), "ES_1min.csv", bad)
    assert ok == [] and len(bad) == 1


# --------------------------------------------------------------------------
# Free-data providers (mock only; NOT verified live)
# --------------------------------------------------------------------------
def test_free_sources_parse_documented_schemas():
    from app.data import free_sources as fs
    fred = fs.fetch_fred_series("VIXCLS", "k", lambda u, p: {"observations": [
        {"date": "2026-01-02", "value": "14.5"}, {"date": "2026-01-03", "value": "."}, {"date": "2026-01-06", "value": "16"}]})
    assert list(fred.values) == [14.5, 16.0]
    td = fs.fetch_twelve_data_ohlcv("SPY", "1min", "k", http_get=lambda u, p: {"values": [
        {"datetime": "2026-01-02 14:31:00", "open": "2", "high": "3", "low": "1", "close": "2.5", "volume": "10"},
        {"datetime": "2026-01-02 14:30:00", "open": "1", "high": "2", "low": "1", "close": "2", "volume": "5"}]})
    assert td["timestamp"].is_monotonic_increasing and len(td) == 2
    pg = fs.fetch_polygon_ohlcv("SPY", 1, "minute", "2026-01-02", "2026-01-02", "k", http_get=lambda u, p: {"results": [
        {"t": 1767364200000, "o": 1, "h": 2, "l": 1, "c": 2, "v": 3}]})
    assert pg["timestamp"].iloc[0] == pd.Timestamp("2026-01-02 14:30:00")
    import pytest as _p
    with _p.raises(fs.DataSourceError):
        fs.fetch_fred_series("X", "")


def test_saving_other_keys_does_not_wipe_new_provider_keys(tmp_path):
    from app.accounts import api_keys as ak
    ak.save_settings(ak.ApiKeysSettings(fred_api_key="a", twelve_data_api_key="b", databento_api_key="c"))
    import dataclasses
    cur = ak.load_settings()
    ak.save_settings(dataclasses.replace(cur, openai_api_key="z"))
    again = ak.load_settings()
    assert (again.fred_api_key, again.twelve_data_api_key, again.databento_api_key, again.openai_api_key) == ("a", "b", "c", "z")


def test_web_api_keys_page_and_save_roundtrip():
    from app.web.server import app as flask_app
    from app.accounts import api_keys as ak
    c = flask_app.test_client()
    r = c.get("/settings/api-keys")
    assert r.status_code == 200 and b"Twelve Data" in r.data
    c.post("/settings/api-keys/save-data", data={"tiingo_api_key": "TT", "fred_api_key": "FF"})
    c.post("/settings/api-keys/save-ai", data={"openai_api_key": "OO"})
    s = ak.load_settings()
    assert (s.tiingo_api_key, s.fred_api_key, s.openai_api_key) == ("TT", "FF", "OO")


# --------------------------------------------------------------------------
# F-DAILY: session-anchored daily/multi-hour bars for CME data
# --------------------------------------------------------------------------
def _cme_minutes(start, end):
    t = pd.date_range(start, end, freq="1min", tz="UTC")
    ct = t.tz_convert("America/Chicago")
    keep = (ct.dayofweek < 5) & (ct.hour != 16)           # 16:00-17:00 CT halt, weekdays only (simplified)
    keep &= ~((ct.dayofweek == 4) & (ct.hour >= 16))       # Fri close
    t = t[keep]
    n = len(t)
    px = 100 + np.arange(n) * 0.01
    return pd.DataFrame({"timestamp": t, "open": px, "high": px + 0.1, "low": px - 0.1, "close": px + 0.05, "volume": 1.0})


def test_daily_bars_follow_cme_trading_day_and_dst():
    from app.data.timeframe_resample import resample_ohlcv, looks_like_cme_session_data
    for start, end, utc_hour in (("2026-01-05", "2026-02-27", 23), ("2026-06-01", "2026-07-24", 22)):
        df = _cme_minutes(start, end)
        assert looks_like_cme_session_data(df)
        d = resample_ohlcv(df, "1d")
        assert (d["timestamp"].dt.hour == utc_hour).all()        # 17:00 CT in UTC, DST-aware
        # one bar per trading day, none split: first bar open == first minute of the session
        assert d["open"].iloc[1] == df.loc[df["timestamp"] >= d["timestamp"].iloc[1], "open"].iloc[0]
        h4 = resample_ohlcv(df, "4h")
        assert set(h4["timestamp"].dt.tz_convert("America/Chicago").dt.hour) <= {17, 21, 1, 5, 9, 13}


def test_resample_is_unchanged_for_24h_markets_and_short_bars():
    from app.data.timeframe_resample import resample_ohlcv
    t = pd.date_range("2026-01-05", periods=6000, freq="1min")
    fx = pd.DataFrame({"timestamp": t, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0})
    assert resample_ohlcv(fx, "1d").equals(resample_ohlcv(fx, "1d", anchor="utc"))
    cme = _cme_minutes("2026-01-05", "2026-02-27")
    assert resample_ohlcv(cme, "1h").equals(resample_ohlcv(cme, "1h", anchor="utc"))


# --------------------------------------------------------------------------
# F-ROLLFOLD: rolling walk-forward folds
# --------------------------------------------------------------------------
def test_rolling_folds_all_survive_full_size_and_tests_do_not_overlap():
    from app.validation.walk_forward_opt import build_folds
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=6000, freq="5min"),
                       "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})
    fs = build_folds(df, n_folds=5, window_mode="rolling", train_frac=0.6)
    assert len(fs) == 5
    assert {len(f.train_df) for f in fs} == {1200}
    spans = [(f.test_start_bar, f.test_end_bar) for f in fs]
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0                                   # no overlap between consecutive tests
    for f in fs:
        assert f.train_df["timestamp"].iloc[-1] < f.test_df["timestamp"].iloc[0]   # train strictly before test


# --------------------------------------------------------------------------
# F-FWD: forward test must record the broker's realized P&L on SL/TP exits
# --------------------------------------------------------------------------
def test_forward_test_records_broker_side_stop_out_pnl_and_trips_daily_limit(tmp_path):
    from app.backtest.risk import RiskConfig
    from app.forward_test.engine import ForwardTestConfig, ForwardTestSession
    from app.forward_test.journal import ForwardTestJournal

    class FakeConn:
        login, server = "1", "Demo"
        def get_open_positions(self, symbol=None): return []                      # position vanished = SL hit
        def closed_position_result(self, ticket): return {"exit_price": 1.0950, "pnl": -1250.0}

    j = ForwardTestJournal(tmp_path / "ft.db")
    sid = j.start_session("python", "x.py", "EURUSD", 15, "1", "Demo")
    risk = RiskConfig(initial_balance=50_000.0, risk_value=1.0, daily_loss_limit_pct=2.0)
    s = ForwardTestSession(strategy=None, strategy_type="python", strategy_filename="x.py", connector=FakeConn(),
                           journal=j, config=ForwardTestConfig(symbol="EURUSD", timeframe_minutes=15, risk=risk))
    s._session_id = sid
    s._open_trade_row_id = j.record_open(sid, 42, 1, 1.0, 1.1000, 1.0950, None)
    s.status.open_position_ticket = 42
    s._reconcile_closed_trade()
    closed = j.closed_trade_stats(sid)
    assert closed["n_trades"] == 1 and closed["net_pnl"] == -1250.0     # previously exit=entry, pnl=0
    assert s._daily_realized_pnl == -1250.0 and s._daily_loss_breached()  # 2% of 50K = 1,000 -> breaker trips


# ---- from test_audit_findings_2026_09.py ----

def _flat_df(n=1000, tz=None, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz=tz)
    price = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "timestamp": ts,
        "open": price, "high": price + 0.3, "low": price - 0.3, "close": price,
        "volume": 100.0,
    })


def _long_short_signal(n):
    return pd.Series(np.where(np.arange(n) % 20 < 10, 1, -1))


class _FixedSignalStrategy:
    """Minimal Strategy stand-in returning a fixed pre-built signal series."""
    source_type = "manual"

    def __init__(self, signals, stop_loss_pips=50, take_profit_pips=80):
        self._signals = signals
        self._stop = stop_loss_pips
        self._take = take_profit_pips

    def generate(self, df):
        return StrategyResult(
            name="fixed", source_type="manual", signals=self._signals,
            stop_loss_pips=self._stop, take_profit_pips=self._take,
        )


def _manual_ema_cross_config():
    return {
        "name": "ema cross",
        "entry_conditions": {
            "long": [{
                "left": {"type": "ema", "period": 5, "field": "close"},
                "operator": "cross above",
                "right": {"type": "ema", "period": 15, "field": "close"},
            }],
            "short": [{
                "left": {"type": "ema", "period": 5, "field": "close"},
                "operator": "cross below",
                "right": {"type": "ema", "period": 15, "field": "close"},
            }],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": {
            "stop_type": "atr", "stop_value": 2.0, "stop_atr_period": 14,
            "target_type": "atr", "target_value": 3.0, "target_atr_period": 14,
            "opposite_signal_exit": True,
        },
    }


class TestRisk001AccountSizeMismatch:
    def test_mismatch_is_detected(self):
        msg = account_size_mismatch_message(100_000.0, 50_000.0)
        assert msg is not None
        assert "100,000" in msg and "50,000" in msg

    def test_no_mismatch_returns_none(self):
        assert account_size_mismatch_message(100_000.0, 100_000.0) is None

    def test_with_prop_safety_defaults_reconciles_initial_balance(self):
        """The two independently-editable values must not silently stay
        divergent -- with_prop_safety_defaults is the one chokepoint every
        pipeline (Full Pipeline, Evolution Lab, Speed Run) already calls,
        so PropRules.account_size must win there."""
        risk = RiskConfig(initial_balance=100_000.0)
        prop_rules = PropRules(account_size=50_000.0, daily_loss_limit_pct=5.0, max_drawdown_pct=10.0)
        reconciled = with_prop_safety_defaults(risk, prop_rules)
        assert reconciled.initial_balance == 50_000.0

    def test_daily_loss_circuit_breaker_and_prop_verdict_agree_after_fix(self):
        """The actual failure mode: without reconciliation, a raw backtest's
        intrabar daily-loss circuit breaker (keyed to risk.initial_balance)
        and the post-hoc prop-firm verdict (keyed to prop_rules.account_size)
        could fire at two different dollar floors for the "same" 5% rule on
        the exact same trade sequence. After the fix, both are keyed to the
        same $50,000 account."""
        risk = RiskConfig(initial_balance=100_000.0, pip_size=1.0)
        prop_rules = PropRules(account_size=50_000.0, daily_loss_limit_pct=5.0, max_drawdown_pct=90.0)
        reconciled = with_prop_safety_defaults(risk, prop_rules)
        # 5% of the RECONCILED $50,000 account = $2,500 -- this must be the
        # daily floor actually enforced during raw execution.
        assert reconciled.daily_loss_limit_pct == 5.0
        assert reconciled.initial_balance == prop_rules.account_size

        n = 5
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": [100] * n, "high": [100] * n, "low": [80] * n, "close": [100] * n,
            "volume": [100] * n,
        })
        sig = pd.Series([1, 0, 0, 0, 0])
        trades, _ = run_execution(
            df=df, signals=sig, risk=reconciled,
            stop_loss_pips=1000, take_profit_pips=None,
            stop_loss_distance=None, take_profit_distance=None,
            trailing_stop_distance=None, breakeven_trigger_r=None, partial_exit_config=None,
            adaptive_risk=None,
        )
        # Whatever the raw backtest actually realized, simulate_account
        # (using prop_rules.account_size) must be checking the SAME
        # dollar account the backtest just enforced its circuit breaker
        # against -- not a silently different one.
        pnls = [t.pnl for t in trades]
        dates = [t.entry_time for t in trades]
        result = simulate_account(pnls, dates, prop_rules)
        assert result.final_balance <= prop_rules.account_size * 1.5  # sanity: no runaway mismatch-scale blowup


class TestExec002ReentryCooldown:
    def _whipsaw_df(self):
        ts = pd.date_range("2024-01-01", periods=5, freq="h")
        return pd.DataFrame({
            "timestamp": ts,
            "open": [100, 100, 99, 99, 99],
            "high": [100.5, 100.2, 99.5, 99.5, 99.5],
            "low": [99.8, 95.0, 98.5, 98.5, 98.5],
            "close": [100, 99, 99, 99, 99],
            "volume": [100] * 5,
        })

    def _run(self, cooldown):
        df = self._whipsaw_df()
        sig = pd.Series([1, 1, 1, 0, 0])
        risk = RiskConfig(
            initial_balance=10_000.0, risk_value=1.0, pip_size=1.0,
            spread_pips=0.0, slippage_pips=0.0, commission_per_trade=0.0,
            reentry_cooldown_bars=cooldown,
            # B2-2 (2026-10-04): fill lag is orthogonal to the cooldown --
            # lag=0 keeps this test about same-bar re-entry (now filled at
            # the bar's open instead of the old same-bar close).
            entry_fill_lag_bars=0,
        )
        trades, _ = run_execution(
            df=df, signals=sig, risk=risk,
            stop_loss_pips=2.0, take_profit_pips=None,
            stop_loss_distance=None, take_profit_distance=None,
            trailing_stop_distance=None, breakeven_trigger_r=None, partial_exit_config=None,
            adaptive_risk=None,
        )
        return trades

    def test_explicit_zero_reproduces_original_no_cooldown_behavior(self):
        """Backward compatibility: reentry_cooldown_bars=0 explicitly set
        must reproduce the ORIGINAL (undocumented) same-bar reentry
        behavior -- a stop-out on bar 1 followed by a fresh same-direction
        entry straight after it. Under the B2-2 fill model a market order
        fills at a bar's OPEN, and the stop-out happens inside that same
        bar, so a fill "on the stop-out bar" would pre-date the stop (an
        impossible fill). The engine therefore re-enters at the NEXT bar's
        open -- the earliest honest moment -- and cooldown=0 imposes no
        further wait (contrast cooldown=2, which suppresses the re-entry)."""
        trades = self._run(cooldown=0)
        assert len(trades) == 2
        assert trades[0].exit_reason == "stop_loss"
        assert trades[1].entry_time >= trades[0].exit_time
        assert trades[1].entry_time - trades[0].exit_time <= pd.Timedelta(hours=1)

    def test_default_now_blocks_same_bar_reentry(self):
        """P2-1: the default is now reentry_cooldown_bars=1 -- a stop-out
        on bar 1 can no longer re-enter at that same bar's close, even
        though the signal is still on. The cooldown only blocks the
        close bar itself, so the next bar (still signalling) re-enters:
        the second trade's entry must NOT be the same bar as the first
        trade's exit."""
        df = self._whipsaw_df()
        sig = pd.Series([1, 1, 1, 0, 0])
        risk = RiskConfig(
            initial_balance=10_000.0, risk_value=1.0, pip_size=1.0,
            spread_pips=0.0, slippage_pips=0.0, commission_per_trade=0.0,
        )
        assert risk.reentry_cooldown_bars == 1
        trades, _ = run_execution(
            df=df, signals=sig, risk=risk,
            stop_loss_pips=2.0, take_profit_pips=None,
            stop_loss_distance=None, take_profit_distance=None,
            trailing_stop_distance=None, breakeven_trigger_r=None, partial_exit_config=None,
            adaptive_risk=None,
        )
        assert len(trades) == 2
        assert trades[0].exit_reason == "stop_loss"
        assert trades[1].entry_time != trades[0].exit_time

    def test_cooldown_suppresses_same_bar_reentry(self):
        trades = self._run(cooldown=2)
        assert len(trades) == 1
        assert trades[0].exit_reason == "stop_loss"


class TestMC004MethodologyNote:
    def _trades(self):
        df = _flat_df(n=400)
        strat = _FixedSignalStrategy(_long_short_signal(400))
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        bt = run_backtest(df, strat, risk)
        assert bt.trades, "fixture must produce trades"
        return bt.trades

    def test_methodology_note_present_and_describes_method(self):
        trades = self._trades()
        rules = PropRules(account_size=10_000.0)
        result = run_monte_carlo(trades, rules, MonteCarloConfig(n_simulations=200, method="bootstrap"))
        assert result.methodology_note
        assert "bootstrap" in result.methodology_note.lower()
        assert "fixed" in result.methodology_note.lower() or "calendar" in result.methodology_note.lower()

    def test_selection_bias_caveat_changes_wording_not_numbers(self):
        trades = self._trades()
        rules = PropRules(account_size=10_000.0)
        cfg = MonteCarloConfig(n_simulations=200, method="bootstrap", random_seed=42)
        plain = run_monte_carlo(trades, rules, cfg, selection_bias_caveat=False)
        flagged = run_monte_carlo(trades, rules, cfg, selection_bias_caveat=True)
        assert "selection" in flagged.methodology_note.lower() or "optimization" in flagged.methodology_note.lower()
        assert "selection" not in plain.methodology_note.lower()
        # Same seed/config/trades -> identical numeric results; only the
        # note's wording should differ.
        assert plain.evaluation_pass_probability == flagged.evaluation_pass_probability
        assert plain.risk_of_ruin_pct == flagged.risk_of_ruin_pct


class TestVal005FoldOverlapAndSilentDrop:
    def test_build_folds_warns_on_silent_drop(self):
        """With n_folds=4 on 1000 bars (this app's own Full Pipeline/Quick
        Optimize defaults), the 4th rolling fold's train slice is empty
        and used to be dropped with NO warning at all."""
        # 2026-10-12: the rolling-window geometry was fixed (every requested fold now keeps a
        # full-size train window), so the old "4th fold has an empty train slice" trigger no longer
        # exists. A fold whose TEST slice is too small to use is still dropped WITH a warning:
        df = _flat_df(n=1000)
        warnings: list[str] = []
        folds = build_folds(df, n_folds=4, window_mode="rolling", train_frac=0.999, warnings=warnings)
        assert len(folds) == 0
        assert any("dropped" in w for w in warnings)
        assert any("fold 3" in w for w in warnings)

    def test_max_test_bar_used_matches_actual_fold_bounds(self):
        df = _flat_df(n=1000)
        folds = build_folds(df, n_folds=4, window_mode="rolling", train_frac=0.6)
        assert len(folds) == 4
        max_bar = max(f.test_end_bar for f in folds)
        assert max_bar == 1000  # 2026-10-12: all 4 folds are built; the last window ends at (n_folds+1)*span
        # ...and the GA itself reserves a tail (reserve_tail_frac=0.2), so its max_test_bar_used is 800
        ga_folds = build_folds(df.iloc[:800], n_folds=4, window_mode="rolling", train_frac=0.6)
        assert max(f.test_end_bar for f in ga_folds) == 800

    def test_embargo_eliminates_overlap_with_ga_folds(self):
        """The core VAL-005 regression: Step 4's fold windows must not
        overlap the bars Step 2's GA already selected the winner against,
        once embargo_start_bar is supplied."""
        df = _flat_df(n=1000)
        ga_folds = build_folds(df.iloc[:800], n_folds=4, window_mode="rolling", train_frac=0.6)  # GA reserves a 20% tail
        ga_touched_ranges = [(f.test_start_bar, f.test_end_bar) for f in ga_folds]
        max_test_bar_used = max(f.test_end_bar for f in ga_folds)

        config = _manual_ema_cross_config()
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)

        result = run_walk_forward(
            df, lambda: ManualStrategy(config), risk,
            n_folds=4, metric="profit_factor", embargo_start_bar=max_test_bar_used,
        )
        assert result is not None
        assert result.embargo_applied_bars == max_test_bar_used
        for fold in result.folds:
            # Every WalkForwardFold reports its test_period as
            # timestamps; recover bar positions via the dataset's own
            # timestamp index to check for overlap against the GA's
            # touched ranges.
            test_start_ts, test_end_ts = fold.test_period
            start_idx = df.index[df["timestamp"] == pd.Timestamp(test_start_ts)][0]
            assert start_idx >= max_test_bar_used, (
                f"Step 4 fold {fold.fold_index} starts at bar {start_idx}, "
                f"which is before the GA's last touched bar {max_test_bar_used}"
            )

    def test_no_embargo_reproduces_original_overlapping_behavior(self):
        """Sanity check that embargo is opt-in: omitting it must reproduce
        the original (overlapping) fold construction unchanged."""
        df = _flat_df(n=1000)
        config = _manual_ema_cross_config()
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        result = run_walk_forward(df, lambda: ManualStrategy(config), risk, n_folds=4, metric="profit_factor")
        assert result is not None
        assert result.embargo_applied_bars == 0


class TestVal006TimezoneHandling:
    def test_trade_timestamps_preserve_tz(self):
        df = _flat_df(n=400, tz="UTC")
        strat = _FixedSignalStrategy(_long_short_signal(400))
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        bt = run_backtest(df, strat, risk)
        assert bt.trades
        assert bt.trades[0].entry_time.tzinfo is not None
        assert str(bt.trades[0].entry_time.tzinfo) == "UTC"

    def test_equity_curve_timestamp_preserves_tz(self):
        df = _flat_df(n=400, tz="UTC")
        strat = _FixedSignalStrategy(_long_short_signal(400))
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        bt = run_backtest(df, strat, risk)
        assert bt.equity_curve["timestamp"].dt.tz is not None

    def test_naive_input_stays_naive(self):
        """The fix must be a no-op for the (still very common) tz-naive
        case -- never introduce a tz where none existed."""
        df = _flat_df(n=400, tz=None)
        strat = _FixedSignalStrategy(_long_short_signal(400))
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        bt = run_backtest(df, strat, risk)
        assert bt.trades
        assert bt.trades[0].entry_time.tzinfo is None
        assert bt.equity_curve["timestamp"].dt.tz is None

    def test_regime_matrix_does_not_raise_on_tz_aware_data(self):
        """The actual reported crash: build_regime_matrix used to raise
        'Cannot compare tz-naive and tz-aware datetime-like objects' on
        any tz-aware input dataset."""
        df = _flat_df(n=400, tz="UTC")
        strat = _FixedSignalStrategy(_long_short_signal(400))
        risk = RiskConfig(initial_balance=10_000.0, pip_size=0.01)
        bt = run_backtest(df, strat, risk)
        result = build_regime_matrix(df, bt.trades, initial_balance=risk.initial_balance)
        assert result is not None


# ---- from test_review_regressions.py ----

ROOT = Path(__file__).resolve().parent.parent.parent


STATIC = ROOT / "app" / "web" / "static"


TEMPLATES = ROOT / "app" / "web" / "templates"


def test_widget_pins_the_properties_page_css_would_leak_into_it():
    """index.html & ~20 other pages define a bare `textarea { min-height:140px }`;
    without an explicit reset the chat input rendered 140px tall on those pages
    (min-height beats the widget's max-height)."""
    css = (TEMPLATES / "_project_chat_widget.html").read_text(encoding="utf-8")
    assert ".t58-pc-input, .t58-pc-project-select" in css and "min-height: 0" in css
    assert "width: auto" in css.split(".t58-pc-input {")[1].split("}")[0] or "width: auto" in css


def test_reaper_runs_before_listing_and_can_close_a_job():
    jm = JobManager()
    job = jm.create(tool="Evolution Lab")
    jm.add_reaper(lambda: jm.finish(job))
    assert jm.list_jobs()[0]["done"] is True


def test_a_broken_reaper_never_breaks_the_feed():
    jm = JobManager()
    jm.create()
    def boom(): raise RuntimeError("reaper bug")
    jm.add_reaper(boom)
    assert len(jm.list_jobs()) == 1


@pytest.fixture
def server(monkeypatch):
    from app.web import server as s
    monkeypatch.setattr(s, "_EVOLUTION_RUNNER", None)
    monkeypatch.setattr(s, "_EVOLUTION_JOB_ID", None)
    monkeypatch.setattr(s, "_EVOLUTION_STARTED", False)
    return s


class _Runner:
    def __init__(self, running): self.is_running = running


def test_reaper_closes_the_entry_of_a_run_that_stopped_on_its_own(server):
    job = server.JOB_MANAGER.create(tool="Evolution Lab", progress_kind="evolution")
    server._EVOLUTION_RUNNER, server._EVOLUTION_JOB_ID, server._EVOLUTION_STARTED = _Runner(False), job, True
    server._reap_evolution_job()
    assert server.JOB_MANAGER.get(job)["done"] is True
    assert server.JOB_MANAGER.get_progress(job)["done"] is True


def test_reaper_leaves_a_running_run_alone(server):
    job = server.JOB_MANAGER.create(tool="Evolution Lab")
    server._EVOLUTION_RUNNER, server._EVOLUTION_JOB_ID, server._EVOLUTION_STARTED = _Runner(True), job, True
    server._reap_evolution_job()
    assert server.JOB_MANAGER.get(job)["done"] is False


def test_reaper_does_not_close_a_brand_new_job_whose_runner_has_not_started(server):
    """Between registering the job and runner.start(), the (old) runner reads as
    'not running'; a poll in that window used to close the NEW job."""
    job = server.JOB_MANAGER.create(tool="Evolution Lab")
    server._EVOLUTION_RUNNER, server._EVOLUTION_JOB_ID, server._EVOLUTION_STARTED = _Runner(False), job, False
    server._reap_evolution_job()
    assert server.JOB_MANAGER.get(job)["done"] is False


def test_finish_evolution_job_targets_the_id_it_was_given_and_is_idempotent(server):
    old = server.JOB_MANAGER.create(); new = server.JOB_MANAGER.create()
    server._EVOLUTION_JOB_ID = new
    server._finish_evolution_job(old)
    assert server.JOB_MANAGER.get(old)["done"] is True and server.JOB_MANAGER.get(new)["done"] is False
    server._finish_evolution_job(old); server._finish_evolution_job("unknown"); server._finish_evolution_job(None)
    server._finish_evolution_job()
    assert server.JOB_MANAGER.get(new)["done"] is True


def test_the_reaper_is_registered_on_the_shared_job_manager(server):
    assert server._reap_evolution_job in server.JOB_MANAGER._reapers


def test_evolution_log_feeds_the_tracker_without_growing_the_job_log(server):
    job = server.JOB_MANAGER.create(tool="Evolution Lab", progress_kind="evolution")
    server._EVOLUTION_JOB_ID = job
    server._evolution_log("===== GENERATION 4 =====")
    assert server.JOB_MANAGER.get_progress(job)["generation"] == 4
    assert server.JOB_MANAGER.get(job)["log"] == []


def test_sixteen_populated_families_do_not_overlap():
    """The whole taxonomy is 16 groups; with the original fixed radius the
    neighbouring clusters on the ring collided."""
    families = [f"fam{i}" for i in range(16)]
    dots = [un.Dot(f"{f}-{i}", f, "ES", "5M", 1.0, 1.0, "tested", "search", score=float(i))
            for f in families for i in range(60)]
    out = un.layout_universe(dots)
    cl = out["clusters"]
    assert len(cl) == 16
    for i in range(16):
        for j in range(i + 1, 16):
            gap = math.dist((cl[i]["cx"], cl[i]["cy"]), (cl[j]["cx"], cl[j]["cy"]))
            assert gap > cl[i]["r"] + cl[j]["r"] - 1e-9, (cl[i]["family"], cl[j]["family"])
    assert all(-1.0 <= d["x"] <= 1.0 and -1.0 <= d["y"] <= 1.0 for d in out["dots"])


def test_a_search_db_deleted_between_glob_and_stat_is_skipped(tmp_path):
    """A dangling entry (what a prune mid-scan looks like) made sorted(key=stat) raise."""
    os.symlink(tmp_path / "gone.db", tmp_path / "search_dangling.db")
    assert un.load_search_dots(tmp_path) == []


NODE = shutil.which("node")


POLL_HARNESS = r"""
const vm = require('vm'), fs = require('fs');
function run(responses, key) {
  const timers = []; let fetches = 0;
  const el = (extra) => Object.assign({ hidden: true, children: [], innerHTML: '', textContent: '', className: '',
    classList: { toggle() {} }, appendChild() {}, addEventListener() {} }, extra);
  const text = el(), pips = el(), stats = el();
  const banner = el({
    getAttribute: (n) => ({ 'data-progress-url': '/p', 'data-progress-key': key || null })[n],
    querySelector: (sel) => ({ '[data-role=text]': text, '[data-role=pips]': pips, '[data-role=stats]': stats })[sel],
  });
  const document = { readyState: 'complete', addEventListener() {},
    querySelectorAll: (sel) => sel === '[data-t58-phase-banner]' ? [banner] : [] };
  const ctx = { document, console, setTimeout: (fn) => { timers.push(fn); return timers.length; }, clearTimeout() {},
    fetch: () => { const r = responses[Math.min(fetches, responses.length - 1)]; fetches++;
                   return Promise.resolve({ json: () => Promise.resolve(r) }); } };
  ctx.globalThis = ctx; vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(process.env.TELEMETRY_JS, 'utf8'), ctx);
  return (async () => {
    for (let i = 0; i < 6; i++) { await new Promise(r => setImmediate(r)); const t = timers.shift(); if (t) t(); }
    await new Promise(r => setImmediate(r));
    return { fetches, shown: !banner.hidden, text: text.textContent, stats: stats.innerHTML };
  })();
}
(async () => {
  const snap = { banner: 'Phase 2 · Breeding · 1,240 candidates · 38/s', phase_index: 2, done: false,
                 candidates_done: 1240, candidates_total: 2000, survivors: 7, elapsed_seconds: 65,
                 best_fitness: '<b>x</b>' };
  console.log(JSON.stringify({
    live: await run([{ found: true, done: false, progress: snap }]),
    done: await run([{ found: true, done: false, progress: { ...snap, done: true } }]),
    untracked: await run([{ found: true, done: false, progress: null }]),
    missing: await run([{ found: false }]),
    evolutionIdle: await run([{ started: false }], 'progress'),
    evolutionKey: await run([{ progress: { ...snap, done: true } }], 'progress'),
  }));
})();
"""


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_banner_polls_while_live_and_stops_when_done_missing_or_untracked():
    proc = subprocess.run([NODE, "-e", POLL_HARNESS], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                          env={**os.environ, "TELEMETRY_JS": str(STATIC / "telemetry.js")})
    assert proc.returncode == 0, proc.stderr
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    assert r["live"]["fetches"] == 7 and r["live"]["shown"]                 # keeps polling a live run
    assert r["live"]["text"] == "Phase 2 · Breeding · 1,240 candidates · 38/s"
    assert "&lt;b&gt;x&lt;/b&gt;" in r["live"]["stats"] and "<b>x</b>" not in r["live"]["stats"]   # escaped
    assert r["done"]["fetches"] == 1 and r["done"]["shown"]                 # finished: render once, stop
    assert r["untracked"]["fetches"] == 1 and not r["untracked"]["shown"]   # never-tracked job: stop, stay hidden
    assert r["missing"]["fetches"] == 1                                     # server restarted: stop
    assert r["evolutionIdle"]["fetches"] > 1                                # no run yet: keep waiting for one
    assert r["evolutionKey"]["fetches"] == 1
