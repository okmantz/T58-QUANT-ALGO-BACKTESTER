"""Auto-Fix: acceptance rules, the two repairs, the verification loop, and the web flow end to end."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy
from app.validation import auto_fix, simple_report
from app.validation.auto_fix import (
    check_score, fix_regime, regime_holdout_verdict, run_auto_fix, tail_verdict, window_metrics, with_regime_exclusion,
)


# ---------------------------------------------------------------- synthetic market

def _market(n=12000, seed=3, block=400):
    """Smooth persistent trends alternate with violent mean-reverting chop. An EMA-cross strategy makes
    money in the trends and bleeds in the chop, so there is a real, repeating losing regime to find."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    price = [1900.0]
    for i in range(1, n):
        if (i // block) % 2 == 0:
            step = 0.15 * (1 if (i // (block * 2)) % 2 == 0 else -1) + rng.normal(0, 0.25)
        else:
            step = -0.35 * (price[-1] - price[max(0, i - 8)]) / 8 + rng.normal(0, 1.4)
        price.append(price[-1] + step)
    p = np.array(price)
    return pd.DataFrame({"timestamp": ts, "open": p, "high": p + np.abs(rng.normal(0.3, 0.15, n)),
                         "low": p - np.abs(rng.normal(0.3, 0.15, n)), "close": p, "volume": 100.0})


_CFG = {"name": "ema cross",
        "indicators": [{"type": "ema", "period": 10, "column": "close", "as": "ema_fast"},
                       {"type": "ema", "period": 30, "column": "close", "as": "ema_slow"}],
        "long_entry": "ema_fast > ema_slow", "short_entry": "ema_fast < ema_slow",
        "stop_loss_pips": 60, "take_profit_pips": 120}
_RISK = RiskConfig(initial_balance=100000, risk_mode="percent", risk_value=1.0, pip_size=0.1)
_NO_BUILD = lambda mode, code: (_ for _ in ()).throw(AssertionError("manual strategies never need the code builder"))


# ---------------------------------------------------------------- acceptance rules (exact numbers)

def _m(trades, net, pf, exp=None):
    return {"trades": trades, "net_profit": net, "profit_factor": pf, "expectancy": (net / trades if trades else 0.0) if exp is None else exp}


def test_regime_verdict_accepts_a_filter_that_helps_unseen_data():
    ok, why = regime_holdout_verdict(_m(50, 100.0, 1.2), _m(45, 150.0, 1.4))
    assert ok and why == []


@pytest.mark.parametrize("after,needle", [
    (_m(45, 90.0, 1.4), "net profit gets worse"),
    (_m(45, 150.0, 1.1), "profit factor gets worse"),
    (_m(10, 150.0, 1.4), "removes too much"),
    (_m(50, 100.0, 1.2), "no difference"),
])
def test_regime_verdict_rejects(after, needle):
    ok, why = regime_holdout_verdict(_m(50, 100.0, 1.2), after)
    assert ok and any(needle in w for w in why), why


def test_regime_verdict_unverifiable_when_unseen_window_is_thin():
    ok, why = regime_holdout_verdict(_m(6, 100.0, 1.2), _m(6, 200.0, 2.0))
    assert not ok and "too few" in why[0]


def test_tail_verdict():
    base = _m(60, 6000.0, 1.5)                                         # 100 per trade
    assert tail_verdict(base, _m(40, 4200.0, 1.9)) == []               # fewer trades, better quality -> fine
    assert any("loses money" in r for r in tail_verdict(base, _m(40, -5.0, 1.9)))
    assert any("profit factor" in r for r in tail_verdict(base, _m(60, 6000.0, 1.0)))
    assert any("per trade" in r for r in tail_verdict(base, _m(60, 3000.0, 1.6)))
    assert any("half as many" in r for r in tail_verdict(base, _m(20, 3000.0, 1.6)))
    assert "only 2 trades" in tail_verdict(base, _m(2, 500.0, 3.0))[0]


def test_window_metrics_caps_profit_factor_and_handles_empty():
    t = lambda p: SimpleNamespace(pnl=p)
    assert window_metrics([])["trades"] == 0
    assert window_metrics([t(5), t(7)])["profit_factor"] == 10.0                    # no losers
    assert window_metrics([t(1000), t(-1)])["profit_factor"] == 10.0               # absurd PF capped
    m = window_metrics([t(30), t(-10), t(-10)])
    assert m["net_profit"] == 10 and m["profit_factor"] == pytest.approx(1.5) and m["win_rate"] == pytest.approx(100 / 3)


def test_check_score_direction_is_higher_is_better():
    assert check_score("regime", {"n_disable_recommended": 3}) > check_score("regime", {"n_disable_recommended": 4})
    assert check_score("monte_carlo", {"evaluation_pass_probability": 60}) > check_score("monte_carlo", {"evaluation_pass_probability": 50})
    assert check_score("pbo", {"pbo": 0.2}) > check_score("pbo", {"pbo": 0.6})
    assert check_score("wfga", {"overfitting_gap": 1.0, "in_sample_fitness": 10}) > check_score("wfga", {"overfitting_gap": 5.0, "in_sample_fitness": 10})
    assert check_score("sensitivity", {"n_parameters": 4, "n_cliffs": 0}) > check_score("sensitivity", {"n_parameters": 4, "n_cliffs": 2})
    assert check_score("walk_forward", None) is None and check_score("nope", {}) is None


# ---------------------------------------------------------------- strategy editing

def test_manual_exclusion_merges_without_duplicates():
    s = ManualStrategy({**_CFG, "filters": {"regime_exclude": [{"volatility": "extreme"}]}})
    new, text, ext = with_regime_exclusion(s, [{"volatility": "extreme"}, {"volatility": "high", "environment": "ranging"}], _NO_BUILD)
    assert ext == "json"
    assert json.loads(text)["filters"]["regime_exclude"] == [{"volatility": "extreme"}, {"volatility": "high", "environment": "ranging"}]
    assert s.config["filters"]["regime_exclude"] == [{"volatility": "extreme"}]       # original object untouched


def test_pinescript_exclusion_keeps_cells_the_file_already_declares():
    code = "//@version=5\nstrategy('x')\n// T58_EXCLUDE_REGIMES=volatility:extreme\n"
    s = SimpleNamespace(source_type="pinescript", code=code)
    built = {}
    new, text, ext = with_regime_exclusion(s, [{"volatility": "high", "environment": "ranging"}],
                                           lambda mode, c: built.setdefault("code", c) or SimpleNamespace(source_type=mode, code=c))
    assert ext == "pine" and built["code"] == text
    last = text.strip().splitlines()[-1]
    assert "volatility:extreme" in last and "volatility:high,environment:ranging" in last


def test_shrink_gene_keeps_base_inside_and_integers_integer():
    from app.optimize.parameter_space import GeneMeta
    from app.optimize.walkforward_ga import _shrink_gene
    g = GeneMeta(path=("a",), kind="period", is_int=True, lo=2.0, hi=102.0, base_value=20.0, label="a")
    h = _shrink_gene(g, 0.25)
    assert h.lo <= 20.0 <= h.hi and (h.hi - h.lo) < (g.hi - g.lo) and h.lo == int(h.lo) and h.hi == int(h.hi)
    assert h.lo >= g.lo and h.hi <= g.hi


# ---------------------------------------------------------------- regime repair on a synthetic market

def test_fix_regime_finds_losing_regimes_and_proves_it_on_unseen_data():
    df = _market()
    res = fix_regime(df, ManualStrategy(_CFG), _RISK, build_from_code=_NO_BUILD)
    assert res.attempted and res.accepted, (res.headline, res.reasons)
    d = res.details
    assert d["excluded_cells"] and all(isinstance(c, dict) for c in d["excluded_cells"])
    # discovered on the first 70%, judged on the last 30% -- and genuinely better there
    assert "first 70%" in d["discovery_window"] and "last 30%" in d["verification_window"]
    assert d["after_holdout"]["net_profit"] >= d["before_holdout"]["net_profit"]
    assert d["after_holdout"]["profit_factor"] >= d["before_holdout"]["profit_factor"]
    # the repaired strategy really carries the filter, and the original was not modified
    assert res.strategy.config["filters"]["regime_exclude"] == d["excluded_cells"]
    assert "filters" not in _CFG
    assert json.loads(res.source_text)["filters"]["regime_exclude"] == d["excluded_cells"]


def test_fix_regime_with_no_trades_is_not_attempted():
    flat = _market()
    flat["close"] = flat["open"] = flat["high"] = flat["low"] = 1900.0
    res = fix_regime(flat, ManualStrategy(_CFG), _RISK, build_from_code=_NO_BUILD)
    assert not res.attempted and not res.accepted and res.strategy is None


# ---------------------------------------------------------------- orchestrator

def test_run_auto_fix_reports_nothing_fixable_for_monte_carlo_only():
    out = run_auto_fix(_market(), ManualStrategy(_CFG), _RISK, PropRules(account_size=100000), failed_checks=["monte_carlo"],
                       before_items=[{"check": "monte_carlo", "label": "Monte Carlo", "status": "fail", "headline": "x"}],
                       form={}, build_from_code=_NO_BUILD)
    assert not out.accepted and out.strategy is None and "Optimize" in " ".join(out.notes)


def test_run_auto_fix_repairs_then_reruns_the_real_checks():
    from app.validation.runner import dispatch_validation_checks
    from app.backtest.engine import run_backtest
    df, rules, strat = _market(), PropRules(account_size=100000), ManualStrategy(_CFG)
    form = {"mc_sims": 200}
    before = dispatch_validation_checks(df, strat, _RISK, rules, ["monte_carlo", "regime"], form,
                                        baseline_trades=run_backtest(df, strat, _RISK).trades)
    items = [simple_report.interpret_check(r) for r in before]
    assert {i["check"]: i["status"] for i in items}["regime"] == "fail"
    events = []
    out = run_auto_fix(df, strat, _RISK, rules, failed_checks=["regime"], before_items=items,
                       before_raw={r["check"]: r["summary"] for r in before if r["ok"]}, form=form,
                       build_from_code=_NO_BUILD, verify_checks=["monte_carlo", "regime"],
                       event_cb=lambda k, **kw: events.append(k))
    assert out.accepted, (out.headline, out.notes)
    assert [e for e in events if e in ("stage_start", "stage_done", "verify_start")] == ["stage_start", "stage_done", "verify_start"]
    after = {r["check"]: r for r in out.after}
    assert set(after) == {"monte_carlo", "regime"}                         # the SAME checks were re-run on the copy
    flagged_before = before[1]["summary"]["n_disable_recommended"]
    flagged_after = next(r for r in out.after_results if r["check"] == "regime")["summary"]["n_disable_recommended"]
    assert flagged_after < flagged_before
    assert out.improved or out.newly_passing
    assert not [r for r in out.regressions if r["status"] in ("fail", "error")]
    assert out.source_text and out.strategy is not strat
    json.dumps(out.to_dict(), default=float)                               # the outcome is serialisable for the page


def test_overfit_repair_runs_with_narrowed_ranges_and_never_touches_the_original():
    from app.optimize.walkforward_ga import run_walkforward_aware_refinement
    from app.optimize.refinement import RefinementConfig
    from app.monte_carlo.engine import MonteCarloConfig
    df, rules = _market(6000), PropRules(account_size=100000)
    cfg = RefinementConfig(population_size=4, generations=1, search_monte_carlo_sims=100)
    mc = MonteCarloConfig(n_simulations=100)
    full = run_walkforward_aware_refinement(df, ManualStrategy(_CFG), _RISK, rules, mc, cfg, n_folds=2, parallel=False)
    narrow = run_walkforward_aware_refinement(df, ManualStrategy(_CFG), _RISK, rules, mc, cfg, n_folds=2, parallel=False,
                                              gene_range_scale=0.25)
    assert len(full.genes) == len(narrow.genes) > 0
    assert all((n.hi - n.lo) <= (f.hi - f.lo) for f, n in zip(full.genes, narrow.genes))
    assert any((n.hi - n.lo) < (f.hi - f.lo) for f, n in zip(full.genes, narrow.genes))
    assert all(n.lo <= n.base_value <= n.hi for n in narrow.genes)
    assert "stop_loss_pips" not in json.dumps(_CFG.get("filters", {}))


# ---------------------------------------------------------------- web flow, end to end

@pytest.fixture()
def web(tmp_path, monkeypatch):
    import app.strategy.library as library
    import app.web.server as server
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path / "lib")
    library.get_strategy_library_dir()
    monkeypatch.setattr(server, "VALIDATION_RUNS_DIR", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield server, library, c


def _wait(client, job_id, timeout=240):
    end = time.time() + timeout
    data = {}
    while time.time() < end:
        data = client.get(f"/validate-fix/job/{job_id}/status.json").get_json()
        if data.get("done"):
            return data
        time.sleep(0.3)
    raise AssertionError(f"fix job did not finish: {data}")


def test_fix_flow_saves_a_new_copy_and_a_report_without_touching_the_original(web, monkeypatch):
    from app.backtest.engine import run_backtest
    from app.validation import report_export
    from app.validation.runner import dispatch_validation_checks
    server, library, client = web

    df, rules = _market(), PropRules(account_size=100000)
    orig_text = json.dumps(_CFG, indent=2)
    library.save_strategy_text(orig_text, "orig.json", "manual")

    # 1) a finished validation run saved as a report (what the Validate page does at the end)
    strat = ManualStrategy(_CFG)
    form = {"existing_dataset": "synthetic.csv", "strategy_pick": "manual::orig.json", "mc_sims": "200"}
    results = dispatch_validation_checks(df, strat, _RISK, rules, ["monte_carlo", "regime"], form,
                                         baseline_trades=run_backtest(df, strat, _RISK).trades)
    report = simple_report.build_report(results, context={})
    report["strategy"] = "orig.json"
    doc = report_export.build_report_document(
        report, raw={r["check"]: r["summary"] for r in results if r["ok"]}, meta={"strategy": "orig.json", "dataset": "synthetic.csv"},
        form=form, requested_checks=["monte_carlo", "regime"])
    report_export.save_report(server.VALIDATION_RUNS_DIR, doc)
    assert "regime" in server._vs_fixable_checks(report)

    # 2) the fix job re-loads the same data / strategy / settings from that report
    monkeypatch.setattr(server, "_resolve_dataset", lambda f, files: (df, "synthetic.csv", "", None))
    monkeypatch.setattr(server, "_simple_load_strategy", lambda f: (ManualStrategy(_CFG), ("manual", "orig.json")))
    monkeypatch.setattr(server, "_simple_prop_rules", lambda f: rules)
    monkeypatch.setattr(server, "_simple_base_risk", lambda f, r: _RISK)

    r = client.post("/validate-fix/start", data={"report_id": doc["report_id"], "checks": "regime"})
    assert r.status_code == 302 and "/validate-fix/job/" in r.headers["Location"]
    job_id = r.headers["Location"].rsplit("/", 1)[-1]
    data = _wait(client, job_id)
    assert not data["error"], data["error"]
    out = data["outcome"]
    assert out["accepted"], out["headline"]
    assert {a["check"] for a in out["after"]} == {"monte_carlo", "regime"}

    # 3) a NEW library strategy was saved, carrying the filter; the original is byte-identical
    fx = data["fixed_strategy"]
    assert fx and fx["filename"] != "orig.json" and fx["filename"].startswith("orig_fixed")
    saved = json.loads(library.load_strategy_text("manual", fx["filename"]))
    assert saved["filters"]["regime_exclude"] and saved["name"].endswith("(fixed)")
    assert library.load_strategy_text("manual", "orig.json") == orig_text
    meta = library.load_strategy_metadata("manual", fx["filename"])
    assert meta["auto_fix"]["fixed_from"] == "orig.json"
    assert meta["last_validate_run"]["report_id"] == data["new_report_id"]    # re-run results count for the NEW copy

    # 4) the fixed copy's own report is downloadable and says what changed
    page = client.get(data["downloads"]["html"]).get_data(as_text=True)
    assert "Auto-Fix" in page and "Regime filter" in page
    assert client.get("/validate-fix/job/" + job_id).status_code == 200


def test_fix_start_refuses_unknown_report_and_unfixable_report(web):
    server, library, client = web
    r = client.post("/validate-fix/start", data={"report_id": "does-not-exist"})
    assert r.status_code == 302 and "error=" in r.headers["Location"]
    from app.validation import report_export
    results = [{"check": "monte_carlo", "ok": True, "error": None, "summary": {"evaluation_pass_probability": 30.0}}]
    report = simple_report.build_report(results, context={})
    doc = report_export.build_report_document(report, raw={"monte_carlo": results[0]["summary"]}, meta={"strategy": "s"}, form={})
    report_export.save_report(server.VALIDATION_RUNS_DIR, doc)
    r = client.post("/validate-fix/start", data={"report_id": doc["report_id"]})
    assert r.status_code == 302 and "error=" in r.headers["Location"]          # Monte Carlo alone is not auto-fixable


def test_results_page_offers_download_and_autofix_buttons(web):
    server, library, client = web
    page = client.get("/validate-simple/job/not-a-job").get_data(as_text=True)   # not_found branch still renders
    assert "Validate" in page
    html = open(server.app.root_path + "/templates/validate_simple_job.html", encoding="utf-8").read()
    for needle in ("download-card", "Auto-fix this", "/validate-fix/start", "Saved reports", "fixable_checks"):
        assert needle in html, needle
