"""v9.8 lifecycle consolidation:

- the sidebar's seven lifecycle steps are single links now (children/tool
  tabs moved onto the section pages' Individual tools grids) while the
  rest of the sidebar (Quant Lab / Hedge Fund Manager / Account /
  Education) is untouched;
- every dataset picker groups files under instrument headers (ES1! then
  its files sorted, then NQ1!, ...) via ONE shared helper
  (app.data.storage.list_datasets_grouped_for_picker);
- the Create page takes market data as ONE grouped multi-select and its
  timeframes as a real multi-select;
- the Test page (Full Pipeline) strategy picker is ONE saved-strategy
  dropdown (no Manual/Python/PineScript/MQL5 tab row in the primary
  flow);
- Deploy (forward test + deploy live) has a saved-strategy dropdown
  wired into the same strategy_mode / existing_strategy_* fields the
  backends have always read;
- every run page carries an explicit `timeframe` selector that actually
  resamples the data before the run (5m from 1m bars -> 1/5 the bars)
  and refuses a pick finer than the data's native bars with a 400 and
  the reason, instead of silently running 1m.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.data.storage as storage
import app.web.server as server


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


@pytest.fixture()
def raw_dir(tmp_path, monkeypatch):
    """Point storage at a raw dir with flat, instrument-named CSVs."""
    raw = tmp_path / "raw"
    raw.mkdir()
    header = "timestamp,open,high,low,close,volume\n"
    body = "".join(f"2024-01-01 09:{i:02d}:00,1,1,1,1,10\n" for i in range(3))
    for name in ("ES1!_5m.csv", "ES1!_15m.csv", "NQ1!_5m.csv"):
        (raw / name).write_text(header + body)
    (raw / "EURUSD").mkdir()
    (raw / "EURUSD" / "EURUSD5.csv").write_text(header + body)
    monkeypatch.setattr(storage, "get_raw_data_dir", lambda: raw)
    return raw


# ------------------------------------------------------------- (a) sidebar

def test_sidebar_lifecycle_steps_are_single_links(client):
    body = client.get("/dashboard").get_data(as_text=True)
    start = body.index('id="t58-sidebar"')
    sidebar = body[start:body.index('<div class="t58-main"', start)]
    for href in ('href="/start-here/create"', 'href="/full-pipeline"',
                 'href="/optimize-simple"', 'href="/validate-simple"',
                 'href="/champion-simple"', 'href="/forward-test"',
                 'href="/graveyard"'):
        assert href in sidebar, href
    # Children moved out of the sidebar onto the section pages
    for href in ('href="/cpcv"', 'href="/quick-optimize"',
                 'href="/walk-forward-opt"', 'href="/evolution"'):
        assert href not in sidebar, href
    # Untouched groups
    assert 'href="/quant-lab"' in sidebar
    assert 'href="/settings/notifications"' in sidebar
    assert sidebar.index("/settings/notifications") < sidebar.index("/settings/api-keys")
    assert 'href="/data-center"' in sidebar
    assert 'href="/support"' in sidebar


# ------------------------------------------------- (b) grouping helper

def test_dataset_instrument_label_examples():
    assert storage.dataset_instrument_label("ES1!_5m.csv") == "ES1!"
    assert storage.dataset_instrument_label("EURUSD/EURUSD5.csv") == "EURUSD"
    assert storage.dataset_instrument_label("AAA_5M.csv") == "AAA"
    assert storage.dataset_instrument_label("") == "(UNGROUPED)"


def test_grouped_picker_sorts_groups_and_files(monkeypatch):
    def fake(count_rows=False):
        return [{"instrument": "(ungrouped)", "files": [
            {"name": "NQ1!_5m.csv", "full_name": "NQ1!_5m.csv", "size_bytes": 100, "rows": 10, "empty": False},
            {"name": "ES1!_5m.csv", "full_name": "ES1!_5m.csv", "size_bytes": 100, "rows": 10, "empty": False},
            {"name": "ES1!_15m.csv", "full_name": "ES1!_15m.csv", "size_bytes": 100, "rows": 10, "empty": False},
            {"name": "EURUSD5.csv", "full_name": "EURUSD/EURUSD5.csv", "size_bytes": 100, "rows": 10, "empty": False},
        ], "file_count": 4, "empty_count": 0}]

    monkeypatch.setattr(storage, "list_datasets_by_instrument", fake)
    groups = storage.list_datasets_grouped_for_picker()
    assert [g["instrument"] for g in groups] == ["ES1!", "EURUSD", "NQ1!"]
    by_label = {g["instrument"]: [f["name"] for f in g["files"]] for g in groups}
    assert by_label["ES1!"] == ["ES1!_15m.csv", "ES1!_5m.csv"]
    assert by_label["NQ1!"] == ["NQ1!_5m.csv"]
    assert by_label["EURUSD"] == ["EURUSD5.csv"]


# ------------------------------------------------ (c) grouped pickers

def test_validate_page_grouped_optgroups_and_timeframe(client, raw_dir):
    body = client.get("/validate-simple").get_data(as_text=True)
    assert '<optgroup label="ES1!">' in body
    assert '<option value="ES1!_5m.csv">' in body
    assert 'name="timeframe"' in body
    assert "Dataset's native timeframe (default)" in body


def test_optimize_and_champion_pages_carry_timeframe(client, raw_dir):
    for url in ("/optimize-simple", "/champion-simple"):
        body = client.get(url).get_data(as_text=True)
        assert 'name="timeframe"' in body, url
        assert '<optgroup label="ES1!">' in body, url


def test_create_page_single_multi_select(client, raw_dir):
    body = client.get("/start-here/create").get_data(as_text=True)
    assert 'name="datasets" multiple' in body
    assert '<optgroup label="ES1!">' in body
    assert 'id="create_timeframes_pick"' in body
    assert 'name="timeframes"' in body


# ---------------------------------------------------- (d) one dropdown

def test_full_pipeline_one_simple_strategy_dropdown(client, raw_dir):
    body = client.get("/full-pipeline").get_data(as_text=True)
    assert 'id="lc-strategy-pick"' in body
    assert 'data-mode="pinescript"' not in body
    assert 'id="existing_strategy_select"' not in body
    assert 'name="timeframe"' in body
    assert '<optgroup label="ES1!">' in body


def test_forward_test_has_saved_strategy_dropdown(client):
    body = client.get("/forward-test").get_data(as_text=True)
    assert 'id="lc-deploy-strategy-pick"' in body
    assert 'id="ft-existing-strategy"' in body
    assert 'href="/deploy-live"' in body


# ------------------------------------------- (e) timeframe is wired

def _fake_strategy_patches(monkeypatch, captured):
    monkeypatch.setattr(server, "load_strategy_text", lambda t, f: "code")
    monkeypatch.setattr(server, "build_strategy_from_code",
                        lambda t, code: SimpleNamespace(name="Stub"))
    stats = SimpleNamespace(net_profit=10.0, win_rate=50.0, max_drawdown_pct=1.0)

    def fake_backtest(df_, s, risk):
        captured["bars"] = len(df_)
        return SimpleNamespace(trades=[], statistics=stats)

    monkeypatch.setattr(server, "run_backtest", fake_backtest)
    monkeypatch.setattr(server, "run_cpcv", lambda *a, **k: SimpleNamespace(
        n_paths=3, metric="profit_factor", mean_oos_metric=1.2,
        median_oos_metric=1.1, mean_is_metric=1.3, is_robust=True))
    monkeypatch.setattr(
        server, "run_monte_carlo",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("mc not ticked")))


def _ohlcv(bars, start, freq):
    import pandas as pd
    ts = pd.date_range(start, periods=bars, freq=freq)
    return pd.DataFrame({
        "timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.5, "volume": 10.0,
    })


def test_validate_timeframe_pick_resamples_before_the_run(client, monkeypatch):
    df = _ohlcv(300, "2024-01-01 09:00", "1min")
    captured = {}
    monkeypatch.setattr(server, "_resolve_dataset",
                        lambda form, files: (df, "tiny.csv", "", None))
    _fake_strategy_patches(monkeypatch, captured)

    r = client.post("/validate-simple/start", data={
        "strategy_pick": "python::stub.py",
        "existing_dataset": "tiny.csv",
        "check_cpcv": "on",
        "timeframe": "5m",
        "initial_balance": "100000", "account_size": "100000",
    })
    assert r.status_code == 200
    assert captured["bars"] == 60  # 300 one-minute bars -> 60 five-minute bars


def test_validate_manual_library_pick_actually_loads(client, monkeypatch):
    """v9.8 audit fix: a 'manual::' strategy_pick used to die with
    'Unknown strategy mode: manual' because _simple_load_strategy called
    build_strategy_from_code (no manual branch). It now goes through
    _build_strategy, like every other run page."""
    import json as _json

    df = _ohlcv(300, "2024-01-01 09:00", "1min")
    captured = {}
    monkeypatch.setattr(server, "_resolve_dataset",
                        lambda form, files: (df, "tiny.csv", "", None))
    _fake_strategy_patches(monkeypatch, captured)
    # Then override the seams _fake_strategy_patches set: the manual JSON
    # comes from the library loader, and build_strategy_from_code -- the
    # function the bug lived behind -- must not be needed at all.
    monkeypatch.setattr(server, "load_strategy_text", lambda t, f: _json.dumps({
        "name": "Stub Manual",
        "indicators": [
            {"type": "sma", "period": 20, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 50, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow", "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow", "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20, "take_profit_pips": 40,
    }))
    monkeypatch.setattr(server, "build_strategy_from_code",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("manual pick bypassed _build_strategy")))

    r = client.post("/validate-simple/start", data={
        "strategy_pick": "manual::stub.json",
        "existing_dataset": "tiny.csv",
        "check_cpcv": "on",
        "initial_balance": "100000", "account_size": "100000",
    })
    assert r.status_code == 200
    assert captured["bars"] == 300


def test_champion_rerun_manual_candidate_actually_loads(client, monkeypatch):
    """Same audit fix on the Champion page: the re-run handler used to
    call build_strategy_from_code directly, so a 'manual::' candidate
    could never load there either."""
    import json as _json

    df = _ohlcv(300, "2024-01-01 09:00", "1min")
    captured = {}
    monkeypatch.setattr(server, "_resolve_dataset",
                        lambda form, files: (df, "tiny.csv", "", None))
    _fake_strategy_patches(monkeypatch, captured)
    monkeypatch.setattr(server, "load_strategy_text", lambda t, f: _json.dumps({
        "name": "Stub Manual",
        "indicators": [
            {"type": "sma", "period": 20, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 50, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow", "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow", "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20, "take_profit_pips": 40,
    }))
    monkeypatch.setattr(server, "build_strategy_from_code",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("manual candidate bypassed _build_strategy")))

    r = client.post("/champion-simple/rerun", data={
        "candidate": "manual::stub.json",
        "existing_dataset": "tiny.csv",
        "check_cpcv": "on",
        "initial_balance": "100000", "account_size": "100000",
    })
    assert r.status_code == 200
    assert "Re-run results" in r.get_data(as_text=True)
    assert captured["bars"] == 300


def test_handoff_prefill_marks_options_selected(client, monkeypatch, raw_dir):
    """Guided flow: the previous step's Next button puts the picks in
    the URL; the receiving page renders them selected."""
    monkeypatch.setattr(server, "_simple_strategy_options",
                        lambda: [{"type": "manual", "name": "stub.json",
                                  "filename": "stub.json"}])
    body = client.get(
        "/validate-simple?strategy_pick=manual::stub.json"
        "&existing_dataset=ES1!_5m.csv&timeframe=5m&prop_preset=lucid_50k"
    ).get_data(as_text=True)
    assert '<option value="manual::stub.json" selected>' in body
    assert '<option value="ES1!_5m.csv" selected>' in body
    assert '<option value="lucid_50k" selected>' in body
    assert '<option value="5m" selected>' in body

    body = client.get(
        "/champion-simple?existing_dataset=ES1!_5m.csv&timeframe=15m"
    ).get_data(as_text=True)
    assert '<option value="ES1!_5m.csv" selected>' in body
    assert '<option value="15m" selected>' in body


def test_full_pipeline_and_deploy_prefill(client, monkeypatch, raw_dir):
    body = client.get(
        "/full-pipeline?existing_dataset=ES1!_5m.csv&timeframe=5m&prop_preset=lucid_50k"
    ).get_data(as_text=True)
    assert '<option value="ES1!_5m.csv" selected>' in body
    assert '<option value="5m" selected>' in body
    assert 'const want = "lucid_50k"' in body

    monkeypatch.setattr(server, "_simple_strategy_options",
                        lambda: [{"type": "manual", "name": "stub.json",
                                  "filename": "stub.json"}])
    body = client.get("/forward-test?strategy=manual::stub.json").get_data(as_text=True)
    assert 'pick.value = "manual::stub.json"' in body


def test_next_step_buttons_present(client, raw_dir):
    for url in ("/full-pipeline", "/optimize-simple", "/validate-simple",
                "/champion-simple"):
        assert 'id="lc-next-step"' in client.get(url).get_data(as_text=True), url
    assert "Next step: Monitor" in client.get("/forward-test").get_data(as_text=True)
    assert "Next step: Test" in client.get("/start-here/create").get_data(as_text=True)


def test_full_pipeline_timeframe_error_renders_not_500(client, monkeypatch):
    """v9.8 audit fix: /full-pipeline/start referenced tf_note without
    ever calling _apply_timeframe_choice (the wiring had landed in the
    batch path only), so ANY Full Pipeline run 500'd with NameError,
    and the error render itself crashed on the new prefill fields.
    An unbuildable timeframe pick must come back as an honest 400."""
    df = _ohlcv(300, "2024-01-01 09:00", "15min")
    monkeypatch.setattr(server, "_resolve_dataset",
                        lambda form, files: (df, "tiny.csv", "", None))
    r = client.post("/full-pipeline/start", data={
        "strategy_mode": "manual", "existing_dataset": "tiny.csv",
        "timeframe": "5m",
    })
    assert r.status_code == 400
    assert "FINER" in r.get_data(as_text=True)


def test_validate_timeframe_finer_than_native_is_an_honest_400(client, monkeypatch):
    df = _ohlcv(300, "2024-01-01 09:00", "15min")
    captured = {}
    monkeypatch.setattr(server, "_resolve_dataset",
                        lambda form, files: (df, "tiny.csv", "", None))
    _fake_strategy_patches(monkeypatch, captured)

    r = client.post("/validate-simple/start", data={
        "strategy_pick": "python::stub.py",
        "existing_dataset": "tiny.csv",
        "check_cpcv": "on",
        "timeframe": "5m",
        "initial_balance": "100000", "account_size": "100000",
    })
    assert r.status_code == 400
    assert "FINER" in r.get_data(as_text=True)
    assert "bars" not in captured  # the run never happened on wrong data
