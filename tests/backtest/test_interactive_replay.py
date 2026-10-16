"""Tests for Interactive Replay (/replay, /replay/prepare, /replay/view,
/replay/data) -- pure visualization of an already-computed backtest, so
these tests focus on data-shape correctness and error paths rather than
backtest correctness itself (that's covered by the backtest engine's own
test suite; replay never runs its own execution/fill logic)."""
from __future__ import annotations

from pathlib import Path

from app.web.server import app

SAMPLE_CSV = Path("data/examples/EURUSD_5M_sample.csv")


def _client():
    app.config["TESTING"] = True
    return app.test_client()


def _prepare_replay(client, **overrides):
    data = {
        "strategy_mode": "manual", "sma_fast": "10", "sma_slow": "30", "sl_pips": "20", "tp_pips": "40",
        "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0", "pip_size": "0.0001",
        "max_bars": "1500",
    }
    data.update(overrides)
    with open(SAMPLE_CSV, "rb") as f:
        data["csv_file"] = (f, "EURUSD_5M_sample.csv")
        resp = client.post("/replay/prepare", data=data, content_type="multipart/form-data")
    return resp


def test_replay_form_renders():
    r = _client().get("/replay")
    assert r.status_code == 200
    assert "Interactive replay" in r.get_data(as_text=True)


def test_replay_prepare_runs_a_real_backtest_and_redirects():
    client = _client()
    resp = _prepare_replay(client)
    assert resp.status_code == 302
    assert "/replay/view/" in resp.headers["Location"]


def test_replay_view_page_renders_for_a_prepared_replay():
    client = _client()
    resp = _prepare_replay(client)
    r = client.get(resp.headers["Location"])
    assert r.status_code == 200
    assert "not found" not in r.get_data(as_text=True).lower()


def test_replay_view_page_shows_not_found_for_an_unknown_id():
    r = _client().get("/replay/view/not-a-real-id")
    assert r.status_code == 200
    assert "not found" in r.get_data(as_text=True).lower()


def test_replay_data_json_shape_is_correct():
    client = _client()
    resp = _prepare_replay(client)
    replay_id = resp.headers["Location"].rstrip("/").split("/")[-1]
    r = client.get(f"/replay/data/{replay_id}.json")
    assert r.status_code == 200
    payload = r.get_json()

    assert len(payload["bars"]) > 0
    assert len(payload["bars"]) <= 1500  # respects max_bars
    bar = payload["bars"][0]
    assert set(bar.keys()) == {"time", "open", "high", "low", "close", "volume"}
    assert isinstance(bar["time"], int)

    # Bars must be in non-decreasing time order -- a scrubber/player that
    # assumes chronological order would silently misbehave otherwise.
    times = [b["time"] for b in payload["bars"]]
    assert times == sorted(times)

    if payload["trades"]:
        trade = payload["trades"][0]
        assert set(trade.keys()) == {
            "entry_time", "exit_time", "direction", "entry_price", "exit_price", "pnl", "exit_reason",
            # UPGRADE (Interactive Replay: TP/SL fields, running balance):
            # equity_after is the backtest engine's own already-tracked
            # per-trade balance; stop_loss_price is derived from
            # initial_risk (None when the trade wasn't risk-sized off a
            # stop); take_profit_price is only ever populated when
            # exit_reason is actually "take_profit" -- never fabricated
            # for any other exit reason.
            "equity_after", "stop_loss_price", "take_profit_price",
        }
        assert trade["direction"] in (1, -1)
        # A trade can't exit before it entered.
        assert trade["exit_time"] >= trade["entry_time"]
        # take_profit_price is only ever set for an actual TP exit.
        if trade["exit_reason"] == "take_profit":
            assert trade["take_profit_price"] == trade["exit_price"]
        else:
            assert trade["take_profit_price"] is None

    if payload["equity"]:
        eq_point = payload["equity"][0]
        assert set(eq_point.keys()) == {"time", "equity"}

    assert payload["initial_balance"] == 100000.0
    assert "statistics" in payload


def test_replay_data_json_404s_for_unknown_id():
    r = _client().get("/replay/data/not-a-real-id.json")
    assert r.status_code == 404
    assert "error" in r.get_json()


def test_replay_prepare_surfaces_dataset_errors_without_a_traceback():
    client = _client()
    r = client.post("/replay/prepare", data={
        "strategy_mode": "manual", "sma_fast": "10", "sma_slow": "30", "sl_pips": "20", "tp_pips": "40",
    })
    assert r.status_code == 400
    assert "error" in r.get_data(as_text=True).lower()


def test_two_replays_prepared_with_the_same_inputs_produce_identical_trade_counts():
    """Replay must not introduce any new randomness -- it's a
    visualization of run_backtest's own deterministic output."""
    client = _client()
    resp1 = _prepare_replay(client)
    resp2 = _prepare_replay(client)
    id1 = resp1.headers["Location"].rstrip("/").split("/")[-1]
    id2 = resp2.headers["Location"].rstrip("/").split("/")[-1]
    data1 = client.get(f"/replay/data/{id1}.json").get_json()
    data2 = client.get(f"/replay/data/{id2}.json").get_json()
    assert len(data1["trades"]) == len(data2["trades"])
    assert data1["trades"] == data2["trades"]


def test_replay_form_has_library_and_pip_detect_and_prop_preset_ui():
    """UPGRADE (Interactive Replay overhaul): the form used to hard-code
    strategy_mode=manual with no way to pick a saved strategy, no pip-
    size detection, and no prop-firm preset -- all three now exist."""
    body = _client().get("/replay").get_data(as_text=True)
    assert "tab-library" in body  # strategy-library picker
    assert "detectPipSize" in body  # reuses the existing shared endpoint
    assert "initPropFirmPresetDropdown" in body  # reuses the shared preset dropdown


def test_replay_prepare_can_load_a_saved_strategy_from_the_library():
    """The 'library' strategy_mode path -- loads a real saved strategy by
    (type, name) via the same helper Full Pipeline's batch queue uses,
    instead of only ever building the fixed quick SMA-crossover config."""
    from app.strategy.library import delete_saved_strategy, save_strategy_text

    code = (
        "import numpy as np\n"
        "SMA_FAST = 10\nSMA_SLOW = 30\n"
        "def generate_signals(df):\n"
        "    fast = df['close'].rolling(SMA_FAST).mean()\n"
        "    slow = df['close'].rolling(SMA_SLOW).mean()\n"
        "    return np.where(fast > slow, 1, np.where(fast < slow, -1, 0))\n"
    )
    filename = "test_replay_library_strategy.py"
    save_strategy_text(code, filename, "python", overwrite=True)
    try:
        client = _client()
        resp = _prepare_replay(
            client, strategy_mode="library", library_strategy_type="python", library_strategy_name=filename,
        )
        assert resp.status_code == 302
        replay_id = resp.headers["Location"].rstrip("/").split("/")[-1]
        data = client.get(f"/replay/data/{replay_id}.json").get_json()
        assert len(data["trades"]) > 0
    finally:
        delete_saved_strategy("python", filename)


def test_replay_prepare_missing_library_selection_is_a_clean_error():
    client = _client()
    resp = _prepare_replay(client, strategy_mode="library", library_strategy_type="", library_strategy_name="")
    assert resp.status_code == 400
    assert "Pick a saved strategy" in resp.get_data(as_text=True)


def test_replay_data_includes_prop_account_size_for_the_balance_panel():
    """UPGRADE (Interactive Replay: running balance panel): the replay
    payload now carries prop_account_size separately from the backtest's
    own initial_balance, since a Prop Account size set on the form can
    legitimately differ from what the backtest itself was sized against
    (see replay_view.html's own comment on this exact point)."""
    client = _client()
    resp = _prepare_replay(client, account_size="50000")
    replay_id = resp.headers["Location"].rstrip("/").split("/")[-1]
    data = client.get(f"/replay/data/{replay_id}.json").get_json()
    assert data["prop_account_size"] == 50000.0
