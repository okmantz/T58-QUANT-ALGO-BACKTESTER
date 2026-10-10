"""v9.15 WS-G: AI Assistant repair -- three Windows failures from Owen's
v9.14 run, reproduced and fixed headlessly:

(a) BEST MARKETS headline was raw per-symbol feed errors
    ("EURUSD: No H1 bars returned for this symbol. | ..."). Symbols
    with no bars anywhere are now skipped quietly; rankings compute
    from real bars (feed first, then the app's own local datasets).
(b) BASIC OUTLOOK showed "OSError: could not get source code" --
    inspect.getsource on a compiled build. The capability grounding
    now never raises and the outlook keeps its deterministic text.
(c) AI DIRECTOR sat on "Loading... / Generating..." indefinitely --
    the scan + Ollama halves are now each bounded through the shared
    v9.14 transport, and the endpoint returns the deterministic list
    with the transport's named error instead of hanging.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pandas as pd
import pytest
from flask import Flask

from app.ai import capability_reference as cr
from app.ai import market_intelligence, market_scanner, trading_assistant
from app.ai.news_forexfactory import CalendarResult
from app.ai.ollama_settings import OllamaSettings
from app.web import ai_assistant_routes as routes


def _make_client():
    app = Flask(__name__)
    app.register_blueprint(routes.ai_assistant_bp)
    return app.test_client()


def _bars(n=360, start=1.1000, step=0.0002, freq_minutes=60):
    ts = pd.date_range("2024-01-01", periods=n, freq=f"{freq_minutes}min", tz="UTC")
    close = [start + step * i for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts,
        "open": close, "high": [c + 0.0005 for c in close],
        "low": [c - 0.0005 for c in close], "close": close,
        "volume": [100.0] * n,
    })


# ----------------------------------------------------------------------
# (a) Best Markets: no-data symbols skipped quietly, rankings survive
# ----------------------------------------------------------------------

def test_rank_markets_skips_no_data_symbol_quietly():
    def fake_fetcher(symbol, timeframe_minutes, count):
        if symbol == "EURUSD":
            return _bars(n=360, freq_minutes=timeframe_minutes)
        return None

    rankings, errors = market_scanner.rank_markets(
        {"forex": ["EURUSD", "ZZZNODATA"]}, bar_fetcher=fake_fetcher,
    )
    assert [r.symbol for r in rankings] == ["EURUSD"]
    assert errors == []
    assert "No H1 bars" not in str(errors)


def _write_bars_csv(path, n=4200):
    df = _bars(n=n, freq_minutes=5)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df.to_csv(path, index=False)
    return path


def test_local_fallback_computes_rankings_and_skips_unknown(monkeypatch, tmp_path):
    csv_path = _write_bars_csv(tmp_path / "EURUSD_5M_sample.csv")
    dataset = SimpleNamespace(name="EURUSD_5M_sample.csv", path=csv_path, size_bytes=csv_path.stat().st_size)

    monkeypatch.setattr(market_intelligence, "_feed_bars", lambda *a: None)
    monkeypatch.setattr(
        market_intelligence, "_find_local_dataset",
        lambda symbol: dataset if symbol == "EURUSD" else None,
    )
    market_intelligence._LOCAL_DATASET_CACHE.clear()

    rankings, errors = market_intelligence.compute_rankings(
        news_result=CalendarResult(events=[], error=None),
        universe={"forex": ["EURUSD", "USDJPY"]},
    )
    assert [r.symbol for r in rankings] == ["EURUSD"]  # USDJPY: no data anywhere -> quiet skip
    assert errors == []
    assert "No H1 bars" not in str(errors)


def test_local_mapping_is_honest_by_underlying():
    mi = market_intelligence
    assert mi._dataset_match_score("ES1!/ES1!_2024.parquet", mi._local_instrument_candidates("SPX500")) is not None
    assert mi._dataset_match_score("NQ1!/part.parquet", mi._local_instrument_candidates("NAS100")) is not None
    assert mi._dataset_match_score("GC1!/part.parquet", mi._local_instrument_candidates("MGC")) is not None
    assert mi._dataset_match_score("EURUSD_5M_sample.csv", mi._local_instrument_candidates("EURUSD")) is not None
    # No honest mapping -> no match (never "close enough"):
    assert mi._dataset_match_score("ES_5m_synth.csv", mi._local_instrument_candidates("US30")) is None
    assert mi._dataset_match_score("ES_5m_synth.csv", mi._local_instrument_candidates("EURUSD")) is None


def test_feed_still_wins_over_local(monkeypatch, tmp_path):
    csv_path = _write_bars_csv(tmp_path / "EURUSD_5M_sample.csv", n=100)
    dataset = SimpleNamespace(name="EURUSD_5M_sample.csv", path=csv_path, size_bytes=csv_path.stat().st_size)
    feed_df_bars = [{
        "time": int(pd.Timestamp("2025-01-01", tz="UTC").timestamp()),
        "open": 9.0, "high": 9.1, "low": 8.9, "close": 9.05, "volume": 1.0,
    }]
    monkeypatch.setattr(market_intelligence, "_feed_bars", lambda *a: feed_df_bars)
    monkeypatch.setattr(market_intelligence, "_find_local_dataset", lambda symbol: dataset)
    out = market_intelligence.bar_fetcher("EURUSD", 60, 10)
    assert out is not None and float(out["close"].iloc[0]) == 9.05


def test_rankings_api_has_no_feed_error_headline(monkeypatch):
    def fake_fetcher(symbol, timeframe_minutes, count):
        return _bars(n=360, freq_minutes=timeframe_minutes) if symbol == "EURUSD" else None

    rankings, errors = market_scanner.rank_markets({"forex": ["EURUSD", "ZZZNODATA"]}, bar_fetcher=fake_fetcher)
    monkeypatch.setattr(routes, "_compute_rankings", lambda: (rankings, errors))
    monkeypatch.setattr(routes, "_compute_news", lambda: CalendarResult(events=[], error=None))
    routes._cache.clear()
    try:
        resp = _make_client().get("/assistant/api/rankings")
    finally:
        routes._cache.clear()
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "No H1 bars" not in body
    data = resp.get_json()
    assert [r["symbol"] for r in data["rankings"]] == ["EURUSD"]
    assert data["errors"] == []


# ----------------------------------------------------------------------
# (b) Basic Outlook: getsource OSError can never reach the response
# ----------------------------------------------------------------------

def test_manual_condition_kinds_survive_getsource_oserror(monkeypatch):
    cr.manual_condition_kinds.cache_clear()
    def _boom(module):
        raise OSError("could not get source code")

    monkeypatch.setattr(cr.inspect, "getsource", _boom)
    try:
        kinds = cr.manual_condition_kinds()  # falls back to reading the .py file
        assert "time_of_day" in kinds
        note = cr.chat_capability_note()
        assert note["manual_json_condition_types"] == kinds
    finally:
        cr.manual_condition_kinds.cache_clear()


def test_manual_condition_kinds_compiled_fallback_is_conservative(monkeypatch):
    import app.strategy.manual as manual_module

    cr.manual_condition_kinds.cache_clear()
    def _boom(module):
        raise OSError("could not get source code")

    monkeypatch.setattr(cr.inspect, "getsource", _boom)
    monkeypatch.setattr(manual_module, "__file__", "/nonexistent/manual.pyc")
    try:
        assert cr.manual_condition_kinds() == []  # no source -> invent nothing
    finally:
        cr.manual_condition_kinds.cache_clear()


def test_outlook_context_and_api_survive_getsource_oserror(monkeypatch):
    cr.manual_condition_kinds.cache_clear()
    def _boom(module):
        raise OSError("could not get source code")

    monkeypatch.setattr(cr.inspect, "getsource", _boom)
    try:
        context = trading_assistant.build_context(rankings=[], news_events=[])
        assert "engine_capabilities" in context
        deterministic = trading_assistant.build_deterministic_outlook(context)
        assert "MACRO UPDATE" in deterministic
        assert "OSError" not in deterministic

        monkeypatch.setattr(routes, "_compute_rankings", lambda: ([], []))
        monkeypatch.setattr(routes, "_compute_news", lambda: CalendarResult(events=[], error=None))
        monkeypatch.setattr(routes, "load_ollama_settings", lambda: OllamaSettings(enabled=False))
        routes._cache.clear()
        try:
            resp = _make_client().get("/assistant/api/outlook")
        finally:
            routes._cache.clear()
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["error"] is None
        assert "MACRO UPDATE" in data["text"]
        assert "OSError" not in data["text"]
        assert "could not get source code" not in resp.get_data(as_text=True)
    finally:
        cr.manual_condition_kinds.cache_clear()


# ----------------------------------------------------------------------
# (c) AI Director: stalled Ollama returns bounded with a named error
# ----------------------------------------------------------------------

_STUB_MODEL = "stub-model"


class _DirectorStubHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/tags":
            self._json({"models": [{"name": f"{_STUB_MODEL}:latest"}]})
        elif self.path == "/api/ps":
            self._json({"models": [{"name": f"{_STUB_MODEL}:latest"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            pass
        if self.path != "/api/chat":
            self._json({"error": "not found"}, 404)
            return
        # Stalled generation: one chunk, a long silence, then the rest.
        self.protocol_version = "HTTP/1.1"
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def _chunk(data: bytes):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        try:
            _chunk((json.dumps({"message": {"role": "assistant", "content": "First "}, "done": False}) + "\n").encode())
            time.sleep(5.0)  # the stall the stall-timeout must name
            _chunk((json.dumps({"message": {"role": "assistant", "content": "rest"}, "done": False}) + "\n").encode())
            _chunk((json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}) + "\n").encode())
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def director_stub():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _DirectorStubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _patch_director(monkeypatch, url):
    settings = OllamaSettings(enabled=True, host=url, model=_STUB_MODEL)
    monkeypatch.setattr(routes, "load_ollama_settings", lambda: settings)
    monkeypatch.setattr(
        routes, "_compute_director_directives",
        lambda: ([], {"total": 0, "by_verdict": {}, "top_strategies": {}}, None),
    )
    monkeypatch.setattr(routes, "DIRECTOR_OLLAMA_TOTAL_TIMEOUT_S", 10.0)
    monkeypatch.setattr(routes, "DIRECTOR_OLLAMA_STALL_TIMEOUT_S", 1.0)
    monkeypatch.setattr(routes, "DIRECTOR_OLLAMA_FIRST_TOKEN_TIMEOUT_S", 5.0)


def test_director_returns_bounded_with_named_error_when_ollama_stalls(monkeypatch, director_stub):
    _patch_director(monkeypatch, director_stub)
    started = time.monotonic()
    resp = _make_client().get("/assistant/api/director")
    elapsed = time.monotonic() - started
    assert elapsed < 10.0  # bounded: never the old 600s / indefinite hang
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["directives"] == []  # deterministic list still returned
    assert "T58 AI DIRECTOR" in data["text"]
    assert "went quiet" in data["text"]  # the transport's named OllamaError


def test_director_stream_yields_terminal_error_chunk_when_ollama_stalls(monkeypatch, director_stub):
    _patch_director(monkeypatch, director_stub)
    started = time.monotonic()
    resp = _make_client().post("/assistant/api/director/stream", json={})
    elapsed = time.monotonic() - started
    assert elapsed < 10.0
    chunks = [json.loads(line) for line in resp.get_data(as_text=True).splitlines() if line.strip()]
    assert chunks, "stream must produce a terminal chunk, not hang"
    assert any("went quiet" in (c.get("error") or "") for c in chunks)
