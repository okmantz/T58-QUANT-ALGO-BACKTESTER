"""v9.14 WS-5: Ollama must actually work -- shared streaming transport.

Every AI call site used to roll its own non-streaming requests.post
with a single 90-120s timeout that had to cover model load + prompt
+ every token: near-guaranteed timeouts on CPU-only local models,
then a silent "default without it". app.ai.ollama_transport now owns
the health check (exact reason, fast), the cold-load warm-up (own
budget), and streamed completions (first-token / stall / total
deadlines). These tests drive the TRANSPORT and each app client
against a faithful stub Ollama HTTP server -- healthy, slow first
token, stalling, model-not-pulled, malformed-line, and refused
modes -- and assert the outputs actually LAND (parsed, returned,
used), not discarded.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.ai.ollama_settings import OllamaSettings

MODEL = "stub-model"


class _StubState:
    mode = "healthy"          # healthy | stall | total | malformed | slow_first
    ps_loaded = True
    tags_models = [f"{MODEL}:latest", "llava:latest"]
    generate_text = "Hello from the stub model."
    chat_text = "Stub chat reply."


class _StubHandler(BaseHTTPRequestHandler):
    state = _StubState()

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
            self._json({"models": [{"name": n} for n in self.state.tags_models]})
        elif self.path == "/api/ps":
            self._json({"models": [{"name": f"{MODEL}:latest"}] if self.state.ps_loaded else []})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            payload = {}
        mode = self.state.mode
        if self.path == "/api/embeddings":
            self._json({"embedding": [0.1, 0.2, 0.3]})
            return
        if self.path == "/api/generate" and payload.get("prompt", None) == "" and not payload.get("stream", True):
            self._json({"response": "", "done": True})  # warm-up probe
            return
        if self.path == "/api/chat":
            key, text = "message", self.state.chat_text
            wrap = lambda piece, done: {"message": {"role": "assistant", "content": piece}, "done": done}  # noqa: E731
        elif self.path == "/api/generate":
            text = payload.get("_stub_text") or self.state.generate_text
            wrap = lambda piece, done: {"response": piece, "done": done}  # noqa: E731
        else:
            self._json({"error": "not found"}, 404)
            return

        if mode == "slow_first":
            time.sleep(3.0)  # cold-load-like delay before the first byte
        # Chunked transfer encoding, like the real (Go) Ollama server:
        # without it the client cannot see lines incrementally.
        self.protocol_version = "HTTP/1.1"
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def _chunk(data: bytes):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        pieces = [text[i:i + 7] for i in range(0, len(text), 7)] or [""]
        if mode == "malformed":
            _chunk(b"this is not json\n")
        for i, piece in enumerate(pieces):
            if mode == "stall" and i == 1:
                time.sleep(5.0)
            if mode == "total":
                time.sleep(0.4)
            _chunk((json.dumps(wrap(piece, False)) + "\n").encode())
        _chunk((json.dumps(wrap("", True)) + "\n").encode())
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


@pytest.fixture
def stub():
    _StubHandler.state = _StubState()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    yield url, _StubHandler.state
    srv.shutdown()


def _settings(url, **kw):
    return OllamaSettings(enabled=True, host=url, model=MODEL, **kw)


# ----------------------------------------------------------------------
# Transport behavior
# ----------------------------------------------------------------------

def test_transport_generate_healthy_streams_and_assembles(stub):
    from app.ai import ollama_transport as t

    url, _state = stub
    pieces = list(t.stream_generate(_settings(url), "Say hello"))
    assert "".join(pieces) == "Hello from the stub model."
    assert len(pieces) > 1  # it really streamed


def test_transport_slow_first_token_still_succeeds(stub):
    """The pre-v9.14 killer: 3s before the first byte (a cold model
    load is far slower) -- health+warm own that budget, the answer's
    clock starts at the first token."""
    from app.ai import ollama_transport as t

    url, state = stub
    state.mode = "slow_first"
    assert t.generate(_settings(url), "Say hello") == "Hello from the stub model."


def test_transport_warms_cold_model_first(stub):
    from app.ai import ollama_transport as t

    url, state = stub
    state.ps_loaded = False  # /api/ps empty -> warm-up probe, then generate
    assert t.generate(_settings(url), "Say hello") == "Hello from the stub model."


def test_transport_stall_is_named(stub):
    from app.ai import ollama_transport as t

    url, state = stub
    state.mode = "stall"
    with pytest.raises(t.OllamaError) as ei:
        t.generate(_settings(url), "Say hello", stall_timeout=1.0)
    assert "went quiet" in str(ei.value)


def test_transport_total_ceiling_is_named(stub):
    from app.ai import ollama_transport as t

    url, state = stub
    state.mode = "total"
    with pytest.raises(t.OllamaError) as ei:
        t.generate(_settings(url), "Say hello", total_timeout=1.0)
    assert "budget" in str(ei.value)


def test_transport_malformed_line_is_skipped(stub):
    from app.ai import ollama_transport as t

    url, state = stub
    state.mode = "malformed"
    assert t.generate(_settings(url), "Say hello") == "Hello from the stub model."


def test_transport_model_not_pulled_is_named(stub):
    from app.ai import ollama_transport as t

    url, state = stub
    state.tags_models = ["llava:latest"]
    with pytest.raises(t.OllamaError) as ei:
        t.generate(_settings(url), "Say hello")
    msg = str(ei.value)
    assert "isn't pulled" in msg and "ollama pull" in msg


def test_transport_refused_is_named():
    from app.ai import ollama_transport as t

    # Nothing listens on 127.0.0.1:9 (discard) -- a refused connection,
    # not a hang.
    with pytest.raises(t.OllamaError) as ei:
        t.generate(_settings("http://127.0.0.1:9"), "Say hello")
    assert "Couldn't reach" in str(ei.value)


# ----------------------------------------------------------------------
# Every app call site, driven against the stub
# ----------------------------------------------------------------------

def test_ollama_client_suggestions_land(stub):
    from dataclasses import dataclass

    from app.ai.ollama_client import OllamaClient

    @dataclass
    class G:
        label: str
        is_int: bool
        lo: float
        hi: float
        base_value: float

    url, state = stub
    state.generate_text = "[[20, 25.5]]"
    genes = [G("emaFast", True, 6.0, 60.0, 20.0), G("T58_SL_PIPS", False, 7.5, 75.0, 25.0)]
    result = OllamaClient(_settings(url)).suggest_parameter_adjustments(
        "Stub Strat", "manual", genes, {"net_profit": -5.0}, {"account_size": 50000})
    assert result.error is None
    assert result.genomes == [[20, 25.5]]


def test_llm_text_client_completes(stub):
    from app.ai.llm_client import OllamaTextClient

    url, _state = stub
    client = OllamaTextClient(settings=_settings(url))
    assert client.complete("Say hello") == "Hello from the stub model."


def test_trading_assistant_chat_lands(stub):
    from app.ai.trading_assistant import TradingAssistantClient

    url, _state = stub
    reply, err = TradingAssistantClient(_settings(url))._chat("sys", "hi")
    assert err is None and reply == "Stub chat reply."


def test_project_chat_lands(stub):
    from app.ai.project_chat import ProjectChatClient

    url, _state = stub
    reply, err = ProjectChatClient(_settings(url)).chat("Proj", "hi")
    assert err is None and reply == "Stub chat reply."


def test_research_agent_call_lands(stub):
    from app.ai.research_agent import ResearchAgent

    url, _state = stub
    text, err = ResearchAgent(_settings(url))._call_ollama("question")
    assert err is None and text == "Hello from the stub model."


def test_strategy_generator_returns_code(stub):
    from app.ai.strategy_generator import generate_strategy

    url, state = stub
    state.generate_text = "```python\ndef generate_signals(df):\n    return df['close'] * 0\n```"
    result = generate_strategy(_settings(url), "python", "EMA crossover on ES 5m")
    assert result.error is None
    assert "generate_signals" in (result.code or "")


def test_research_loop_hypothesis_lands_and_fallback_logs_reason(stub, caplog):
    import logging

    from app.ai.research_loop import _ask_ollama_next_hypothesis

    url, _state = stub
    text, from_ai = _ask_ollama_next_hypothesis(
        _settings(url), "buy the open", {"suggestion": "losses cluster at 14:00"}, timeout=30)
    assert from_ai and "stub model" in text.lower()

    with caplog.at_level(logging.WARNING, logger="t58.ollama"):
        text2, from_ai2 = _ask_ollama_next_hypothesis(
            _settings("http://127.0.0.1:9"), "buy the open", {"suggestion": "losses cluster at 14:00"}, timeout=30)
    assert not from_ai2 and "14:00" in text2  # computed fallback, still useful
    assert "Couldn't reach" in caplog.text     # ...and the reason is in the log


def test_embedder_lands(stub):
    from app.ai.vector_store import OllamaEmbedder

    url, _state = stub
    vec, err = OllamaEmbedder(_settings(url), model=MODEL).embed_one("some text")
    assert err is None and vec == [0.1, 0.2, 0.3]


def test_settings_timeout_is_configurable(stub, monkeypatch):
    """The total budget resolves settings -> env -> default, so Owen
    can raise it for a slow machine without touching code."""
    from app.ai import ollama_transport as t

    url, _state = stub
    assert t.resolve_total_timeout(_settings(url)) == t.DEFAULT_TOTAL_TIMEOUT_S
    assert t.resolve_total_timeout(_settings(url, timeout_seconds=123)) == 123.0
    monkeypatch.setenv("T58_OLLAMA_TIMEOUT_S", "77")
    assert t.resolve_total_timeout(_settings(url)) == 77.0
    assert t.resolve_total_timeout(_settings(url), explicit=5) == 5.0
