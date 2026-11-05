"""End-to-end Ollama transport against a fake local streaming server (no real Ollama needed)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.ai import ollama_settings as os_, ollama_transport as tr


class _H(BaseHTTPRequestHandler):
    delay_first = 0.0

    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": "llama3.1:latest"}]})
        elif self.path == "/api/ps":
            self._json({"models": [{"name": "llama3.1:latest"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        time.sleep(type(self).delay_first)
        for i, w in enumerate(["Hello", " from", " fake", " ollama"]):
            line = json.dumps({"response": w, "message": {"content": w}, "done": False}).encode() + b"\n"
            self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()
        line = json.dumps({"response": "", "message": {"content": ""}, "done": True}).encode() + b"\n"
        self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n0\r\n\r\n")
        self.wfile.flush()


@pytest.fixture()
def server(tmp_path, monkeypatch):
    os_._invalidate_cache()
    monkeypatch.setattr(os_, "_config_dir", lambda: tmp_path)
    monkeypatch.setattr(os_, "remote_api_key", lambda p: "")
    tr._HEALTHY_CACHE.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_generate_and_chat_stream_from_local_server(server):
    s = os_.OllamaSettings(enabled=True, host=server, model="llama3.1")
    assert tr.generate(s, "hi") == "Hello from fake ollama"
    assert tr.chat(s, [{"role": "user", "content": "hi"}]) == "Hello from fake ollama"


def test_slow_first_token_within_budget_succeeds(server):
    _H.delay_first = 1.5
    try:
        s = os_.OllamaSettings(enabled=True, host=server, model="llama3.1")
        assert tr.generate(s, "hi", first_token_timeout=10, stall_timeout=10, total_timeout=30)
    finally:
        _H.delay_first = 0.0


def test_missing_model_reports_pull_hint(server):
    s = os_.OllamaSettings(enabled=True, host=server, model="nope")
    with pytest.raises(tr.OllamaError, match="ollama pull nope"):
        tr.generate(s, "hi")
