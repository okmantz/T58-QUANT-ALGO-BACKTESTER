"""Shared Ollama HTTP transport (v9.14).

Before v9.14, every AI feature in this app rolled its own requests.post
against Ollama -- mostly NON-streaming with a single 90-120s timeout.
On a CPU-only local model that is almost a guaranteed failure: the
one timeout has to cover (a) cold-loading the model into RAM/VRAM
(tens of seconds to minutes), (b) the prompt evaluation, and (c) EVERY
generated token, and a non-streaming request gets zero bytes back
until all three are done. So the app "always timed out and defaulted
without it" (Owen's report) even with a perfectly healthy Ollama --
the model was working fine; the transport was just impatient in the
wrong place and silent about why.

This module is the one transport every call site now shares:

  * health first, fast: GET /api/tags with a short timeout, so "not
    running" / "wrong host" fails in seconds with that exact reason
    instead of after a 90s hang -- and "model not pulled" is detected
    BEFORE asking for a completion (message names what's available
    and the pull command).
  * warm before the clock: if the model isn't resident (GET /api/ps),
    a tiny num_predict=1 request loads it under its own generous
    first-token budget, so model-load time is never charged against
    the answer's budget.
  * streaming completions: tokens flow immediately; the deadlines
    that remain are the ones that actually signal trouble -- a STALL
    limit (max silence between chunks), a FIRST-TOKEN limit (after
    warm-up, waiting for generation to start), and a TOTAL ceiling.
  * no silent anything: every failure raises OllamaError carrying the
    concrete reason (unreachable / not pulled / stalled / ceiling /
    HTTP error / empty response) and is logged at WARNING, so the
    feature that falls back also says why. First-token latency and
    total time are logged at INFO -- "the model takes a while" is now
    a number in the log, not a mystery.

Timeouts are configurable: an explicit `total_timeout` argument wins,
then OllamaSettings.timeout_seconds (persisted, 0 = default), then
T58_OLLAMA_TIMEOUT_S, then DEFAULT_TOTAL_TIMEOUT_S.
"""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger("t58.ollama")

CONNECT_TIMEOUT_S = 5.0
TAGS_TIMEOUT_S = 10.0
DEFAULT_FIRST_TOKEN_TIMEOUT_S = 300.0   # cold model load on CPU hardware
DEFAULT_STALL_TIMEOUT_S = 90.0         # max silence between streamed chunks
DEFAULT_TOTAL_TIMEOUT_S = 600.0         # hard ceiling for one completion
EMBED_TIMEOUT_S = 120.0

_HEALTHY_CACHE: dict[tuple[str, str], float] = {}
_HEALTHY_TTL_S = 30.0


class OllamaError(RuntimeError):
    """One failure, one human-readable reason. Callers surface str(exc)."""


def _host(settings) -> str:
    host = (getattr(settings, "host", "") or "").rstrip("/")
    if not host:
        raise OllamaError("No Ollama host configured.")
    return host


def _headers(settings) -> dict:
    headers = {"Content-Type": "application/json"}
    api_key = getattr(settings, "api_key", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def resolve_total_timeout(settings, explicit: float | None = None) -> float:
    if explicit:
        return float(explicit)
    configured = getattr(settings, "timeout_seconds", 0) or 0
    if configured:
        return float(configured)
    env = os.environ.get("T58_OLLAMA_TIMEOUT_S", "").strip()
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return DEFAULT_TOTAL_TIMEOUT_S


def list_models(settings, timeout: float = TAGS_TIMEOUT_S) -> list[str]:
    """GET /api/tags -> pulled model names. Raises OllamaError with the
    reachability reason on any failure."""
    import requests

    host = _host(settings)
    try:
        resp = requests.get(f"{host}/api/tags", headers=_headers(settings), timeout=(CONNECT_TIMEOUT_S, timeout))
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise OllamaError(f"Couldn't reach Ollama at {host} (is `ollama serve` running?).") from None
    except requests.exceptions.Timeout:
        raise OllamaError(f"Ollama at {host} didn't answer a status check within {timeout:.0f}s.") from None
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"Ollama status check failed at {host}: {exc}") from None
    try:
        return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def model_available(names: list[str], model: str) -> bool:
    return any(n == model or n.startswith(f"{model}:") for n in names)


def ensure_ready(settings, model: str | None = None, *, warm: bool = True,
                 first_token_timeout: float = DEFAULT_FIRST_TOKEN_TIMEOUT_S) -> str:
    """Verifies (and by default warms) the model a completion is about
    to use. Returns the resolved model name. Raises OllamaError with
    the precise reason: unreachable, not pulled (naming alternatives),
    or didn't finish loading in time. Healthy results are cached for
    _HEALTHY_TTL_S so search loops don't pay a status round-trip on
    every call (a failure is never cached)."""
    import requests

    model = model or settings.model
    host = _host(settings)
    key = (host, model)
    if _HEALTHY_CACHE.get(key, 0.0) > time.monotonic() - _HEALTHY_TTL_S:
        return model

    names = list_models(settings)
    if names and not model_available(names, model):
        available = ", ".join(names[:8])
        raise OllamaError(
            f"Ollama is running at {host}, but model '{model}' isn't pulled. "
            f"Available: {available}. Run: ollama pull {model}"
        )

    if warm:
        loaded = _loaded_models(settings)
        if loaded is not None and not model_available(loaded, model):
            t0 = time.monotonic()
            try:
                resp = requests.post(
                    f"{host}/api/generate",
                    headers=_headers(settings),
                    json={"model": model, "prompt": "", "stream": False,
                          "options": {"num_predict": 1}},
                    timeout=(CONNECT_TIMEOUT_S, first_token_timeout),
                )
                resp.raise_for_status()
            except requests.exceptions.Timeout:
                raise OllamaError(
                    f"Model '{model}' didn't finish loading within {first_token_timeout:.0f}s "
                    f"at {host} -- the machine may be short on RAM, or the model too large for it. "
                    "Try a smaller model or raise the timeout."
                ) from None
            except requests.exceptions.ConnectionError:
                raise OllamaError(f"Couldn't reach Ollama at {host} (is `ollama serve` running?).") from None
            except Exception as exc:  # noqa: BLE001
                raise OllamaError(f"Ollama warm-up request failed at {host}: {exc}") from None
            logger.info("ollama: model '%s' loaded in %.1fs", model, time.monotonic() - t0)

    _HEALTHY_CACHE[key] = time.monotonic()
    return model


def _loaded_models(settings) -> list[str] | None:
    """GET /api/ps -> currently resident models; None if the endpoint
    can't answer (older server) -- callers then just try the warm-up
    only when they must."""
    import requests

    host = _host(settings)
    try:
        resp = requests.get(f"{host}/api/ps", headers=_headers(settings), timeout=(CONNECT_TIMEOUT_S, TAGS_TIMEOUT_S))
        resp.raise_for_status()
        return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return None


def _stream_pieces(settings, path: str, payload: dict, *,
                   total_timeout: float, stall_timeout: float,
                   first_token_timeout: float, text_of):
    """Core streaming loop shared by generate/chat. Yields text pieces
    (via text_of(chunk)->str) as they arrive; raises OllamaError with
    the concrete reason on stall/ceiling/HTTP/connection failures.

    The precise deadlines (first-token vs stall-vs-total) can't be
    expressed with requests' single fixed read timeout, so a daemon
    reader thread pushes lines into a queue and THIS generator enforces
    each deadline with queue waits; on a deadline it closes the
    response, which unblocks the reader."""
    import queue
    import threading

    import requests

    host = _host(settings)
    t0 = time.monotonic()
    first_at: float | None = None
    saw_any = False
    try:
        resp = requests.post(
            f"{host}{path}", headers=_headers(settings), json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT_S, max(stall_timeout, first_token_timeout, 1.0)),
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise OllamaError(f"Couldn't reach Ollama at {host} (is `ollama serve` running?).") from None
    except requests.exceptions.Timeout:
        raise OllamaError(f"Ollama at {host} accepted the connection but sent nothing.") from None
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.text[:200] if exc.response is not None else ""
        except Exception:  # noqa: BLE001
            pass
        raise OllamaError(f"Ollama at {host} returned HTTP {exc.response.status_code if exc.response is not None else '?'}: {detail}".strip()) from None
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"Ollama request failed at {host}: {exc}") from None

    lines_q: "queue.Queue" = queue.Queue()

    def _reader():
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                lines_q.put(("line", raw))
        except Exception as exc:  # noqa: BLE001 -- delivered in order, below
            lines_q.put(("error", exc))
        finally:
            lines_q.put(("eof", None))

    threading.Thread(target=_reader, daemon=True, name="t58-ollama-reader").start()
    try:
        while True:
            now = time.monotonic()
            remaining_total = total_timeout - (now - t0)
            if remaining_total <= 0:
                raise OllamaError(
                    f"Ollama at {host} exceeded the {total_timeout:.0f}s completion budget and was stopped "
                    "(it was still generating). Raise T58_OLLAMA_TIMEOUT_S or the AI timeout setting, or use a smaller/faster model."
                )
            wait = first_token_timeout if first_at is None else stall_timeout
            try:
                kind, value = lines_q.get(timeout=max(0.0, min(wait, remaining_total)))
            except queue.Empty:
                if time.monotonic() - t0 >= total_timeout:
                    raise OllamaError(
                        f"Ollama at {host} exceeded the {total_timeout:.0f}s completion budget and was stopped "
                        "(it was still generating). Raise T58_OLLAMA_TIMEOUT_S or the AI timeout setting, or use a smaller/faster model."
                    ) from None
                if first_at is None:
                    raise OllamaError(
                        f"Ollama at {host} produced no output within {first_token_timeout:.0f}s of starting generation "
                        "(model load should already have happened -- check `ollama ps` / machine load)."
                    ) from None
                raise OllamaError(
                    f"Ollama at {host} went quiet mid-generation (no new output for over {stall_timeout:.0f}s) -- "
                    "the model may be stuck, or the machine is out of resources. Try a smaller/faster model."
                ) from None
            if kind == "eof":
                break
            if kind == "error":
                raise OllamaError(f"Ollama stream from {host} broke mid-reply: {value}") from None
            line = value
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue  # a malformed line is skipped, never sinks the stream
            if chunk.get("error"):
                raise OllamaError(f"Ollama error from {payload.get('model')}: {chunk['error']}")
            saw_any = True
            if first_at is None:
                first_at = time.monotonic()
                logger.info("ollama: first token from '%s' after %.1fs", payload.get("model"), first_at - t0)
            piece = text_of(chunk)
            if piece:
                yield piece
            if chunk.get("done"):
                break
    finally:
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass
    logger.info("ollama: completion from '%s' finished in %.1fs", payload.get("model"), time.monotonic() - t0)
    if not saw_any:
        raise OllamaError(f"Ollama at {host} closed the response without producing anything (empty reply from '{payload.get('model')}').")


def stream_generate(settings, prompt: str, *, model: str | None = None, system: str | None = None,
                    format_json: bool = False, images: list[str] | None = None,
                    options: dict | None = None, keep_alive: str | None = None,
                    total_timeout: float | None = None,
                    stall_timeout: float = DEFAULT_STALL_TIMEOUT_S,
                    first_token_timeout: float = DEFAULT_FIRST_TOKEN_TIMEOUT_S):
    """Streaming POST /api/generate. Yields response text pieces."""
    resolved = ensure_ready(settings, model)
    total = resolve_total_timeout(settings, total_timeout)
    payload: dict = {"model": resolved, "prompt": prompt, "stream": True}
    if system:
        payload["system"] = system
    if format_json:
        payload["format"] = "json"
    if images:
        payload["images"] = images
    if options:
        payload["options"] = options
    if keep_alive:
        payload["keep_alive"] = keep_alive
    yield from _stream_pieces(settings, "/api/generate", payload,
                              total_timeout=total, stall_timeout=stall_timeout,
                              first_token_timeout=first_token_timeout,
                              text_of=lambda c: c.get("response", "") or "")


def generate(settings, prompt: str, **kw) -> str:
    """Non-generator generate: returns the full text (the transport
    still streams underneath, so first-byte no longer waits for the
    entire completion). Raises OllamaError."""
    return "".join(stream_generate(settings, prompt, **kw))


def stream_chat(settings, messages: list[dict], *, model: str | None = None,
                options: dict | None = None, keep_alive: str | None = None,
                total_timeout: float | None = None,
                stall_timeout: float = DEFAULT_STALL_TIMEOUT_S,
                first_token_timeout: float = DEFAULT_FIRST_TOKEN_TIMEOUT_S):
    """Streaming POST /api/chat. Yields message-content pieces."""
    resolved = ensure_ready(settings, model)
    total = resolve_total_timeout(settings, total_timeout)
    payload: dict = {"model": resolved, "messages": messages, "stream": True}
    if options:
        payload["options"] = options
    if keep_alive:
        payload["keep_alive"] = keep_alive
    yield from _stream_pieces(settings, "/api/chat", payload,
                              total_timeout=total, stall_timeout=stall_timeout,
                              first_token_timeout=first_token_timeout,
                              text_of=lambda c: (c.get("message") or {}).get("content", "") or "")


def chat(settings, messages: list[dict], **kw) -> str:
    return "".join(stream_chat(settings, messages, **kw))


def embed(settings, text: str, *, model: str, total_timeout: float = EMBED_TIMEOUT_S) -> list[float]:
    """POST /api/embeddings (single text). Raises OllamaError."""
    import requests

    host = _host(settings)
    ensure_ready(settings, model, warm=False)
    try:
        resp = requests.post(
            f"{host}/api/embeddings", headers=_headers(settings),
            json={"model": model, "prompt": text},
            timeout=(CONNECT_TIMEOUT_S, total_timeout),
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        raise OllamaError(f"Couldn't reach Ollama at {host} (is it running?).") from None
    except requests.exceptions.Timeout:
        raise OllamaError(f"Ollama at {host} didn't return an embedding within {total_timeout:.0f}s.") from None
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"Ollama embedding request failed at {host}: {exc}") from None
    vector = data.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise OllamaError(
            f"Ollama responded, but returned no embedding -- is '{model}' pulled? (Try: ollama pull {model})"
        )
    return [float(v) for v in vector]
