"""T58's own MT5 bridge CLIENT (replaces the third-party `mt5linux` package).

Presents the same API surface as the official `MetaTrader5` module (initialize, login, copy_rates_from_pos,
order_send, ..., plus its TIMEFRAME_*/ORDER_*/TRADE_* constants) but forwards each call over TCP to
app/forward_test/mt5_rpc_server.py running with the Windows Python next to the terminal. Results come back as
plain local objects: numpy structured arrays for rates/ticks, attribute-style rows (with ``_asdict()``) for
account/position/deal/order records. Needs only `rpyc` (pure Python) on this side.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import os
import threading

import numpy as np

PROTOCOL = 1


class BridgeError(RuntimeError):
    pass


class Row:
    """Attribute-style record like MetaTrader5's namedtuples (login, balance, ticket, ...)."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_d", dict(data))

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_d")[name]
        except KeyError:
            raise AttributeError(name) from None

    def _asdict(self):
        return dict(self._d)

    def __iter__(self):
        return iter(self._d.values())

    def __len__(self):
        return len(self._d)

    def __getitem__(self, i):
        return list(self._d.values())[i] if isinstance(i, int) else self._d[i]

    def __repr__(self):
        return "Row(" + ", ".join(f"{k}={v!r}" for k, v in list(self._d.items())[:8]) + ")"


def _decode(v):
    if isinstance(v, dict):
        if "__nd__" in v:
            d = v["__nd__"]
            dtype = np.dtype([(n, s) for n, s in d["descr"]])
            return np.frombuffer(base64.b64decode(d["b64"]), dtype=dtype).copy()
        if "__na__" in v:
            d = v["__na__"]
            return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()
        if "__row__" in v:
            return Row({k: _decode(x) for k, x in v["__row__"].items()})
        return {k: _decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return tuple(_decode(x) for x in v)
    return v


def _encode_arg(v):
    if isinstance(v, _dt.datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=_dt.timezone.utc)
        return {"__dt__": v.isoformat()}
    if isinstance(v, (list, tuple)):
        return [_encode_arg(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _encode_arg(x) for k, x in v.items()}
    if isinstance(v, np.generic):
        return v.item()
    return v


class MT5BridgeClient:
    """Drop-in stand-in for the ``MetaTrader5`` module."""

    def __init__(self, host: str = "localhost", port: int = 18812, token: str | None = None, timeout: float = 300.0):
        self._host, self._port = host, int(port)
        self._token = token if token is not None else os.environ.get("T58_MT5_BRIDGE_TOKEN", "")
        self._timeout = timeout
        self._conn = None
        self._lock = threading.RLock()
        self._consts: dict = {}
        self._open()                      # raises on failure: callers report "could not reach the MT5 bridge"
        self._load_constants()

    # ---- connection ----
    def _open(self):
        import rpyc
        self._close()
        self._conn = rpyc.connect(self._host, self._port, config={"sync_request_timeout": self._timeout,
                                                                  "allow_public_attrs": False})
        info = json.loads(self._conn.root.ping())
        proto = int(info.get("protocol", 0))
        if proto != PROTOCOL:
            raise BridgeError(f"bridge protocol mismatch (server {proto}, app {PROTOCOL}): restart the bridge server so it "
                              "uses the app's current mt5_rpc_server.py")

    def _close(self):
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        self._conn = None

    def _load_constants(self):
        self._consts = json.loads(self._conn.root.constants(self._token))

    def close(self):
        self._close()

    def ping(self) -> dict:
        with self._lock:
            return json.loads(self._conn.root.ping())

    # ---- API ----
    def _call(self, name, *args, **kwargs):
        payload = (json.dumps([_encode_arg(a) for a in args]), json.dumps({k: _encode_arg(x) for k, x in kwargs.items()}))
        with self._lock:
            for attempt in (0, 1):
                try:
                    if self._conn is None or self._conn.closed:
                        self._open()
                    raw = self._conn.root.call(name, payload[0], payload[1], self._token)
                    return _decode(json.loads(raw))
                except (EOFError, ConnectionError, OSError, TimeoutError) as exc:
                    if attempt == 1:
                        raise BridgeError(f"lost the MT5 bridge at {self._host}:{self._port} ({type(exc).__name__}: {exc})") from exc
                    self._close()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        consts = object.__getattribute__(self, "_consts")
        if name in consts:
            return consts[name]
        if name.isupper():
            raise AttributeError(name)
        from app.forward_test import mt5_rpc_server as _s  # allow-list is shared; the module only imports rpyc
        if name not in _s.ALLOWED:
            raise AttributeError(name)
        return lambda *a, **k: self._call(name, *a, **k)
