#!/usr/bin/env python3
"""T58's own MT5 bridge SERVER. Runs with the WINDOWS Python that has the MetaTrader5 package (inside Wine on
Linux/macOS, or on a Windows PC/VPS). Needs only:  pip install MetaTrader5 rpyc numpy

    python mt5_rpc_server.py --host 127.0.0.1 --port 18812 [--token SECRET]

The app (app/forward_test/mt5_bridge.py) connects over TCP. Unlike the old `mt5linux` package this speaks a tiny,
fixed protocol: the client names an allow-listed MetaTrader5 function and sends JSON arguments; the server returns
JSON (numpy rate arrays travel as raw bytes + dtype). No pickling, no remote code execution, no version coupling to
a third-party client. This file is standalone on purpose (it is copied into / run from the Wine prefix) -- it
imports nothing from the app.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
import sys
import threading

import rpyc

PROTOCOL = 1

ALLOWED = {
    "initialize", "login", "shutdown", "version", "last_error", "account_info", "terminal_info", "symbols_total",
    "symbols_get", "symbol_info", "symbol_info_tick", "symbol_select", "market_book_add", "market_book_get",
    "market_book_release", "copy_rates_from", "copy_rates_from_pos", "copy_rates_range", "copy_ticks_from",
    "copy_ticks_range", "orders_total", "orders_get", "order_calc_margin", "order_calc_profit", "order_check",
    "order_send", "positions_total", "positions_get", "history_orders_total", "history_orders_get",
    "history_deals_total", "history_deals_get",
}
DATETIME_ARGS = {"copy_rates_from", "copy_rates_range", "copy_ticks_from", "copy_ticks_range", "history_orders_total",
                 "history_orders_get", "history_deals_total", "history_deals_get"}

_LOCK = threading.RLock()      # the MetaTrader5 package is one process-global connection: serialise every call
_MT5 = None


def _mt5():
    global _MT5
    if _MT5 is None:
        import MetaTrader5 as m  # imported lazily so --selftest / import of this file never needs the terminal
        _MT5 = m
    return _MT5


def encode(v):
    """MetaTrader5 result -> JSON-able value."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        np = None
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if np is not None and isinstance(v, np.ndarray):
        if v.dtype.names:
            return {"__nd__": {"descr": [[n, v.dtype.fields[n][0].str] for n in v.dtype.names],
                               "b64": base64.b64encode(v.tobytes()).decode("ascii")}}
        return {"__na__": {"dtype": v.dtype.str, "shape": list(v.shape), "b64": base64.b64encode(v.tobytes()).decode("ascii")}}
    if np is not None and isinstance(v, np.generic):
        return v.item()
    if hasattr(v, "_asdict"):
        return {"__row__": {k: encode(x) for k, x in v._asdict().items()}}
    if isinstance(v, dict):
        return {str(k): encode(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [encode(x) for x in v]
    return str(v)


def decode_arg(v):
    if isinstance(v, dict) and "__dt__" in v:
        # naive datetimes would be read as local time by MT5; callers send UTC-aware values
        return _dt.datetime.fromisoformat(v["__dt__"])
    if isinstance(v, list):
        return [decode_arg(x) for x in v]
    if isinstance(v, dict):
        return {k: decode_arg(x) for k, x in v.items()}
    return v


class T58MT5Service(rpyc.Service):
    ALIASES = ["T58MT5"]
    token = ""

    def exposed_ping(self):
        return json.dumps({"protocol": PROTOCOL, "python": sys.version.split()[0], "platform": sys.platform})

    def exposed_constants(self, token=""):
        self._auth(token)
        m = _mt5()
        out = {}
        for name in dir(m):
            if name.isupper():
                val = getattr(m, name)
                if isinstance(val, (int, float, str)):
                    out[name] = val
        return json.dumps(out)

    def exposed_call(self, name, args_json="[]", kwargs_json="{}", token=""):
        self._auth(token)
        if name not in ALLOWED:
            raise ValueError(f"MT5 function not allowed: {name}")
        args = [decode_arg(a) for a in json.loads(args_json)]
        kwargs = {k: decode_arg(x) for k, x in json.loads(kwargs_json).items()}
        with _LOCK:
            res = getattr(_mt5(), name)(*args, **kwargs)
        return json.dumps(encode(res))

    def _auth(self, token):
        if self.token and token != self.token:
            raise PermissionError("bad bridge token")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T58 MT5 bridge server (run with the Windows Python that has MetaTrader5)")
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this machine only (default). 0.0.0.0 exposes it to the network: use --token")
    ap.add_argument("--port", type=int, default=18812)
    ap.add_argument("--token", default=os.environ.get("T58_MT5_BRIDGE_TOKEN", ""))
    a = ap.parse_args(argv)
    T58MT5Service.token = a.token or ""
    from rpyc.utils.server import ThreadedServer
    srv = ThreadedServer(T58MT5Service, hostname=a.host, port=a.port,
                         protocol_config={"allow_public_attrs": False, "allow_pickle": False, "sync_request_timeout": 300})
    print(f"T58 MT5 bridge listening on {a.host}:{a.port}", flush=True)
    srv.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
