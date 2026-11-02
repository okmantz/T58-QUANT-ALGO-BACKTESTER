# MetaTrader 5 on Windows, Linux and macOS

The app's MT5 features (Forward Test, Live Market bars, Deploy-Live MT5 adapter, overnight autopilot) all go
through `app/forward_test/mt5_connector.py`. It picks a backend automatically:

| Where the app runs | Backend | What you need |
|---|---|---|
| Windows | **native** — official `MetaTrader5` package | MT5 terminal installed (nothing else) |
| Linux / macOS | **bridge** — the app's own built-in bridge | MT5 terminal under Wine (or MetaQuotes' macOS app / CrossOver) + `pip install rpyc` |
| any OS, terminal on another PC/VPS | **bridge** | Windows host running `mt5_rpc_server.py` |

MetaQuotes only publishes the Python package for Windows, which is why non-Windows systems need a bridge: a Windows
Python (inside Wine, or on a Windows machine) runs the MT5 package and exposes it on a TCP port; this app connects
to that port and calls the same API. The bridge is T58's own (`app/forward_test/mt5_bridge.py` client,
`app/forward_test/mt5_rpc_server.py` server). It no longer uses the third-party `mt5linux` package, whose newer
releases changed their API and caused `No module named 'mt5linux.metatrader5'`.

## Linux / macOS: automatic

1. Install Wine and your broker's MT5 once, and log in to the demo account once.
2. `pip install -r config/requirements.txt` (installs `rpyc`).
3. Forward Test tab → enter the account → *Save & Test Connection*.

On the first connection the app, by itself: finds your Wine prefix and `terminal64.exe`; installs Windows Python
inside Wine if it is missing; installs `MetaTrader5`, `rpyc`, `numpy` inside Wine (one time, needs internet); starts
the bridge server on `127.0.0.1:18812`; then connects. Later runs start the bridge in a couple of seconds. If a step
cannot be done, the message names the step and gives the one command to run by hand.

Check the link any time with `python scripts/mt5_bridge_server.py --check`; `--setup` runs the one-time preparation
without connecting.

## Remote terminal (Windows PC / VPS)

On the Windows machine: `pip install MetaTrader5 rpyc numpy`, copy `app/forward_test/mt5_rpc_server.py` there and run

    python mt5_rpc_server.py --host 0.0.0.0 --port 18812 --token <a long secret>

Set the same secret here as `T58_MT5_BRIDGE_TOKEN`, and set `T58_MT5_BRIDGE_HOST` / `T58_MT5_BRIDGE_PORT` (or the
bridge host/port in the saved MT5 settings). Only an allow-list of MetaTrader5 functions can be called, but anyone
who can reach the port and knows the token can trade the account: keep it on a VPN / SSH tunnel / firewall, never on
the open internet.

## Troubleshooting

* *"MT5 bridge unavailable …"* — the message states which step failed (Wine missing, Python/MetaTrader5 install inside
  Wine failed, remote host unreachable).
* *protocol mismatch* — restart the bridge so it uses the app's current `mt5_rpc_server.py`.
* *-10003 IPC initialize failed* — the Windows-side Python could not find the terminal; give the path in *Terminal path*.
* Tested in CI with a stub MetaTrader5 module and a fake Wine (tests/data/test_mt5_rpc_bridge.py); not exercised
  against a live broker terminal under real Wine.
