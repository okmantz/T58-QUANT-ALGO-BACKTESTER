# MetaTrader 5 on Windows, Linux and macOS

The app's MT5 features (Forward Test, Live Market bars, Deploy-Live MT5 adapter, overnight autopilot) all go
through `app/forward_test/mt5_connector.py`. It picks a backend automatically:

| Where the app runs | Backend | What you need |
|---|---|---|
| Windows | **native** — official `MetaTrader5` package | MT5 terminal installed (nothing else) |
| Linux | **bridge** — `mt5linux` RPyC client | MT5 terminal under Wine + bridge server |
| macOS | **bridge** | MT5 under MetaQuotes' macOS app / CrossOver / Wine + bridge server |
| any OS, terminal on another PC/VPS | **bridge** | Windows host running the bridge server |

MetaQuotes only publishes the Python package for Windows, which is why non-Windows systems need the bridge: a
Windows Python (inside Wine, or on a Windows machine) runs the MT5 package and exposes it on a TCP port; this app
connects to that port and calls the identical API, so no other code differs by OS.

## Setup (Linux / macOS)

`python scripts/mt5_bridge_server.py` prints the commands tailored to your machine. In short:

1. Install Wine (or CrossOver / MetaQuotes' macOS installer) and your broker's MT5 setup; log in to the demo account once.
2. Install Windows Python in the same Wine prefix and run `wine python -m pip install MetaTrader5 mt5linux`.
3. Start the server and leave it running: `wine python -m mt5linux --host 127.0.0.1 --port 18812`.
4. In the app's environment: `pip install mt5linux` (already in `config/requirements.txt` for non-Windows).
5. Forward Test tab → *Save & Test Connection*. *Auto-detect* searches Wine, CrossOver, Bottles and MetaQuotes-app prefixes for
   `terminal64.exe`; host paths are translated to `C:\...` for the bridge automatically.

Check the link any time with `python scripts/mt5_bridge_server.py --check`.

## Remote terminal

Run the server on the Windows machine (`python -m mt5linux --host 0.0.0.0 --port 18812`), open that port to the app only
(firewall / VPN / SSH tunnel — the bridge has **no authentication**; never expose it to the internet), then set
`T58_MT5_BRIDGE_HOST` / `T58_MT5_BRIDGE_PORT` or the bridge host/port in the saved MT5 settings.

## Troubleshooting

* *"MT5 bridge unavailable … could not reach the MT5 bridge"* — the server isn't running, or host/port differ.
* *-10003 IPC initialize failed* — the terminal wasn't found by the Windows-side Python; give the Wine-side path in *Terminal path*.
* Package versions: the `mt5linux` client and server must be the same version.
* Not verified against a live broker terminal in CI: the connector is tested with fake backends (tests/data/test_mt5_cross_platform.py).
