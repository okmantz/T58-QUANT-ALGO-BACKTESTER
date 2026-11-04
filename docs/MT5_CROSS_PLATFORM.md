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

You do these once:

1. Install Wine and your broker's MT5 (on a Mac, MetaQuotes' official MetaTrader 5 for Mac is the easiest), open it once and log in to a demo account.
2. `pip install -r config/requirements.txt` (installs `rpyc`).
3. Save the account under *Settings > API Keys > Brokers/Prop Firms*.

Then **Forward Test > Start forward test** does everything else and shows a live checklist while it works:

1. checks the setup; 2. starts T58's MT5 helper (the first time it also installs Windows Python and `MetaTrader5`/`rpyc`/`numpy`
inside Wine, about 2 to 5 minutes, needs internet); 3. opens MetaTrader 5 if it is closed; 4. logs in; 5. confirms the **Algo
Trading** button is on; 6. starts your strategy. Later starts take seconds.

The **Ready check** card at the top of the Forward Test page shows before you press Start what is ready, what T58 will do by
itself, and what needs you, with the exact steps and a Copy button for any command.

How the helper is started: T58 copies `mt5_rpc_server.py` into the prefix's own `C:\t58\` folder (so Wine never needs a `Z:`
drive), runs it with the Wine Python, and writes everything the helper prints to `data/logs/mt5_bridge.log`. If `C:` fails it
retries through `Z:`. If it still fails, the page shows the cause in plain words plus the last lines of that log under
*Technical details*.

Check the link any time with `python scripts/mt5_bridge_server.py --check`; `--setup` runs the one-time preparation without
connecting; `--log` prints the helper's log.

Trades are sized in lots from the broker's own contract size, volume step, minimum and maximum. A signal whose risk budget is
below the broker's minimum lot is skipped (and logged) rather than oversized. T58 sends the orders itself through the MT5 API, so no
Expert Advisor has to be built or attached; the Algo Trading button only has to be green.

## Remote terminal (Windows PC / VPS)

On the Windows machine: `pip install MetaTrader5 rpyc numpy`, copy `app/forward_test/mt5_rpc_server.py` there and run

    python mt5_rpc_server.py --host 0.0.0.0 --port 18812 --token <a long secret>

Set the same secret here as `T58_MT5_BRIDGE_TOKEN`, and set `T58_MT5_BRIDGE_HOST` / `T58_MT5_BRIDGE_PORT` (or the
bridge host/port in the saved MT5 settings). Only an allow-list of MetaTrader5 functions can be called, but anyone
who can reach the port and knows the token can trade the account: keep it on a VPN / SSH tunnel / firewall, never on
the open internet.

## Troubleshooting

The Forward Test page turns each of these into a plain-language card with numbered steps. For reference:

* *Wine missing / MT5 not found* — install them (the card gives the command for your OS), then press Start again.
* *Needs Rosetta* (Apple Silicon) — `softwareupdate --install-rosetta --agree-to-license`, then Start again.
* *Wine couldn't open T58's helper file* (exit code 2) — T58 retries through a second path; if both fail, check the Z: drive mapping in `winecfg`.
* *Algo Trading is switched off* — click the **Algo Trading** button in MT5's toolbar so it turns green.
* *MT5 rejected the account details* — compare login and server letter for letter with MT5's login window; use the trading password, not the investor password.
* *Terminal didn't respond in time (-10004/-10005)* — close any pop-up in MT5, run MT5 and T58 as the same normal user, keep one MT5 open.
* *Protocol mismatch* — an old helper is still running; quit MT5/Wine (or restart) and press Start.
* *-10003 IPC initialize failed* — T58 couldn't find the terminal; give the path in the account's *Terminal path*.
* Tested in CI with a stub MetaTrader5 module and a fake Wine (tests/data/test_mt5_rpc_bridge.py, test_mt5_helper_startup.py,
  test_mt5_diagnostics.py, tests/web/test_forward_test_start_flow.py); not exercised against a live broker terminal under real Wine.
