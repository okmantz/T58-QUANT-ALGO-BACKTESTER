# Test layout

Run everything with `pytest -q tests` (desktop GUI tests need a display: `xvfb-run -a pytest tests/ui` on Linux).

| Folder | Covers |
|---|---|
| `web/` | Flask routes, templates, web job pages, web-side parity |
| `ui/` | Desktop (Tk) UI, themes, navigation, tours |
| `backtest/` | Engine, indicators, fills, portfolios, instruments, data-frame mechanics |
| `strategy/` | Strategy objects, families, library, presets, ML/Python strategies |
| `search/` | Search Lab, Quick Optimize, Speed Run, pipelines, batch runners, job management |
| `evolution/` | Evolution Lab, GA, Forge, gene/parameter spaces |
| `validation/` | Walk-forward, CPCV/PBO, Monte Carlo, robustness, look-ahead, regime and report gates |
| `prop/` | Prop-firm rules, simulators, recommender, risk/sizing |
| `ai/` | AI assistant, research agents, Ollama, news/calendar feeds |
| `data/` | Importers, storage, data center, Alpaca, forward-test data |
| `hedge_fund/` | Hedge-fund manager modules, options/vol/order-flow math |
| `licensing/` | License client/gate, API keys, account safety |
| `packaging/` | Release packaging, CLI, audit regressions, consolidated bundles |

Shared helpers stay at the top level: `conftest.py`, `ui_source.py` (the desktop UI source as one text
blob, for tests that assert on source text), `validate_job_helper.py`, `fixtures/`, `reference/`.
