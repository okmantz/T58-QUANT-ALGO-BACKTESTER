# Strategy Discovery Engine (2026-10-25)

Search Lab, Evolution Lab, Forge, Speed Run and the hypothesis layer are ONE engine with presets (`app/discovery_engine`).
One objective (`edge_lb_r`), one entry floor (100+ trades, mean-R t >= 1, cost < 0.15R), one lifetime trial ledger clustered to
effective trials, one sealed slice read once, one promotion sequence:
reproduction -> sequence-deflated Sharpe -> random-timing null gate -> pooled cross-market edge -> sealed slice.
A passing champion is a forward-test candidate, never a "winner".

## Run
    python -m scripts.bootstrap_data                      # which pooled markets do you actually have? (needs >= 8)
    python -m scripts.run_surrogate_study --null-runs 60  # false-discovery rate on no-edge data + power on planted +0.10..0.30R edges
    python -m scripts.run_discovery_engine --preset search_lab --data data/raw/NQ_5m.csv --symbol NQ --pool ES=... --retest
    python -m scripts.run_preregistered_families NQ=... ES=... --v13-batch=1

The 100 report indicators are in `app/strategy/indicators_v13*.py`; the 100 families (92 runnable, 8 blocked with reasons) in
`app/search/families_v13.py`, pre-registered with kill conditions in `app/discovery/hypothesis.py`.
New family batches stay CLOSED until `reports/surrogate_gate.json` (written by the surrogate study, valid 90 days, bound to the
current engine code) shows < 5% false discoveries over >= 20 null runs (~60 needed to prove it) and >= 50% power on +0.20/+0.30R plants.
Requires `numba` for speed (pure-Python fallback is ~100x slower for the HMM/analog indicators).

## Known limits
- Surrogate FDR on your real data has not been measured; the shipped gate is closed.
- Ext feeds (VIX term, COT, skew, ...) need your CSV drop-ins; those families are `needs_data`.
- Pooled test needs >= 8 markets and fails closed with fewer.
- Blocked families: 6, 95 (scale-in), 18 (calendar spread), 39 (closing imbalance), 45 (roll calendar), 56, 57, 59 (tick/book).
- No Johansen test (PC-spread + AR(1) gate instead); family 29 uses OLS lines, not Theil-Sen.
