"""
Deploy Live -- connecting a validated strategy to a REAL, funded prop-firm
account for automated live trading.

This is deliberately a separate package from app.forward_test, even though
it reuses the same MT5 connection code: forward_test exists specifically
for demo accounts (see that package's own docstring), and every safety
default there assumes "nothing real is at stake." Live deployment code
must never quietly inherit those assumptions, so it gets its own settings
storage (app.live_deploy.live_settings, supporting multiple named live
accounts rather than the one demo account forward_test manages) and its
own curated prop-firm reference data (app.live_deploy.prop_firms).

As of this module's introduction, actual live order placement is NOT
wired up -- the UI (see MainWindow._build_deploy_live_tab) supports
saving/testing live account connections end-to-end, but the "start live
trading" action is an intentional stub. Turning that on for real is a
separate, deliberate step once the account-management plumbing here has
been used and reviewed.

v7 UPDATE (2026-10-05): the note above is STALE -- live order placement
IS wired up now. ``LiveExecutionSession`` (app.live_deploy.
execution_engine) runs the real poll -> signal -> size -> order loop
against any ``BrokerAdapter``, launched from the desktop Deploy Live tab
and the web ``/deploy-live/start`` route (the latter disabled by default
-- see app.live_deploy.web_deploy_config). v7 also closed the critical
safety gaps: sizing units are converted to broker-native quantity in
every adapter (``BrokerAdapter.to_broker_qty``), futures-prop guardrails
ship enforced (news blackout, no weekend hold, no hedging, per-order lot
cap, live max-drawdown halt), and repeated poll failures best-effort
flatten and halt the session. The "not yet battle-tested" honesty note
on the individual adapters still stands: run a demo account first.
"""
