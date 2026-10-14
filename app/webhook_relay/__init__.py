"""
TradingView webhook relay (TradersPost-style).

A Pine Script strategy runs on TradingView and fires an alert with a webhook
URL; this package receives that alert, turns it into a position change, and
(1) always tracks it on a T58 paper account -- the forward test -- and
(2) optionally mirrors it to a DEMO/practice broker account through the same
broker adapters Deploy uses. See app/web/webhook_relay_routes.py for the
HTTP side and the setup page.
"""
