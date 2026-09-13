"""
A curated, informational reference list of prop firms and which trading
platform(s) each is known to use -- so the Deploy Live tab can show a
dropdown with a sensible starting point instead of a blank text field.

This is NOT a live directory and nothing here is fetched from the firms
themselves -- server names, exact platform lineups, and which firms are
even still operating all change over time (one well-known firm, for
example, shut down with little warning in February 2026). Always confirm
the exact server name from your own account-issued email or firm
dashboard before connecting; the `notes` field flags anything more
important than that to know up front.

UPDATED: this app's live connectors now cover five platform families
instead of one -- MT4/MT5 (app.live_deploy.broker_mt5, wrapping the
existing app.forward_test.mt5_connector), cTrader Open API
(broker_ctrader.py), Tradovate (broker_tradovate.py, the futures-prop-firm
gap this list used to flag as entirely unaddressed), TradeLocker
(broker_tradelocker.py), and DXtrade (broker_dxtrade.py). See each
adapter module's own docstring for its setup steps and, importantly, its
honesty note: none of the four new adapters have been exercised against a
live account by anyone on this project yet, unlike the MT5 path which the
Forward Test tab already runs routinely against demo accounts. Rithmic
and NinjaTrader are still NOT connectable -- both require either a
proprietary desktop plugin architecture (NinjaTrader) or a licensed data
agreement with Rithmic that isn't obtainable as a simple free API
registration the way cTrader/Tradovate/TradeLocker/DXtrade are, so they
were left out of this pass rather than shipped half-working.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_CONNECTABLE_PLATFORMS = {"MT4", "MT5", "cTrader", "DXtrade", "TradeLocker", "Tradovate"}


@dataclass
class PropFirm:
    name: str
    platforms: list[str]          # what this firm offers; connectable ones are in _CONNECTABLE_PLATFORMS
    asset_focus: str              # "Forex/CFD", "Futures", etc. -- informational
    connectable_today: bool       # True if at least one listed platform has a working adapter
    notes: str = ""


PROP_FIRMS: list[PropFirm] = [
    PropFirm(
        "FTMO", ["MT4", "MT5", "cTrader", "DXtrade"], "Forex/CFD", True,
        "MT4/MT5 accounts use the mature, demo-tested connector. cTrader and DXtrade accounts "
        "now have a connector too, but it's untested against a live account -- run it against "
        "FTMO's demo/free trial first.",
    ),
    PropFirm(
        "FundedNext", ["MT5", "cTrader", "TradeLocker"], "Forex/CFD", True,
        "MT5 uses the mature connector. cTrader/TradeLocker accounts now have a connector, "
        "untested against a live account -- verify against FundedNext's own demo first.",
    ),
    PropFirm(
        "The5%ers", ["MT5", "DXtrade"], "Forex/CFD", True,
        "MT5 uses the mature connector. Some newer The5%ers account tiers use DXtrade, which "
        "now has a connector -- untested against a live account, and DXtrade is white-labeled "
        "per firm, so confirm the exact API host from your own dashboard first.",
    ),
    PropFirm(
        "E8 Markets", ["MT5", "cTrader"], "Forex/CFD", True,
        "MT5 uses the mature connector; cTrader now has a connector (untested against a live account).",
    ),
    PropFirm(
        "Blue Guardian", ["MT4", "MT5"], "Forex/CFD", True,
        "MT4/MT5 accounts work with the mature connector.",
    ),
    PropFirm(
        "Apex Trader Funding", ["Tradovate", "Rithmic", "NinjaTrader", "TradingView"], "Futures", True,
        "Tradovate-based accounts are now connectable via the new Tradovate adapter -- untested "
        "against a live account, start on Apex's/Tradovate's demo environment. Rithmic and "
        "NinjaTrader accounts are still NOT connectable (see module docstring). Apex explicitly "
        "permits automated/EA trading on its current account lineup -- verify that's still true "
        "before relying on it.",
    ),
    PropFirm(
        "Topstep", ["TopStepX / ProjectX", "NinjaTrader", "Tradovate", "Rithmic"], "Futures", True,
        "Tradovate-routed accounts are now connectable (untested against a live account). "
        "TopStepX/ProjectX, NinjaTrader, and Rithmic accounts are still NOT connectable.",
    ),
    PropFirm(
        "MyFundedFutures", ["Tradovate", "Rithmic", "NinjaTrader"], "Futures", True,
        "Tradovate-routed accounts are now connectable (untested against a live account). "
        "Reversed an earlier ban on automated trading in mid-2025 -- check current rules before "
        "assuming any given automation is still allowed. Rithmic/NinjaTrader accounts are still NOT connectable.",
    ),
    PropFirm(
        "Other / not listed", ["MT4", "MT5", "cTrader", "Tradovate", "TradeLocker", "DXtrade", "Other"], "Unknown", True,
        "Pick this and select the matching platform if your firm isn't listed here -- the "
        "connection works the same way regardless of firm name as long as the platform matches. "
        "If your firm uses Rithmic or NinjaTrader specifically, it isn't connectable yet.",
    ),
]


def find(name: str) -> PropFirm | None:
    return next((f for f in PROP_FIRMS if f.name == name), None)


def is_platform_connectable(platform: str) -> bool:
    return platform in _CONNECTABLE_PLATFORMS
