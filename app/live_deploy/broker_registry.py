"""
Factory: turns a saved LiveAccount (app.live_deploy.live_settings) into
the right BrokerAdapter instance. This is the one place that needs to
know about every platform module -- everything downstream (execution_engine,
the Deploy Live routes) just gets back a BrokerAdapter and never imports
a platform-specific module directly.
"""
from __future__ import annotations

from app.live_deploy.broker_base import BrokerAdapter
from app.live_deploy.live_settings import LiveAccount

PLATFORM_MT4_MT5 = "MT4/MT5"
PLATFORM_CTRADER = "cTrader"
PLATFORM_TRADOVATE = "Tradovate"
PLATFORM_TRADELOCKER = "TradeLocker"
PLATFORM_DXTRADE = "DXtrade"

SUPPORTED_PLATFORMS = [
    PLATFORM_MT4_MT5, PLATFORM_CTRADER, PLATFORM_TRADOVATE, PLATFORM_TRADELOCKER, PLATFORM_DXTRADE,
]


def build_adapter(account: LiveAccount) -> BrokerAdapter:
    """Raises ValueError for an unknown/unsupported platform string, or
    KeyError-style ValueError if a platform-specific required field is
    missing from account.extra_credentials -- both are meant to surface
    as a clear form-validation-style error in the Deploy Live UI, not a
    traceback."""
    platform = account.platform

    if platform == PLATFORM_MT4_MT5:
        from app.live_deploy.broker_mt5 import MT5BrokerAdapter
        return MT5BrokerAdapter(
            login=account.login, password=account.password,
            server=account.server, terminal_path=account.terminal_path,
        )

    extra = account.extra_credentials or {}

    if platform == PLATFORM_CTRADER:
        from app.live_deploy.broker_ctrader import CTraderBrokerAdapter
        _require(extra, ["client_id", "client_secret", "refresh_token", "ctid_trader_account_id"], platform)
        return CTraderBrokerAdapter(
            client_id=extra["client_id"], client_secret=extra["client_secret"],
            refresh_token=extra["refresh_token"],
            ctid_trader_account_id=int(extra["ctid_trader_account_id"]),
            host=extra.get("host", "demo"),
        )

    if platform == PLATFORM_TRADOVATE:
        from app.live_deploy.broker_tradovate import TradovateBrokerAdapter
        _require(extra, ["app_id", "app_secret", "cid", "sec"], platform)
        return TradovateBrokerAdapter(
            username=account.login, password=account.password,
            app_id=extra["app_id"], app_secret=extra["app_secret"],
            cid=extra["cid"], sec=extra["sec"],
            is_live=extra.get("is_live", "false").lower() == "true" if isinstance(extra.get("is_live"), str) else bool(extra.get("is_live")),
        )

    if platform == PLATFORM_TRADELOCKER:
        from app.live_deploy.broker_tradelocker import TradeLockerBrokerAdapter
        return TradeLockerBrokerAdapter(
            email=account.login, password=account.password, server=account.server,
            is_live=extra.get("is_live", "false").lower() == "true" if isinstance(extra.get("is_live"), str) else bool(extra.get("is_live")),
        )

    if platform == PLATFORM_DXTRADE:
        from app.live_deploy.broker_dxtrade import DXtradeBrokerAdapter
        _require(extra, ["base_url"], platform)
        return DXtradeBrokerAdapter(
            base_url=extra["base_url"], username=account.login, password=account.password,
            domain=extra.get("domain", "default"),
        )

    raise ValueError(f"Unsupported platform '{platform}'. Supported: {SUPPORTED_PLATFORMS}.")


def _require(extra: dict, keys: list[str], platform: str) -> None:
    missing = [k for k in keys if not extra.get(k)]
    if missing:
        raise ValueError(f"Missing required {platform} credential field(s): {', '.join(missing)}.")
