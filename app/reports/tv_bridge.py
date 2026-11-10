"""TradingView bridge: every backtest gets a short *settings ID* that a Pine script turns back into the exact configuration.

Backtest -> live chart in one copy-paste: copy the ID from the report, paste it into the "T58 Settings" indicator's single
input, and the chart shows the same strategy, market, timeframe, sizing and account protections the backtest ran with.

The ID is plain text on purpose (Pine has str.split / str.tonumber but no zlib or base64):

    T58.1|n=Opening Range|s=ES|tf=15m|rm=percent|rv=0.5|mt=3|dl=2|dd=4|ab=50000|ar=balanced|ps=90|pt=3000|md=2|cs=50|c=4F2A

Fields are single-token keys; the last field `c` is a checksum of everything before it, so a half-copied ID is caught by the
Python decoder here and (via the same checksum) by the Pine script, instead of silently showing wrong numbers.
"""
from __future__ import annotations

import re
import zlib

VERSION = "T58.1"
SEP = "|"
_FIELDS = {   # key -> (label shown on the chart, numeric?)
    "n": ("Strategy", False), "s": ("Market", False), "tf": ("Timeframe", False),
    "rm": ("Risk mode", False), "rv": ("Risk per trade", True), "mt": ("Max trades / day", True),
    "dl": ("Daily loss limit %", True), "dd": ("Max drawdown %", True), "ab": ("Account size", True),
    "ar": ("Adaptive risk", False), "ps": ("Profit split %", True), "pt": ("Profit target $", True),
    "md": ("Min trading days", True), "cs": ("Consistency %", True),
    "sf": ("Session from", False), "st": ("Session to", False),
}
_BAD = re.compile(r"[|=\r\n]")


class SettingsIdError(ValueError):
    pass


def _clean(value) -> str:
    return _BAD.sub(" ", str(value)).strip()[:48]


def _num(value) -> str | None:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return None


def checksum(body: str) -> str:
    return f"{zlib.crc32(body.encode('utf-8')) & 0xFFFF:04X}"


def encode(settings: dict) -> str:
    """Build the ID from a {key: value} dict (unknown keys and empty values are skipped, order is fixed)."""
    parts = [VERSION]
    for key, (_label, numeric) in _FIELDS.items():
        value = settings.get(key)
        if value in (None, ""):
            continue
        text = _num(value) if numeric else _clean(value)
        if text:
            parts.append(f"{key}={text}")
    body = SEP.join(parts)
    return f"{body}{SEP}c={checksum(body)}"


def decode(settings_id: str) -> dict:
    """Inverse of encode(); raises SettingsIdError on a wrong version, a truncated paste or a failed checksum."""
    text = (settings_id or "").strip()
    head, sep, tail = text.rpartition(f"{SEP}c=")
    if not sep or not head.startswith(VERSION):
        raise SettingsIdError("That does not look like a T58 settings ID (it should start with T58.1 and end with c=XXXX).")
    if checksum(head) != tail.strip().upper():
        raise SettingsIdError("The settings ID failed its check - it was probably cut off when copying. Copy it again in full.")
    out = {}
    for piece in head.split(SEP)[1:]:
        key, _, value = piece.partition("=")
        if key in _FIELDS:
            out[key] = float(value) if _FIELDS[key][1] else value
    return out


def describe(settings: dict) -> list[tuple[str, str]]:
    """(label, value) rows exactly as the Pine table shows them - handy for the report and for tests."""
    rows = []
    for key, (label, numeric) in _FIELDS.items():
        if key in settings:
            value = settings[key]
            rows.append((label, f"{value:g}" if numeric and not isinstance(value, str) else str(value)))
    return rows


def settings_from_report(report: dict, adaptive_mode: str | None = None) -> dict:
    """Pull the bridge fields out of a build_report() dict (strategy block, risk_config, prop_firm_rules)."""
    strat = report.get("strategy") or {}
    risk = report.get("risk_config") or {}
    rules = report.get("prop_firm_rules") or {}
    return {
        "n": strat.get("name"), "s": strat.get("instrument"), "tf": risk.get("exec_timeframe") or strat.get("timeframe"),
        "rm": risk.get("risk_mode"), "rv": risk.get("risk_value"), "mt": risk.get("max_trades_per_day"),
        "dl": risk.get("daily_loss_limit_pct") or rules.get("daily_loss_limit_pct"),
        "dd": risk.get("max_account_drawdown_pct") or rules.get("max_drawdown_pct"),
        "ab": risk.get("initial_balance") or rules.get("account_size"),
        "ar": adaptive_mode, "ps": rules.get("profit_split_pct"),
        "pt": rules.get("profit_target_amount") or rules.get("evaluation_profit_target_pct"),
        "md": rules.get("min_trading_days"), "cs": rules.get("consistency_rule_pct"),
    }


def settings_id_for_report(report: dict, adaptive_mode: str | None = None) -> str:
    return encode(settings_from_report(report, adaptive_mode))


_PINE = """//@version=5
// T58 Settings - paste the settings ID from your T58 backtest report into the single input below.
// The table shows the exact configuration the backtest ran with.
indicator("T58 Settings", overlay=true)

settingsId = input.string("", "T58 settings ID")

keys   = array.from(__KEYS__)
labels = array.from(__LABELS__)

var table t = table.new(position.top_right, 2, 20, bgcolor=color.new(color.black, 15), border_width=1)

if barstate.islast
    table.clear(t, 0, 0, 1, 19)
    parts = str.split(settingsId, "|")
    n = array.size(parts)
    // A complete ID starts with the version tag and ends with c=XXXX (4 characters). A cut-off paste fails this.
    last = n > 0 ? array.get(parts, n - 1) : ""
    ok = n > 2 and array.get(parts, 0) == "__VERSION__" and str.length(last) == 6 and str.startswith(last, "c=")
    if not ok
        table.cell(t, 0, 0, "T58: paste the complete settings ID in this indicator's settings", text_color=color.orange, text_size=size.small)
    else
        table.cell(t, 0, 0, "T58 backtest settings", text_color=color.white, text_size=size.normal)
        row = 1
        for i = 1 to n - 2
            kv = str.split(array.get(parts, i), "=")
            if array.size(kv) == 2
                idx = array.indexof(keys, array.get(kv, 0))
                if idx >= 0 and row < 20
                    table.cell(t, 0, row, array.get(labels, idx), text_color=color.gray, text_size=size.small)
                    table.cell(t, 1, row, array.get(kv, 1), text_color=color.white, text_size=size.small)
                    row += 1
"""


def pine_indicator() -> str:
    """The one Pine v5 script the person adds once: it splits the ID and prints every field in a table on the chart.

    Pine cannot compute the checksum, so on the chart it only confirms the ID is complete; decode() above verifies the
    checksum when an ID is pasted back into the app."""
    keys = ", ".join(f'"{k}"' for k in _FIELDS)
    labels = ", ".join(f'"{label}"' for label, _n in _FIELDS.values())
    return _PINE.replace("__KEYS__", keys).replace("__LABELS__", labels).replace("__VERSION__", VERSION)


_PINE_STRATEGY = """//@version=5
// T58 Settings Strategy - the same settings ID, but it also ENFORCES the account protections from your backtest.
// Paste the ID below. Risk per trade, max trades per day, daily loss limit and max drawdown come from the ID.
// The entry signal (EMA cross) and the ATR stop are placeholders: replace the two lines marked YOUR ENTRY LOGIC with your own.
strategy("T58 Settings Strategy", overlay=true, initial_capital=50000, default_qty_type=strategy.fixed, pyramiding=0)

settingsId = input.string("", "T58 settings ID")
atrMult    = input.float(1.5, "Stop = ATR x", minval=0.1, step=0.1)
rewardR    = input.float(2.0, "Target (R multiple)", minval=0.1, step=0.1)

parts = str.split(settingsId, "|")
nParts = array.size(parts)
lastPart = nParts > 0 ? array.get(parts, nParts - 1) : ""
idOk = nParts > 2 and array.get(parts, 0) == "__VERSION__" and str.length(lastPart) == 6 and str.startswith(lastPart, "c=")

f_get(key) =>
    out = ""
    for i = 1 to nParts - 2
        kv = str.split(array.get(parts, i), "=")
        if array.size(kv) == 2 and array.get(kv, 0) == key
            out := array.get(kv, 1)
    out

f_num(key, fallback) =>
    txt = f_get(key)
    txt == "" ? fallback : nz(str.tonumber(txt), fallback)

riskMode     = f_get("rm")
riskValue    = f_num("rv", 0.5)
maxTrades    = f_num("mt", 10)
dailyLossPct = f_num("dl", 0)
maxDdPct     = f_num("dd", 0)
account      = f_num("ab", 50000)

var float dayStartEquity = na
var float peakEquity = na
var int tradesToday = 0
newDay = ta.change(time("D")) != 0
if na(dayStartEquity) or newDay
    dayStartEquity := strategy.equity
    tradesToday := 0
peakEquity := na(peakEquity) ? strategy.equity : math.max(peakEquity, strategy.equity)

dayLossHit = dailyLossPct > 0 and strategy.equity < dayStartEquity - account * dailyLossPct / 100
drawdownHit = maxDdPct > 0 and strategy.equity < peakEquity - account * maxDdPct / 100
canTrade = idOk and not dayLossHit and not drawdownHit and tradesToday < maxTrades

stopDist = ta.atr(14) * atrMult                                   // YOUR ENTRY LOGIC: replace the stop rule
longSignal = ta.crossover(ta.ema(close, 9), ta.ema(close, 21))    // YOUR ENTRY LOGIC: replace the entry condition
shortSignal = ta.crossunder(ta.ema(close, 9), ta.ema(close, 21))

riskUsd = riskMode == "fixed" ? riskValue : strategy.equity * riskValue / 100
qty = stopDist > 0 and syminfo.pointvalue > 0 ? riskUsd / (stopDist * syminfo.pointvalue) : 0.0

if canTrade and qty > 0 and strategy.position_size == 0
    if longSignal
        strategy.entry("Long", strategy.long, qty=qty)
        tradesToday += 1
    if shortSignal
        strategy.entry("Short", strategy.short, qty=qty)
        tradesToday += 1
strategy.exit("Long exit", "Long", stop=strategy.position_avg_price - stopDist, limit=strategy.position_avg_price + stopDist * rewardR)
strategy.exit("Short exit", "Short", stop=strategy.position_avg_price + stopDist, limit=strategy.position_avg_price - stopDist * rewardR)

var table t = table.new(position.top_right, 2, 8, bgcolor=color.new(color.black, 15), border_width=1)
if barstate.islast
    table.clear(t, 0, 0, 1, 7)
    if not idOk
        table.cell(t, 0, 0, "T58: paste the complete settings ID in this strategy's settings", text_color=color.orange, text_size=size.small)
    else
        table.cell(t, 0, 0, "T58 " + f_get("n") + " on " + f_get("s") + " " + f_get("tf"), text_color=color.white, text_size=size.normal)
        table.cell(t, 0, 1, "Risk per trade", text_color=color.gray, text_size=size.small)
        table.cell(t, 1, 1, (riskMode == "fixed" ? "$" : "") + str.tostring(riskValue) + (riskMode == "fixed" ? "" : "%"), text_color=color.white, text_size=size.small)
        table.cell(t, 0, 2, "Trades today", text_color=color.gray, text_size=size.small)
        table.cell(t, 1, 2, str.tostring(tradesToday) + " / " + str.tostring(maxTrades), text_color=color.white, text_size=size.small)
        table.cell(t, 0, 3, "Daily loss limit", text_color=color.gray, text_size=size.small)
        table.cell(t, 1, 3, dayLossHit ? "HIT: standing down" : (dailyLossPct > 0 ? str.tostring(dailyLossPct) + "%" : "off"), text_color=dayLossHit ? color.red : color.white, text_size=size.small)
        table.cell(t, 0, 4, "Max drawdown", text_color=color.gray, text_size=size.small)
        table.cell(t, 1, 4, drawdownHit ? "HIT: standing down" : (maxDdPct > 0 ? str.tostring(maxDdPct) + "%" : "off"), text_color=drawdownHit ? color.red : color.white, text_size=size.small)
"""


def pine_strategy() -> str:
    """A Pine v5 *strategy* template driven by the same settings ID: it sizes by the backtest's risk setting and stops trading when the
    ID's max trades / daily loss / drawdown limits are hit. The entry logic is a clearly marked placeholder the person replaces."""
    return _PINE_STRATEGY.replace("__VERSION__", VERSION)
