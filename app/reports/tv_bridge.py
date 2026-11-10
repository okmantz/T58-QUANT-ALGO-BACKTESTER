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
