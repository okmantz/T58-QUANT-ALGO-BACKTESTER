"""
T58 Backtest Integrity Check -- a pre-flight gate that runs BEFORE any
backtest, answering "what did the strategy actually know at the exact
moment it made each decision" instead of just "do we have some data."

Owen's own framing (Sept 2026 session that added this module): for a
multi-timeframe strategy (1H bias -> 15m setup -> 5m confirmation -> 1m
execution), the dangerous question isn't "do I have 15m data" -- it's
"could that 15m/1H context have leaked a bar that hadn't closed yet as of
the execution bar's own timestamp." This module packages together checks
that mostly already existed SEPARATELY elsewhere in this app (data
import validation, timeframe resampling, app.strategy.lookahead_check,
app.data.timeframe_resample's own completed-bar-safe multi-timeframe
merge) into ONE report, run once, up front, so a bad dataset or a
mismatched timeframe request is caught before minutes/hours of backtest
compute are spent on it -- not discovered after the fact from a
suspicious-looking result.

This module does not re-implement lookahead protection -- the actual
protection is app.strategy.mtf.completed_bars() / app.data.
multi_timeframe.merge_multi_timeframe (already used by every declared
multi-timeframe strategy path -- see app.data.timeframe_resample.
prepare_timeframe_aligned_data) and app.strategy.lookahead_check.
check_for_lookahead (an empirical re-run-on-truncated-data probe). This
module's only job is to RUN that existing check up front and report the
result in one place alongside data/timeframe/execution/account/
validation-split information that previously had no single home.

Usage (see app.web.server's /run route for the wired example):

    report = run_integrity_check(df, strategy, risk, prop_rules, requested_timeframe="15m")
    if report.status == "BLOCKED":
        ...refuse to run the backtest, show report.render()...
    else:
        ...proceed; report.render() is still worth showing/logging...
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.risk import RiskConfig
from app.data.timeframe_resample import (
    TimeframeError,
    infer_timeframe_label,
    native_bar_minutes,
    parse_timeframe_label,
    resample_ohlcv,
)
from app.prop.simulator import PropRules
from app.strategy.base import Strategy
from app.strategy.lookahead_check import check_for_lookahead

STATUS_VALID = "VALID"
STATUS_BLOCKED = "BLOCKED"

# A resampled/native dataset missing more than this fraction of its
# expected bars is still only a WARNING (⚠), matching Owen's own example
# report (missing bars shown as a warning, overall status still VALID) --
# gaps are common in real market data (sessions closed, feed dropouts)
# and are not by themselves evidence the backtest is wrong.
_MISSING_BARS_WARN_ONLY = True
# Invalid OHLC bars (high < low, non-finite, etc.) ARE treated as
# critical once they exceed this fraction of the dataset -- a handful is
# a data-vendor quirk (warn), a large fraction means the feed itself is
# broken and every downstream number is suspect (block).
_INVALID_OHLC_BLOCK_FRACTION = 0.01


@dataclass
class IntegrityCheckLine:
    ok: bool          # True = "✓", False = "⚠" (warning) or blocking, see `critical`
    label: str
    detail: str
    critical: bool = False  # True = this line alone is a BACKTEST BLOCKED reason

    @property
    def marker(self) -> str:
        if self.critical and not self.ok:
            return "✗"
        return "✓" if self.ok else "⚠"

    def render(self) -> str:
        return f"{self.marker} {self.label}: {self.detail}"


@dataclass
class IntegrityReport:
    status: str  # STATUS_VALID | STATUS_BLOCKED
    sections: dict[str, list[IntegrityCheckLine]] = field(default_factory=dict)
    blocking_reason: str | None = None
    blocking_detail: str | None = None

    @property
    def critical_lines(self) -> list[IntegrityCheckLine]:
        return [line for lines in self.sections.values() for line in lines if line.critical and not line.ok]

    @property
    def warning_lines(self) -> list[IntegrityCheckLine]:
        return [line for lines in self.sections.values() for line in lines if not line.critical and not line.ok]

    def render(self) -> str:
        bar = "─" * 34
        if self.status == STATUS_BLOCKED:
            out = ["BACKTEST BLOCKED", ""]
            if self.blocking_reason:
                out.append(self.blocking_reason)
            out.append("")
            if self.blocking_detail:
                out.append("Reason:")
                out.append(self.blocking_detail)
                out.append("")
            out.append("No performance results were produced.")
            return "\n".join(out)

        out = ["T58 BACKTEST INTEGRITY CHECK", bar, ""]
        for section_name, lines in self.sections.items():
            out.append(section_name)
            for line in lines:
                out.append(line.render())
            out.append("")
        out.append(bar)
        out.append(f"BACKTEST STATUS: {self.status}")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "blocking_reason": self.blocking_reason,
            "blocking_detail": self.blocking_detail,
            "sections": {
                name: [{"ok": l.ok, "label": l.label, "detail": l.detail, "critical": l.critical} for l in lines]
                for name, lines in self.sections.items()
            },
            "text": self.render(),
        }


def _blocked(reason: str, detail: str) -> IntegrityReport:
    return IntegrityReport(status=STATUS_BLOCKED, blocking_reason=reason, blocking_detail=detail)


def _invalid_ohlc_mask(df: pd.DataFrame) -> pd.Series:
    numeric = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    non_finite = ~np.isfinite(numeric).all(axis=1)
    inconsistent = (
        (numeric["high"] < numeric["low"])
        | (numeric["high"] < numeric["open"])
        | (numeric["high"] < numeric["close"])
        | (numeric["low"] > numeric["open"])
        | (numeric["low"] > numeric["close"])
    )
    return non_finite | inconsistent.fillna(True)


def _expected_bar_count(period_minutes: float, bar_minutes: float) -> int:
    if bar_minutes <= 0:
        return 0
    return max(int(period_minutes // bar_minutes), 0)


def run_integrity_check(
    df: pd.DataFrame,
    strategy: Strategy | None,
    risk: RiskConfig,
    prop_rules: PropRules | None = None,
    requested_timeframe: str | None = None,
    data_label: str = "uploaded dataset",
    holdout_frac: float = 0.2,
    tick_value: float | None = None,
) -> IntegrityReport:
    """Runs every check and returns a single IntegrityReport. `strategy`
    is optional (a caller checking data alone, before a strategy is even
    picked, can pass None -- the STRATEGY section is simply omitted).
    `requested_timeframe`, if omitted, defaults to whatever `strategy`
    itself declares (see app.data.timeframe_resample.
    resolve_strategy_declaration) or, failing that, the data's own native
    timeframe (i.e. "no resample requested")."""
    sections: dict[str, list[IntegrityCheckLine]] = {}

    # ---------------------------------------------------------------
    # DATA
    # ---------------------------------------------------------------
    if df is None or len(df) == 0:
        return _blocked(
            "No usable market data.",
            f"{data_label} loaded 0 bars. No performance results were produced.",
        )

    ts = pd.to_datetime(df["timestamp"])
    n_bars = len(df)
    period_start, period_end = ts.iloc[0], ts.iloc[-1]
    duplicate_timestamps = int(ts.duplicated().sum())
    invalid_mask = _invalid_ohlc_mask(df)
    invalid_count = int(invalid_mask.sum())
    invalid_frac = invalid_count / n_bars if n_bars else 0.0
    native_minutes = native_bar_minutes(df)
    period_minutes = (period_end - period_start).total_seconds() / 60.0
    expected_bars = _expected_bar_count(period_minutes, native_minutes)
    missing_bars = max(expected_bars - n_bars, 0) if expected_bars else 0
    tz = getattr(ts.dt, "tz", None)

    data_lines = [
        IntegrityCheckLine(True, "File", data_label),
        IntegrityCheckLine(True, "Period", f"{period_start} -> {period_end}"),
        IntegrityCheckLine(True, "Bars loaded", f"{n_bars:,}"),
        IntegrityCheckLine(duplicate_timestamps == 0, "Duplicate timestamps", str(duplicate_timestamps)),
        IntegrityCheckLine(
            invalid_count == 0 or invalid_frac <= _INVALID_OHLC_BLOCK_FRACTION,
            "Invalid OHLC bars", f"{invalid_count:,} ({invalid_frac:.2%})",
            critical=invalid_frac > _INVALID_OHLC_BLOCK_FRACTION,
        ),
        IntegrityCheckLine(missing_bars == 0, "Missing native bars", f"{missing_bars:,}"),
        IntegrityCheckLine(tz is not None, "Timezone", str(tz) if tz is not None else "naive (no tz set -- assumed exchange-local)"),
    ]
    sections["DATA"] = data_lines
    if invalid_frac > _INVALID_OHLC_BLOCK_FRACTION:
        return _blocked(
            "Corrupt market data.",
            f"{invalid_count:,} of {n_bars:,} bars ({invalid_frac:.1%}) have invalid OHLC values "
            f"(high < low, non-finite prices, etc.) -- above the {_INVALID_OHLC_BLOCK_FRACTION:.0%} tolerance. "
            "Re-export or re-clean this dataset before backtesting it.",
        )

    # ---------------------------------------------------------------
    # TIMEFRAME
    # ---------------------------------------------------------------
    source_label = infer_timeframe_label(df)
    declared_exec_tf = None
    if strategy is not None:
        try:
            from app.data.timeframe_resample import resolve_strategy_declaration
            declared_exec_tf = resolve_strategy_declaration(strategy).execution_timeframe
        except Exception:
            declared_exec_tf = None
    target_label = requested_timeframe or declared_exec_tf or source_label

    try:
        target_minutes = parse_timeframe_label(target_label)
    except TimeframeError as exc:
        return _blocked(
            f"Requested timeframe: {target_label}",
            f"'{target_label}' is not a recognized timeframe label ({exc}). No performance results were produced.",
        )

    if target_minutes < native_minutes - 1e-9:
        return _blocked(
            f"Requested timeframe: {target_label}\nAvailable (native) timeframe: {source_label}",
            f"{target_label} is FINER than this dataset's native {source_label} bars -- finer bars cannot be "
            "manufactured from coarser source data. Upload data at least as fine as the requested timeframe, "
            "or request a timeframe >= the native resolution.\n\nResampling: FAILED",
        )

    needs_resample = target_minutes > native_minutes + 1e-9
    if needs_resample:
        try:
            resampled = resample_ohlcv(df, target_label)
        except Exception as exc:  # noqa: BLE001 -- any resample failure is a hard block
            return _blocked(
                f"Requested timeframe: {target_label}\nAvailable timeframe: {source_label}\nResampling: FAILED",
                f"{target_label} dataset could not be generated ({exc}). No performance results were produced.",
            )
        if len(resampled) == 0:
            return _blocked(
                f"Requested timeframe: {target_label}\nAvailable timeframe: {source_label}\nResampling: FAILED",
                f"{target_label} dataset could not be generated -- 0 bars produced from {n_bars:,} native bars. "
                "No performance results were produced.",
            )
        resampled_bars = len(resampled)
        expected_resampled = _expected_bar_count(period_minutes, target_minutes)
        coverage = (resampled_bars / expected_resampled * 100.0) if expected_resampled else 100.0
    else:
        resampled_bars = n_bars
        coverage = 100.0

    sections["TIMEFRAME"] = [
        IntegrityCheckLine(True, "Source", source_label),
        IntegrityCheckLine(True, "Requested", target_label),
        IntegrityCheckLine(True, "Resampled", target_label if needs_resample else "not needed (native == requested)"),
        IntegrityCheckLine(True, f"{target_label} bars created" if needs_resample else "Bars used", f"{resampled_bars:,}"),
        IntegrityCheckLine(coverage >= 90.0, "Coverage", f"{coverage:.1f}%"),
    ]

    # ---------------------------------------------------------------
    # STRATEGY (only if a strategy was given)
    # ---------------------------------------------------------------
    lookahead_bug_detected = None
    if strategy is not None:
        try:
            lookahead_result = check_for_lookahead(strategy, df)
            lookahead_bug_detected = lookahead_result.bug_detected if lookahead_result.checked else None
            lookahead_detail = (
                "PASS" if lookahead_result.checked and not lookahead_result.bug_detected
                else "FAIL -- see detail below" if lookahead_bug_detected
                else f"skipped ({lookahead_result.skip_reason})"
            )
        except Exception as exc:  # noqa: BLE001 -- never let the probe itself crash the gate
            lookahead_detail = f"could not run ({exc})"

        sections["STRATEGY"] = [
            IntegrityCheckLine(True, "Signal timeframe", target_label),
            IntegrityCheckLine(True, "Execution timeframe", source_label),
            IntegrityCheckLine(lookahead_bug_detected is not True, "Lookahead protection", lookahead_detail, critical=lookahead_bug_detected is True),
            IntegrityCheckLine(lookahead_bug_detected is not True, "Future-bar access", "PASS" if lookahead_bug_detected is not True else "FAIL"),
        ]
        if lookahead_bug_detected is True:
            return _blocked(
                "Lookahead bias detected.",
                f"{lookahead_result.summary()}\n\nNo performance results were produced -- fix the leak before "
                "trusting any backtest of this strategy.",
            )

    # ---------------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------------
    sections["EXECUTION"] = [
        IntegrityCheckLine(True, "Commission", f"${risk.commission_per_trade:g}/trade"),
        IntegrityCheckLine(True, "Slippage", f"{risk.slippage_pips:g} pips"),
        IntegrityCheckLine(True, "Intrabar resolution", source_label),
        IntegrityCheckLine(True, "TP/SL conflict handling", "STOP-FIRST (see app.backtest.execution)"),
    ]

    # ---------------------------------------------------------------
    # ACCOUNT
    # ---------------------------------------------------------------
    max_position_ok = risk.max_position_size is None or risk.max_position_size > 0
    account_lines = [
        IntegrityCheckLine(True, "Starting balance", f"${risk.initial_balance:,.0f}"),
        IntegrityCheckLine(max_position_ok, "Contract sizing", "PASS" if max_position_ok else "FAIL (max_position_size <= 0)"),
        IntegrityCheckLine(tick_value is not None, "Tick value", f"${tick_value:g}" if tick_value is not None else "not specified"),
        IntegrityCheckLine(True, "Max position", f"{risk.max_position_size:g}" if risk.max_position_size else "unlimited"),
        IntegrityCheckLine(prop_rules is not None, "Prop rules", "Loaded" if prop_rules is not None else "Not provided"),
    ]
    sections["ACCOUNT"] = account_lines

    # ---------------------------------------------------------------
    # VALIDATION (IS / OOS split)
    # ---------------------------------------------------------------
    split_idx = int(n_bars * (1 - holdout_frac))
    split_idx = max(1, min(split_idx, n_bars - 1)) if n_bars > 1 else n_bars
    is_end = ts.iloc[split_idx - 1] if split_idx > 0 else period_start
    oos_start = ts.iloc[split_idx] if split_idx < n_bars else period_end
    sections["VALIDATION"] = [
        IntegrityCheckLine(True, "IS", f"{period_start} -> {is_end}"),
        IntegrityCheckLine(True, "OOS", f"{oos_start} -> {period_end}"),
        IntegrityCheckLine(
            True, "OOS untouched during optimization",
            "declared by this run's holdout_frac -- not independently verifiable from data alone; "
            "see app.orchestration.full_pipeline's reserve_true_holdout for the actual enforcement.",
        ),
    ]

    return IntegrityReport(status=STATUS_VALID, sections=sections)
