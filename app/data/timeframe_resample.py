"""
Timeframe resampling and strategy-declared multi-timeframe support.

Historically, every strategy in this app ran against whatever bar size the
loaded price file actually was, full stop -- a manual strategy's "ema40"
parameter or a Python strategy's own indicator math had no way to say
"compute this against 15-minute bars" if the loaded file happened to be
1-minute data; it silently computed against 1-minute bars instead, no
matter what the strategy's own name or intent implied. A strategy
literally named "... 1H" run against a 1-minute parquet file was computing
every one of its EMA/ADX/etc. periods against 1-minute bars the entire
time -- a strategy designed as a multi-day swing system was actually
trading a several-hour scalping system, with no warning anywhere that
this had happened.

This module lets a strategy DECLARE what timeframe(s) it actually needs
-- its own execution/entry timeframe, plus optionally one or more coarser
"context" timeframes for a bias/filter (e.g. "trade on the 15m, but only
with the 1h trend") -- and automatically resamples whatever base data was
loaded (e.g. 1-minute GC/ES/NQ) into exactly that shape before the
strategy ever sees it. No strategy needs to be told what timeframe the
loaded file is; declaring what IT needs is enough.

How a strategy declares its timeframe(s):

  Manual (visual-builder) strategies -- top-level `"timeframe"` key on the
  strategy config is the EXECUTION timeframe (what trades are placed and
  filled on). Any individual indicator/operand (inside `entry_conditions`/
  `exit_conditions`, or the legacy `indicators` list) may additionally
  carry its own `"timeframe"` key naming a COARSER context timeframe for
  that one condition alone -- e.g. an EMA(50) tagged `"timeframe": "1h"`
  used as a bias filter for a strategy whose top-level `"timeframe"` is
  "15m". All of `"timeframe"` are optional; a strategy that sets none of
  them behaves exactly as before this module existed (raw loaded data,
  unchanged).

  Python strategies -- an optional module-level `TIMEFRAME = "15m"`
  (execution timeframe) and/or `HTF_TIMEFRAMES = ["1h"]` (one or more
  coarser context timeframes, whose raw OHLCV becomes available as
  `tf60_open`/`tf60_high`/`tf60_low`/`tf60_close`/`tf60_volume` columns on
  the dataframe `generate_signals(df)` receives -- the strategy's own code
  computes whatever it wants from those). Omitting both is identical to
  every Python strategy's existing behavior.

  PineScript / MQL5 strategies -- same directive-comment convention
  app.strategy.pinescript/app.strategy.mql5 already use for
  T58_SL_PIPS/T58_TP_PIPS (neither language has a native way to express
  this, so a `//`-comment directive is the only option): an optional
  `// T58_TIMEFRAME=15m` (execution timeframe) and/or
  `// T58_HTF=1h` or `// T58_HTF=1h,4h` (one or more comma-separated
  coarser context timeframes, merged on as the same tfNN_* raw-OHLCV
  columns Python strategies get -- there is no way for a line-based
  parser to know what a Pine/MQL5 script would have DONE with an
  indicator computed at that timeframe, so unlike Manual/Python, no
  indicator can be tagged to a context timeframe here, only raw OHLCV).
  The directive can appear anywhere in the script, on its own line or
  trailing other code, same as the SL/TP directives. Omitting it is
  identical to every PineScript/MQL5 strategy's existing behavior.

This is invoked from exactly one place -- app.backtest.engine.run_backtest
-- so every tool that ultimately calls run_backtest (Run & Report, Full
Pipeline, Quick Optimize, Search Lab, Evolution Lab, Forge, Speed Run,
CPCV, Walk-Forward GA/Opt, Multi-Objective, Ensemble, Portfolio, and
anything else built on top of it) gets this automatically, with no
per-tool changes needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from app.data.multi_timeframe import infer_timeframe_minutes, merge_multi_timeframe
from app.strategy.indicators import build_indicator_series

# Kinds that resolve to a plain OHLCV column rather than a computed
# indicator -- an operand/indicator tagged with a context timeframe and
# one of these kinds needs no precomputation at all; the raw tfNN_* column
# from the context-timeframe merge already covers it.
_RAW_PRICE_KINDS = {"price", "open", "high", "low", "close", "volume"}
_NON_INDICATOR_KINDS = _RAW_PRICE_KINDS | {"value", "constant", "number"}

_UNIT_MINUTES = {"m": 1.0, "min": 1.0, "minute": 1.0, "minutes": 1.0,
                 "h": 60.0, "hr": 60.0, "hour": 60.0, "hours": 60.0,
                 "d": 60.0 * 24, "day": 60.0 * 24, "days": 60.0 * 24}

_LABEL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$")


class TimeframeError(Exception):
    """Raised when a declared timeframe label can't be parsed."""


def parse_timeframe_label(label: str) -> float:
    """Parses a human-typed timeframe label into minutes. Accepts anything
    reasonable: "15m", "15min", "1h", "1H", "4h", "4hr", "1d", "D",
    "daily", a bare number ("15" -> 15 minutes, "60" -> 60 minutes), or
    "40m" (an unusual but perfectly valid bar size -- this is a generic
    N-minutes parser, not a fixed enum of "standard" sizes, so any whole
    number of minutes/hours/days works)."""
    if label is None:
        raise TimeframeError("Empty timeframe label.")
    text = str(label).strip()
    if not text:
        raise TimeframeError("Empty timeframe label.")
    low = text.lower()
    if low in {"d", "1d", "day", "daily"}:
        return 60.0 * 24
    m = _LABEL_RE.match(text)
    if not m:
        raise TimeframeError(f"Unrecognized timeframe label: {label!r}")
    number, unit = m.group(1), m.group(2).lower()
    if not unit:
        # A bare number with no unit means minutes -- matches how every
        # other numeric field in this app (stop_loss_pips, period, etc.)
        # is already unit-less-by-convention-of-minutes for time fields.
        unit = "m"
    if unit not in _UNIT_MINUTES:
        raise TimeframeError(f"Unrecognized timeframe unit in {label!r}.")
    return float(number) * _UNIT_MINUTES[unit]


def normalize_timeframe_label(label: str) -> str:
    """Canonical display form for a timeframe label -- "60m", "1h", "1H",
    and "60" all normalize to "1h" so the SAME context timeframe declared
    two different ways (by two different conditions in the same strategy,
    say) is recognized as one timeframe rather than two, and so report
    text always reads the same way regardless of how the strategy typed
    it."""
    minutes = parse_timeframe_label(label)
    if minutes >= 60 * 24 and minutes % (60 * 24) == 0:
        days = int(minutes // (60 * 24))
        return "1d" if days == 1 else f"{days}d"
    if minutes >= 60 and minutes % 60 == 0:
        hours = int(minutes // 60)
        return f"{hours}h"
    return f"{int(minutes) if minutes == int(minutes) else minutes}m"


def native_bar_minutes(df: pd.DataFrame) -> float:
    """The loaded data's own native bar spacing, in minutes."""
    return infer_timeframe_minutes(df)


def infer_timeframe_label(df: pd.DataFrame) -> str:
    """Best-effort real timeframe label ("5m", "1h", "1d", ...) inferred
    from a loaded dataframe's own bar spacing -- the shared implementation
    both the desktop app and the web app call, so a desktop run and a web
    run against the same data always compute the identical label (and
    therefore the identical graveyard_path_for(instrument, timeframe) key
    for the tools -- run_search, run_search_loop, run_forge_loop -- that
    use this value to pick which shared graveyard file to write to).
    Falls back to "unknown" only if inference itself fails (e.g. too few
    rows, or a malformed/irregular timestamp column) -- never raises.

    NOTE: switching one of those three call sites from a hardcoded
    "unknown" to this real inferred value changes which graveyard file
    NEW runs write to for non-"unknown"-shaped data -- rejections already
    recorded under the old "unknown" bucket stay there rather than being
    retroactively reclassified. This is the accepted, understood cost of
    fixing "unknown" to mean something real going forward.
    """
    try:
        return normalize_timeframe_label(str(native_bar_minutes(df)))
    except Exception:
        return "unknown"


def resample_ohlcv(df: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
    """Resamples a standardized OHLCV dataframe (timestamp/open/high/low/
    close[/volume]) up to a coarser bar size. Standard aggregation: open
    = first, high = max, low = min, close = last, volume = sum (if
    present). Bins with zero source rows are dropped (never synthesized/
    forward-filled) -- a resampled bar only ever exists where real source
    data actually does. The returned dataframe's `timestamp` column is
    the bin's START time (pandas' own resample default), a plain
    RangeIndex, matching every other OHLCV dataframe convention already
    used across this app (see app.data.multi_timeframe)."""
    minutes = parse_timeframe_label(timeframe_label)
    freq = f"{round(minutes)}min" if minutes == int(minutes) else f"{minutes}min"
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work = work.sort_values("timestamp").set_index("timestamp")

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in work.columns:
        agg["volume"] = "sum"
    agg = {k: v for k, v in agg.items() if k in work.columns}

    resampled = work.resample(freq).agg(agg).dropna(subset=["close"]) if "close" in agg else work.resample(freq).agg(agg)
    return resampled.reset_index()


def _merge_htf_onto_base(base_df: pd.DataFrame, htf_native_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """As-of/backward merge of one already-resampled, native-frequency
    higher-timeframe dataframe onto `base_df`, with its columns
    (including any indicator columns already attached to it) renamed
    under `prefix`. Reuses app.data.multi_timeframe.merge_multi_timeframe
    (already correct about not leaking a still-forming HTF bar into an
    earlier base row -- see that module's MTF-LOOKAHEAD-001 fix) rather
    than re-implementing the same merge-asof logic a second time."""
    merged, _labels = merge_multi_timeframe([base_df, htf_native_df])
    # merge_multi_timeframe derives its own prefix from inferred minutes;
    # rename to the caller's canonical prefix if it differs (it won't, in
    # practice, since htf_native_df was built by OUR OWN resample_ohlcv
    # for this exact label -- this is just a safety net, not the normal
    # path).
    inferred_minutes = round(infer_timeframe_minutes(htf_native_df))
    inferred_prefix = f"tf{inferred_minutes}"
    if inferred_prefix != prefix:
        rename = {c: c.replace(f"{inferred_prefix}_", f"{prefix}_", 1)
                   for c in merged.columns if c.startswith(f"{inferred_prefix}_")}
        merged = merged.rename(columns=rename)
    return merged


@dataclass
class _HtfIndicatorSpec:
    kind: str
    period: int
    field: str
    lookback: int

    @property
    def alias(self) -> str:
        return f"{self.kind}_{self.period}_{self.field}"


@dataclass
class StrategyTimeframeDeclaration:
    execution_timeframe: str | None = None
    context_timeframes: set = field(default_factory=set)
    context_indicators: dict = field(default_factory=dict)  # label -> list[_HtfIndicatorSpec]

    @property
    def declares_anything(self) -> bool:
        return bool(self.execution_timeframe or self.context_timeframes)


def _scan_manual_operand(operand, decl: StrategyTimeframeDeclaration) -> None:
    if not isinstance(operand, dict):
        return
    tf = operand.get("timeframe")
    if not tf:
        return
    try:
        label = normalize_timeframe_label(tf)
    except TimeframeError:
        return
    decl.context_timeframes.add(label)
    kind = str(operand.get("type", operand.get("source", "close"))).lower().strip()
    if kind in _NON_INDICATOR_KINDS:
        return  # plain price/constant operand -- the raw tfNN_* merge already covers it
    period = max(int(operand.get("period", 14) or 14), 1)
    field_ = str(operand.get("field", "close")).lower().strip()
    lookback = max(int(operand.get("lookback", period) or period), 1)
    decl.context_indicators.setdefault(label, [])
    spec = _HtfIndicatorSpec(kind=kind, period=period, field=field_, lookback=lookback)
    if spec.alias not in {s.alias for s in decl.context_indicators[label]}:
        decl.context_indicators[label].append(spec)


def _resolve_manual_declaration(config: dict) -> StrategyTimeframeDeclaration:
    decl = StrategyTimeframeDeclaration()
    exec_tf = config.get("timeframe")
    if exec_tf:
        try:
            decl.execution_timeframe = normalize_timeframe_label(exec_tf)
        except TimeframeError:
            pass

    entries = config.get("entry_conditions", {}) or {}
    exits = config.get("exit_conditions", {}) or {}
    for conditions in (entries.get("long", []), entries.get("short", []),
                       exits.get("long", []), exits.get("short", [])):
        for cond in conditions or []:
            if not isinstance(cond, dict):
                continue
            _scan_manual_operand(cond.get("left", cond.get("source")), decl)
            _scan_manual_operand(cond.get("right", cond.get("value")), decl)

    for ind in config.get("indicators", []) or []:
        if not isinstance(ind, dict):
            continue
        tf = ind.get("timeframe")
        if not tf:
            continue
        try:
            label = normalize_timeframe_label(tf)
        except TimeframeError:
            continue
        decl.context_timeframes.add(label)
        period = max(int(ind.get("period", 14) or 14), 1)
        field_ = str(ind.get("column", "close")).lower().strip()
        decl.context_indicators.setdefault(label, [])
        spec = _HtfIndicatorSpec(kind=str(ind.get("type", "")).lower().strip(), period=period, field=field_, lookback=period)
        if spec.alias not in {s.alias for s in decl.context_indicators[label]}:
            decl.context_indicators[label].append(spec)

    return decl


def _resolve_python_declaration(strategy) -> StrategyTimeframeDeclaration:
    decl = StrategyTimeframeDeclaration()
    module_attr = getattr(strategy, "module_attr", None)
    if module_attr is None:
        return decl
    exec_tf = module_attr("TIMEFRAME", None)
    if exec_tf:
        try:
            decl.execution_timeframe = normalize_timeframe_label(exec_tf)
        except TimeframeError:
            pass
    htf = module_attr("HTF_TIMEFRAMES", None)
    if htf:
        if isinstance(htf, str):
            htf = [htf]
        for tf in htf:
            try:
                decl.context_timeframes.add(normalize_timeframe_label(tf))
            except TimeframeError:
                continue
    return decl


# PineScript / MQL5 directive-comment convention -- see this module's own
# docstring for the full explanation of why a `//`-comment directive is
# the mechanism for these two languages (neither has a native way to
# express this) and exactly what T58_TIMEFRAME/T58_HTF mean. Same regex
# style (bare `KEY\s*=\s*value`, no `//` required in the pattern itself)
# as app.strategy.pinescript/app.strategy.mql5's own pre-existing
# T58_SL_PIPS/T58_TP_PIPS directives, which this is a sibling of.
_TIMEFRAME_DIRECTIVE_RE = re.compile(r"T58_TIMEFRAME\s*=\s*(\S+)")
_HTF_DIRECTIVE_RE = re.compile(r"T58_HTF\s*=\s*(\S+)")


def _resolve_directive_declaration(strategy) -> StrategyTimeframeDeclaration:
    """PineScript and MQL5 both resolve through this one function -- see
    this module's docstring ("PineScript / MQL5 strategies") for the
    `// T58_TIMEFRAME=15m` / `// T58_HTF=1h,4h` convention. Unlike Manual/
    Python, no indicator can be tagged to a context timeframe here (a
    line-based parser can't know what a Pine/MQL5 script would have DONE
    with an indicator at that timeframe) -- T58_HTF only ever contributes
    raw OHLCV tfNN_* columns, never a computed indicator. A script with
    neither directive declares nothing, exactly like today."""
    decl = StrategyTimeframeDeclaration()
    code = getattr(strategy, "code", None)
    if not code:
        return decl
    exec_match = _TIMEFRAME_DIRECTIVE_RE.search(code)
    if exec_match:
        try:
            decl.execution_timeframe = normalize_timeframe_label(exec_match.group(1))
        except TimeframeError:
            pass
    htf_match = _HTF_DIRECTIVE_RE.search(code)
    if htf_match:
        for tf in htf_match.group(1).split(","):
            tf = tf.strip()
            if not tf:
                continue
            try:
                decl.context_timeframes.add(normalize_timeframe_label(tf))
            except TimeframeError:
                continue
    return decl


def resolve_strategy_declaration(strategy) -> StrategyTimeframeDeclaration:
    """What timeframe(s), if any, `strategy` has declared it needs. See
    this module's docstring for exactly how each source type declares
    this. Returns an empty (declares_anything == False) declaration for
    any strategy that declares nothing -- the fully backward-compatible
    default."""
    source_type = getattr(strategy, "source_type", None)
    if source_type == "manual" and isinstance(getattr(strategy, "config", None), dict):
        return _resolve_manual_declaration(strategy.config)
    if source_type == "python":
        return _resolve_python_declaration(strategy)
    if source_type in ("pinescript", "mql5"):
        return _resolve_directive_declaration(strategy)
    return StrategyTimeframeDeclaration()


def describe_resolved_timeframe(strategy, raw_df: pd.DataFrame | None = None) -> str:
    """A short, human-readable label for whatever timeframe `strategy`
    actually resolves to -- e.g. "1h", "15m (bias: 1h)", or "native
    (~1m, unspecified)" for a strategy that declares nothing. Meant for
    the report's "Timeframe" field, which used to just be a hardcoded
    "unknown" string with no real connection to what the strategy
    actually ran against (see app.orchestration.full_pipeline's
    generate_full_report call, which this replaces)."""
    decl = resolve_strategy_declaration(strategy)
    if not decl.declares_anything:
        if raw_df is not None and len(raw_df):
            return f"native (~{native_bar_minutes(raw_df):.0f}m, unspecified)"
        return "native (unspecified)"
    label = decl.execution_timeframe or "native"
    if decl.context_timeframes:
        ctx = ", ".join(sorted(decl.context_timeframes, key=parse_timeframe_label))
        return f"{label} (bias: {ctx})"
    return label


def prepare_timeframe_aligned_data(raw_df: pd.DataFrame, strategy) -> tuple[pd.DataFrame, list[str]]:
    """The main entry point, called once by app.backtest.engine.
    run_backtest before a strategy ever sees the data.

    Returns (df_to_use, warnings): `df_to_use` is `raw_df` UNCHANGED
    whenever `strategy` declares no timeframe at all (the default --
    every strategy that predates this module, and every strategy that
    simply doesn't need this, is completely unaffected). Otherwise it's
    `raw_df` resampled to the strategy's declared execution timeframe,
    with any declared coarser context timeframe(s) merged on as tfNN_*
    columns (raw OHLCV, plus any indicator the strategy tagged with that
    timeframe, computed at that timeframe's own native frequency -- never
    computed against the base timeframe's upsampled/repeated values,
    which would silently give a wrong period).
    """
    decl = resolve_strategy_declaration(strategy)
    if not decl.declares_anything:
        return raw_df, []

    warnings: list[str] = []
    native_minutes = native_bar_minutes(raw_df)

    if decl.execution_timeframe:
        exec_minutes = parse_timeframe_label(decl.execution_timeframe)
        if exec_minutes < native_minutes - 1e-6:
            warnings.append(
                f"Strategy declared execution timeframe '{decl.execution_timeframe}' is FINER than "
                f"the loaded data's native ~{native_minutes:.0f}-minute bars -- cannot resample finer "
                f"than the source data. Using the source's native {native_minutes:.0f}-minute bars "
                "instead; upload finer-grained source data if you need genuine "
                f"{decl.execution_timeframe} execution."
            )
            base_df = raw_df
            exec_label = f"native ~{native_minutes:.0f}m"
            exec_minutes_effective = native_minutes
        elif abs(exec_minutes - native_minutes) < 1e-6:
            base_df = raw_df
            exec_label = decl.execution_timeframe
            exec_minutes_effective = exec_minutes
        else:
            base_df = resample_ohlcv(raw_df, decl.execution_timeframe)
            exec_label = decl.execution_timeframe
            exec_minutes_effective = exec_minutes
    else:
        base_df = raw_df
        exec_label = f"native ~{native_minutes:.0f}m"
        exec_minutes_effective = native_minutes

    merged = base_df
    used_context_labels = []
    for label in sorted(decl.context_timeframes, key=parse_timeframe_label):
        ctx_minutes = parse_timeframe_label(label)
        if ctx_minutes <= exec_minutes_effective + 1e-6:
            warnings.append(
                f"Declared context/bias timeframe '{label}' is not coarser than the execution "
                f"timeframe ({exec_label}) -- a bias timeframe must be coarser than what you trade "
                "on. Skipped; that condition falls back to reading the execution-timeframe data "
                "directly."
            )
            continue
        htf_native = resample_ohlcv(raw_df, label)
        for spec in decl.context_indicators.get(label, []):
            if spec.alias in htf_native.columns:
                continue
            try:
                htf_native[spec.alias] = build_indicator_series(
                    htf_native, spec.kind, period=spec.period, column=spec.field, lookback=spec.lookback,
                )
            except Exception:  # noqa: BLE001
                # A malformed/unsupported HTF indicator spec shouldn't take
                # down the whole run -- the condition referencing it will
                # simply find its column missing and raise a clear
                # StrategyError at evaluation time instead.
                continue
        prefix = f"tf{round(ctx_minutes)}"
        merged = _merge_htf_onto_base(merged, htf_native, prefix)
        used_context_labels.append(f"{prefix} ({label})")

    detail = f"execution timeframe: {exec_label}"
    if used_context_labels:
        detail += f"; context/bias timeframe(s): {', '.join(used_context_labels)}"
    warnings.append(
        f"Timeframe-aware data pipeline: resampled from the loaded data's native "
        f"~{native_minutes:.0f}-minute bars -- {detail}."
    )
    return merged, warnings
