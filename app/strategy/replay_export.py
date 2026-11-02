"""Trade-replay export: makes ANY strategy (including hand-written Python) portable to PineScript v5 or MQL5.

Hand-written strategy code (pandas expressions, loops, custom indicators) has no safe, general reduction to the
condition grammar the structured translators use. Instead of refusing, this module RUNS the strategy on a dataset,
records every position change (time, direction, stop/target distance), and writes a Pine / MQL5 strategy that replays
exactly those entries and exits on the same bars.

Honest limits (also written into the generated file header):
  * It reproduces what the strategy did on THAT dataset; it does not contain the strategy's logic, so it will not
    produce new signals on other data. For live use re-generate it, or port the logic by hand.
  * Only entry/exit bars, direction and the stop/take-profit distance at entry are replayed (not trailing, break-even,
    partial exits, or resting limit orders).
  * Bar timestamps are matched in UTC. Tell the tool the timezone of the CSV's clock if it is not UTC.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_EVENTS = 2000
CHUNK_CHARS = 3200      # Pine string literals are limited to 4096 chars


class ReplayError(Exception):
    pass


@dataclass
class ReplayData:
    name: str
    events: list = field(default_factory=list)      # [(epoch_ms, dir, sl_dist, tp_dist)]
    n_bars: int = 0
    first: str = ""
    last: str = ""
    notes: list = field(default_factory=list)
    truncated: bool = False


def _dist_at(series, i: int) -> float:
    if series is None:
        return 0.0
    try:
        v = float(series.iloc[i] if hasattr(series, "iloc") else series)
    except Exception:  # noqa: BLE001
        return 0.0
    return v if np.isfinite(v) and v > 0 else 0.0


def collect_replay(strategy, df: pd.DataFrame, name: str, tz: str = "UTC", max_bars: int | None = None,
                   max_events: int = MAX_EVENTS) -> ReplayData:
    """Run `strategy` on `df` and record position changes. `df` must have a DatetimeIndex or a 'timestamp' column."""
    if df is None or len(df) == 0:
        raise ReplayError("The dataset is empty.")
    work = df
    if max_bars and len(work) > max_bars:
        work = work.tail(max_bars).reset_index(drop=True)
    if "timestamp" in work.columns:
        times = pd.DatetimeIndex(pd.to_datetime(work["timestamp"]))
    elif isinstance(work.index, pd.DatetimeIndex):
        times = work.index
    else:
        raise ReplayError("The dataset needs a timestamp column so signals can be matched to bars.")
    res = strategy.generate(work)       # the SAME frame shape the backtest engine passes
    if getattr(res, "entry_orders", None):
        raise ReplayError("This strategy uses resting limit/zone orders, which a bar-by-bar replay cannot reproduce faithfully.")
    sig = np.asarray(res.signals.values, dtype=int)
    idx = times
    if idx.tz is None:
        try:
            idx = idx.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        except Exception as exc:  # noqa: BLE001
            raise ReplayError(f"Unknown timezone '{tz}': {exc}") from exc
    idx = idx.tz_convert("UTC")
    ms = idx.asi8 // 1_000_000
    sl_s, tp_s = getattr(res, "stop_loss_distance", None), getattr(res, "take_profit_distance", None)
    out = ReplayData(name=name, n_bars=len(work), first=str(idx[0]), last=str(idx[-1]))
    if (sl_s is None and getattr(res, "stop_loss_pips", None)) or (tp_s is None and getattr(res, "take_profit_pips", None)):
        out.notes.append("The strategy uses a fixed pip stop/target; the replay carries no stop/target for it (exits follow the recorded signal).")
    prev = 0
    for i in range(len(sig)):
        cur = int(sig[i])
        if cur == prev:
            continue
        if ms[i] < 0:        # NaT from a DST gap
            prev = cur
            continue
        if len(out.events) >= max_events:
            out.truncated = True
            break
        sl = _dist_at(sl_s, i) if cur != 0 else 0.0
        tp = _dist_at(tp_s, i) if cur != 0 else 0.0
        out.events.append((int(ms[i]), cur, round(sl, 6), round(tp, 6)))
        prev = cur
    if not out.events:
        raise ReplayError("The strategy produced no trades on this dataset, so there is nothing to export. Pick a longer dataset.")
    if out.truncated:
        out.notes.append(f"Only the first {max_events} position changes are included (script size limits).")
    if getattr(res, "trailing_stop_distance", None) is not None or getattr(res, "breakeven_trigger_r", None) or getattr(res, "partial_exit", None):
        out.notes.append("Trailing stop / break-even / partial exits of the original are not replayed.")
    return out


def _fmt(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".") or "0"


def _chunks(rd: ReplayData) -> list[str]:
    parts, cur = [], ""
    for ms, d, sl, tp in rd.events:
        item = f"{ms},{d},{_fmt(sl)},{_fmt(tp)};"
        if len(cur) + len(item) > CHUNK_CHARS:
            parts.append(cur)
            cur = ""
        cur += item
    if cur:
        parts.append(cur)
    return parts


def _header(rd: ReplayData, comment: str) -> list[str]:
    lines = [f"{comment} T58 TRADE REPLAY of '{rd.name}'",
             f"{comment} Source: a hand-written strategy that cannot be reduced to indicator conditions, so this file replays the",
             f"{comment} {len(rd.events)} position changes it made on {rd.n_bars:,} bars ({rd.first} -> {rd.last}).",
             f"{comment} It does NOT contain the strategy's logic and will not trade other date ranges. Timestamps are UTC bar-open times",
             f"{comment} of the bar whose CLOSE produced the signal; orders are placed on the next bar."]
    lines += [f"{comment} NOTE: {n}" for n in rd.notes]
    return lines


def to_pinescript_replay(rd: ReplayData) -> str:
    name = re.sub(r'["\\\n]', " ", rd.name)
    lines = ["//@version=5", *_header(rd, "//"),
             f'strategy("{name} (T58 replay)", overlay=true, default_qty_type=strategy.fixed, default_qty_value=1, calc_on_every_tick=false)',
             "", "var array<int>   evT  = array.new_int()", "var array<int>   evD  = array.new_int()",
             "var array<float> evSL = array.new_float()", "var array<float> evTP = array.new_float()", "",
             "f_load(string chunk) =>", "    rows = str.split(chunk, \";\")", "    for r in rows",
             "        if str.length(r) > 0", "            f = str.split(r, \",\")",
             "            array.push(evT,  math.round(str.tonumber(array.get(f, 0))))",
             "            array.push(evD,  math.round(str.tonumber(array.get(f, 1))))",
             "            array.push(evSL, str.tonumber(array.get(f, 2)))",
             "            array.push(evTP, str.tonumber(array.get(f, 3)))", "", "if barstate.isfirst"]
    for c in _chunks(rd):
        lines.append(f'    f_load("{c}")')
    lines += ["", "var int idx = 0",
              "if barstate.isconfirmed",
              "    while idx < array.size(evT) and array.get(evT, idx) < time",
              "        idx += 1",
              "    if idx < array.size(evT) and array.get(evT, idx) == time",
              "        d  = array.get(evD, idx)",
              "        sl = array.get(evSL, idx)",
              "        tp = array.get(evTP, idx)",
              "        if d == 0",
              "            strategy.close_all(comment=\"replay exit\")",
              "        else if d == 1",
              "            strategy.entry(\"L\", strategy.long)",
              "            strategy.exit(\"XL\", \"L\", stop=sl > 0 ? close - sl : na, limit=tp > 0 ? close + tp : na)",
              "        else",
              "            strategy.entry(\"S\", strategy.short)",
              "            strategy.exit(\"XS\", \"S\", stop=sl > 0 ? close + sl : na, limit=tp > 0 ? close - tp : na)",
              "        idx += 1"]
    return "\n".join(lines) + "\n"


def to_mql5_replay(rd: ReplayData) -> str:
    name = re.sub(r'["\\\n]', " ", rd.name)
    chunks = _chunks(rd)
    lines = [*_header(rd, "//"), "#property strict", f'#property description "T58 replay of {name}"',
             "#include <Trade\\Trade.mqh>", "CTrade trade;",
             'input double InpLots = 0.10;', 'input long   InpMagic = 5805;', "",
             "long   g_t[]; int g_d[]; double g_sl[]; double g_tp[]; int g_idx = 0; datetime g_lastBar = 0;", "",
             "void AddChunk(string chunk)", "{", "   string rows[]; int n = StringSplit(chunk, ';', rows);",
             "   for(int i = 0; i < n; i++)", "   {", "      if(StringLen(rows[i]) == 0) continue;",
             "      string f[]; if(StringSplit(rows[i], ',', f) < 4) continue;",
             "      int k = ArraySize(g_t); ArrayResize(g_t, k + 1); ArrayResize(g_d, k + 1); ArrayResize(g_sl, k + 1); ArrayResize(g_tp, k + 1);",
             "      g_t[k] = (long)StringToInteger(f[0]); g_d[k] = (int)StringToInteger(f[1]);",
             "      g_sl[k] = StringToDouble(f[2]); g_tp[k] = StringToDouble(f[3]);", "   }", "}", "",
             "int OnInit()", "{", "   trade.SetExpertMagicNumber(InpMagic);"]
    for c in chunks:
        lines.append(f'   AddChunk("{c}");')
    lines += ["   return(INIT_SUCCEEDED);", "}", "",
              "bool HasPos(ENUM_POSITION_TYPE &t)",
              "{",
              "   for(int i = PositionsTotal() - 1; i >= 0; i--)",
              "   {",
              "      ulong tk = PositionGetTicket(i);",
              "      if(tk == 0 || !PositionSelectByTicket(tk)) continue;",
              "      if(PositionGetString(POSITION_SYMBOL) != _Symbol || PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;",
              "      t = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); return true;",
              "   }",
              "   return false;",
              "}", "",
              "void CloseMine()",
              "{",
              "   for(int i = PositionsTotal() - 1; i >= 0; i--)",
              "   {",
              "      ulong tk = PositionGetTicket(i);",
              "      if(tk == 0 || !PositionSelectByTicket(tk)) continue;",
              "      if(PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic) trade.PositionClose(tk);",
              "   }",
              "}", "",
              "void OnTick()",
              "{",
              "   datetime barClosed = iTime(_Symbol, _Period, 1);",
              "   datetime cur = iTime(_Symbol, _Period, 0);",
              "   if(cur == g_lastBar) return;          // act once per new bar",
              "   g_lastBar = cur;",
              "   int n = ArraySize(g_t);",
              "   while(g_idx < n && g_t[g_idx] < (long)barClosed) g_idx++;",
              "   if(g_idx >= n || g_t[g_idx] != (long)barClosed) return;",
              "   int d = g_d[g_idx]; double sl = g_sl[g_idx], tp = g_tp[g_idx]; g_idx++;",
              "   ENUM_POSITION_TYPE pt;",
              "   if(d == 0) { CloseMine(); return; }",
              "   if(HasPos(pt))",
              "   {",
              "      if((d == 1 && pt == POSITION_TYPE_BUY) || (d == -1 && pt == POSITION_TYPE_SELL)) return;",
              "      CloseMine();",
              "   }",
              "   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK), bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);",
              "   if(d == 1)  trade.Buy(InpLots, _Symbol, ask, sl > 0 ? ask - sl : 0, tp > 0 ? ask + tp : 0, \"T58 replay\");",
              "   else        trade.Sell(InpLots, _Symbol, bid, sl > 0 ? bid + sl : 0, tp > 0 ? bid - tp : 0, \"T58 replay\");",
              "}"]
    return "\n".join(lines) + "\n"


def render_replay(rd: ReplayData, target: str) -> str:
    target = target.lower()
    if target == "pinescript":
        return to_pinescript_replay(rd)
    if target == "mql5":
        return to_mql5_replay(rd)
    if target == "python":
        raise ReplayError("A Python strategy is already Python -- pick PineScript or MQL5.")
    raise ReplayError(f"Unsupported target '{target}'.")


def replay_summary_json(rd: ReplayData) -> str:
    return json.dumps({"name": rd.name, "events": len(rd.events), "bars": rd.n_bars, "first": rd.first, "last": rd.last})
