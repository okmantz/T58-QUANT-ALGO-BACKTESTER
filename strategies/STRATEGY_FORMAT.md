# Strategy Format — how the backtester reads your strategy

This is the contract between your strategy and the backtester. Write to it
and your strategy loads, backtests, optimizes, and deploys. Break it and
you get a clear error naming the problem — never a silently wrong backtest.

Four ways to make a strategy, one shared contract:

1. **Manual** — the visual Manual Strategy Builder (no code; produces a config).
2. **Python** (`.py`) — a `generate_signals(df)` function.
3. **Pine Script** (`.pine`, v5 subset) — `strategy.entry()` / `strategy.close()` calls.
4. **MQL5** (`.mq5`, EA subset) — `trade.Buy()` / `trade.Sell()` / `trade.PositionClose()` calls.

## The universal contract (all four sources)

**Signals are `-1` (short), `0` (flat), `1` (long)** — one value per bar.
A change `0 → 1` opens a long, `0 → -1` opens a short, a change back to
`0` closes the position, and an opposite-direction signal while a position
is open reverses it immediately. (The Manual Builder has an "Opposite
Signal Exit" toggle; when off, opposite signals are ignored while a
position is open.)

**One position at a time.** No pyramiding, no hedging, no partial fills in
the backtest. Size is set separately in the lab (risk % per trade), not in
your strategy code.

**Fills are next-bar-open by default.** Your signal is computed on bar N;
the trade fills at bar N+1's open. There is no same-bar fill fantasy. If a
single bar touches both your stop and your target, the stop wins.

**Costs are lab settings, not strategy code.** Spread, slippage, and
commission are configured where you run the backtest (sensible per-
instrument defaults are applied automatically). Your strategy never
prices its own costs.

**pip_size comes from the instrument, not your file.** When you pick a
dataset in any lab, the app auto-detects the instrument spec (pip_size,
contract_size, commission) and fills it in — an explicit value you typed
is never overridden, but if you hand-set one, double-check it matches the
instrument. Getting pip_size wrong is the #1 silent killer: an FX-scale
`0.0001` on ES data turns every fixed-pip stop into 0.002 points and
every candidate dies on the first tick.

**A stop-loss is not optional in practice.** If your strategy defines no
stop at all, the engine applies a 1%-of-entry-price protective stop and
warns you (counted in the report as fallback stops). Always define a real
stop — see "Stops and targets" below.

**Your strategy cannot see its own trade outcomes.** Signal generation runs
once, statelessly, over the whole dataset before any trade exists. A "stop
trading after N losses today" counter inside strategy code is silently a
no-op — use the lab's risk settings (daily loss limit) for that instead.

---

## 1. Manual (visual builder, no code)

The Manual Strategy Builder produces a **data config** (JSON-like dict),
not source code — so there is nothing to hand-format. What it stores:

- **Entry / exit conditions** — operands (indicators, price fields like
  `close`/`high`/`low`, session highs/lows, previous-day levels, opening
  range, time-of-day, day-of-week, liquidity sweeps, ATR expansion/
  contraction legs, RSI divergence, …) combined with comparisons and
  `and`/`or`/`not` into long-entry, short-entry, long-exit, short-exit
  rule sets.
- **Risk management** — stop-loss / take-profit (fixed pips or ATR
  multiples), trailing stop, breakeven trigger, time-based exits,
  max-bars-in-trade.
- **Filters** (optional) — exclude weekdays (`filters.days_of_week`,
  0=Monday…6=Sunday) or exclude market regimes the strategy already loses
  in (`filters.regime_exclude`, fed straight from the Regime Survival
  Matrix). Filtered bars force the signal flat regardless of the rules.

Two builder behaviors worth knowing:

- **Impossible-condition check.** Comparing a bounded oscillator against a
  threshold outside its range (e.g. `RSI > 102`) can never be true — the
  builder warns you, because that branch of your logic would be dead code
  with nothing in the numbers telling you so.
- **Opposite Signal Exit toggle.** On (default): an opposite entry signal
  reverses the open position. Off: the position can only close via its
  own exits, stop, target, or time exit.

If you can click it in the builder, the engine can read it — the builder
only offers constructs the backtester supports.

### The JSON format (for hand-written or shared manual strategies)

A manual strategy is a JSON object. The files in `strategies/manual/`
(e.g. `ES1! VWMA Trend Pullback.json`) are exactly this format, and the
app loads them through the Strategy Library as `strategy_type: "manual"`
— the JSON is parsed into a config dict and run by the same Manual engine
as builder-made strategies. You can write one by hand, share it, or
export one from the builder and edit it.

```json
{
  "name": "My Pullback",
  "timeframe": "1h",
  "entry_conditions": {
    "long": [
      {
        "left":     { "type": "close", "period": 1, "field": "close" },
        "operator": ">",
        "right":    { "type": "ema", "period": 200, "field": "close" }
      },
      {
        "left":     { "type": "rsi", "period": 14, "field": "close" },
        "operator": "<=",
        "right":    { "type": "value", "value": 40 }
      }
    ],
    "long_connectors": ["AND"],
    "short": [
      {
        "left":     { "type": "close", "period": 1, "field": "close" },
        "operator": "<",
        "right":    { "type": "ema", "period": 200, "field": "close" }
      }
    ]
  },
  "exit_conditions": {
    "long": [
      {
        "left":     { "type": "close", "period": 1, "field": "close" },
        "operator": "cross above",
        "right":    { "type": "vwma", "period": 20, "field": "close" }
      }
    ],
    "short": []
  },
  "risk_management": {
    "stop_type": "atr",
    "stop_value": 1.5,
    "stop_atr_period": 14,
    "target_type": "atr",
    "target_value": 3.0,
    "target_atr_period": 14,
    "opposite_signal_exit": true,
    "max_bars_in_trade": 20
  },
  "filters": {
    "days_of_week": { "exclude": [6] }
  }
}
```

How the engine reads it, field by field:

- **`entry_conditions.long` / `.short`** — arrays of conditions. Each
  condition is `{left, operator, right}` where `left`/`right` are
  **operands** and `operator` is a comparison. All conditions in one side
  must agree to fire an entry for that side.
- **Operands** — `{ "type": ..., "period": ..., "field": ... }`:
  - `"type": "close"` (also `"open"`, `"high"`, `"low"`) with
    `"period": N` = that price N bars ago (`period: 1` = current bar).
    `"field"` is accepted but the price types read their own series.
  - `"type": "ema" | "vwma" | "rsi" | "sma" | "atr" | …` — any indicator
    the engine knows (same set the builder offers, incl. the v7
    additions), computed on `"field"` (usually `"close"`) with
    `"period"`. Unknown types raise a clear error listing what's
    supported.
  - `"type": "value", "value": 40` — a constant number.
  - Time/session operands (`"session_high"`, `"previous_day_high"`,
    `"liquidity_sweep"`, `"time_of_day"`, `"day_of_week"`, …) take extra
    keys per operand — the builder writes them correctly; copy an
    existing file's shape when hand-writing one.
- **Operators** — `">"`, `">="`, `"<"`, `"<="`, `"=="`, `"!="`
  (word aliases like `"greater than"` also work), `"cross above"` /
  `"cross below"` (true only on the bar the cross happens), `"is true"` /
  `"is false"`. Anything else raises `Unsupported condition operator`.
- **`long_connectors` / `short_connectors`** — `"AND"` / `"OR"` between
  consecutive conditions (one fewer than the condition count; missing
  entries default to `"AND"`).
- **`exit_conditions.long` / `.short`** — same condition shape; true =
  close that side's position. Empty array = no rule-based exits (position
  still closes on opposite signal, stop, target, or time exit).
- **`risk_management`** —
  - `stop_type` / `target_type`: `"atr"` (multiple of ATR — `stop_value`
    × ATR(`stop_atr_period`)) or `"pips"` (fixed pip distance). ATR is
    the scale-safe choice; see the pip_size note in the universal
    contract.
  - `opposite_signal_exit`: `true` = opposite entry reverses the
    position (default); `false` = only exits/stops/targets close it.
  - `max_bars_in_trade`: force-close after N bars (omit or 0 = no limit).
- **`filters`** (optional) — `{ "days_of_week": { "exclude": [6] } }`
  forces signals flat on those weekdays (0=Monday…6=Sunday);
  `{ "regime_exclude": [{ "volatility": "extreme" }] }` forces signals
  flat on bars in those market regimes (same cells the Regime Survival
  Matrix reports).
- **`timeframe`** — `"1h"` resamples the loaded data to 1h bars before
  anything runs; omit it to trade the file's native bar size.

A malformed file fails loudly: not-valid-JSON, unknown indicator type,
or unsupported operator each raise a named error instead of backtesting
something you didn't write.

---

## 2. Python (`.py`)

**Required:** one top-level function.

```python
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    fast = close.rolling(10).mean()
    slow = close.rolling(30).mean()
    long_cond = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    short_cond = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    signals = pd.Series(0, index=df.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals  # -1 / 0 / 1, one value per row of df
```

Rules:

- `df` columns are `timestamp, open, high, low, close, volume`
  (lowercase). If you declare `HTF_TIMEFRAMES` (below), you also get
  `tfNN_open/high/low/close/volume` columns for each context timeframe.
- Return a Series the **same length as `df`**, containing `-1`, `0`, `1`.
  Anything else is coerced (`fillna(0)`, clipped to [-1, 1], rounded) —
  don't rely on that; return clean values.
- `generate_signals(df)` is called **once, statelessly**, on a copy of the
  data, before any trade exists. No trade counters, no P&L-aware logic.
- The module is imported in isolation; any exception while loading or
  running your function becomes a clear `StrategyError`, not a silent
  mis-backtest.

**Optional module-level constants** (all read without calling your
function):

```python
STRATEGY_NAME = "My Strategy"   # shown in reports; defaults to filename
STOP_LOSS_PIPS = 20             # fixed stop, whole backtest
TAKE_PROFIT_PIPS = 40           # fixed target, whole backtest
TIMEFRAME = "15m"               # resample data to 15m bars BEFORE your
                                # function runs; trades fill on these bars
HTF_TIMEFRAMES = ["1h"]         # coarser context bars merged on as
                                # tf60_open/high/low/close/volume —
                                # lookahead-safe: only fully closed HTF
                                # bars are ever visible per row
WARMUP_BARS = 200               # force first N signals flat; set to at
                                # least your longest indicator lookback
EXCLUDE_DAYS_OF_WEEK = [6]      # 0=Monday..6=Sunday; signal forced flat
                                # on these weekdays
```

**Dynamic (per-trade) stops and targets** — attach to the returned
Series' `.attrs`, in raw price units, one value per bar (only the entry
bar's value is read):

```python
signals.attrs["stop_loss_distance"]     # |entry - stop|, e.g. 1.5 * atr
signals.attrs["take_profit_distance"]   # |entry - target|
signals.attrs["trailing_stop_distance"] # raw-price trailing distance
signals.attrs["breakeven_trigger_r"]    # scalar float, e.g. 1.0 == "+1R"
```

This is the **only** path a computed stop/target reaches the engine. If
you compute one and don't attach it here, the engine never sees it and
falls back to its generic stop — your risk management is silently
discarded.

**The #1 Python bug: lookahead in multi-timeframe filters.** If you
resample to a higher timeframe yourself and filter with
`htf[htf.index < timestamp]`, you leak the still-forming current HTF bar
(a resampled bar is labeled by its start time). This exact bug has
manufactured entire fake "edges" in real uploaded strategies. Prefer
`HTF_TIMEFRAMES` (safe by construction), or use
`app.strategy.mtf.completed_bars()` / `last_completed_bar()` if you
hand-roll the resample.

**ML-style strategies:** set `RETRAIN_PER_FOLD = True` and read
`df.attrs.get("wf_train_end_index")` — during walk-forward validation the
engine calls your function once on train+test concatenated with the real
split index marked, so you retrain on the true training window instead of
guessing one. Absent (`None`) on every other call path; fall back to your
own logic then.

---

## 3. Pine Script (`.pine`, v5 subset)

A line-based parser, not a full Pine runtime. Anything outside the subset
below raises a clear error naming the unsupported construct. Cosmetic
lines (the `strategy()` header, `plot()`, alerts) are silently ignored.

**You may use:**

- Price: `open, high, low, close, hl2, hlc3, ohlc4`
- `x = input.int(20, ...)` / `input.float(1.5, ...)` → constant from the
  default value (no other `input.*` types)
- `ta.sma(src, len)`, `ta.ema(src, len)`, `ta.wma(src, len)`,
  `ta.rsi(src, len)`
- `ta.crossover(a, b)`, `ta.crossunder(a, b)`
- `ta.atr(len)`, `ta.vwap()` (session VWAP), `ta.highest(src, len)`,
  `ta.lowest(src, len)`, `ta.stdev(src, len)` — standalone **or embedded**
  inside larger expressions, e.g. `stopDist = ta.atr(14) * 1.5`
- `[m, s, h] = ta.macd(src, fast, slow, signal)` — 3-way destructuring
  (the only destructuring supported)
- Plain arithmetic over defined series/constants:
  `spreadPct = (fastMA - slowMA) / slowMA`
- Boolean rule variables from comparisons / `and` / `or` / `not`:
  `longCondition = ta.crossover(fast, slow) and rsiVal < 70`

**Entries** (inline `when=` or inside an `if` block) — direction comes
from `strategy.long` / `strategy.short`:

```pine
strategy.entry("Long", strategy.long, when=longCondition)
strategy.entry("Short", strategy.short, when=shortCondition)

if longCondition
    strategy.entry("Long", strategy.long)
```

**Exits:** `strategy.close("Long", when=exitLong)` — the trade id decides
the side: contains "short" → closes shorts, "long" → closes longs,
anything else → closes both.

**Stops and targets** — special directive comments (Pine's
`strategy.exit()` price offsets aren't portable across instruments, so
they're not read):

```pine
// T58_SL_PIPS=20
// T58_TP_PIPS=40
```

or, **preferred for anything that isn't FX** (gold, indices, crypto —
a fixed pip count is only correct at the one pip_size it was tuned for):

```pine
// T58_SL_ATR_MULT=1.5
// T58_TP_ATR_MULT=3.0
// T58_ATR_PERIOD=14     (optional, defaults to 14)
```

ATR-mult wins if both styles are present. It computes a per-bar
stop/target in raw price units — scale-independent, no pip trap.

**Timeframes** (this parser can't use `security()` — multi-timeframe
requests are rejected):

```pine
// T58_TIMEFRAME=15m     (resample to 15m execution bars first)
// T58_HTF=1h,4h         (merge raw coarser bars; indicators can't be
                          computed at HTF in Pine — raw bars only)
```

**Weekday filter:** `// T58_EXCLUDE_DAYS=6` (0=Monday…6=Sunday,
comma-separated; signal forced flat those days).

**Not supported:** custom functions, arrays/matrices, `security()`,
repainting constructs, and any `ta.*` beyond the list above.

---

## 4. MQL5 (`.mq5`, EA subset)

Also a line-based parser. Anything outside the subset raises a clear
error naming the construct.

**You may use:**

- Direct-value indicator calls (simplified/legacy style):

```mql5
double fastMA = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_SMA, PRICE_CLOSE);
double slowMA = iMA(_Symbol, PERIOD_CURRENT, 30, 0, MODE_EMA, PRICE_CLOSE);
double rsiVal = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
double atrVal = iATR(_Symbol, PERIOD_CURRENT, 14);
double bandTop = iBands(_Symbol, PERIOD_CURRENT, 20, 2, 0, PRICE_CLOSE, MODE_UPPER);
double hh     = iHighest(_Symbol, PERIOD_CURRENT, MODE_HIGH, 20, 0);
double ll     = iLowest(_Symbol, PERIOD_CURRENT, MODE_LOW, 20, 0);
```

`iMA` modes: `MODE_SMA` / `MODE_EMA` / `MODE_LWMA`. `iBands` selectors:
`MODE_UPPER` / `MODE_LOWER` / `MODE_MAIN`. The symbol / timeframe /
shift / applied-price arguments are accepted but not used — the engine
always runs on the single imported dataset bar-by-bar.

- Plain arithmetic over defined variables:
  `double stopDist = atrVal * 1.5;`
- Boolean conditions with C-style operators: `> < >= <= == != && || !`
- `if (condition) { ... }` or single-statement `if (condition) statement;`
  (nesting handled)

**Entries** inside a condition's guard:

```mql5
if (longCondition) { trade.Buy(0.1); }
// or: OrderSend(_Symbol, ORDER_TYPE_BUY, 0.1, ...);   (OP_BUY also accepted)
if (shortCondition) { trade.Sell(0.1); }
```

(Lot-size arguments are accepted but ignored — sizing comes from the
lab's risk settings, exactly like every other source.)

**Exits** inside a condition's guard:
`trade.PositionClose(ticket);` / `OrderClose(...);`

**Stops, targets, timeframes, weekday filter** — identical directive
comments to Pine Script:

```mql5
// T58_SL_PIPS=20            (or the ATR-mult trio below — preferred off-FX)
// T58_TP_PIPS=40
// T58_SL_ATR_MULT=1.5
// T58_TP_ATR_MULT=3.0
// T58_ATR_PERIOD=14
// T58_TIMEFRAME=15m
// T58_HTF=1h,4h
// T58_EXCLUDE_DAYS=6
```

**Not supported:** `CopyBuffer()` indicator handles, custom indicators,
arrays/structs, multi-symbol or multi-timeframe logic, trailing stops,
and any indicator beyond `iMA` / `iRSI` / `iATR` / `iBands` /
`iHighest` / `iLowest`.

---

## Stops and targets — precedence, in one place

When several are defined, the engine uses the first available:

1. **Per-trade dynamic** — Python `.attrs` distances, or Pine/MQL5
   `T58_*_ATR_MULT` directives (raw price units, computed per bar).
2. **Fixed pips** — `STOP_LOSS_PIPS` / `TAKE_PROFIT_PIPS`, or
   `T58_SL_PIPS` / `T58_TP_PIPS` directives (interpreted with the
   instrument's pip_size — see the pip_size note up top).
3. **Protective fallback** — 1% of entry price, with a loud warning
   counted in the report. A backtest full of fallback stops is telling
   you the strategy has no real risk management.

Trailing stops and breakeven triggers are supported for Python
(`.attrs`) and the Manual Builder; Pine/MQL5 express them via the
builder, not directives.

---

## Quick checklist before you upload or generate

1. **One required entry point per language** — `generate_signals(df)`
   (Python); at least one `strategy.entry(...)` (Pine);
   `trade.Buy/Sell(...)` or `OrderSend(...)` inside an `if` (MQL5);
   a JSON with `entry_conditions.long` or `.short` non-empty (Manual).
   Missing it = immediate, named error.
2. **Only the listed indicators/functions.** Anything else fails to load
   on purpose — a rejected strategy beats a silently mis-backtested one.
3. **A real stop is defined** — not the 1% fallback. Prefer ATR-based
   stops for anything that isn't FX.
4. **pip_size matches the instrument** — or leave it to auto-detect and
   confirm the green confirmation line names your instrument.
5. **No lookahead** — no negative shifts, no hand-rolled HTF filter with
   `htf.index < timestamp`; every backtest runs the behavioral
   lookahead check and fails loudly on a leak.
6. **Run 05 Run & Report first, then 15 Full Pipeline** before trusting
   any numbers — whether you wrote the strategy, downloaded it, or the
   AI Generate tab built it.

## Minimal complete examples

**Python** — EMA cross with ATR stop:

```python
import pandas as pd

STRATEGY_NAME = "EMA Cross 10/30"
WARMUP_BARS = 30

def generate_signals(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    fast = close.ewm(span=10).mean()
    slow = close.ewm(span=30).mean()
    tr = (df["high"] - df["low"]).abs()
    atr = tr.rolling(14).mean()

    long_cond = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    short_cond = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    signals = pd.Series(0, index=df.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    signals.attrs["stop_loss_distance"] = 1.5 * atr      # per-trade ATR stop
    signals.attrs["take_profit_distance"] = 3.0 * atr    # per-trade ATR target
    return signals
```

**Pine** — RSI washout reversal:

```pine
rsiLen = input.int(14, "RSI length")
rsiVal = ta.rsi(close, rsiLen)
longCondition = ta.crossunder(rsiVal, 30)
shortCondition = ta.crossover(rsiVal, 70)

strategy.entry("Long", strategy.long, when=longCondition)
strategy.entry("Short", strategy.short, when=shortCondition)
strategy.close("Long", when=ta.crossover(rsiVal, 55))
strategy.close("Short", when=ta.crossunder(rsiVal, 45))

// T58_SL_ATR_MULT=1.5
// T58_TP_ATR_MULT=3.0
```

**MQL5** — Donchian-style breakout:

```mql5
double hh = iHighest(_Symbol, PERIOD_CURRENT, MODE_HIGH, 20, 0);
double ll = iLowest(_Symbol, PERIOD_CURRENT, MODE_LOW, 20, 0);
double atrVal = iATR(_Symbol, PERIOD_CURRENT, 14);

if (close > hh)
   trade.Buy(0.1);
if (close < ll)
   trade.Sell(0.1);

// T58_SL_ATR_MULT=2.0
// T58_TP_ATR_MULT=4.0
```

(Conditions are evaluated bar-by-bar over the full history, producing a
boolean series — an entry fires on each bar where its guard is true and no
position is already open, exactly like the Pine subset.)
