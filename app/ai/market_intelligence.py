"""Shared "gather real market facts across forex/crypto/futures" logic,
used by BOTH the web AI Assistant blueprint (app.web.ai_assistant_routes)
and the desktop AI Assistant tab (app.ui.main_window).

BUGFIX (Sep 2026): this logic used to live ONLY inline inside the web
blueprint. The desktop tab's own `_ai_market_context()` always passed
`rankings=[]` into app.ai.trading_assistant.build_context() -- so Daily
Brief/chat/the new Basic Outlook button on the desktop app silently never
saw "which markets moved best" even though the identical feature on the
web app already worked correctly. Extracting the real implementation here
(rather than copy-pasting it into main_window.py a second time) means the
two UIs can no longer drift apart on this again.

Data source: the app's existing global MT5 connection (app.web.live_market
-- the same one the Live Market tab manages) for forex/futures/broker-
crypto-CFDs, with app.data's Alpaca source as a key-free-tier fallback for
crypto when MT5 isn't configured or doesn't list a symbol. Both desktop
and web already depend on app.web.live_market for other features, so
importing it from the desktop side here is not a new dependency.

Macro bias caveat (same one the web dashboard surfaces in `macro_note`):
this app has no live rates/positioning feed, so `macro_bias_by_symbol` is a
plain technical PROXY (daily EMA50 vs EMA200 trend) and is what
t58_strategy_engine's score is actually computed from -- that hasn't
changed. `fundamental_bias_by_symbol` (Sep 2026) is a second, independent
read built from recent FRED/ForexFactory data surprises (actual vs.
forecast on already-released high/medium-impact events -- see
app.ai.news_forexfactory.recent_data_surprise_bias_by_currency), carried
on each ranking and passed into Owen AI chat's context so the model can
reason about technical and fundamental agreeing/conflicting, rather than
being blended into the technical score itself (which stays deterministic
and unchanged). compute_market_structure_notes() below is a further,
separate signal again -- real, deterministic BOS/ChoCH (break of
structure / change of character) and Wyckoff spring/upthrust/SOS/SOW +
phase detection (app.quant_lab.market_structure, ported from the HyperTA
project's Structures module) -- fed into Owen AI's chat context (see
app.ai.trading_assistant.build_context) so the model has real computed
structure facts for its top-ranked symbols instead of only an EMA-cross
proxy, or having to eyeball structure from price alone.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time

import pandas as pd

from app.ai import market_scanner, news_forexfactory
from app.quant_lab.market_structure import MarketStructureError, summarize_market_structure
from app.strategy.indicators import ema

logger = logging.getLogger(__name__)

# v9.15: one feed attempt (MT5, then Alpaca) gets this long before the
# local-dataset fallback takes over. Without a bound, a hung terminal /
# SDK call stalls compute_rankings symbol-by-symbol and the AI Director
# panel sits on "Loading..." forever (Owen's Windows report).
FEED_ATTEMPT_TIMEOUT_S = 8.0

# ---------------------------------------------------------------------------
# v9.15 LOCAL dataset fallback. When neither feed returns bars, look for
# a matching dataset the app ALREADY has on disk (app.data.storage's
# registry + app.data.importer's loader -- the same pair every dataset
# picker uses). Mapping is by underlying instrument only, never by
# "close enough": NAS100->NQ and SPX500->ES (the CME futures on the same
# indices), MES/MNQ/MGC->their full-size ES/NQ/GC contracts, forex/crypto
# -> same-symbol files. Anything with no real mapping (e.g. US30 with
# only ES/NQ files on disk) finds nothing and is skipped quietly.
# ---------------------------------------------------------------------------
_LOCAL_INSTRUMENT_CANDIDATES: dict[str, list[str]] = {
    "EURUSD": ["EURUSD"], "GBPUSD": ["GBPUSD"], "USDJPY": ["USDJPY"],
    "AUDUSD": ["AUDUSD"], "USDCAD": ["USDCAD"], "USDCHF": ["USDCHF"],
    "NZDUSD": ["NZDUSD"], "XAUUSD": ["XAUUSD"],
    "BTCUSD": ["BTCUSD", "BTC"], "ETHUSD": ["ETHUSD", "ETH"],
    "NAS100": ["NAS100", "NQ", "MNQ"],
    "SPX500": ["SPX500", "ES", "MES", "SPX"],
    "US30": ["US30", "YM", "MYM"],
    "MES": ["MES", "ES"], "MNQ": ["MNQ", "NQ"], "MGC": ["MGC", "GC"],
}
_LOCAL_DATASET_CACHE: dict[str, pd.DataFrame | None] = {}


# When a feed attempt TIMES OUT (genuinely hung, not just empty),
# later symbols stop paying that cost one-by-one: feed attempts are
# skipped for a short cooldown and the scan goes straight to local
# data. A merely-empty feed result does NOT trip this -- a connected
# feed that lacks one symbol must still be tried for the next.
_FEED_DOWN_UNTIL = 0.0
_FEED_DOWN_COOLDOWN_S = 30.0


def _feed_bars(symbol: str, timeframe_minutes: int, count: int):
    global _FEED_DOWN_UNTIL
    if time.monotonic() < _FEED_DOWN_UNTIL:
        return None
    from app.web import live_market

    def _attempt():
        bars = live_market.fetch_mt5_bars(symbol, timeframe_minutes, count)
        if not bars:
            bars = live_market.fetch_alpaca_bars(symbol, "Crypto", timeframe_minutes, count)
        return bars

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="t58-feed")
    try:
        return executor.submit(_attempt).result(timeout=FEED_ATTEMPT_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        _FEED_DOWN_UNTIL = time.monotonic() + _FEED_DOWN_COOLDOWN_S
        logger.debug("market_intelligence: feed attempt for %s timed out; cooling down", symbol)
        return None
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)


def _local_instrument_candidates(symbol: str) -> list[str]:
    sym = (symbol or "").strip().upper()
    return _LOCAL_INSTRUMENT_CANDIDATES.get(sym, [sym] if sym else [])


def _dataset_match_score(dataset_name: str, candidates: list[str]) -> int | None:
    """Score how honestly `dataset_name` matches one of `candidates`
    (best candidate wins). Matches the instrument LABEL (subfolder or
    filename prefix, via app.data.storage.dataset_instrument_label),
    allowing the app's own "1!" continuous-future suffix (ES1! -> ES).
    Returns None for no match -- never a substring guess."""
    try:
        from app.data.storage import dataset_instrument_label

        label = dataset_instrument_label(dataset_name).upper()
    except Exception:  # noqa: BLE001
        return None
    label_root = label.rstrip("!")
    if label_root.endswith("1"):
        label_root = label_root[:-1]
    best: int | None = None
    for idx, cand in enumerate(candidates):
        cand_root = cand.upper().rstrip("!")
        score = None
        if label == cand.upper() or label_root == cand_root:
            score = 100 - idx
        if score is not None and (best is None or score > best):
            best = score
    return best


def _find_local_dataset(symbol: str):
    """The best-matching stored dataset for `symbol`, or None. Uses
    app.data.storage.list_stored_datasets() -- the app's own registry.
    Ties break toward the largest file (longest history, most bars to
    resample from). Never raises."""
    try:
        from app.data.storage import list_stored_datasets

        datasets = list_stored_datasets()
    except Exception:  # noqa: BLE001
        return None
    candidates = _local_instrument_candidates(symbol)
    if not candidates:
        return None
    best = None
    best_key: tuple[int, int] | None = None
    for ds in datasets:
        try:
            if getattr(ds, "size_bytes", 0) <= 32:
                continue
            score = _dataset_match_score(ds.name, candidates)
        except Exception:  # noqa: BLE001
            continue
        if score is None:
            continue
        key = (score, int(ds.size_bytes))
        if best_key is None or key > best_key:
            best, best_key = ds, key
    return best


def _load_local_dataset(path) -> pd.DataFrame | None:
    """Load one stored dataset via app.data.importer.import_csv (the
    loader every web dataset picker funnels through), normalized to
    timestamp/open/high/low/close/volume and cached by path+mtime+size
    so a full-universe scan loads each file at most once. Never raises."""
    try:
        from pathlib import Path

        p = Path(path)
        st = p.stat()
        cache_key = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    except Exception:  # noqa: BLE001
        return None
    if cache_key in _LOCAL_DATASET_CACHE:
        return _LOCAL_DATASET_CACHE[cache_key]
    df: pd.DataFrame | None = None
    try:
        from app.data.importer import import_csv

        result = import_csv(str(p))
        raw = getattr(result, "dataframe", None)
        if raw is not None and not raw.empty:
            work = raw.copy()
            work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
            work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
            for col in ("open", "high", "low", "close"):
                work[col] = pd.to_numeric(work[col], errors="coerce")
            if "volume" not in work.columns:
                work["volume"] = 0.0
            work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
            work = work.dropna(subset=["close"])
            df = work[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
            if df.empty:
                df = None
    except Exception:  # noqa: BLE001
        df = None
    _LOCAL_DATASET_CACHE[cache_key] = df
    return df


def _timeframe_label(timeframe_minutes: int) -> str:
    minutes = int(timeframe_minutes)
    if minutes >= 1440 and minutes % 1440 == 0:
        return "1d" if minutes == 1440 else f"{minutes // 1440}d"
    if minutes >= 60 and minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def local_bars(symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame | None:
    """Bars for `symbol` from a matching LOCAL stored dataset, resampled
    to the requested timeframe with the app's own resampling
    (app.data.timeframe_resample.resample_ohlcv). Returns None -- a
    quiet skip -- when no dataset honestly maps to the symbol, when
    the local data is COARSER than requested (never upsampled /
    fabricated), or when anything fails to load. Never raises."""
    try:
        ds = _find_local_dataset(symbol)
        if ds is None:
            return None
        df = _load_local_dataset(ds.path)
        if df is None or df.empty:
            return None
        from app.data.multi_timeframe import infer_timeframe_minutes

        try:
            native_minutes = float(infer_timeframe_minutes(df))
        except Exception:  # noqa: BLE001
            native_minutes = float(timeframe_minutes)
        if native_minutes > float(timeframe_minutes) + 1e-6:
            return None
        if abs(native_minutes - float(timeframe_minutes)) < 1e-6:
            out = df.tail(int(count))
        else:
            from app.data.timeframe_resample import resample_ohlcv

            out = resample_ohlcv(df, _timeframe_label(timeframe_minutes)).tail(int(count))
        if out is None or out.empty:
            return None
        return out.reset_index(drop=True)
    except Exception:  # noqa: BLE001
        logger.debug("market_intelligence: local fallback failed for %s", symbol, exc_info=True)
        return None


def bar_fetcher(symbol: str, timeframe_minutes: int, count: int):
    """Adapts the app's existing data sources to app.ai.market_scanner's
    BarFetcher shape. Feed first (shared MT5 connection, then Alpaca
    crypto fallback), each attempt time-bounded; when the feed returns
    nothing, falls back to a matching LOCAL stored dataset (v9.15 --
    see _LOCAL_INSTRUMENT_CANDIDATES). Returns None when no source has
    real bars for this symbol; callers skip such symbols quietly."""
    bars = _feed_bars(symbol, timeframe_minutes, count)
    if bars:
        try:
            df = pd.DataFrame(bars)
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception:  # noqa: BLE001 -- malformed feed rows -> try local data instead
            pass
    return local_bars(symbol, timeframe_minutes, count)


def daily_trend_bias(symbol: str) -> str:
    """Technical PROXY for macro_bias -- see module docstring's caveat.
    Daily EMA50 above EMA200 -> 'bullish', below -> 'bearish', otherwise
    'neutral'."""
    try:
        df = bar_fetcher(symbol, 1440, 260)
        if df is None or len(df) < 210:
            return "neutral"
        e50, e200 = ema(df["close"], 50), ema(df["close"], 200)
        if float(e50.iloc[-1]) > float(e200.iloc[-1]):
            return "bullish"
        if float(e50.iloc[-1]) < float(e200.iloc[-1]):
            return "bearish"
        return "neutral"
    except Exception:
        return "neutral"


def get_universe() -> dict[str, list[str]]:
    return market_scanner.DEFAULT_UNIVERSE


# Base/quote for the standard FX pairs in DEFAULT_UNIVERSE/CURRENCY_TO_SYMBOLS
# -- lets fundamental_bias_by_symbol() apply the right sign convention (base
# strength = bullish pair, quote strength = bearish pair). Deliberately
# excludes metals (XAUUSD/XAGUSD), crypto (BTCUSD/ETHUSD) and equity indices
# (US30/NAS100/SPX500): a USD data surprise doesn't move those with a
# reliable, well-known sign the way it does a standard FX pair (gold and
# crypto are usually but not always inversely correlated to USD strength;
# indices can go either way depending on the regime), so guessing a
# direction for them would be asserting a "fact" this app can't actually
# back up. Those symbols get "neutral" from fundamental_bias_for_symbol
# below rather than a fabricated call.
FX_PAIR_BASE_QUOTE: dict[str, tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "AUDUSD": ("AUD", "USD"),
    "NZDUSD": ("NZD", "USD"), "USDJPY": ("USD", "JPY"), "USDCAD": ("USD", "CAD"),
    "USDCHF": ("USD", "CHF"), "EURJPY": ("EUR", "JPY"), "EURGBP": ("EUR", "GBP"),
    "GBPJPY": ("GBP", "JPY"),
}
_BIAS_SCORE = {"bullish": 1, "neutral": 0, "bearish": -1}


def fundamental_bias_for_symbol(symbol: str, currency_bias: dict[str, str]) -> str:
    """Combines two currencies' recent-data-surprise biases (see
    app.ai.news_forexfactory.recent_data_surprise_bias_by_currency) into a
    directional call on the pair itself: base-currency strength is bullish
    for the pair, quote-currency strength is bearish for it. Symbols with
    no known base/quote convention (see FX_PAIR_BASE_QUOTE's docstring)
    return "neutral" rather than a guess."""
    pair = FX_PAIR_BASE_QUOTE.get(symbol)
    if not pair:
        return "neutral"
    base, quote = pair
    total = _BIAS_SCORE.get(currency_bias.get(base, "neutral"), 0) - _BIAS_SCORE.get(currency_bias.get(quote, "neutral"), 0)
    if total > 0:
        return "bullish"
    if total < 0:
        return "bearish"
    return "neutral"


def compute_news() -> "news_forexfactory.CalendarResult":
    """ForexFactory's live feed merged with FRED's official release
    calendar (when a FRED key is configured) -- see app.ai.news_fred's
    module docstring for why this uses two sources instead of one, and
    app.accounts.api_keys for where the FRED key comes from (the same
    key the API Keys page's "Test Connection" already validates)."""
    ff_result = news_forexfactory.fetch_calendar()
    try:
        from app.accounts.api_keys import load_settings as load_api_keys_settings
        fred_key = load_api_keys_settings().fred_api_key
    except Exception:  # noqa: BLE001 -- a settings-load hiccup must never take down the News panel
        fred_key = ""
    if not fred_key:
        return ff_result

    from app.ai import news_fred
    fred_result = news_fred.fetch_calendar(fred_key)
    return news_fred.merge_calendars(ff_result, fred_result)


def compute_rankings(news_result=None, universe: dict[str, list[str]] | None = None) -> tuple[list, list[str]]:
    """Returns (rankings, errors). Pass an already-fetched `news_result`
    (e.g. from a caller that also needs it separately) to avoid a second
    calendar fetch. Pass `universe` to scan a specific subset instead of
    the full get_universe() (e.g. app.web.ai_assistant_routes.
    api_trade_of_the_day scans just DEFAULT_UNIVERSE["micro_futures"])."""
    universe = universe if universe is not None else get_universe()
    all_symbols = [s for symbols in universe.values() for s in symbols]
    macro_bias = {s: daily_trend_bias(s) for s in all_symbols}

    if news_result is None:
        news_result = compute_news()
    news_risk = {s: news_forexfactory.news_risk_for_symbol(news_result, s) for s in all_symbols}
    currency_bias = news_forexfactory.recent_data_surprise_bias_by_currency(news_result)
    fundamental_bias = {s: fundamental_bias_for_symbol(s, currency_bias) for s in all_symbols}

    return market_scanner.rank_markets(
        universe, bar_fetcher=bar_fetcher, macro_bias_by_symbol=macro_bias, news_risk_by_symbol=news_risk,
        fundamental_bias_by_symbol=fundamental_bias,
    )


def market_structure_note_for_symbol(symbol: str) -> str:
    """One-line, plain-language BOS/ChoCH/Wyckoff summary for `symbol` on
    daily bars -- see this module's docstring for why this exists
    alongside (not instead of) daily_trend_bias()'s EMA-cross proxy.
    Returns a short "no data" note rather than raising, matching
    daily_trend_bias()'s own fail-safe convention (a chat context should
    degrade gracefully for one bad symbol, not break the whole context)."""
    try:
        df = bar_fetcher(symbol, 1440, 260)
        if df is None or len(df) < 50:
            return "not enough daily bars for structure analysis"
        summary = summarize_market_structure(df)
        note = f"trend={summary.latest_trend}, bias={summary.bias}"
        if summary.latest_structure_event:
            note += f", last event={summary.latest_structure_event}"
        if summary.latest_wyckoff_phase:
            note += f", wyckoff={summary.latest_wyckoff_phase}"
        return note
    except MarketStructureError as exc:
        return f"structure analysis unavailable: {exc}"
    except Exception:
        return "structure analysis unavailable"


def compute_market_structure_notes(symbols: list[str], max_symbols: int = 5) -> dict[str, str]:
    """Computes market_structure_note_for_symbol() for up to `max_symbols`
    of `symbols` -- capped because this fetches+analyzes daily bars per
    symbol, and the chat context only needs this for the handful of
    symbols actually relevant to the current conversation (typically the
    top-ranked markets), not the whole universe on every message."""
    return {s: market_structure_note_for_symbol(s) for s in symbols[:max_symbols]}
