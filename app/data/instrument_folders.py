"""Instrument-folder naming for market data (Oct 2026).

ONE place that answers "which instrument does this data file belong to?" so that

  * importing a folder / the market-data pack keeps every instrument together
    (data/raw/EURUSD/EURUSD1.csv, EURUSD5.csv, fx_EUR_USD_1m.parquet ...) instead of
    flattening every file into data/raw/, and
  * every dataset dropdown shows ONE header per instrument ("EURUSD") with all of its
    timeframes listed under it, not a header per file prefix ("EURUSD1", "EURUSD5",
    "FUTURES", "FX" ...).

Before this module the dropdown header was a regex over the first characters of the file
name. That made "EURUSD1.csv" -> "EURUSD1", "EURUSD5.csv" -> "EURUSD5", and lumped
"futures_ES.F_1m.parquet", "futures_NQ.F_1m.parquet", "futures_GC.F_1m.parquet" together under a
single "FUTURES" header (so ES / NQ / GC looked missing) and every "fx_*" file under "FX".

Pure functions, no I/O, no heavy imports -- safe to import anywhere.
"""
from __future__ import annotations

import re

# Canonical folder name (exactly the folder names already used in data/raw/) -> every spelling of
# that instrument that shows up in data file names. Matching is on a normalised key (upper-case,
# letters and digits only), so "AUD_USD", "AUDUSD" and "aud-usd" are all the same alias.
_FOLDER_ALIASES: dict[str, tuple[str, ...]] = {
    "AUDUSD": ("AUDUSD",),
    "BITCOIN": ("BTCUSD", "BTC", "BITCOIN", "XBTUSD"),
    "BRENT CRUDE OIL": ("BRENTCMDUSD", "BCOUSD", "BRENT", "BRENTCRUDEOIL", "UKOIL"),
    "DOGECOIN": ("DOGEUSD", "DOGE", "DOGECOIN"),
    "DOW JONES 30": ("USA30IDXUSD", "US30USD", "US30", "DJ30", "DJI", "DOWJONES30", "DOWJONES"),
    "CL1!": ("CL", "CL1", "CLF"),
    "ES1!": ("ES", "ES1", "ESF"),
    "ETHEREUM": ("ETHUSD", "ETH", "ETHEREUM"),
    "EURUSD": ("EURUSD",),
    "GBPUSD": ("GBPUSD",),
    "GC1!": ("GC", "GC1", "GCF"),
    "MES1!": ("MES", "MES1", "MESF"),
    "MGC1!": ("MGC", "MGC1", "MGCF"),
    "MNQ1!": ("MNQ", "MNQ1", "MNQF"),
    "NASDAQ 100": ("USATECHIDXUSD", "NAS100USD", "NAS100", "NDX", "USTEC", "NASDAQ100"),
    "NQ1!": ("NQ", "NQ1", "NQF"),
    "NVIDIA": ("NVDA", "NVIDIA"),
    "NZDUSD": ("NZDUSD",),
    "RUSSELL 2000": ("RUSSELL2000", "RTY", "US2000"),
    "S&P 500": ("USA500IDXUSD", "SPX500USD", "SPX500", "SPX", "SP500", "US500"),
    "SI1!": ("SI", "SI1", "SIF"),
    "SOLANA": ("SOLUSD", "SOL", "SOLANA"),
    "USDCAD": ("USDCAD",),
    "USDCHF": ("USDCHF",),
    "USDJPY": ("USDJPY",),
    "XAGUSD (SILVER)": ("XAGUSD", "XAG", "SILVER"),
    "XAUUSD (GOLD)": ("XAUUSD", "XAU", "GOLD"),
    "XCUUSD (COPPER)": ("XCUUSD", "XCU", "COPPER"),
    "XPTUSD (PLATINUM)": ("XPTUSD", "XPT", "PLATINUM"),
    "YAHOO": ("YHOO", "YAHOO"),
    "YM1!": ("YM", "YM1", "YMF"),
    "RTY1!": ("RTY", "RTY1", "RTYF"),
    "ZN1!": ("ZN", "ZN1", "ZNF"),
    "ZB1!": ("ZB", "ZB1", "ZBF"),
    "6E1!": ("6E", "6E1", "6EF"),
}

# Words a data vendor / exporter puts in FRONT of the symbol ("futures_ES.F_1m", "fx_EUR_USD_1m").
_NOISE_TOKENS = frozenset({
    "FUTURES", "FUTURE", "FX", "FOREX", "CRYPTO", "STOCK", "STOCKS", "EQUITY", "EQUITIES",
    "INDEX", "INDICES", "CFD", "SPOT", "METAL", "METALS", "COMMODITY", "COMMODITIES",
})

# Folder names that only describe where a file was picked from, never an instrument.
_GENERIC_FOLDERS = frozenset({
    "DATA", "RAW", "IMPORT", "IMPORTS", "UPLOAD", "UPLOADS", "DOWNLOAD", "DOWNLOADS", "DESKTOP",
    "DOCUMENTS", "MARKETDATA", "T58MARKETDATA", "MARKET", "DATASETS", "DATASET", "FILES", "CSV",
    "PARQUET", "TMP", "TEMP",
})

_MAX_FOLDER_LEN = 60
_UNSAFE_FOLDER_CHARS = re.compile(r'[<>:"|?*\\/\x00-\x1f]')


def _norm(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _build_key_map() -> dict[str, str]:
    keys: dict[str, str] = {}
    for folder, aliases in _FOLDER_ALIASES.items():
        keys[_norm(folder)] = folder
        for alias in aliases:
            keys.setdefault(_norm(alias), folder)
    return keys


_KEY_TO_FOLDER = _build_key_map()


def _lookup(key: str) -> str | None:
    """Folder for a normalised key, also accepting a trailing timeframe number
    ("EURUSD1440", "USATECHIDXUSD5") when the key without it is a known instrument."""
    if not key:
        return None
    hit = _KEY_TO_FOLDER.get(key)
    if hit:
        return hit
    stripped = key.rstrip("0123456789")
    if stripped and stripped != key:
        return _KEY_TO_FOLDER.get(stripped)
    return None


def _stem_tokens(filename: str) -> list[str]:
    base = str(filename).replace("\\", "/").split("/")[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    tokens = [t for t in re.split(r"[\s_\-.]+", stem) if t]
    # A leading vendor/asset-class word is never the instrument.
    while tokens and _norm(tokens[0]) in _NOISE_TOKENS:
        tokens.pop(0)
    return tokens


def instrument_from_filename(filename: str) -> tuple[str, bool]:
    """(instrument folder name, known) for one data file name.

    ``known`` is True when the file's symbol matched a canonical instrument folder; otherwise the
    folder name is a best-effort guess from the file's leading symbol token. Never raises;
    returns ("", False) when there is nothing readable.

        EURUSD5.csv                -> ("EURUSD", True)
        fx_EUR_USD_1m.parquet      -> ("EURUSD", True)
        futures_ES.F_1m.parquet    -> ("ES1!", True)
        nvda-1999-2014.csv         -> ("NVIDIA", True)
        USATECHIDXUSD15.csv        -> ("NASDAQ 100", True)
        AAA_5M.csv                 -> ("AAA", False)
    """
    tokens = _stem_tokens(filename)
    if not tokens:
        return "", False
    # Longest symbol first: ["AUD", "USD", "1m"] -> "AUDUSD" before "AUD".
    for n in range(min(3, len(tokens)), 0, -1):
        hit = _lookup(_norm("".join(tokens[:n])))
        if hit:
            return hit, True
    first = _norm(tokens[0])
    if not first:
        return "", False
    letters = first.rstrip("0123456789")
    return (letters if len(letters) >= 2 else first), False


def _clean_folder_name(name: str) -> str:
    cleaned = _UNSAFE_FOLDER_CHARS.sub("", str(name)).strip().strip(".")
    return cleaned[:_MAX_FOLDER_LEN].strip().strip(".")


def split_import_relpath(relpath: str) -> tuple[list[str], str]:
    """(folder parts, file name) of a client/folder-relative import path, with any wrapper the
    market-data pack adds stripped: "T58-Market-Data/data/raw/EURUSD/EURUSD1.csv" ->
    (["EURUSD"], "EURUSD1.csv"); "EURUSD/EURUSD1.csv" -> (["EURUSD"], "EURUSD1.csv")."""
    parts = [p for p in str(relpath or "").replace("\\", "/").split("/") if p and p not in (".", "..")]
    if not parts:
        return [], ""
    folders, name = parts[:-1], parts[-1]
    lowered = [f.lower() for f in folders]
    for i in range(len(folders) - 1):
        if lowered[i] == "data" and lowered[i + 1] == "raw":
            return folders[i + 2:], name
    if "raw" in lowered:
        return folders[lowered.index("raw") + 1:], name
    return folders, name


def instrument_folder_for_import(relpath: str, existing_folders: list[str] | None = None) -> str:
    """The data/raw/<folder> a file imported as ``relpath`` should be stored under ("" = keep it at
    the top level because nothing about it identifies an instrument).

    Priority: (1) the file name's own symbol, (2) a known instrument folder it already sits in,
    (3) any other meaningful parent folder the person organised it in, (4) the file name's leading
    token. An existing folder that differs only by case/spacing/punctuation is reused, so
    importing never creates "eurusd" next to "EURUSD"."""
    folders, name = split_import_relpath(relpath)
    chosen = ""
    guess, known = instrument_from_filename(name)
    if known:
        chosen = guess
    else:
        for folder in folders[:1] + folders[-1:]:
            hit = _lookup(_norm(folder))
            if hit:
                chosen = hit
                break
        if not chosen:
            for folder in folders[:1]:
                if _norm(folder) and _norm(folder) not in _GENERIC_FOLDERS:
                    chosen = _clean_folder_name(folder)
                    break
        # (A bare, unrecognised file name with no folder stays loose at the top level, as before:
        # inventing a folder for "notes.csv" would just scatter one-file folders.)
    chosen = _clean_folder_name(chosen)
    if chosen and existing_folders:
        wanted = _norm(chosen)
        for existing in existing_folders:
            if _norm(existing) == wanted:
                return existing
    return chosen


def picker_label(name: str) -> str:
    """Dropdown group header for one stored dataset (relative path under data/raw/).

    A subfolder is the instrument ("EURUSD/EURUSD5.csv" -> "EURUSD"); a loose file is mapped by its
    own symbol ("fx_EUR_USD_1m.parquet" -> "EURUSD", "futures_ES.F_1m.parquet" -> "ES1!"). Never
    raises; "(UNGROUPED)" when nothing is readable."""
    parts = [p for p in str(name).replace("\\", "/").split("/") if p]
    if not parts:
        return "(UNGROUPED)"
    if len(parts) > 1:
        folder = parts[0]
        return _KEY_TO_FOLDER.get(_norm(folder), folder)
    guess, _known = instrument_from_filename(parts[-1])
    return guess or "(UNGROUPED)"


# ---------------------------------------------------------------------------
# Timeframe from a file name, so a dropdown lists 1m -> 5m -> 15m ... -> 1D in bar-size order
# ---------------------------------------------------------------------------
_UNIT_SECONDS = {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hr": 3600, "d": 86400, "day": 86400,
                 "w": 604800}
_TF_TOKEN = re.compile(r"^(\d+)(s|sec|m|min|h|hr|d|day|w)$", re.IGNORECASE)
_TRAILING_MINUTES = re.compile(r"[A-Za-z!]\.?(\d{1,4})$")


def timeframe_seconds(filename: str) -> int | None:
    """Bar size in seconds implied by a data file name, or None when it can't be told.

    "EURUSD1440.csv" -> 86400 (bare trailing numbers are minutes, the MetaTrader export convention),
    "fx_EUR_USD_1m.parquet" -> 60, "XCU_USD_15s.parquet" -> 15, "ES_5min_2026...csv" -> 300."""
    base = str(filename).replace("\\", "/").split("/")[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    for token in re.split(r"[\s_\-.]+", stem):
        m = _TF_TOKEN.match(token)
        if m:
            return int(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()]
    tokens = [t for t in re.split(r"[\s_\-.]+", stem) if t]
    if tokens:
        m = _TRAILING_MINUTES.search(tokens[0])
        if m:
            minutes = int(m.group(1))
            if minutes in (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 45, 60, 120, 180, 240, 360, 480, 720, 1440, 10080):
                return minutes * 60
    return None


def timeframe_label(filename: str) -> str:
    """Short bar-size label ("1m", "5m", "1h", "4h", "1D") or "" when unknown."""
    seconds = timeframe_seconds(filename)
    if not seconds:
        return ""
    if seconds % 604800 == 0:
        return f"{seconds // 604800}W"
    if seconds % 86400 == 0:
        return f"{seconds // 86400}D"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def dataset_sort_key(filename: str) -> tuple:
    """Order files inside one instrument by bar size (finest first), then by name; files whose
    timeframe can't be read go last."""
    seconds = timeframe_seconds(filename)
    return (seconds is None, seconds or 0, str(filename).lower())
