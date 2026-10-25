"""Cross-instrument transfer check: "a real ES edge should show up, at least weakly, on NQ".

The final strategy -- parameters FROZEN, nothing re-tuned -- is run on a sibling instrument's own data (same timeframe,
that instrument's own tick/pip/cost resolved by ``resolve_leg_risk``) over the window the primary was developed on. An edge
that is a property of the market's behaviour (index futures share most of their intraday microstructure) transfers with the
same sign and a visible fraction of the size; an edge that is a fit to one instrument's noise does not.

Verdict states
  PASS        sibling mean R is positive AND (one-sided day-block bootstrap p <= ``weak_p`` OR it keeps at least
              ``min_retention`` of the primary's mean R)                      -> "at least weakly" transfers
  FAIL        enough sibling trades, but the mean R is non-positive, or positive yet indistinguishable from zero AND
              below the retention floor
  NOT_TESTED  no sibling data on disk, no overlap, or too few sibling trades to judge. NEVER a failure: missing evidence
              is not negative evidence.

Pure math lives in ``judge_transfer`` (unit-testable without data); ``run_transfer_check`` does the data discovery and the
backtest. Nothing here can promote a verdict -- the pipeline uses it only to demote (``transfer_mode="cap"``) or annotate
(``"advisory"``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# Root symbol -> siblings to try, most similar first. Index futures share session structure; the micro contracts track
# their minis; metals track each other. Only siblings that actually have data on disk are used.
SIBLINGS: dict[str, tuple[str, ...]] = {
    "ES": ("NQ", "MNQ", "YM", "RTY"), "MES": ("MNQ", "NQ", "ES"),
    "NQ": ("ES", "MES", "YM", "RTY"), "MNQ": ("MES", "ES", "NQ"),
    "YM": ("ES", "NQ"), "RTY": ("ES", "NQ"),
    "GC": ("MGC", "SI"), "MGC": ("GC", "SI"), "SI": ("GC", "MGC"),
    "CL": ("BZ",), "BZ": ("CL",),
}
_ROOT_RE = re.compile(r"(?<![A-Za-z])(MES|MNQ|MGC|ES|NQ|YM|RTY|GC|SI|CL|BZ)(?![A-Za-z])", re.I)
_ALIASES = {"SPX500": "ES", "S&P 500": "ES", "NAS100": "NQ", "NASDAQ 100": "NQ", "US30": "YM", "DOW JONES 30": "YM",
            "XAUUSD": "GC", "GOLD": "GC", "XAGUSD": "SI", "SILVER": "SI"}


def root_symbol(label: str | None) -> str | None:
    """'ES1!' / 'ES_5min_2026.csv' / 'futures_ES.F_1m.parquet' / 'NQ1!/NQ_5min.csv' -> 'ES' / 'NQ'; None if unknown."""
    if not label:
        return None
    s = str(label).replace("\\", "/")
    up = s.upper()
    for alias, root in _ALIASES.items():
        if alias in up:
            return root
    # prefer the folder segment ("NQ1!/..."), then the file stem
    for part in [p for p in s.split("/") if p]:
        m = _ROOT_RE.search(re.sub(r"\d+!?", " ", part))
        if m:
            return m.group(1).upper()
    return None


def sibling_candidates(label: str | None) -> list[str]:
    root = root_symbol(label)
    return list(SIBLINGS.get(root, ())) if root else []


@dataclass
class TransferResult:
    status: str                          # PASS | FAIL | NOT_TESTED
    primary_label: str = ""
    sibling: str = ""
    sibling_dataset: str = ""
    n_trades: int = 0
    mean_r: float | None = None
    primary_mean_r: float | None = None
    retention: float | None = None       # sibling mean R / primary mean R (None if primary <= 0)
    p_value: float | None = None         # one-sided day-block bootstrap, H0: mean R <= 0
    reason: str = ""
    tried: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def judge_transfer(sibling_r, primary_mean_r, p_value, *, min_trades: int = 30, weak_p: float = 0.30,
                   min_retention: float = 0.25) -> tuple[str, str, float | None, float | None]:
    """(status, reason, mean_r, retention) from the sibling's per-trade R values."""
    r = np.asarray([x for x in sibling_r if x is not None and np.isfinite(x)], dtype=float)
    n = int(r.size)
    if n < min_trades:
        return "NOT_TESTED", f"only {n} sibling trades (need {min_trades}) -- too few to judge transfer", None, None
    mean_r = float(r.mean())
    retention = (mean_r / primary_mean_r) if (primary_mean_r is not None and primary_mean_r > 0) else None
    if mean_r <= 0:
        return "FAIL", f"mean {mean_r:+.3f}R/trade over {n} sibling trades -- the edge does not appear at all", mean_r, retention
    weak_by_p = p_value is not None and p_value <= weak_p
    weak_by_size = retention is not None and retention >= min_retention
    if weak_by_p or weak_by_size:
        bits = [f"mean {mean_r:+.3f}R over {n} trades"]
        if p_value is not None:
            bits.append(f"p={p_value:.2f}")
        if retention is not None:
            bits.append(f"keeps {retention * 100:.0f}% of the primary's R")
        return "PASS", "; ".join(bits), mean_r, retention
    return ("FAIL", f"mean {mean_r:+.3f}R over {n} trades is positive but indistinguishable from zero"
            + (f" (p={p_value:.2f})" if p_value is not None else "")
            + (f" and keeps only {retention * 100:.0f}% of the primary's R" if retention is not None else ""),
            mean_r, retention)


def trade_r_values(trades) -> list[float]:
    """Net R per trade = pnl / (initial_risk * size), for trades that recorded an initial risk."""
    out = []
    for t in trades or []:
        ir, sz = getattr(t, "initial_risk", None), getattr(t, "size", None)
        if ir and sz and ir * sz > 0:
            out.append(float(t.pnl) / (float(ir) * float(sz)))
    return out


def find_sibling_dataset(label: str | None, timeframe_seconds: int | None, listing: list[dict] | None = None):
    """(sibling_root, stored dataset full_name) for the first sibling with data at the primary's bar size, else None.

    ``listing`` is ``app.data.storage.list_datasets_by_instrument(count_rows=False)`` (injectable for tests)."""
    from app.data.instrument_folders import timeframe_seconds as _tfs
    if listing is None:
        from app.data.storage import list_datasets_by_instrument
        listing = list_datasets_by_instrument(count_rows=False)
    tried = []
    for sib in sibling_candidates(label):
        for group in listing:
            if root_symbol(group.get("instrument")) != sib:
                continue
            files = [f for f in group.get("files", []) if not f.get("empty")]
            same_tf = [f for f in files if timeframe_seconds and _tfs(f["name"]) == timeframe_seconds]
            pick = same_tf[0] if same_tf else None
            if pick:
                return sib, pick["full_name"], tried
            tried.append(f"{sib}: no {timeframe_seconds}s dataset")
    return None, None, tried


def _window(sib_df, dev_df):
    """Restrict the sibling to the primary's development window (never the locked holdout)."""
    try:
        t0, t1 = dev_df["timestamp"].iloc[0], dev_df["timestamp"].iloc[-1]
        ts = sib_df["timestamp"]
        out = sib_df[(ts >= t0) & (ts <= t1)].reset_index(drop=True)
        return out
    except Exception:  # noqa: BLE001
        return sib_df


def run_transfer_check(strategy, dev_df, primary_trades, risk, instrument_label: str | None, *,
                       sibling_df=None, sibling_label: str | None = None, timeframe_seconds: int | None = None,
                       min_trades: int = 30, weak_p: float = 0.30, min_retention: float = 0.25) -> TransferResult:
    """Run the frozen ``strategy`` on a sibling instrument and judge transfer. Never raises: any failure is NOT_TESTED."""
    res = TransferResult(status="NOT_TESTED", primary_label=str(instrument_label or ""))
    try:
        primary_r = trade_r_values(primary_trades)
        res.primary_mean_r = float(np.mean(primary_r)) if primary_r else None
        sib_name = sibling_label
        if sibling_df is None:
            from app.data.importer import import_csv
            from app.data.storage import resolve_stored_dataset
            sib_root, full_name, tried = find_sibling_dataset(instrument_label, timeframe_seconds)
            res.tried = tried
            if not full_name:
                res.reason = ("no sibling instrument data on disk at this bar size"
                              + (f" ({'; '.join(tried)})" if tried else "")
                              if sibling_candidates(instrument_label) else "no known sibling instrument for this market")
                return res
            path = resolve_stored_dataset(full_name)
            imp = import_csv(path, historical_complete=True) if path else None
            if imp is None or imp.dataframe is None:
                res.reason = f"could not load sibling dataset {full_name}"
                return res
            sibling_df, sib_name, res.sibling_dataset = imp.dataframe, sib_root, full_name
        if "timestamp" not in sibling_df.columns:
            sibling_df = sibling_df.reset_index().rename(columns={sibling_df.index.name or "index": "timestamp"})
        sib = _window(sibling_df, dev_df)
        res.sibling = str(sib_name or "")
        if len(sib) < 500:
            res.reason = f"sibling data overlaps the development window by only {len(sib)} bars"
            return res
        from app.backtest.engine import run_backtest
        from app.search.instrument_risk import resolve_leg_risk
        leg_risk, _notes = resolve_leg_risk(risk, sib, sib_name)
        bt = run_backtest(sib, strategy, leg_risk)
        r = trade_r_values(bt.trades)
        res.n_trades = len(r)
        p = None
        if len(r) >= 5:
            from app.research.preregistered_tests import bootstrap_pvalue_positive
            p = bootstrap_pvalue_positive(r, [str(t.entry_time)[:10] for t in bt.trades if getattr(t, "initial_risk", None)
                                              and getattr(t, "size", None)], n_boot=2000)
        res.p_value = p
        res.status, res.reason, res.mean_r, res.retention = judge_transfer(
            r, res.primary_mean_r, p, min_trades=min_trades, weak_p=weak_p, min_retention=min_retention)
        return res
    except Exception as exc:  # noqa: BLE001 -- missing evidence is not negative evidence
        res.status, res.reason = "NOT_TESTED", f"transfer check could not run: {exc}"
        return res
