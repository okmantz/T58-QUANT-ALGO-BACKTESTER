"""
Strategy Graveyard -- Owen's ask: "every rejected strategy should be
stored with why it failed... eventually you build a map of dead
strategy space."

app.search.family_health already answers "has this whole FAMILY ever
produced a real survivor" (a coarse, binary, cross-run signal). This
module answers the finer-grained question underneath that: for a
candidate that got far enough to be genuinely interesting (it survived
the cheap pre-filter and reached robustness/OOS/Monte Carlo/CPCV/stress
scoring) but still didn't make the leaderboard, WHY not, specifically --
so the next run doesn't spend another four hours re-discovering that
"RSI-extreme-reversion with a 68-period lookback and 1.8R target is
71% likely to fail Monte Carlo" the hard way.

Storage: JSON Lines, one row per rejected candidate, append-only --
same pattern as app.evolution.checkpoint's tested_candidates.jsonl and
app.evolution.knowledge_graph's own log, deliberately NOT a new
database dependency. Cheap to write (one line, no locking needed
beyond append), cheap to read back in full for a single run's worth of
history (tens of thousands of lines at most).

This module does not duplicate the pre-filter's own rejection log
(app.evolution.checkpoint's "stage": "prefilter" rows already cover
"91/91 unprofitable" at near-zero cost per candidate) -- it exists
specifically for candidates that got EXPENSIVE (robustness, Monte
Carlo, CPCV, stress) before dying, where the "why" is worth a full
sentence, not just a rejection-reason keyword.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.data.storage import get_app_base_dir


def default_graveyard_path() -> Path:
    return get_app_base_dir() / "data" / "evolution" / "strategy_graveyard.jsonl"


def _sanitize_path_component(s: str) -> str:
    # "-" (never "_") for anything not alnum/hyphen, so a sanitized
    # component can never itself contain the "__" double-underscore this
    # module uses as the filename's field delimiter (graveyard_path_for) --
    # an instrument name containing punctuation (e.g. "ES1!") would
    # otherwise sometimes produce a trailing "_" that collides with the
    # delimiter and makes the instrument/timeframe unrecoverable from the
    # filename in list_graveyard_files.
    s = (s or "unknown").strip().lower()
    return "".join(c if (c.isalnum() or c == "-") else "-" for c in s) or "unknown"


def graveyard_path_for(instrument: str, timeframe: str = "unknown") -> Path:
    """A stable, PERSISTENT graveyard path shared by every run against the
    same instrument+timeframe, so rejections actually accumulate across
    runs instead of disappearing into a one-off file nothing ever reads
    again. Scoped per-instrument (not one single global file) because a
    parameter neighborhood dead on ES futures says nothing about whether
    the same neighborhood is dead on EURUSD -- collapsing everything into
    one file would let an irrelevant market's history suppress a
    hypothesis that was never actually tried on THIS data.
    """
    fname = f"strategy_graveyard__{_sanitize_path_component(instrument)}__{_sanitize_path_component(timeframe)}.jsonl"
    return get_app_base_dir() / "data" / "evolution" / fname


@dataclass
class GraveyardEntry:
    candidate_id: str
    family: str
    generation: int | None
    stage_died: str                 # "cpcv" | "stress" | "not_ranked" -- where it was cut
    reason: str                     # one-sentence, human-readable primary reason
    oos_result: str | None = None                # "negative" / "positive" / None if not computed
    neighbor_robustness_pct: float | None = None  # parameter-neighborhood stability, 0-100
    monte_carlo_failure_pct: float | None = None  # 100 - eval pass probability
    prop_sim_pass_pct: float | None = None        # raw eval pass probability (in-sample MC)
    cpcv_oos_pass_pct: float | None = None        # honest held-out estimate, if computed
    pbo: float | None = None
    fitness_score: float | None = None
    param_signature: str | None = None            # coarse "same neighborhood" fingerprint
    notes: list = field(default_factory=list)

    # -- "Why Did This Strategy Fail?" engine (see app.search.failure_diagnosis) --
    # Populated by Forge Strategy (app.orchestration.forge) for candidates that
    # got far enough to be worth a structured diagnosis, not just a reason
    # string. All optional/None for entries written by older callers (Search
    # Lab, Evolution Lab) that don't build a full diagnosis -- this is a
    # strictly additive extension of the existing entry shape.
    primary_failure: str | None = None             # e.g. "daily loss limit"
    secondary_failure: str | None = None           # e.g. "excessive losing streak"
    strength: str | None = None                    # e.g. "excellent target achievement"
    weakness: str | None = None                    # e.g. "losses cluster during high-volatility periods"
    suggested_mutation: str | None = None          # e.g. "add a volatility filter"
    related_successful_family: str | None = None   # the best-performing family in this same run, if different

    # -- "Map of dead strategy space" fields (2026-10-24) ------------------
    # Optional, same additive pattern as the failure-diagnosis block above:
    # older rows and callers that cannot fill them simply leave them None.
    n_trades: int | None = None                    # trades behind the numbers below
    net_profit: float | None = None                # $ net at modelled costs
    avg_r_net: float | None = None                 # mean net R-multiple per trade (pnl / initial risk $)
    profit_factor: float | None = None             # net PF at modelled costs (None when undefined/inf)
    null_gate_p: float | None = None               # random-entry null-gate p-value, ONLY when that gate really ran
    bootstrap_p: float | None = None               # cheap day-block bootstrap p(mean pnl > 0); a screen, not the null gate
    doubled_cost_net_profit: float | None = None   # $ net with round-trip friction charged twice
    doubled_cost_profit_factor: float | None = None
    doubled_cost_avg_r: float | None = None
    instrument: str | None = None                  # also recorded in the row, not only in the filename
    timeframe: str | None = None
    data_start: str | None = None                  # ISO date of the first bar the candidate was evaluated on
    data_end: str | None = None
    data_bars: int | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self) -> str:
        lines = [f"{self.candidate_id}", "FAILED", "", f"Reason:\n{self.reason}", ""]
        if self.primary_failure is not None:
            lines += [f"Primary failure:\n{self.primary_failure}", ""]
        if self.secondary_failure is not None:
            lines += [f"Secondary failure:\n{self.secondary_failure}", ""]
        if self.strength is not None:
            lines += [f"Strength:\n{self.strength}", ""]
        if self.weakness is not None:
            lines += [f"Weakness:\n{self.weakness}", ""]
        if self.suggested_mutation is not None:
            lines += [f"Potential mutation:\n{self.suggested_mutation}", ""]
        if self.related_successful_family is not None:
            lines += [f"Related successful family:\n{self.related_successful_family}", ""]
        if self.oos_result is not None:
            lines += [f"OOS:\n{self.oos_result}", ""]
        if self.neighbor_robustness_pct is not None:
            lines += [f"Neighbor robustness:\n{self.neighbor_robustness_pct:.0f}%", ""]
        if self.monte_carlo_failure_pct is not None:
            lines += [f"Monte Carlo:\nFailure probability {self.monte_carlo_failure_pct:.0f}%", ""]
        if self.prop_sim_pass_pct is not None:
            lines += [f"Prop simulation:\nPASS probability {self.prop_sim_pass_pct:.0f}%", ""]
        if self.n_trades is not None:
            bits = [f"{self.n_trades} trades"]
            if self.avg_r_net is not None:
                bits.append(f"net {self.avg_r_net:+.2f}R/trade")
            if self.profit_factor is not None:
                bits.append(f"PF {self.profit_factor:.2f}")
            lines += ["Edge:\n" + ", ".join(bits), ""]
        if self.null_gate_p is not None:
            lines += [f"Null gate:\np = {self.null_gate_p:.3f}", ""]
        elif self.bootstrap_p is not None:
            lines += [f"Bootstrap (not the null gate):\np = {self.bootstrap_p:.3f}", ""]
        if self.doubled_cost_net_profit is not None:
            pf = f", PF {self.doubled_cost_profit_factor:.2f}" if self.doubled_cost_profit_factor is not None else ""
            lines += [f"At doubled costs:\nnet ${self.doubled_cost_net_profit:,.0f}{pf}", ""]
        if self.data_start or self.data_end:
            where = " ".join(x for x in (self.instrument, self.timeframe) if x)
            lines += [f"Data window:\n{where + ' ' if where else ''}{self.data_start or '?'} to {self.data_end or '?'}", ""]
        return "\n".join(lines).rstrip()


def _round_sig(v: float, digits: int = 1) -> float:
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return 0.0


def param_signature(family: str, config: dict | None, keys: tuple[str, ...] = ()) -> str:
    """A coarse fingerprint for 'is this the same neighborhood as
    something already in the graveyard' -- rounds numeric params to 1
    significant change so that gen258/gen260/gen264's near-identical
    mutations of the same elite (see app.evolution.engine's mutation
    step) collapse to one signature instead of three separate entries.
    Not a hash of the full config (irrelevant fields like display names
    would fragment the signature); if `keys` is empty, uses every
    numeric top-level key found. Best-effort: a config shape this can't
    walk simply signs as the family name alone, which still dedupes
    exact-family repeats even without parameter-level granularity.
    """
    if not config:
        return family
    try:
        items = []
        source = config if not keys else {k: config.get(k) for k in keys if k in config}
        for k, v in sorted(source.items()):
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                items.append(f"{k}={_round_sig(v)}")
        return family + "|" + ",".join(items) if items else family
    except Exception:  # noqa: BLE001 -- signature is a best-effort dedupe key, never load-bearing
        return family


# ---------------------------------------------------------------------------
# Starter seed: a few hundred REAL, measured failures shipped with the app (see scripts/build_graveyard_seed.py and
# scripts/export_graveyard_seed.py). On first use they are merged -- idempotently, by candidate_id -- into the
# per-instrument graveyard files, so a fresh download already has a populated Graveyard to browse and learn from.
# ---------------------------------------------------------------------------
SEED_PATH = Path(__file__).parent / "seed_data" / "graveyard_seed.jsonl"
_SEED_CHECKED = False


def _seed_marker() -> Path:
    return get_app_base_dir() / "data" / "evolution" / ".graveyard_seed_installed"


def ensure_seed_installed(force: bool = False) -> int:
    """Merge the bundled seed into the persistent graveyard files. Returns how many rows were added (0 when already
    installed). Idempotent (rows are keyed by candidate_id, and a marker stores the seed's content hash so a
    rebuilt seed merges only its new rows), never raises, and never touches rows a user's own runs wrote."""
    global _SEED_CHECKED
    if _SEED_CHECKED and not force:
        return 0
    try:
        import hashlib

        if not SEED_PATH.exists():
            _SEED_CHECKED = True
            return 0
        raw = SEED_PATH.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        marker = _seed_marker()
        if not force and marker.exists() and marker.read_text(encoding="utf-8").strip() == digest:
            _SEED_CHECKED = True
            return 0
        grouped: dict[tuple[str, str], list[dict]] = {}
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            inst, tf = row.pop("_seed_instrument", None), row.pop("_seed_timeframe", None)
            if inst and tf:
                grouped.setdefault((inst, tf), []).append(row)
        added = 0
        for (inst, tf), rows in grouped.items():
            path = graveyard_path_for(inst, tf)
            have = {r.get("candidate_id") for r in load_graveyard(path, _install=False)}
            new = [r for r in rows if r.get("candidate_id") not in have]
            if new:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    for r in new:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                added += len(new)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(digest, encoding="utf-8")
        _SEED_CHECKED = True
        return added
    except Exception:  # noqa: BLE001 -- the seed is a convenience; it must never break a run or the page
        _SEED_CHECKED = True
        return 0


def list_graveyard_files() -> list[dict]:
    """Every persistent, instrument-scoped graveyard file on disk (see
    graveyard_path_for), newest-modified first, for a browsing UI to pick
    from. Each dict: {instrument, timeframe, path, n_rows, modified}."""
    ensure_seed_installed()
    base = get_app_base_dir() / "data" / "evolution"
    if not base.exists():
        return []
    out = []
    for p in base.glob("strategy_graveyard__*__*.jsonl"):
        stem = p.stem  # strategy_graveyard__<instrument>__<timeframe>
        parts = stem.split("__")
        instrument = parts[1] if len(parts) > 1 else "unknown"
        timeframe = parts[2] if len(parts) > 2 else "unknown"
        try:
            n_rows = sum(1 for _ in p.open("r", encoding="utf-8"))
        except Exception:  # noqa: BLE001
            n_rows = 0
        out.append({
            "instrument": instrument, "timeframe": timeframe, "path": str(p),
            "n_rows": n_rows, "modified": p.stat().st_mtime,
        })
    out.sort(key=lambda d: d["modified"], reverse=True)
    return out


def record_rejection(entry: GraveyardEntry, path: Path | str | None = None) -> None:
    path = Path(path) if path else default_graveyard_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict(), default=str) + "\n")


def record_rejections(entries: list[GraveyardEntry], path: Path | str | None = None) -> None:
    if not entries:
        return
    path = Path(path) if path else default_graveyard_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict(), default=str) + "\n")


def load_graveyard(path: Path | str | None = None, limit: int = 50_000, _install: bool = True) -> list[dict]:
    if _install:
        ensure_seed_installed()
    path = Path(path) if path else default_graveyard_path()
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:  # noqa: BLE001 -- a corrupt/partial file must not crash a caller just reading history
        return []
    return rows[-limit:]


@dataclass
class GraveyardCluster:
    """One 'dead neighborhood' -- every rejection sharing a
    param_signature, collapsed into one entry with a count, so 40
    near-identical mutations of the same failed idea show up as
    ONE line ('tested 40 times, still dead'), not 40 lines."""
    signature: str
    family: str
    n_tested: int
    most_common_reason: str
    worst_stage_reached: str
    best_prop_sim_pass_pct: float | None
    example_candidate_ids: list

    def to_dict(self) -> dict:
        return dict(self.__dict__)


_STAGE_RANK = {"not_ranked": 0, "cpcv": 1, "stress": 2}


def summarize_graveyard(rows: list[dict], top_n: int = 25) -> list[GraveyardCluster]:
    """Groups graveyard rows by param_signature (falling back to family
    alone for rows with no signature) and returns the `top_n` clusters
    with the most repeated attempts first -- that ordering is exactly
    'don't waste another 4 hours here,' the biggest neighborhood T58 has
    already proven dead, first."""
    by_sig: dict[str, list[dict]] = {}
    for r in rows:
        sig = r.get("param_signature") or r.get("family") or "unknown"
        by_sig.setdefault(sig, []).append(r)

    clusters: list[GraveyardCluster] = []
    for sig, group in by_sig.items():
        reasons = [g.get("reason") for g in group if g.get("reason")]
        most_common = max(set(reasons), key=reasons.count) if reasons else "unknown"
        best_pct = max(
            (g.get("prop_sim_pass_pct") for g in group if g.get("prop_sim_pass_pct") is not None),
            default=None,
        )
        worst_stage = max((g.get("stage_died", "not_ranked") for g in group), key=lambda s: _STAGE_RANK.get(s, 0))
        clusters.append(GraveyardCluster(
            signature=sig, family=group[0].get("family", "?"), n_tested=len(group),
            most_common_reason=most_common, worst_stage_reached=worst_stage,
            best_prop_sim_pass_pct=best_pct,
            example_candidate_ids=[g.get("candidate_id") for g in group[:3]],
        ))
    clusters.sort(key=lambda c: c.n_tested, reverse=True)
    return clusters[:top_n]


def render_graveyard_report(clusters: list[GraveyardCluster]) -> str:
    lines = ["Strategy Graveyard -- dead neighborhoods, most-tested first", ""]
    if not clusters:
        lines.append("(empty -- nothing has reached full evaluation and failed yet)")
        return "\n".join(lines)
    header = f"{'Family':<28}{'Tested':>8}{'Worst stage':>14}{'Best pass %':>12}   Reason"
    lines.append(header)
    lines.append("-" * len(header))
    for c in clusters:
        best = f"{c.best_prop_sim_pass_pct:.0f}%" if c.best_prop_sim_pass_pct is not None else "--"
        lines.append(f"{c.family:<28}{c.n_tested:>8}{c.worst_stage_reached:>14}{best:>12}   {c.most_common_reason}")
    return "\n".join(lines)


def is_known_dead_neighborhood(
    family: str, config: dict | None, rows: list[dict] | None = None,
    min_attempts: int = 8, keys: tuple[str, ...] = (),
    path: Path | str | None = None,
) -> tuple[bool, int]:
    """Cheap check before spending a full generation's compute on a
    candidate: has something with this exact param_signature already
    been tried at least `min_attempts` times and always died? Returns
    (is_dead, n_previous_attempts). rows can be pre-loaded (e.g. once
    per generation) to avoid re-reading the file per-candidate; if not
    given, loads from `path`/the default location every call."""
    sig = param_signature(family, config, keys=keys)
    rows = rows if rows is not None else load_graveyard(path)
    matches = [r for r in rows if (r.get("param_signature") or r.get("family")) == sig]
    return (len(matches) >= min_attempts, len(matches))


# ---------------------------------------------------------------------------
# Edge / cost / data-window diagnostics shared by Search Lab, Evolution Lab and Forge
# ---------------------------------------------------------------------------
def _finite_or_none(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _pf(pnls) -> float | None:
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p <= 0)
    return _finite_or_none(gp / gl) if gl > 0 else None


def edge_diagnostics(trades, risk=None, df=None, instrument: str | None = None,
                     timeframe: str | None = None, null_gate_p: float | None = None,
                     bootstrap: bool = True) -> dict:
    """Optional graveyard fields (trade count, net R / PF, doubled-cost result, bootstrap p, data window) from a
    finished trade list. Best-effort: every field degrades to None, and it never raises.

    * ``avg_r_net``: mean of pnl / (initial_risk * size) over trades that recorded an initial risk.
    * doubled costs: the round-turn friction the engine already charged (``size * 2 * side_cost_price`` + the
      trade's commission) is charged a second time. Analytic re-costing of the same trades -- it does not re-run
      the engine, so fills that only exist at the original costs are not re-simulated.
    * ``null_gate_p`` is passed through ONLY when the caller really ran the random-entry null gate; the cheap
      day-block bootstrap goes to ``bootstrap_p`` so the two are never confused.
    """
    out: dict = {}
    try:
        out["instrument"] = instrument
        out["timeframe"] = timeframe
        if df is not None and len(df):
            idx = df["timestamp"] if "timestamp" in getattr(df, "columns", ()) else df.index
            idx = list(idx[[0, -1]]) if hasattr(idx, "iloc") is False else [idx.iloc[0], idx.iloc[-1]]
            out["data_start"] = str(idx[0])[:19]
            out["data_end"] = str(idx[-1])[:19]
            out["data_bars"] = int(len(df))
        out["null_gate_p"] = _finite_or_none(null_gate_p)
        trades = list(trades or [])
        out["n_trades"] = len(trades)
        if not trades:
            return out
        pnls = [float(t.pnl) for t in trades]
        out["net_profit"] = _finite_or_none(sum(pnls))
        out["profit_factor"] = _pf(pnls)
        rs = []
        for t in trades:
            ir = getattr(t, "initial_risk", None)
            sz = getattr(t, "size", None)
            if ir and sz and ir * sz > 0:
                rs.append(float(t.pnl) / (float(ir) * float(sz)))
        out["avg_r_net"] = _finite_or_none(sum(rs) / len(rs)) if rs else None
        if risk is not None:
            side = float(risk.side_cost_price())
            extra = [abs(float(t.size)) * 2.0 * side + float(getattr(t, "commission", 0.0) or 0.0) for t in trades]
            dpnl = [p - e for p, e in zip(pnls, extra)]
            out["doubled_cost_net_profit"] = _finite_or_none(sum(dpnl))
            out["doubled_cost_profit_factor"] = _pf(dpnl)
            drs = []
            for t, e in zip(trades, extra):
                ir = getattr(t, "initial_risk", None)
                sz = getattr(t, "size", None)
                if ir and sz and ir * sz > 0:
                    drs.append((float(t.pnl) - e) / (float(ir) * float(sz)))
            out["doubled_cost_avg_r"] = _finite_or_none(sum(drs) / len(drs)) if drs else None
        if bootstrap and len(trades) >= 10:
            try:
                from app.research.preregistered_tests import bootstrap_pvalue_positive
                days = [str(getattr(t, "entry_time", ""))[:10] for t in trades]
                out["bootstrap_p"] = _finite_or_none(bootstrap_pvalue_positive(pnls, days, n_boot=500))
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 -- diagnostics must never affect a run
        pass
    return out


# ---------------------------------------------------------------------------
# Read-back index (Search Lab / Evolution Lab skip signatures that already died repeatedly)
# ---------------------------------------------------------------------------
def readback_enabled() -> bool:
    """False when ``T58_DISABLE_GRAVEYARD_READBACK=1`` (the test suite sets it so repeated runs that append to the shared
    graveyard file cannot start skipping their own candidates)."""
    import os
    return os.environ.get("T58_DISABLE_GRAVEYARD_READBACK", "") not in ("1", "true", "True")


def dead_signature_counts(rows: list[dict]) -> dict:
    """{param_signature-or-family: n_rows}, built once per run so per-candidate lookups are O(1)
    (``is_known_dead_neighborhood`` rescans every row on every call)."""
    counts: dict = {}
    for r in rows or []:
        k = r.get("param_signature") or r.get("family")
        if k:
            counts[k] = counts.get(k, 0) + 1
    return counts


def is_dead_indexed(family: str, config: dict | None, counts: dict, min_attempts: int = 8,
                    keys: tuple = ()) -> tuple[bool, int]:
    """Same answer as ``is_known_dead_neighborhood`` using a precomputed ``dead_signature_counts`` index."""
    n = counts.get(param_signature(family, config, keys=keys), 0)
    return (n >= min_attempts, n)
