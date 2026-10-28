"""The searched-variant ledger: every variant any discovery run has ever tested, kept for good.

Why it exists
-------------
A Discovery run tests thousands of variants (8,000+ on "deep") but until now kept only the handful that died at the last
stages (the Strategy Graveyard's JSONL). Everything else was thrown away, and because the search used a fixed seed, the next
run walked the same space and landed on the same strategy again.

This ledger is the missing memory. It is a SQLite file (thousands to millions of rows is fine) with one row per
(scope, variant) where *scope* is the market being searched (e.g. "ES1!"):

  * ``is_known``          -> the next run skips variants already tested instead of re-testing them
  * ``top_performers``    -> the next run starts evolution from the best past variants (warm start) and breeds NEW ones from them
  * ``family_scores``     -> the next run spends more of its random budget on the strategy types that have produced the most
                            train-passing variants, and less on the ones that never did
  * ``lifetime_signal_sets`` -> the multiple-testing bar rises with every variant ever tried, so a lead found after 40,000 tries is
                            judged as harshly as it should be
  * ``summary`` / ``recent_runs`` / ``family_table`` -> what the Graveyard page shows ("37,412 variants searched over 6 runs")

Nothing here ever deletes a variant. A ledger row is a statement of fact ("this exact variant was tested on this market and
scored X"); it is not a verdict that the variant is bad on other data, so each scope (market) has its own memory.

Stages, lowest to highest (a row only ever moves up):  searched -> train_pass -> validated -> near_miss -> winner
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

STAGES = ("searched", "train_pass", "validated", "near_miss", "winner")
_RANK = {s: i for i, s in enumerate(STAGES)}
FLUSH_EVERY = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS variants (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    sig TEXT,
    source TEXT,
    family TEXT,
    bar_minutes INTEGER,
    spec TEXT,
    label TEXT,
    stage TEXT NOT NULL DEFAULT 'searched',
    stage_rank INTEGER NOT NULL DEFAULT 0,
    fit REAL,
    train_t REAL,
    train_n INTEGER,
    times_seen INTEGER NOT NULL DEFAULT 1,
    first_run INTEGER,
    last_run INTEGER,
    first_seen REAL,
    last_seen REAL,
    PRIMARY KEY (scope, key)
);
CREATE INDEX IF NOT EXISTS idx_variants_scope_fit ON variants(scope, fit);
CREATE INDEX IF NOT EXISTS idx_variants_family ON variants(scope, family);
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL,
    finished_at REAL,
    source TEXT,
    scope TEXT,
    effort TEXT,
    seed INTEGER,
    tested INTEGER DEFAULT 0,
    new_variants INTEGER DEFAULT 0,
    skipped_known INTEGER DEFAULT 0,
    warm_started INTEGER DEFAULT 0,
    winners INTEGER DEFAULT 0,
    near_misses INTEGER DEFAULT 0
);
"""


def stable_key(*parts) -> str:
    """16-hex-char key from canonical JSON of ``parts`` (Python's hash() is salted per process, so it cannot be used)."""
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def default_path() -> Path:
    from app.data.storage import get_app_base_dir
    d = get_app_base_dir() / "data" / "discovery"
    d.mkdir(parents=True, exist_ok=True)
    return d / "variant_ledger.sqlite"


class VariantLedger:
    """Thread-safe (one lock) and crash-safe (WAL; rows are buffered and flushed in batches, always in a ``finally``)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._con.row_factory = sqlite3.Row
        try:
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass
        self._con.executescript(_SCHEMA)
        self._con.commit()
        self._buf: dict[tuple, dict] = {}
        self._known: dict[str, set] = {}
        self.run_id: Optional[int] = None
        self.skipped_known = 0
        self.new_variants = 0
        self.warm_started = 0

    # ------------------------------------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        with self._lock:
            try:
                self.flush()
            finally:
                self._con.close()

    def run_count(self, scope: str | None = None) -> int:
        with self._lock:
            if scope:
                return int(self._con.execute("SELECT COUNT(*) FROM runs WHERE scope=?", (scope,)).fetchone()[0])
            return int(self._con.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    def begin_run(self, *, source: str, scope: str, effort: str = "", seed: int = 0) -> int:
        with self._lock:
            cur = self._con.execute("INSERT INTO runs(started_at, source, scope, effort, seed) VALUES (?,?,?,?,?)",
                                    (time.time(), source, scope, effort, int(seed)))
            self._con.commit()
            self.run_id = int(cur.lastrowid)
            self.skipped_known = self.new_variants = self.warm_started = 0
            return self.run_id

    def finish_run(self, *, tested: int = 0, winners: int = 0, near_misses: int = 0) -> None:
        with self._lock:
            self.flush()
            if self.run_id is None:
                return
            self._con.execute("UPDATE runs SET finished_at=?, tested=?, new_variants=?, skipped_known=?, warm_started=?, winners=?, near_misses=? WHERE run_id=?",
                              (time.time(), int(tested), self.new_variants, self.skipped_known, self.warm_started, int(winners), int(near_misses), self.run_id))
            self._con.commit()

    # ------------------------------------------------------------------------------------------------ known / record
    def _known_set(self, scope: str) -> set:
        s = self._known.get(scope)
        if s is None:
            s = {r[0] for r in self._con.execute("SELECT key FROM variants WHERE scope=?", (scope,))}
            self._known[scope] = s
        return s

    def is_known(self, scope: str, key: str) -> bool:
        """True when this exact variant was tested in an EARLIER run (a variant first tested in this run is not 'known'; that is
        the in-run ``seen`` set's job)."""
        with self._lock:
            return key in self._known_set(scope) and (scope, key) not in self._buf

    def count_known(self, scope: str) -> int:
        with self._lock:
            return len(self._known_set(scope))

    def record(self, scope: str, key: str, *, source: str, family: str = "", bar_minutes: int = 0, spec: dict | None = None,
               label: str = "", stage: str = "searched", fit: float | None = None, train_t: float | None = None,
               train_n: int | None = None, sig: str | None = None) -> None:
        with self._lock:
            k = (scope, key)
            row = self._buf.get(k)
            if row is None:
                row = {"scope": scope, "key": key, "sig": sig or key, "source": source, "family": family, "bar_minutes": int(bar_minutes or 0),
                       "spec": json.dumps(spec, sort_keys=True, default=str) if spec is not None else None, "label": label[:300],
                       "stage": "searched", "fit": None, "train_t": None, "train_n": None}
                self._buf[k] = row
            if _RANK.get(stage, 0) >= _RANK[row["stage"]]:
                row["stage"] = stage if stage in _RANK else "searched"
            for name, v in (("fit", fit), ("train_t", train_t), ("train_n", train_n)):
                if v is not None:
                    row[name] = v
            if len(self._buf) >= FLUSH_EVERY:
                self.flush()

    def mark(self, scope: str, key: str, stage: str) -> None:
        """Promote an already-recorded variant to a later stage."""
        with self._lock:
            row = self._buf.get((scope, key))
            if row is not None:
                if _RANK.get(stage, 0) > _RANK[row["stage"]]:
                    row["stage"] = stage
                return
            self._con.execute("UPDATE variants SET stage=?, stage_rank=? WHERE scope=? AND key=? AND stage_rank<?",
                              (stage, _RANK.get(stage, 0), scope, key, _RANK.get(stage, 0)))

    def flush(self) -> None:
        with self._lock:
            if not self._buf:
                self._con.commit()
                return
            now, rid = time.time(), self.run_id
            for (scope, key), r in self._buf.items():
                known = key in self._known_set(scope)
                rank = _RANK.get(r["stage"], 0)
                if known:
                    self._con.execute(
                        "UPDATE variants SET times_seen=times_seen+1, last_run=?, last_seen=?, "
                        "stage=CASE WHEN ?>stage_rank THEN ? ELSE stage END, stage_rank=MAX(stage_rank, ?), "
                        "fit=CASE WHEN ? IS NOT NULL AND (fit IS NULL OR ?>fit) THEN ? ELSE fit END, "
                        "train_t=COALESCE(?, train_t), train_n=COALESCE(?, train_n) WHERE scope=? AND key=?",
                        (rid, now, rank, r["stage"], rank, r["fit"], r["fit"], r["fit"], r["train_t"], r["train_n"], scope, key))
                else:
                    self._con.execute(
                        "INSERT INTO variants(scope,key,sig,source,family,bar_minutes,spec,label,stage,stage_rank,fit,train_t,train_n,times_seen,"
                        "first_run,last_run,first_seen,last_seen) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?,?,?)",
                        (scope, key, r["sig"], r["source"], r["family"], r["bar_minutes"], r["spec"], r["label"], r["stage"], rank,
                         r["fit"], r["train_t"], r["train_n"], rid, rid, now, now))
                    self._known_set(scope).add(key)
                    self.new_variants += 1
            self._buf.clear()
            self._con.commit()

    # ------------------------------------------------------------------------------------------------ learning
    def top_performers(self, scope: str, *, bar_minutes: int | None = None, families: Iterable[str] | None = None, limit: int = 24) -> list[dict]:
        """Best past variants (by train fitness) with their stored spec, for warm-starting the next search."""
        with self._lock:
            self.flush()
            q = "SELECT * FROM variants WHERE scope=? AND fit IS NOT NULL AND spec IS NOT NULL"
            args: list = [scope]
            if bar_minutes:
                q += " AND bar_minutes=?"
                args.append(int(bar_minutes))
            fams = list(families) if families else None
            if fams:
                q += " AND family IN (%s)" % ",".join("?" * len(fams))
                args += fams
            q += " ORDER BY stage_rank DESC, fit DESC LIMIT ?"
            args.append(int(limit))
            out = []
            for r in self._con.execute(q, args):
                d = dict(r)
                try:
                    d["spec"] = json.loads(d["spec"])
                except Exception:  # noqa: BLE001
                    continue
                out.append(d)
            return out

    def family_scores(self, scope: str, bar_minutes: int | None = None) -> dict:
        """{family: {"tested", "train_pass", "rate", "best_fit"}} from everything tried so far on this market."""
        with self._lock:
            self.flush()
            q = ("SELECT family, COUNT(*) n, SUM(CASE WHEN stage_rank>=1 THEN 1 ELSE 0 END) p, MAX(fit) b FROM variants "
                 "WHERE scope=? AND family<>''")
            args: list = [scope]
            if bar_minutes:
                q += " AND bar_minutes=?"
                args.append(int(bar_minutes))
            q += " GROUP BY family"
            return {r["family"]: {"tested": int(r["n"]), "train_pass": int(r["p"] or 0), "rate": (r["p"] or 0) / max(1, r["n"]), "best_fit": r["b"]}
                    for r in self._con.execute(q, args)}

    def family_weights(self, scope: str, families: list, bar_minutes: int | None = None, min_tested: int = 60) -> dict:
        """Integer sampling weights 1..4. A family needs ``min_tested`` past tries before its record can move its weight; below
        that it stays at 1 so unexplored families are never starved. Weights are capped so the search still explores."""
        sc = self.family_scores(scope, bar_minutes)
        rates = [v["rate"] for f, v in sc.items() if f in families and v["tested"] >= min_tested]
        mean = (sum(rates) / len(rates)) if rates else 0.0
        out = {}
        for f in families:
            v = sc.get(f)
            if not v or v["tested"] < min_tested or mean <= 0:
                out[f] = 1
            else:
                out[f] = int(max(1, min(4, round(v["rate"] / mean * 1.5))))
        return out

    def lifetime_signal_sets(self, scope: str) -> int:
        """Distinct signal sets (exit variants collapsed) ever tried on this market -- the multiple-testing count."""
        with self._lock:
            self.flush()
            return int(self._con.execute("SELECT COUNT(DISTINCT sig) FROM variants WHERE scope=?", (scope,)).fetchone()[0])

    # ------------------------------------------------------------------------------------------------ reporting
    def summary(self, scope: str | None = None) -> dict:
        with self._lock:
            self.flush()
            w, a = ("WHERE scope=?", (scope,)) if scope else ("", ())
            r = self._con.execute(f"SELECT COUNT(*) n, SUM(times_seen) seen, SUM(CASE WHEN stage_rank>=1 THEN 1 ELSE 0 END) p, "
                                  f"SUM(CASE WHEN stage_rank>=2 THEN 1 ELSE 0 END) v, SUM(CASE WHEN stage='near_miss' THEN 1 ELSE 0 END) nm, "
                                  f"SUM(CASE WHEN stage='winner' THEN 1 ELSE 0 END) win FROM variants {w}", a).fetchone()
            runs = int(self._con.execute(f"SELECT COUNT(*) FROM runs {w}", a).fetchone()[0])
            return {"variants": int(r["n"] or 0), "evaluations": int(r["seen"] or 0), "train_pass": int(r["p"] or 0), "validated": int(r["v"] or 0),
                    "near_misses": int(r["nm"] or 0), "winners": int(r["win"] or 0), "runs": runs}

    def scopes(self) -> list[dict]:
        with self._lock:
            self.flush()
            return [dict(r) for r in self._con.execute(
                "SELECT scope, COUNT(*) variants, SUM(CASE WHEN stage_rank>=1 THEN 1 ELSE 0 END) train_pass, "
                "SUM(CASE WHEN stage='winner' THEN 1 ELSE 0 END) winners FROM variants GROUP BY scope ORDER BY variants DESC")]

    def family_table(self, scope: str | None = None, limit: int = 40) -> list[dict]:
        with self._lock:
            self.flush()
            w, a = ("WHERE scope=?", (scope,)) if scope else ("", ())
            return [dict(r) for r in self._con.execute(
                f"SELECT family, COUNT(*) tested, SUM(CASE WHEN stage_rank>=1 THEN 1 ELSE 0 END) train_pass, "
                f"SUM(CASE WHEN stage_rank>=2 THEN 1 ELSE 0 END) validated, MAX(fit) best_fit FROM variants {w} "
                f"GROUP BY family ORDER BY tested DESC LIMIT ?", (*a, int(limit)))]

    def recent_runs(self, limit: int = 12) -> list[dict]:
        with self._lock:
            self.flush()
            return [dict(r) for r in self._con.execute("SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", (int(limit),))]

    def top_table(self, scope: str | None = None, limit: int = 15) -> list[dict]:
        with self._lock:
            self.flush()
            w, a = ("AND scope=?", (scope,)) if scope else ("", ())
            return [dict(r) for r in self._con.execute(
                f"SELECT scope, family, label, stage, fit, train_t, train_n, bar_minutes, times_seen FROM variants WHERE fit IS NOT NULL {w} "
                f"ORDER BY stage_rank DESC, fit DESC LIMIT ?", (*a, int(limit)))]

    def compact(self, scope: str | None = None, keep_unscored_per_scope: int = 250_000) -> int:
        """Housekeeping that never touches a scored/promoted variant: if one market's un-scored rows exceed the cap, the OLDEST
        un-scored ones lose their spec/label text (the key stays so they are still never re-tested). Returns rows trimmed."""
        with self._lock:
            self.flush()
            n = 0
            for sc in ([scope] if scope else [r["scope"] for r in self.scopes()]):
                cnt = self._con.execute("SELECT COUNT(*) FROM variants WHERE scope=? AND stage_rank=0 AND spec IS NOT NULL", (sc,)).fetchone()[0]
                if cnt > keep_unscored_per_scope:
                    cur = self._con.execute(
                        "UPDATE variants SET spec=NULL, label=NULL WHERE rowid IN (SELECT rowid FROM variants WHERE scope=? AND stage_rank=0 "
                        "AND spec IS NOT NULL ORDER BY first_seen ASC LIMIT ?)", (sc, cnt - keep_unscored_per_scope))
                    n += cur.rowcount
            self._con.commit()
            return n


_LEDGER: VariantLedger | None = None
_LEDGER_LOCK = threading.Lock()


def get_ledger() -> VariantLedger:
    """Process-wide ledger on the app's data folder (one SQLite connection, shared by every discovery run)."""
    global _LEDGER
    with _LEDGER_LOCK:
        if _LEDGER is None or not Path(_LEDGER.path).exists():
            _LEDGER = VariantLedger()
        return _LEDGER
