"""Append-only, hash-chained trial log + the multiple-testing "noise ceiling" (2026-11-06 audit, "global trial ledger").

``app.validation.trial_ledger`` is a COUNTER: it merges every booking made on the same UTC day into one number, rewrites the whole file on
every update, and ages entries out. That is the right shape for the Deflated-Sharpe penalty, and it stays exactly as it was. What it
cannot do is answer "how many times have I actually looked, and can I prove that history was not edited?". This module is the other half:

    * one JSON line per booking, never rewritten, never merged -- a real event log;
    * each line carries ``prev`` (the hash of the line before it) and ``hash`` = sha256(prev + canonical payload), so editing, deleting
      or reordering any past line breaks every hash after it and ``verify()`` says exactly where;
    * a small ``.head`` file anchors the latest (seq, hash) so quietly TRUNCATING the tail is caught too (a pure hash chain cannot
      see a deleted tail by itself);
    * a cross-process file lock, so the desktop app and the web app booking at the same moment cannot fork the chain.

It is written from the ledger's own choke points (``trial_ledger.record`` / ``book_run`` / ``mark_clock_epoch``), so every tab that already
books to the ledger feeds the chain with no per-tab wiring, and from the handful of searchers that never booked anything (Iterative
Refinement, Multi-Objective, Multi-Market, the plain backtest route). The chain is INFORMATIONAL for those: it does not change any trial
count the Deflated-Sharpe gate or any verdict reads.

The noise ceiling (``judge`` / ``noise_context``): after N independent looks at pure noise, the BEST t-statistic you would see is not 0, it
is ``expected_max_z(N)`` (about 2.8 after 200 looks, 4.9 after a million). A result is only worth taking seriously when its t clears the
ceiling for the number of looks that produced it. The ceiling is shown next to the t-statistic; it is a guard rail, not a gate.

Escape hatches: ``T58_TRIAL_CHAIN=off`` disables booking entirely; ``python -m app.validation.trial_chain --restart "reason"`` archives a
broken chain beside itself and starts a new one whose first line records the old head and the reason (the evidence is kept, the app unblocked).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

GENESIS = "0" * 64
_PATH = Path(__file__).resolve().parents[2] / "reports" / "trial_chain.jsonl"
_THREAD_LOCK = threading.RLock()
_TAIL_BYTES = 65536           # how much of the file's end is read to find the last line before falling back to a full read


class TrialChainError(RuntimeError):
    """The trial chain is corrupt, truncated, forked or unwritable. Booking aborts rather than extending a chain nobody can trust."""


def enabled() -> bool:
    return os.environ.get("T58_TRIAL_CHAIN", "on").strip().lower() not in ("off", "0", "false", "no")


def _resolve(path) -> Path:
    """Explicit path wins; otherwise the chain sits beside the counter ledger (so tests / tools that redirect ``trial_ledger._PATH`` redirect
    the chain with it), falling back to ``reports/trial_chain.jsonl``."""
    if path is not None:
        return Path(path)
    try:
        from app.validation import trial_ledger as _tl
        return Path(_tl._PATH).with_name("trial_chain.jsonl")      # noqa: SLF001
    except Exception:  # noqa: BLE001
        return _PATH


def _head_path(p: Path) -> Path:
    return p.with_name(p.name + ".head")


def _canon(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _digest(prev: str, payload: dict) -> str:
    return hashlib.sha256((prev + "\n" + _canon(payload)).encode("utf-8")).hexdigest()


def _payload_of(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k not in ("prev", "hash")}


_HELD = threading.local()      # per-thread re-entrancy depth: flock on a SECOND fd of the same file would deadlock against ourselves


@contextlib.contextmanager
def _file_lock(p: Path):
    """Thread lock + best-effort cross-process advisory lock on a sidecar file (fcntl on POSIX, msvcrt on Windows). Re-entrant per thread."""
    with _THREAD_LOCK:
        depth = getattr(_HELD, "depth", 0)
        if depth > 0:
            _HELD.depth = depth + 1
            try:
                yield
            finally:
                _HELD.depth -= 1
            return
        _HELD.depth = 1
        try:
            with _file_lock_outer(p):
                yield
        finally:
            _HELD.depth = 0


@contextlib.contextmanager
def _file_lock_outer(p: Path):
    if True:
        fh = None
        locker = None
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            fh = open(p.with_name(p.name + ".lock"), "a+b")
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                locker = ("fcntl", fcntl)
            except ImportError:
                try:
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    locker = ("msvcrt", msvcrt)
                except Exception:  # noqa: BLE001 -- thread lock alone still protects a single process
                    locker = None
            except Exception:  # noqa: BLE001
                locker = None
        except OSError:
            fh = None
        try:
            yield
        finally:
            if fh is not None:
                try:
                    if locker and locker[0] == "fcntl":
                        locker[1].flock(fh.fileno(), locker[1].LOCK_UN)
                    elif locker and locker[0] == "msvcrt":
                        fh.seek(0)
                        locker[1].locking(fh.fileno(), locker[1].LK_UNLCK, 1)
                except Exception:  # noqa: BLE001
                    pass
                fh.close()


def _last_line(p: Path) -> str | None:
    """The file's last non-empty line, reading only its tail when possible. None for a missing / empty file."""
    try:
        size = p.stat().st_size
    except FileNotFoundError:
        return None
    if size == 0:
        return None
    with open(p, "rb") as fh:
        chunk = _TAIL_BYTES
        while True:
            start = max(0, size - chunk)
            fh.seek(start)
            data = fh.read(size - start)
            lines = [ln for ln in data.split(b"\n") if ln.strip()]
            if len(lines) >= 2 or start == 0:
                return lines[-1].decode("utf-8", "replace") if lines else None
            chunk *= 4


def _parse(line: str, where: str) -> dict:
    try:
        e = json.loads(line)
    except ValueError as exc:
        raise TrialChainError(f"Trial chain {where}: a line is not valid JSON ({exc}).") from exc
    if not isinstance(e, dict) or "hash" not in e or "prev" not in e or "seq" not in e:
        raise TrialChainError(f"Trial chain {where}: a line is missing seq/prev/hash.")
    return e


def _check_entry_self(e: dict, where: str) -> None:
    if _digest(str(e["prev"]), _payload_of(e)) != e["hash"]:
        raise TrialChainError(f"Trial chain {where}: entry #{e.get('seq')} does not match its own hash -- it was edited after it was written.")


def append(kind: str, label, n: int = 1, *, fp: str | None = None, run_key: str | None = None, src: str | None = None,
           meta: dict | None = None, now: float | None = None, path=None) -> dict | None:
    """Append one booking. Returns the written entry, or None when booking is off / ``n`` is not positive (markers with kind 'epoch' or
    'restart' may carry n=0). Raises ``TrialChainError`` if the existing chain's tail is damaged or the write fails."""
    if not enabled():
        return None
    n = int(n)
    if n <= 0 and kind not in ("epoch", "restart"):
        return None
    p = _resolve(path)
    with _file_lock(p):
        last = _last_line(p)
        if last is None:
            prev, seq = GENESIS, 1
        else:
            le = _parse(last, str(p))
            _check_entry_self(le, str(p))
            prev, seq = le["hash"], int(le["seq"]) + 1
            hd = _read_head(p)
            if hd is not None and int(hd.get("seq", 0)) > int(le["seq"]):
                raise TrialChainError(f"Trial chain {p} ends at entry #{le['seq']} but its head file says #{hd['seq']}: the tail was truncated.")
        payload = {"seq": seq, "t": float(time.time() if now is None else now), "kind": str(kind), "label": str(label if label is not None else ""),
                   "n": n, "fp": str(fp) if fp else None, "run": str(run_key) if run_key else None, "src": str(src) if src else None,
                   "meta": dict(meta) if meta else {}}
        entry = dict(payload, prev=prev, hash=_digest(prev, payload))
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str) + "\n"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "ab") as fh:
                fh.write(line.encode("utf-8"))
                fh.flush()
                os.fsync(fh.fileno())
            _write_head(p, entry)
        except OSError as exc:
            raise TrialChainError(f"Trial chain {p} could not be written ({exc}); this booking was NOT recorded.") from exc
        return entry


def note_looks(kind: str, label, n: int, *, run_key: str | None = None, src: str | None = None, meta: dict | None = None) -> bool:
    """Best-effort, chain-ONLY booking for searchers that never booked to the counter ledger (single backtest route, Iterative Refinement,
    Multi-Objective, Multi-Market). INFORMATIONAL: never changes a trial count a verdict reads, and never raises into the caller's run --
    a damaged chain is reported by ``verify`` / the trial-ledger page instead. Returns True when a line was written."""
    try:
        return append(kind, label, n, run_key=run_key, src=src, meta=meta) is not None
    except Exception:  # noqa: BLE001
        return False


def _read_head(p: Path) -> dict | None:
    try:
        d = json.loads(_head_path(p).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _write_head(p: Path, entry: dict) -> None:
    try:
        _head_path(p).write_text(json.dumps({"seq": entry["seq"], "hash": entry["hash"], "t": entry["t"]}), encoding="utf-8")
    except OSError:
        pass        # the head file is an anchor, not the record: a failed anchor write never loses a booking


@dataclass
class ChainStatus:
    ok: bool
    n_entries: int = 0
    head_hash: str = GENESIS
    head_seq: int = 0
    first_bad_seq: int | None = None
    reason: str = ""
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["head_short"] = self.head_hash[:12]
        return d


def entries(path=None):
    """Yield every well-formed entry in order WITHOUT verifying it (use ``verify`` for that). Unreadable lines are skipped."""
    p = _resolve(path)
    try:
        fh = open(p, "rb")
    except FileNotFoundError:
        return
    with fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                e = json.loads(raw)
            except ValueError:
                continue
            if isinstance(e, dict) and "seq" in e and "n" in e:
                yield e


def verify(path=None) -> ChainStatus:
    """Walk the whole chain: valid JSON, seq 1..N with no gaps, every ``prev`` equal to the line before, every hash recomputed, and the
    head file (when present) matching the last line. Never raises: the verdict is in the returned status."""
    p = _resolve(path)
    if not p.exists():
        return ChainStatus(ok=True, reason="no chain yet (nothing has been booked)")
    prev, seq, n = GENESIS, 0, 0
    try:
        with open(p, "rb") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    e = json.loads(raw)
                except ValueError:
                    return ChainStatus(False, n, prev, seq, seq + 1, f"line after entry #{seq} is not valid JSON")
                if not isinstance(e, dict) or "hash" not in e or "prev" not in e or "seq" not in e:
                    return ChainStatus(False, n, prev, seq, seq + 1, f"line after entry #{seq} is missing seq/prev/hash")
                if int(e["seq"]) != seq + 1:
                    return ChainStatus(False, n, prev, seq, int(e["seq"]), f"sequence jumps from #{seq} to #{e['seq']} (an entry was removed or reordered)")
                if e["prev"] != prev:
                    return ChainStatus(False, n, prev, seq, int(e["seq"]), f"entry #{e['seq']} does not follow entry #{seq} (chain was spliced)")
                if _digest(str(e["prev"]), _payload_of(e)) != e["hash"]:
                    return ChainStatus(False, n, prev, seq, int(e["seq"]), f"entry #{e['seq']} was edited after it was written")
                prev, seq, n = e["hash"], int(e["seq"]), n + 1
    except OSError as exc:
        return ChainStatus(False, n, prev, seq, None, f"chain could not be read ({exc})")
    st = ChainStatus(True, n, prev, seq, None, "chain intact")
    hd = _read_head(p)
    if hd is not None:
        if int(hd.get("seq", 0)) > seq:
            return ChainStatus(False, n, prev, seq, seq + 1, f"chain ends at #{seq} but the head file says #{hd['seq']}: the tail was truncated")
        if int(hd.get("seq", 0)) == seq and hd.get("hash") != prev:
            return ChainStatus(False, n, prev, seq, seq, "last entry differs from the head file's anchor")
        if int(hd.get("seq", 0)) < seq:
            st.notes.append("head file was behind the log (an interrupted write); it is re-anchored on the next booking")
    elif n:
        st.notes.append("no head file: truncation of the tail cannot be detected until the next booking writes one")
    return st


# ------------------------------------------------------------------------------------------------------------------------------
# Totals
# ------------------------------------------------------------------------------------------------------------------------------
def summary(path=None) -> dict:
    """Raw (never aged, never merged) counts straight from the log."""
    by_kind: dict = {}
    by_label: dict = {}
    total = 0
    first_t = last_t = None
    n_events = 0
    for e in entries(path):
        n = int(e.get("n", 0))
        n_events += 1
        if n > 0:
            total += n
            by_kind[e.get("kind", "?")] = by_kind.get(e.get("kind", "?"), 0) + n
            by_label[e.get("label", "")] = by_label.get(e.get("label", ""), 0) + n
        t = e.get("t")
        if t:
            first_t = t if first_t is None else min(first_t, t)
            last_t = t if last_t is None else max(last_t, t)
    top = sorted(by_label.items(), key=lambda kv: -kv[1])[:20]
    return {"raw_total": int(total), "n_events": n_events, "by_kind": by_kind, "top_labels": [{"label": k, "n": v} for k, v in top],
            "first_t": first_t, "last_t": last_t}


def total(label=None, kinds=None, path=None) -> int:
    """Raw looks booked in the chain (optionally for one label and / or a set of kinds)."""
    kinds = set(kinds) if kinds else None
    out = 0
    for e in entries(path):
        if label is not None and str(e.get("label")) != str(label):
            continue
        if kinds is not None and e.get("kind") not in kinds:
            continue
        out += max(0, int(e.get("n", 0)))
    return out


def restart(reason: str, path=None) -> dict:
    """Archive a broken / abandoned chain beside itself (evidence kept) and start a new one whose first line records the old head."""
    p = _resolve(path)
    with _file_lock(p):
        old = verify(p)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if p.exists():
            p.rename(p.with_name(f"{p.stem}.archived-{stamp}{p.suffix}"))
        with contextlib.suppress(OSError):
            _head_path(p).unlink()
        return append("restart", "chain", 0, src="restart", path=p,
                      meta={"reason": str(reason), "old_ok": old.ok, "old_head": old.head_hash, "old_entries": old.n_entries, "old_problem": old.reason})


# ------------------------------------------------------------------------------------------------------------------------------
# Noise ceiling
# ------------------------------------------------------------------------------------------------------------------------------
def daily_t_stat(trades, min_days: int = 20) -> dict | None:
    """t-statistic of mean DAILY net P&L (trades grouped by exit date, so same-day trades are one observation). None when there are
    fewer than ``min_days`` trading days or no variation -- a t-statistic that thin is not evidence either way."""
    try:
        from app.validation.effective_trials import daily_pnl_series
        s = daily_pnl_series(trades)
    except Exception:  # noqa: BLE001
        return None
    vals = [float(x) for x in s.to_numpy() if x == x]
    n = len(vals)
    if n < max(2, int(min_days)):
        return None
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    if var <= 1e-12:
        return None
    return {"t": mean / math.sqrt(var / n), "days": n, "mean_daily_pnl": mean, "sd_daily_pnl": math.sqrt(var)}


def judge(t: float | None, n_looks: int) -> dict:
    """Is ``t`` bigger than what pure noise reaches after ``n_looks`` independent looks? (normal approximation; t at 20+ days is close)."""
    from app.validation.rulebook import expected_max_z, z_bar
    n = max(1, int(n_looks))
    ceiling = float(expected_max_z(n))
    bonf = float(z_bar(n))
    out = {"n_looks": n, "noise_ceiling_t": ceiling, "bonferroni_t": bonf, "t": None if t is None else float(t)}
    if t is None:
        out.update(clears_noise_ceiling=None, clears_bonferroni=None, margin_over_noise=None)
    else:
        out.update(clears_noise_ceiling=bool(t > ceiling), clears_bonferroni=bool(t >= bonf), margin_over_noise=float(t - ceiling))
    return out


def noise_context(trades=None, *, t: float | None = None, label=None, counted_looks: int | None = None, path=None) -> dict:
    """The t-statistic judged against the noise ceiling at the numbers of looks that matter, in words a trader can act on.

    ``raw_looks`` = every look the chain ever recorded (never fades, over-counts near-duplicate candidates); ``counted_looks`` = what the
    verdict's aged ledger uses (pass it in when you have it); ``label_looks`` = looks recorded under this strategy's own label.
    INFORMATIONAL: no verdict reads this."""
    td = None
    if t is None and trades is not None:
        td = daily_t_stat(trades)
        t = None if td is None else td["t"]
    raw = int(summary(path)["raw_total"])
    judged = {"raw": judge(t, max(1, raw))}
    if counted_looks is not None:
        judged["counted"] = judge(t, max(1, int(counted_looks)))
    if label is not None:
        ll = total(label, path=path)
        if ll > 0:
            judged["label"] = judge(t, ll)
    if t is None:
        text = "Not enough trading days to compute a daily t-statistic, so there is nothing to hold against the noise ceiling."
    else:
        r = judged["raw"]
        text = (f"Daily t-statistic {t:.2f}. After {r['n_looks']:,} recorded looks pure noise alone reaches about t = {r['noise_ceiling_t']:.2f}; "
                + ("this result clears that ceiling." if r["clears_noise_ceiling"] else "this result does NOT clear that ceiling, so it is not distinguishable from the best of that many lucky looks."))
        if "counted" in judged:
            c = judged["counted"]
            text += (f" At the {c['n_looks']:,} looks the verdict counts (older ones fade) the ceiling is {c['noise_ceiling_t']:.2f}: "
                     + ("clears it." if c["clears_noise_ceiling"] else "does not clear it."))
    return {"t": t, "days": None if td is None else td["days"], "judged": judged, "text": text}


def _cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m app.validation.trial_chain", description="Inspect or restart the trial chain.")
    ap.add_argument("--verify", action="store_true", help="verify the chain (default action)")
    ap.add_argument("--restart", metavar="REASON", help="archive the current chain beside itself and start a new one")
    a = ap.parse_args(argv)
    if a.restart:
        e = restart(a.restart)
        print(f"chain restarted; archived the old file; new head {e['hash'][:12]}")
        return 0
    st = verify()
    print(json.dumps(st.to_dict(), indent=1))
    s = summary()
    print(f"{s['raw_total']:,} looks in {s['n_events']:,} events; by kind: {s['by_kind']}")
    return 0 if st.ok else 1


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(_cli())
