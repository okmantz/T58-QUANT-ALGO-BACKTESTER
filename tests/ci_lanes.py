"""CI test lanes: which tests are the FAST lane, and how the full regression is split into shards.

* Fast lane  (``pytest -m "fast and not slow"``): unit + parity tests. Gates every pull request and every build; target < 3 minutes
  on a 4-core runner. Membership is the path prefix list in ``tests/fast_lane.txt`` (one prefix per line), so adding a lane member is
  a one-line change and never means editing 400 test files.
* Full regression (``pytest --shard i/N``): everything, split into N file-level shards that run in parallel on separate runners
  (nightly, on version tags, on demand). Shards are balanced by weight: a file's weight is 1 plus 8 per test listed in
  ``tests/slow_tests.txt``, so the shards finish at about the same time instead of one shard owning all the slow files.

Pure functions only (no pytest import) so ``tests/packaging/test_ci_lanes.py`` can exercise them directly.
"""
from __future__ import annotations

from pathlib import Path

SLOW_WEIGHT = 8
_HERE = Path(__file__).resolve().parent


def read_prefixes(path: Path | None = None) -> list[str]:
    path = path or (_HERE / "fast_lane.txt")
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def is_fast(nodeid: str, prefixes: list[str]) -> bool:
    """True when the test's file (or folder) is on the fast-lane list. ``prefixes`` entries are repo-relative paths such as
    ``tests/backtest/`` (a folder) or ``tests/test_vector_account_parity.py`` (one file)."""
    path = nodeid.split("::", 1)[0].replace("\\", "/")
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in prefixes)


def parse_shard(spec: str | None) -> tuple[int, int] | None:
    """'2/4' -> (2, 4). None / '' -> None (no sharding). Raises ValueError on anything that is not INDEX/TOTAL with 1 <= INDEX <= TOTAL."""
    if not spec:
        return None
    try:
        index_s, total_s = spec.split("/", 1)
        index, total = int(index_s), int(total_s)
    except ValueError as exc:
        raise ValueError(f"--shard must look like INDEX/TOTAL (for example 2/4), got {spec!r}") from exc
    if total < 1 or not 1 <= index <= total:
        raise ValueError(f"--shard INDEX must be between 1 and TOTAL, got {spec!r}")
    return index, total


def file_weights(nodeids: list[str], slow_ids: set[str]) -> dict[str, int]:
    """Weight per test FILE: 1 per collected test is too coarse (a file of 3 slow tests dwarfs a file of 30 tiny ones), so a file
    weighs 1 + SLOW_WEIGHT per slow test it contains."""
    weights: dict[str, int] = {}
    for nodeid in nodeids:
        file = nodeid.split("::", 1)[0]
        base = nodeid.split("[", 1)[0]
        weights[file] = weights.get(file, 0) + (SLOW_WEIGHT if (nodeid in slow_ids or base in slow_ids) else 0)
    return {f: 1 + w for f, w in weights.items()}


def assign_shards(weights: dict[str, int], total: int) -> dict[str, int]:
    """Greedy longest-processing-time packing: heaviest file first into the currently lightest shard. Deterministic (ties broken by path),
    so every runner computes the identical assignment from the same collection. Returns file -> 1-based shard index."""
    loads = [0] * total
    out: dict[str, int] = {}
    for file, weight in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):
        lightest = min(range(total), key=lambda i: (loads[i], i))
        loads[lightest] += weight
        out[file] = lightest + 1
    return out
