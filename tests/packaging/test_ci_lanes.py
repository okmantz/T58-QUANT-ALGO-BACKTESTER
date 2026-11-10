"""The fast lane and the nightly shards: membership, balance, and that no test is lost or run twice."""
from __future__ import annotations

import pytest

from tests import ci_lanes


def test_fast_lane_membership_is_by_folder_or_single_file():
    prefixes = ["tests/backtest/", "tests/test_zones.py"]
    assert ci_lanes.is_fast("tests/backtest/test_engine.py::test_a", prefixes)
    assert ci_lanes.is_fast("tests/test_zones.py::test_z[param]", prefixes)
    assert not ci_lanes.is_fast("tests/backtest_extra/test_x.py::test_a", prefixes)   # a folder prefix must end at the slash
    assert not ci_lanes.is_fast("tests/web/test_web.py::test_a", prefixes)


def test_the_shipped_fast_lane_lists_real_paths():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    prefixes = ci_lanes.read_prefixes()
    assert prefixes, "tests/fast_lane.txt must list at least one prefix"
    for prefix in prefixes:
        assert (root / prefix).exists(), f"fast_lane.txt points at a path that does not exist: {prefix}"


@pytest.mark.parametrize("spec,expected", [(None, None), ("", None), ("1/1", (1, 1)), ("3/4", (3, 4))])
def test_parse_shard(spec, expected):
    assert ci_lanes.parse_shard(spec) == expected


@pytest.mark.parametrize("bad", ["4", "0/4", "5/4", "a/b", "1/0", "-1/3"])
def test_parse_shard_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        ci_lanes.parse_shard(bad)


def test_weights_count_slow_tests_heavier():
    ids = ["tests/a.py::t1", "tests/a.py::t2", "tests/b.py::slow1", "tests/b.py::slow2[x]"]
    slow = {"tests/b.py::slow1", "tests/b.py::slow2"}
    w = ci_lanes.file_weights(ids, slow)
    assert w["tests/a.py"] == 1
    assert w["tests/b.py"] == 1 + 2 * ci_lanes.SLOW_WEIGHT


def test_every_file_lands_in_exactly_one_shard_and_the_load_is_balanced():
    weights = {f"tests/f{i}.py": (60 if i % 17 == 0 else 1 + (i * 7) % 9) for i in range(120)}
    total = 4
    assignment = ci_lanes.assign_shards(weights, total)
    assert set(assignment) == set(weights)                       # nothing lost
    assert set(assignment.values()) <= set(range(1, total + 1))  # nothing outside 1..N
    loads = [sum(w for f, w in weights.items() if assignment[f] == s) for s in range(1, total + 1)]
    assert max(loads) - min(loads) <= max(weights.values())      # LPT guarantee: spread never exceeds the biggest item


def test_assignment_is_deterministic():
    weights = {f"tests/f{i}.py": 1 + i % 5 for i in range(40)}
    assert ci_lanes.assign_shards(weights, 3) == ci_lanes.assign_shards(dict(reversed(list(weights.items()))), 3)
