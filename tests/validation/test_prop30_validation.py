import numpy as np
import pandas as pd
import pytest

from app.prop.presets import get_preset
from app.prop.thirty_day import TradeLedger
from app.validation.prop30_validation import (ForwardRegistry, ObservedRange, Partitions, independence_audit, validate_prop30,
                                              walk_forward_spans)


def _led(pnls, start, every=1):
    d = pd.bdate_range(start, periods=len(pnls) * every)[::every]
    et = np.array([x + pd.Timedelta(hours=15) for x in d], dtype="datetime64[ns]")
    p = np.asarray(pnls, float)
    z = np.zeros(len(p))
    return TradeLedger(et, et + np.timedelta64(1, "h"), p, np.minimum(p, 0), np.maximum(p, 0), z + 5, z + 500, np.ones(len(p), int),
                       np.array(["s"] * len(p), dtype=object), np.array(["MNQ"] * len(p), dtype=object))


R = lambda: get_preset("lucid_50k").to_prop_rules()  # noqa: E731


def test_holdout_overlapping_observed_range_is_not_independent():
    parts = Partitions(("2022-01-01", "2023-01-01"), ("2023-01-01", "2024-06-01"), ("2024-09-01", "2025-06-01"))
    a = independence_audit(parts, [ObservedRange("2025-01-01", "2025-03-01", "prior report holdout")])
    assert not a.holdout_independent and any("already observed" in p for p in a.problems)


def test_embargo_and_ordering_checks():
    parts = Partitions(("2022-01-01", "2023-01-01"), ("2023-01-01", "2024-06-01"), ("2024-06-10", "2025-06-01"))
    assert not independence_audit(parts).holdout_independent
    ok = Partitions(("2022-01-01", "2023-01-01"), ("2023-01-01", "2024-06-01"), ("2024-09-01", "2025-06-01"))
    assert independence_audit(ok).holdout_independent
    assert not independence_audit(Partitions(("2022-01-01", "2023-01-01"), ("2023-01-01", "2024-01-01"), None)).holdout_independent


def test_no_holdout_means_not_established_even_for_a_perfect_ledger():
    led = _led([400.0] * 700, "2022-01-03")
    parts = Partitions(("2022-01-03", "2023-01-02"), ("2023-01-02", "2024-06-03"), None)
    rep = validate_prop30(led, R(), parts)
    assert rep.verdict == "NOT_ESTABLISHED" and any("independent holdout" in r for r in rep.reasons)


def test_independent_holdout_passing_still_needs_forward(tmp_path):
    led = _led([400.0] * 1000, "2022-01-03")
    parts = Partitions(("2022-01-03", "2023-01-02"), ("2023-01-02", "2024-06-03"), ("2024-09-02", "2025-12-01"))
    rep = validate_prop30(led, R(), parts, forward=ForwardRegistry(tmp_path / "f.json"), config_id="c")
    assert rep.verdict in ("ESTABLISHED_ON_HOLDOUT_PENDING_FORWARD", "NOT_ESTABLISHED")
    assert rep.verdict != "CONFIRMED"


def test_losing_ledger_is_not_established():
    led = _led([-200.0] * 700, "2022-01-03")
    parts = Partitions(("2022-01-03", "2023-01-02"), ("2023-01-02", "2024-06-03"), ("2024-09-02", "2025-12-01"))
    assert validate_prop30(led, R(), parts).verdict == "NOT_ESTABLISHED"


def test_forward_registry_summary_and_validation(tmp_path):
    f = ForwardRegistry(tmp_path / "f.json")
    for i in range(8):
        f.record("c", f"2026-0{i + 1}-01", "passed", 20.0)
    s = f.summary("c")
    assert s["n_resolved"] == 8 and s["p_pass"] == 1.0 and s["lower"] < 1.0
    with pytest.raises(ValueError):
        f.record("c", "2026-01-01", "maybe")


def test_walk_forward_spans_cover_range_without_overlap():
    sp = walk_forward_spans("2023-01-01", "2024-01-01", test_days=120)
    assert len(sp) == 3 and all(sp[i][1] == sp[i + 1][0] for i in range(len(sp) - 1))
