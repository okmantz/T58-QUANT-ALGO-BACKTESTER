"""Append-only hash-chained trial log + noise ceiling (2026-11-06 audit)."""
import json

import pytest

from app.validation import trial_chain as tc
from app.validation import trial_ledger as tl


@pytest.fixture
def chain(tmp_path):
    return tmp_path / "trial_chain.jsonl"


def _fill(p, k=4):
    for i in range(k):
        tc.append("record", f"s{i % 2}", 10 + i, path=p, now=1000.0 + i)


def test_chain_verifies_and_counts(chain):
    _fill(chain)
    st = tc.verify(chain)
    assert st.ok and st.n_entries == 4 and st.head_seq == 4
    assert tc.summary(chain)["raw_total"] == 10 + 11 + 12 + 13
    assert tc.total("s0", path=chain) == 10 + 12


def test_edit_detected(chain):
    _fill(chain)
    chain.write_text(chain.read_text().replace('"n":11', '"n":1'))
    st = tc.verify(chain)
    assert not st.ok and st.first_bad_seq == 2


def test_deleted_middle_line_detected(chain):
    _fill(chain)
    lines = chain.read_text().splitlines()
    chain.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    assert not tc.verify(chain).ok


def test_truncated_tail_detected_by_head_anchor(chain):
    _fill(chain)
    lines = chain.read_text().splitlines()
    chain.write_text("\n".join(lines[:-1]) + "\n")
    st = tc.verify(chain)
    assert not st.ok and "truncated" in st.reason
    with pytest.raises(tc.TrialChainError):
        tc.append("record", "x", 1, path=chain)


def test_restart_archives_and_links(chain):
    _fill(chain)
    chain.write_text(chain.read_text().replace('"n":11', '"n":1'))
    e = tc.restart("test", path=chain)
    assert tc.verify(chain).ok
    assert e["meta"]["old_ok"] is False
    assert list(chain.parent.glob("trial_chain.archived-*.jsonl"))


def test_off_switch(chain, monkeypatch):
    monkeypatch.setenv("T58_TRIAL_CHAIN", "off")
    assert tc.append("record", "a", 5, path=chain) is None and not chain.exists()


def test_ledger_mirrors_into_chain_and_book_run_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_PATH", tmp_path / "trial_ledger.json")
    tl.record("A", 5)
    tl.book_run("r1", {"x": 100})
    tl.book_run("r1", {"x": 100})          # retried run: must not double count in the log either
    tl.mark_clock_epoch("e1", "note")
    cp = tmp_path / "trial_chain.jsonl"
    assert tc.verify(cp).ok
    assert tc.summary(cp)["raw_total"] == 105
    assert tc.verify().ok                   # default path follows the redirected ledger


def test_chain_does_not_change_counter_totals(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_PATH", tmp_path / "trial_ledger.json")
    tl.record("A", 7)
    assert tl.raw_total("A") == 7


def test_corrupt_chain_blocks_booking_as_ledger_error(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_PATH", tmp_path / "trial_ledger.json")
    tl.record("A", 1)
    cp = tmp_path / "trial_chain.jsonl"
    cp.write_text(cp.read_text().replace('"n":1', '"n":9'))
    with pytest.raises(tl.TrialLedgerError):
        tl.record("A", 1)


def test_note_looks_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_PATH", tmp_path / "trial_ledger.json")
    (tmp_path / "trial_chain.jsonl").write_text("garbage\n")
    assert tc.note_looks("backtest", "single_backtest", 1) is False


def test_noise_ceiling_grows_with_looks():
    a, b = tc.judge(3.0, 10), tc.judge(3.0, 1_000_000)
    assert a["noise_ceiling_t"] < b["noise_ceiling_t"]
    assert a["clears_noise_ceiling"] is True and b["clears_noise_ceiling"] is False
    assert tc.judge(None, 5)["clears_noise_ceiling"] is None
