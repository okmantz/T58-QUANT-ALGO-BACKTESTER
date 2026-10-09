"""v9.11: Full Pipeline Stop must stop, Step 7 must not look hung, and
the desktop sidebar lists Strategy Library above User Manual.

Three Owen reports (2026-10-09), three regressions pinned here:

1. Stop during Full Pipeline's Step 2 GA did nothing until the whole
   search finished (191.7s measured pre-fix on a synthetic run: stop
   clicked at t=5s, terminal state at t=197s). The pipeline never
   passed its cancel_event into the GA, the GA's pool drained with
   as_completed() + shutdown(wait=True). Now the event reaches the GA,
   the pool polls it every second, workers are terminated on stop, and
   the pipeline translates the GA's cancel exception into its own.

2. A finished run "just kept loading": after the Step 7 attempt-replay
   print, the random-entry null runs 200 FULL engine passes over the
   dev set (~25 min at 377k bars) with no log line at all, then the
   report compiles silently. The verdict math is unchanged -- the fix
   is that every long leg of Step 7 narrates itself (progress every 25
   null runs, bracketing lines around cost stress / report compile),
   the job page shows a running clock + last-updated heartbeat instead
   of retrying failed polls silently, and the status endpoint's
   guidance fields can no longer take the whole payload down.

3. Desktop sidebar: "Strategy Library" now sits directly above
   "User Manual" (OVERVIEW group). Web has no sidebar Strategy Library
   entry (it lives on the Create page / dashboard button), so desktop
   is the only view with both labels to order.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.web.server import app  # noqa: E402

STRATEGY_PATH = ROOT / "strategies" / "manual" / "ES1! Momentum Continuation 5m.json"


def _csv_bytes(n: int = 6000, seed: int = 7) -> bytes:
    """ES-scale 5m bars with alternating drift so the momentum strategy
    actually trades (same shape as the manual repro harness)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-02", periods=n, freq="5min")
    price = 6000.0
    rows = []
    for i in range(n):
        drift = 1.1 if (i // 150) % 2 == 0 else -1.1
        step = drift + rng.normal(0, 2.0)
        o = price
        c = o + step
        rows.append((ts[i], o, max(o, c) + abs(rng.normal(0, 1.0)),
                     min(o, c) - abs(rng.normal(0, 1.0)), c, 1000.0))
        price = c
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def _start(client, **over):
    fields = {
        "strategy_mode": "manual",
        "strategy_code": STRATEGY_PATH.read_text(encoding="utf-8"),
        "initial_balance": "50000", "account_size": "50000",
        "risk_value": "1.0", "contract_size": "5", "commission": "4.2",
        "pip_size": "1", "spread_pips": "0.25", "slippage_pips": "0.25",
        "profit_target": "8", "daily_loss": "5", "max_dd": "10",
        "n_folds": "3", "ga_population": "6", "ga_generations": "2",
        "ga_search_mc_sims": "50", "final_mc_sims": "500", "baseline_mc_sims": "200",
    }
    fields.update({k: str(v) for k, v in over.items()})
    r = client.post(
        "/full-pipeline/start",
        data={"csv_file": (io.BytesIO(_csv_bytes()), "ES_5m_synth.csv"), **fields},
        content_type="multipart/form-data",
    )
    assert r.status_code == 302, r.get_data(as_text=True)[:500]
    return r.headers["Location"].rstrip("/").split("/")[-1]


def _status(client, job_id):
    r = client.get(f"/full-pipeline/job/{job_id}/status.json")
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    return r.get_json()


def _worker_children() -> list[str] | None:
    """This process's live multiprocessing worker children (spawn_main),
    or None when the platform can't answer (assertion is skipped then).
    The resource_tracker child is Python bookkeeping, not a worker."""
    try:
        out = subprocess.run(["ps", "-eo", "ppid=,args="], capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001
        return None
    me = str(os.getpid())
    return [ln.strip() for ln in out.splitlines()
            if ln.split()[:1] == [me] and "spawn_main" in ln]


def test_stop_during_step2_ga_reaches_terminal_state_fast():
    client = app.test_client()
    # Wide GA: cannot possibly finish before Stop lands.
    job_id = _start(client, ga_population=32, ga_generations=10, ga_search_mc_sims=200,
                    final_mc_sims=2000, baseline_mc_sims=500)

    deadline = time.time() + 180
    while time.time() < deadline:
        data = _status(client, job_id)
        if any("Step 2/7" in line for line in data["log"]):
            break
        assert not data["done"], f"job ended before Step 2: {data['error']!r}"
        time.sleep(0.5)
    else:
        raise AssertionError("Step 2 never appeared in the job log")

    t_stop = time.time()
    resp = client.post(f"/full-pipeline/job/{job_id}/stop")
    assert resp.status_code == 200 and resp.get_json()["found"] is True

    deadline = time.time() + 120
    data = None
    while time.time() < deadline:
        data = _status(client, job_id)
        if data["done"] or data["cancelled"] or data["error"]:
            break
        time.sleep(0.5)
    latency = time.time() - t_stop
    assert data is not None and (data["done"] or data["cancelled"]), "job never reached a terminal state after Stop"
    assert data["cancelled"] is True, f"expected a stopped job, got: {data!r}"
    assert data["error"] is None
    # Pre-fix this measured 191.7s (the full 10-generation search). The
    # generous bound still fails that behavior by 4x; expect ~1-5s.
    assert latency < 45, f"stop->terminal took {latency:.1f}s"
    # Partial log is preserved, including the step we stopped inside.
    assert any("Step 2/7" in line for line in data["log"])
    assert any("Stop requested" in line for line in data["log"])

    workers = _worker_children()
    if workers is not None:
        settle_deadline = time.time() + 20
        while workers and time.time() < settle_deadline:
            time.sleep(1)
            workers = _worker_children()
        assert not workers, f"orphan GA worker processes after stop: {workers}"


def test_pipeline_completes_status_payload_and_step7_progress(monkeypatch):
    # tests/conftest.py sets T58_SKIP_EXTRA_GATES=1 suite-wide (speed),
    # which skips the whole Step 7 evidence pass. Owen's real runs have
    # it ON (his log shows the attempt-replay block), and the narration
    # this test pins only exists on that path -- so turn it back on for
    # this one test.
    monkeypatch.setenv("T58_SKIP_EXTRA_GATES", "0")
    client = app.test_client()
    job_id = _start(client)

    deadline = time.time() + 600
    data = None
    while time.time() < deadline:
        data = _status(client, job_id)
        if data["done"] or data["error"]:
            break
        time.sleep(1.0)
    assert data is not None and data["done"], f"job did not finish: {data and data.get('error')!r}"
    assert data["error"] is None
    assert data["cancelled"] is False

    summary = data["summary"]
    assert summary is not None
    assert summary["verdict"] in {"READY", "MARGINAL", "NOT READY"}
    assert summary["report_html"]
    assert data["next_step"], "verdict guidance missing from the terminal payload"

    log_text = "\n".join(data["log"])
    # Step 7 narrates its long legs now (Issue 3): null progress lines
    # and the report-compile bracket must be in the finished log.
    assert "Random-entry null:" in log_text
    assert "runs done..." in log_text
    assert "Compiling the final report" in log_text
    assert "Full Pipeline complete" in log_text

    # The finished job page itself renders (200, verdict card mounts).
    r = client.get(f"/full-pipeline/job/{job_id}")
    assert r.status_code == 200


def test_random_entry_null_progress_callback_is_advisory_only():
    from app.backtest.risk import RiskConfig
    from app.research.director import random_entry_distribution

    rng = np.random.default_rng(3)
    n = 400
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, n))
    df = pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.3,
        "low": close - 0.3, "close": close, "volume": 100.0,
    })
    risk = RiskConfig(initial_balance=50000, pip_size=0.01, contract_size=1000.0)
    seen: list[tuple[int, int]] = []
    out = random_entry_distribution(
        df, risk, observed_value=0.0, n_entries=10, n_seeds=30,
        stop_loss_pips=50, take_profit_pips=100,
        progress_cb=lambda d, t: seen.append((d, t)),
    )
    assert seen == [(25, 30)]
    assert out["n_seeds"] == 30
    assert "p_value" in out


def test_library_and_history_caches_are_fresh_and_identical(tmp_path, monkeypatch):
    """Issue 5: page renders no longer re-scan/re-parse the library and
    run history per load -- but the caches must return EXACTLY what a
    fresh scan returns, and must notice saves/deletes immediately
    (fingerprint invalidation, never TTL)."""
    import json as _json

    from app.reports import run_history
    from app.strategy import library

    base = tmp_path / "appbase"
    (base / "strategies" / "manual").mkdir(parents=True)
    (base / "data").mkdir(parents=True)
    monkeypatch.setattr(library, "get_app_base_dir", lambda: base)
    monkeypatch.setattr(run_history, "get_app_base_dir", lambda: base)

    cfg = {"name": "Cache Probe", "entry_conditions": [], "exit_conditions": []}
    strat = base / "strategies" / "manual" / "Cache Probe.json"
    strat.write_text(_json.dumps(cfg), encoding="utf-8")

    first = library.list_saved_strategies()
    assert [s.name for s in first] == ["Cache Probe.json"]
    second = library.list_saved_strategies()
    assert [(s.name, s.metadata, s.size_bytes) for s in second] == \
           [(s.name, s.metadata, s.size_bytes) for s in first]

    # A save lands immediately (fingerprint changed), a delete too.
    strat2 = base / "strategies" / "manual" / "Cache Probe 2.json"
    strat2.write_text(_json.dumps(cfg), encoding="utf-8")
    assert {s.name for s in library.list_saved_strategies()} == {"Cache Probe.json", "Cache Probe 2.json"}
    strat2.unlink()
    assert [s.name for s in library.list_saved_strategies()] == ["Cache Probe.json"]

    # The web layer's serialized library follows the same invalidation.
    from app.web.server import _saved_strategies_json
    assert "Cache Probe.json" in _saved_strategies_json()
    strat.unlink()
    assert "Cache Probe.json" not in _saved_strategies_json()

    # Run history: a recorded run shows up on the very next load.
    assert run_history.load_runs() == []
    run_history.record_run({"strategy": {"name": "Cache Probe"}}, {"html": "x.html", "json": "x.json"})
    runs = run_history.load_runs()
    assert len(runs) == 1 and runs[0]["strategy_name"] == "Cache Probe"
    assert run_history.load_runs() == runs


def test_next_step_button_text_is_dark_on_every_lifecycle_page():
    """Issue 4: the Create page's 'Next step ->' is an <a class=lc-btn>,
    and .lc-scope a{color:inherit} outranks .lc-btn, so it rendered
    near-white text on the mint accent. The shared rule must name
    a.lc-btn explicitly so anchors and <button>s alike get the dark ink,
    while variant rules (ghost/danger) keep their own colors."""
    css = (ROOT / "app" / "web" / "templates" / "_lifecycle_css.html").read_text(encoding="utf-8")
    m = re.search(r"\.lc-btn,\s*a\.lc-btn\s*\{([^}]*)\}", css)
    assert m, "shared .lc-btn, a.lc-btn rule not found"
    assert "color:var(--lc-acc-ink)" in m.group(1)
    # The dark ink token itself: near-black, not a light color.
    assert "--lc-acc-ink:#03231a" in css
    # The Create page's next-step control is the anchor form of the class.
    create = (ROOT / "app" / "web" / "templates" / "section_start_here.html").read_text(encoding="utf-8")
    assert 'class="lc-btn big"' in create and "Next step" in create


def _nav_block() -> str:
    source = (ROOT / "app" / "ui" / "main_window.py").read_text(encoding="utf-8")
    start = source.index("self._nav_items = [")
    end = source.index("]\n", start)
    return source[start:end]


def _key_pos(block: str, key: str) -> int:
    m = re.search(rf'\(\s*"{re.escape(key)}"\s*,', block)
    assert m, f"nav key {key!r} not found in _nav_items"
    return m.start()


def test_desktop_nav_strategy_library_directly_above_user_manual():
    block = _nav_block()
    lib = _key_pos(block, "stratlibrary")
    manual = _key_pos(block, "manual")
    assert lib < manual, "Strategy Library must sit above User Manual"
    # Exactly one Strategy Library entry in the whole sidebar (the move
    # must not duplicate it), with its entry unchanged.
    assert block.count('"stratlibrary"') == 1
    assert '("stratlibrary", "", "Strategy Library", self.tab_stratlibrary, NEON_VIOLET)' in block
    assert '("manual", "", "User Manual", self.tab_manual' in block
    # Nothing between the two entries: Library is DIRECTLY above Manual.
    between = block[lib:manual]
    assert between.count('("') == 1, f"unexpected nav entries between Library and Manual: {between!r}"
