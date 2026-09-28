"""Regression tests for bugs found while reviewing the telemetry work. Each
test names the failure it guards against."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.orchestration import universe as un
from app.web.job_manager import JobManager

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "app" / "web" / "static"
TEMPLATES = ROOT / "app" / "web" / "templates"


# ---- chat widget must survive page-level CSS -----------------------------------

def test_widget_pins_the_properties_page_css_would_leak_into_it():
    """index.html & ~20 other pages define a bare `textarea { min-height:140px }`;
    without an explicit reset the chat input rendered 140px tall on those pages
    (min-height beats the widget's max-height)."""
    css = (TEMPLATES / "_project_chat_widget.html").read_text(encoding="utf-8")
    assert ".t58-pc-input, .t58-pc-project-select" in css and "min-height: 0" in css
    assert "width: auto" in css.split(".t58-pc-input {")[1].split("}")[0] or "width: auto" in css


# ---- job manager reapers ---------------------------------------------------------

def test_reaper_runs_before_listing_and_can_close_a_job():
    jm = JobManager()
    job = jm.create(tool="Evolution Lab")
    jm.add_reaper(lambda: jm.finish(job))
    assert jm.list_jobs()[0]["done"] is True


def test_a_broken_reaper_never_breaks_the_feed():
    jm = JobManager()
    jm.create()
    def boom(): raise RuntimeError("reaper bug")
    jm.add_reaper(boom)
    assert len(jm.list_jobs()) == 1


# ---- evolution job bookkeeping ---------------------------------------------------

@pytest.fixture
def server(monkeypatch):
    from app.web import server as s
    monkeypatch.setattr(s, "_EVOLUTION_RUNNER", None)
    monkeypatch.setattr(s, "_EVOLUTION_JOB_ID", None)
    monkeypatch.setattr(s, "_EVOLUTION_STARTED", False)
    return s


class _Runner:
    def __init__(self, running): self.is_running = running


def test_reaper_closes_the_entry_of_a_run_that_stopped_on_its_own(server):
    job = server.JOB_MANAGER.create(tool="Evolution Lab", progress_kind="evolution")
    server._EVOLUTION_RUNNER, server._EVOLUTION_JOB_ID, server._EVOLUTION_STARTED = _Runner(False), job, True
    server._reap_evolution_job()
    assert server.JOB_MANAGER.get(job)["done"] is True
    assert server.JOB_MANAGER.get_progress(job)["done"] is True


def test_reaper_leaves_a_running_run_alone(server):
    job = server.JOB_MANAGER.create(tool="Evolution Lab")
    server._EVOLUTION_RUNNER, server._EVOLUTION_JOB_ID, server._EVOLUTION_STARTED = _Runner(True), job, True
    server._reap_evolution_job()
    assert server.JOB_MANAGER.get(job)["done"] is False


def test_reaper_does_not_close_a_brand_new_job_whose_runner_has_not_started(server):
    """Between registering the job and runner.start(), the (old) runner reads as
    'not running'; a poll in that window used to close the NEW job."""
    job = server.JOB_MANAGER.create(tool="Evolution Lab")
    server._EVOLUTION_RUNNER, server._EVOLUTION_JOB_ID, server._EVOLUTION_STARTED = _Runner(False), job, False
    server._reap_evolution_job()
    assert server.JOB_MANAGER.get(job)["done"] is False


def test_finish_evolution_job_targets_the_id_it_was_given_and_is_idempotent(server):
    old = server.JOB_MANAGER.create(); new = server.JOB_MANAGER.create()
    server._EVOLUTION_JOB_ID = new
    server._finish_evolution_job(old)
    assert server.JOB_MANAGER.get(old)["done"] is True and server.JOB_MANAGER.get(new)["done"] is False
    server._finish_evolution_job(old); server._finish_evolution_job("unknown"); server._finish_evolution_job(None)
    server._finish_evolution_job()
    assert server.JOB_MANAGER.get(new)["done"] is True


def test_the_reaper_is_registered_on_the_shared_job_manager(server):
    assert server._reap_evolution_job in server.JOB_MANAGER._reapers


def test_evolution_log_feeds_the_tracker_without_growing_the_job_log(server):
    job = server.JOB_MANAGER.create(tool="Evolution Lab", progress_kind="evolution")
    server._EVOLUTION_JOB_ID = job
    server._evolution_log("===== GENERATION 4 =====")
    assert server.JOB_MANAGER.get_progress(job)["generation"] == 4
    assert server.JOB_MANAGER.get(job)["log"] == []


# ---- universe layout ---------------------------------------------------------------

def test_sixteen_populated_families_do_not_overlap():
    """The whole taxonomy is 16 groups; with the original fixed radius the
    neighbouring clusters on the ring collided."""
    families = [f"fam{i}" for i in range(16)]
    dots = [un.Dot(f"{f}-{i}", f, "ES", "5M", 1.0, 1.0, "tested", "search", score=float(i))
            for f in families for i in range(60)]
    out = un.layout_universe(dots)
    cl = out["clusters"]
    assert len(cl) == 16
    for i in range(16):
        for j in range(i + 1, 16):
            gap = math.dist((cl[i]["cx"], cl[i]["cy"]), (cl[j]["cx"], cl[j]["cy"]))
            assert gap > cl[i]["r"] + cl[j]["r"] - 1e-9, (cl[i]["family"], cl[j]["family"])
    assert all(-1.0 <= d["x"] <= 1.0 and -1.0 <= d["y"] <= 1.0 for d in out["dots"])


def test_a_search_db_deleted_between_glob_and_stat_is_skipped(tmp_path):
    """A dangling entry (what a prune mid-scan looks like) made sorted(key=stat) raise."""
    os.symlink(tmp_path / "gone.db", tmp_path / "search_dangling.db")
    assert un.load_search_dots(tmp_path) == []


# ---- telemetry.js polling behaviour, run under node with a fake DOM -----------------

NODE = shutil.which("node")

POLL_HARNESS = r"""
const vm = require('vm'), fs = require('fs');
function run(responses, key) {
  const timers = []; let fetches = 0;
  const el = (extra) => Object.assign({ hidden: true, children: [], innerHTML: '', textContent: '', className: '',
    classList: { toggle() {} }, appendChild() {}, addEventListener() {} }, extra);
  const text = el(), pips = el(), stats = el();
  const banner = el({
    getAttribute: (n) => ({ 'data-progress-url': '/p', 'data-progress-key': key || null })[n],
    querySelector: (sel) => ({ '[data-role=text]': text, '[data-role=pips]': pips, '[data-role=stats]': stats })[sel],
  });
  const document = { readyState: 'complete', addEventListener() {},
    querySelectorAll: (sel) => sel === '[data-t58-phase-banner]' ? [banner] : [] };
  const ctx = { document, console, setTimeout: (fn) => { timers.push(fn); return timers.length; }, clearTimeout() {},
    fetch: () => { const r = responses[Math.min(fetches, responses.length - 1)]; fetches++;
                   return Promise.resolve({ json: () => Promise.resolve(r) }); } };
  ctx.globalThis = ctx; vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(process.env.TELEMETRY_JS, 'utf8'), ctx);
  return (async () => {
    for (let i = 0; i < 6; i++) { await new Promise(r => setImmediate(r)); const t = timers.shift(); if (t) t(); }
    await new Promise(r => setImmediate(r));
    return { fetches, shown: !banner.hidden, text: text.textContent, stats: stats.innerHTML };
  })();
}
(async () => {
  const snap = { banner: 'Phase 2 · Breeding · 1,240 candidates · 38/s', phase_index: 2, done: false,
                 candidates_done: 1240, candidates_total: 2000, survivors: 7, elapsed_seconds: 65,
                 best_fitness: '<b>x</b>' };
  console.log(JSON.stringify({
    live: await run([{ found: true, done: false, progress: snap }]),
    done: await run([{ found: true, done: false, progress: { ...snap, done: true } }]),
    untracked: await run([{ found: true, done: false, progress: null }]),
    missing: await run([{ found: false }]),
    evolutionIdle: await run([{ started: false }], 'progress'),
    evolutionKey: await run([{ progress: { ...snap, done: true } }], 'progress'),
  }));
})();
"""


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_banner_polls_while_live_and_stops_when_done_missing_or_untracked():
    proc = subprocess.run([NODE, "-e", POLL_HARNESS], capture_output=True, text=True, timeout=30,
                          env={**os.environ, "TELEMETRY_JS": str(STATIC / "telemetry.js")})
    assert proc.returncode == 0, proc.stderr
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    assert r["live"]["fetches"] == 7 and r["live"]["shown"]                 # keeps polling a live run
    assert r["live"]["text"] == "Phase 2 · Breeding · 1,240 candidates · 38/s"
    assert "&lt;b&gt;x&lt;/b&gt;" in r["live"]["stats"] and "<b>x</b>" not in r["live"]["stats"]   # escaped
    assert r["done"]["fetches"] == 1 and r["done"]["shown"]                 # finished: render once, stop
    assert r["untracked"]["fetches"] == 1 and not r["untracked"]["shown"]   # never-tracked job: stop, stay hidden
    assert r["missing"]["fetches"] == 1                                     # server restarted: stop
    assert r["evolutionIdle"]["fetches"] > 1                                # no run yet: keep waiting for one
    assert r["evolutionKey"]["fetches"] == 1
