"""Runs telemetry.js's pure helpers under node (no browser). Skipped when node
isn't installed. Also a syntax check of every front-end script we ship."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "app" / "web" / "static"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

HARNESS = r"""
const vm = require('vm'), fs = require('fs');
const src = fs.readFileSync(process.env.TELEMETRY_JS, 'utf8');
const ctx = { globalThis: null, console };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(src, ctx);          // no `window`/`document`: must not throw or auto-boot
const T = ctx.T58Telemetry;
const out = {
  esc: T.esc('<img src=x onerror="a()">&\'x'),
  escNull: T.esc(null),
  money: [T.formatMoney(1234.5), T.formatMoney(-50), T.formatMoney(null), T.formatMoney(NaN)],
  pct: [T.formatPct(12.34), T.formatPct(null), T.formatPct(5, 0)],
  elapsed: [T.formatElapsed(0), T.formatElapsed(59), T.formatElapsed(61), T.formatElapsed(3725), T.formatElapsed(null)],
  clip: T.clipRange([[0, 1, 2, 3], [-1, 0, 500]]),
  clipEmpty: T.clipRange([]),
  clipFlat: T.clipRange([[5, 5, 5]]),
  clipHuge: T.clipRange([Array.from({length: 200}, (_, i) => i / 10).concat([1e9])]),
  near: [T.nearestDot([{px: 10, py: 10}, {px: 50, py: 50}], 12, 11, 9),
         T.nearestDot([{px: 10, py: 10}], 100, 100, 9),
         T.nearestDot([], 1, 1, 9)],
};
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def result():
    proc = subprocess.run([NODE, "-e", HARNESS], capture_output=True, text=True, timeout=30,
                          env={**os.environ, "TELEMETRY_JS": str(STATIC / "telemetry.js")})
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_html_escaping_neutralises_markup(result):
    assert "<" not in result["esc"] and ">" not in result["esc"] and '"' not in result["esc"]
    assert result["esc"].startswith("&lt;img") and result["escNull"] == ""


def test_formatters(result):
    assert result["money"] == ["$1,235", "-$50", "n/a", "n/a"] or result["money"][0] in ("$1,234", "$1,235")
    assert result["money"][1:] == ["-$50", "n/a", "n/a"]
    assert result["pct"] == ["12.3%", "n/a", "5%"]
    assert result["elapsed"] == ["0s", "59s", "1m 01s", "62m 05s", "0s"]


def test_clip_range_always_includes_zero_and_ignores_outliers(result):
    lo, hi = result["clip"]["lo"], result["clip"]["hi"]
    assert lo < -1 <= 0 < hi
    assert result["clipEmpty"] == {"lo": -1, "hi": 1}
    assert result["clipFlat"]["lo"] < 0 < result["clipFlat"]["hi"]
    assert result["clipHuge"]["hi"] < 100            # the 1e9 outlier is clipped away


def test_nearest_dot(result):
    assert result["near"] == [0, -1, -1]


@pytest.mark.parametrize("name", ["telemetry.js", "project_chat.js", "t58-chrome.js"])
def test_scripts_parse(name):
    proc = subprocess.run([NODE, "--check", str(STATIC / name)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
