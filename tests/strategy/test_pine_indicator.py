"""Pine Indicator Creator: every generated script must pass the linter AND the real Pine grammar (when installed)."""
from __future__ import annotations

import itertools

import pytest

from app.strategy import pine_indicator as P

pynescript = pytest.importorskip("pynescript", reason="pynescript gives the real-grammar check")


def _errors(res):
    return [i for i in res.issues if i["level"] == "error"]


@pytest.mark.parametrize("comp", [c.id for c in P.CATALOG])
def test_each_catalog_component_compiles_alone(comp):
    c = P._BY_ID[comp]
    code = P.assemble([(c, dict(c.defaults))])
    assert not [i for i in P.lint(code) if i.level == "error"], code
    assert P.grammar_check(code)[0] is True, code


def test_every_overlay_pair_and_every_pane_pair_compiles():
    overlays = [c for c in P.CATALOG if c.overlay]
    panes = [c for c in P.CATALOG if not c.overlay]
    for group in (overlays, panes):
        for a, b in itertools.combinations(group, 2):
            code = P.assemble([(a, dict(a.defaults)), (b, dict(b.defaults))])
            assert not [i for i in P.lint(code) if i.level == "error"], (a.id, b.id)
            assert P.grammar_check(code)[0] is True, (a.id, b.id, code)


def test_all_components_together_stay_under_the_plot_limit_and_compile():
    overlays = [(c, dict(c.defaults)) for c in P.CATALOG if c.overlay]
    code = P.assemble(overlays)
    assert [i for i in P.lint(code) if i.level == "error"] == []
    assert P.grammar_check(code)[0] is True


def test_idea_parsing_picks_components_and_numbers():
    parts = {c.id: p for c, p in P.pick_components("EMA 21 50 200 ribbon, RSI 7, bollinger bands, MACD 8 17 9, liquidity sweeps")}
    assert parts["ema"]["periods"] == [21, 50, 200]
    assert parts["rsi"]["period"] == 7
    assert (parts["macd"]["fast"], parts["macd"]["slow"], parts["macd"]["signal"]) == (8, 17, 9)
    assert {"bb", "sweep"} <= set(parts)


def test_mixed_overlay_and_oscillator_idea_is_split_into_two_valid_scripts():
    res = P.build_from_idea("EMA 50 with RSI 14")
    assert res.ok and len(res.extra_scripts) == 1 and res.extra_scripts[0]["ok"]
    assert "overlay=true" in res.code and "overlay=false" in res.extra_scripts[0]["code"]


def test_unrecognised_idea_still_returns_a_valid_starter_script_and_says_so():
    res = P.build_from_idea("zzz qqq")
    assert res.ok and any("starter" in n for n in res.notes)


def test_v4_code_is_upgraded_to_v5_and_passes():
    v4 = "//@version=4\nstudy('Old', overlay=true)\nlen = input(14, type=input.integer)\nx = sma(close, len)\ny = security(syminfo.tickerid, 'D', close)\nplot(x, color=red)\n"
    res = P.check_and_fix(v4)
    assert res.ok, res.issues
    assert "//@version=5" in res.code and "ta.sma(" in res.code and "request.security(" in res.code and "indicator(" in res.code


@pytest.mark.parametrize("bad,needle", [
    ("//@version=5\nindicator('x')\nx = ta.emaa(close, 5)\nplot(x)\n", "Unknown ta.emaa"),
    ("//@version=5\nindicator('x')\nif close > open\n    plot(close)\n", "global scope"),
    ("//@version=5\nindicator('x')\na = 1\na = 2\nplot(a)\n", "already declared"),
    ("//@version=5\nindicator('x')\nplot(ta.sma(close, 5)\n", "never closed"),
    ("//@version=5\nindicator('x')\nx = close\n", "draws nothing"),
    ("//@version=5\nstrategy('x')\nplot(close)\n", "strategy("),
    ("indicator('x')\nplot(close)\n", "@version"),
])
def test_linter_catches_common_compile_errors(bad, needle):
    msgs = " | ".join(i.message for i in P.lint(bad) if i.level == "error")
    assert needle in msgs, msgs


def test_llm_path_repairs_a_bad_draft():
    calls = []

    def fake(prompt, system):
        calls.append(prompt)
        if len(calls) == 1:      # first draft uses a v4 call and an unknown function
            return "```pine\n//@version=5\nindicator('d', overlay=true)\nplot(ta.emaa(close, 9))\n```", None
        return "```pine\n//@version=5\nindicator('d', overlay=true)\nplot(ta.ema(close, 9))\n```", None

    res = P.generate_with_llm("ema 9", complete=fake)
    assert res.ok and res.source == "llm" and len(calls) == 2


def test_llm_failure_falls_back_to_the_offline_builder():
    res = P.generate_with_llm("RSI 14", complete=lambda p, s: (None, "no model"))
    assert res.ok and res.source == "builder" and any("unavailable" in n.lower() for n in res.notes)


def test_llm_that_never_produces_clean_code_falls_back_and_keeps_its_draft():
    res = P.generate_with_llm("RSI 14", complete=lambda p, s: ("```pine\nplot(nonsense(\n```", None), max_repairs=1)
    assert res.source == "builder" and res.ok and any(e["title"].startswith("AI draft") for e in res.extra_scripts)
