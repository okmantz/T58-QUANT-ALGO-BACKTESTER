"""Tests for app.education.content -- the shared lesson data both the web
/education route and the desktop Education tab render from. Pure data
tests only (no tkinter/Flask needed), matching this module's own "no I/O,
no side effects" design."""
from __future__ import annotations

import pytest

from app.education import content as edu


def test_list_sections_nonempty_and_unique_ids():
    sections = edu.list_sections()
    assert len(sections) > 0
    ids = [s.id for s in sections]
    assert len(ids) == len(set(ids))


def test_every_section_has_at_least_one_lesson():
    for section in edu.list_sections():
        assert len(section.lessons) >= 1, f"{section.id} has no lessons"


def test_every_lesson_has_unique_id_within_its_section():
    for section in edu.list_sections():
        lesson_ids = [l.id for l in section.lessons]
        assert len(lesson_ids) == len(set(lesson_ids)), f"duplicate lesson id in {section.id}"


def test_lesson_ids_are_globally_unique():
    """Not structurally required, but keeps anchor links (#lesson-id-style
    URLs, if ever added) unambiguous across the whole tab."""
    all_ids = [l.id for s in edu.list_sections() for l in s.lessons]
    assert len(all_ids) == len(set(all_ids))


def test_every_lesson_has_at_least_one_block():
    for section in edu.list_sections():
        for lesson in section.lessons:
            assert len(lesson.blocks) >= 1, f"{section.id}/{lesson.id} has no content blocks"


def test_every_block_has_the_right_payload_for_its_kind():
    valid_kinds = {"p", "bullets", "steps", "tip", "warn", "example"}
    for section in edu.list_sections():
        for lesson in section.lessons:
            for block in lesson.blocks:
                assert block.kind in valid_kinds, f"{section.id}/{lesson.id} has unknown block kind {block.kind!r}"
                if block.kind in ("bullets", "steps"):
                    assert block.items, f"{section.id}/{lesson.id} has an empty {block.kind} block"
                else:
                    assert block.text, f"{section.id}/{lesson.id} has an empty {block.kind} block"


def test_get_section_returns_expected():
    section = edu.get_section("validation-lab")
    assert section.title == "Validation Lab -- Is Your Edge Real?"
    assert any(l.id == "pbo" for l in section.lessons)


def test_get_section_unknown_id_raises():
    with pytest.raises(KeyError):
        edu.get_section("not_a_real_section")


def test_total_lesson_count_matches_sum_across_sections():
    assert edu.total_lesson_count() == sum(len(s.lessons) for s in edu.list_sections())


def test_block_constructor_helpers_build_expected_blocks():
    assert edu.p("hello") == edu.Block("p", text="hello")
    assert edu.bullets("a", "b") == edu.Block("bullets", items=("a", "b"))
    assert edu.steps("a", "b") == edu.Block("steps", items=("a", "b"))
    assert edu.tip("hi") == edu.Block("tip", text="hi")
    assert edu.warn("careful") == edu.Block("warn", text="careful")
    assert edu.example("e.g.") == edu.Block("example", text="e.g.")


def test_key_concepts_from_this_session_are_actually_covered():
    """Regression guard: the concepts this app-building session repeatedly
    had to explain from scratch (PBO/multiple-comparisons, trailing vs
    static drawdown, consistency-rule staging, fastest_payout) should be
    findable in the Education tab, not just in code comments."""
    all_text = []
    for section in edu.list_sections():
        for lesson in section.lessons:
            all_text.append(lesson.title)
            for block in lesson.blocks:
                all_text.append(block.text)
                all_text.extend(block.items)
    haystack = " ".join(all_text).lower()

    for term in (
        "pbo", "multiple-comparisons", "trailing", "static", "consistency rule",
        "fastest_payout", "walk-forward", "monte carlo", "lookahead",
    ):
        assert term in haystack, f"expected to find {term!r} somewhere in the Education tab content"


def test_see_also_references_use_real_tab_names_not_generic_placeholders():
    """Every see_also entry should look like an actual tab/route name used
    elsewhere in this app, not a vague placeholder -- spot-check a few
    known-real ones appear somewhere in the catalog."""
    all_see_also = [s for sec in edu.list_sections() for l in sec.lessons for s in l.see_also]
    assert any("CPCV" in s for s in all_see_also)
    assert any("Run & Report" in s for s in all_see_also)
    assert any("Prop-Firm Rules" in s for s in all_see_also)
