"""Tests confirming the reusable Start Here section renders correctly
(no double-HTML-escaping, actual content present) on every page it's
been added to so far."""
from __future__ import annotations

import pytest

from app.web.server import app


@pytest.mark.parametrize("path,expected_phrase", [
    ("/", "is the fastest way to backtest ONE strategy"),
    ("/forge", "Forge Strategy is the CREATE section"),
    ("/search", "Search Lab tries many DIFFERENT strategy families"),
    ("/evolution", "Evolution Lab runs a longer"),
    ("/full-pipeline", "Full Pipeline is the complete, validated verdict"),
])
def test_start_here_renders_with_correct_text_and_no_double_escaping(path, expected_phrase):
    app.config["TESTING"] = True
    client = app.test_client()
    r = client.get(path)
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Start here" in body
    assert expected_phrase in body
    # The macro renders `description` through Jinja's normal auto-escaping
    # -- an ampersand in the source description must show up as a plain
    # "&" (escaped once, to &amp; in the raw HTML, which browsers render
    # as "&") and never as a literal double-escaped "&amp;amp;".
    assert "&amp;amp;" not in body


def test_start_here_buttons_link_to_real_pages():
    app.config["TESTING"] = True
    client = app.test_client()
    r = client.get("/forge")
    body = r.get_data(as_text=True)
    assert 'href="/library"' in body
    assert 'href="/generate-strategies"' in body
