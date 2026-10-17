"""Every data-timeframe dropdown offers 1m, and 1m is a valid choice end to end."""
from __future__ import annotations

import re

from app.web import server
from app.web.server import app


def test_choices_start_with_1m():
    assert server.TIMEFRAME_CHOICES[0] == "1m"


def test_pages_render_1m_option():
    c = app.test_client()
    for url in ("/optimize-simple", "/validate-simple", "/full-pipeline"):
        r = c.get(url, follow_redirects=True)
        if r.status_code != 200:
            continue
        html = r.get_data(as_text=True)
        sel = re.search(r'<select[^>]*name="timeframe".*?</select>', html, re.S)
        assert sel and 'value="1m"' in sel.group(0), url
