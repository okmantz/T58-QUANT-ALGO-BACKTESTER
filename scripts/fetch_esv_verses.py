"""Build app/web/static/esv-verses.json (the English Standard Version text for the Personalize verse bar).

The ESV is copyrighted (Crossway), so its text is not committed to this repository. Run this ONCE on your own machine with
your own free API key from https://api.esv.org/ :

    set ESV_API_KEY=your-key            (Windows)      export ESV_API_KEY=your-key     (macOS / Linux)
    python scripts/fetch_esv_verses.py

It reads the verse references out of app/web/static/t58-personalize.js, asks the official ESV API for each one, and writes
the JSON the app loads. Until that file exists the verse bar shows the public-domain King James text, labelled "KJV".
Using ESV text in a distributed or paid product needs Crossway's permission -- see https://www.crossway.org/permissions/.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "app" / "web" / "static" / "t58-personalize.js"
OUT = ROOT / "app" / "web" / "static" / "esv-verses.json"
API = "https://api.esv.org/v3/passage/text/"
COPYRIGHT = ("Scripture quotations are from the ESV® Bible (The Holy Bible, English Standard Version®), "
             "copyright © 2001 by Crossway, a publishing ministry of Good News Publishers. Used by permission. All rights reserved.")


def references() -> list[str]:
    src = JS.read_text(encoding="utf-8")
    return re.findall(r'\["([1-3]? ?[A-Za-z ]+ \d+:[\d\-,]+)",\s*"', src)


def fetch(ref: str, key: str) -> str:
    q = urllib.parse.urlencode({
        "q": ref, "include-passage-references": "false", "include-verse-numbers": "false", "include-footnotes": "false",
        "include-footnote-body": "false", "include-headings": "false", "include-short-copyright": "false",
        "include-passage-horizontal-lines": "false", "include-heading-horizontal-lines": "false"})
    req = urllib.request.Request(f"{API}?{q}", headers={"Authorization": f"Token {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    return re.sub(r"\s+", " ", "".join(data.get("passages") or [])).strip()


def main() -> int:
    key = os.environ.get("ESV_API_KEY", "").strip()
    if not key:
        print("Set ESV_API_KEY first (free key at https://api.esv.org/).", file=sys.stderr)
        return 2
    refs = references()
    if not refs:
        print("No verse references found in t58-personalize.js", file=sys.stderr)
        return 1
    verses: dict[str, str] = {}
    for ref in refs:
        try:
            text = fetch(ref, key)
        except Exception as exc:  # keep going; that verse falls back to KJV
            print(f"  skipped {ref}: {exc}", file=sys.stderr)
            continue
        if text:
            verses[ref] = text
        time.sleep(0.25)
    OUT.write_text(json.dumps({"version": "ESV", "copyright": COPYRIGHT, "verses": verses}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(verses)}/{len(refs)} verses to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
