"""Check every link in research/*/Reading_List.md and optionally download arXiv PDFs.

Run on a machine with normal internet access:

    python scripts/check_paper_links.py              # report dead links
    python scripts/check_paper_links.py --download   # also save arXiv PDFs into research/papers/

Google Scholar search links are skipped (they always load). Exit code is 1 if any link failed.
Nothing here is imported by the app.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "research"
URL = re.compile(r"https?://[^\s)]+")
UA = {"User-Agent": "Mozilla/5.0 (T58 link checker)"}


def check(url: str, timeout: int = 20) -> str:
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, headers=UA, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return "ok" if r.status < 400 else f"http {r.status}"
        except Exception as e:  # try GET after a HEAD failure
            last = str(e)[:60]
    return f"FAIL {last}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="save arxiv.org/pdf links to research/papers/")
    a = ap.parse_args()
    bad = 0
    for lst in sorted(ROOT.glob("*/Reading_List.md")):
        for url in sorted(set(URL.findall(lst.read_text(encoding="utf-8")))):
            url = url.rstrip(".,;")
            if "scholar.google" in url:
                continue
            res = check(url)
            if res != "ok":
                bad += 1
                print(f"{lst.parent.name}: {res}: {url}")
            elif a.download and "arxiv.org/pdf/" in url:
                dest = ROOT / "papers" / lst.parent.name
                dest.mkdir(parents=True, exist_ok=True)
                name = url.rsplit("/pdf/", 1)[1].replace("/", "_") + ".pdf"
                if not (dest / name).exists():
                    req = urllib.request.Request(url, headers=UA)
                    (dest / name).write_bytes(urllib.request.urlopen(req, timeout=60).read())
    print(f"{bad} link(s) need attention")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
