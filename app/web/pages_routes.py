"""Achievements, Playground, Library (books / papers / Bible / focus timer) and the Morning Briefing API.

One Blueprint so server.py needs a single import + register line. Everything here is local: no market data, no network
beyond what the person explicitly opens (research-paper links open the publisher's page in a new tab)."""
from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

pages_bp = Blueprint("pages_extra", __name__)


# ------------------------------------------------------------------ Achievements
@pages_bp.route("/achievements")
def achievements_page():
    from app.achievements import evaluate
    model = evaluate()
    return render_template("achievements.html", active_page="achievements", model=model)


@pages_bp.route("/api/achievements")
def achievements_api():
    from app.achievements import evaluate
    m = evaluate()
    return jsonify({k: m[k] for k in ("counters", "xp", "level", "unlocked", "total", "next_up")})


@pages_bp.route("/api/achievements/new")
def achievements_new():
    """Unlocks the person has not been shown yet (the toast on any page polls this once)."""
    from app.achievements import evaluate
    m = evaluate()
    return jsonify({"new": [{"id": i["id"], "title": i["title"], "tier": i["tier"], "xp": i["xp"], "description": i["description"]} for i in m["new"]],
                    "level": m["level"]})


@pages_bp.route("/api/achievements/ack", methods=["POST"])
def achievements_ack():
    from app.achievements import acknowledge
    data = request.get_json(silent=True) or {}
    ids = data.get("ids")
    acknowledge(ids if isinstance(ids, list) else None)
    return jsonify({"ok": True})


# ------------------------------------------------------------------ Playground
PLAYGROUND_SECTIONS = {
    "fidgets": {"title": "Fidgets", "blurb": "Calm things to fiddle with between runs.", "toys": ["Infinity Cube", "Liquid Playground", "Particle Field"]},
    "arcade": {"title": "Arcade", "blurb": "Quick games that fit in a coffee break.", "toys": ["Chess vs computer", "2048 Quant Edition", "Tetris"]},
    "puzzles": {"title": "Daily Puzzles", "blurb": "A fresh set every day, same for everyone.", "toys": ["Infinite Maze", "Daily chess puzzle", "Chart analyzer"]},
    "quantum": {"title": "Quantum Playground", "blurb": "Physics toys and tiny challenges.", "toys": ["Gravity Well", "Particle Storm", "Orbit Challenge", "Signal Path", "Light Tunnel"]},
}


@pages_bp.route("/playground")
def playground_home():
    return render_template("playground.html", active_page="playground", sections=PLAYGROUND_SECTIONS)


@pages_bp.route("/playground/<section>")
def playground_section(section):
    if section == "library":
        return render_template("books_library.html", active_page="books_library")
    meta = PLAYGROUND_SECTIONS.get(section)
    if not meta:
        from flask import abort
        abort(404)
    return render_template("playground_section.html", active_page="playground_" + section, section=section, meta=meta)


# ------------------------------------------------------------------ Library (books, papers, Bible, focus sessions)
import json as _json
import re as _re
import time as _time
from pathlib import Path as _Path


def _base_dir() -> _Path:
    from app.data.storage import get_app_base_dir
    return get_app_base_dir()


def _lib_dir(*parts: str) -> _Path:
    d = _base_dir() / "data" / "library"
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(text: str) -> str:
    return _re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:80] or "item"


def _research_dir() -> _Path:
    d = _base_dir() / "research"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pretty_category(folder: str) -> str:
    name = _re.sub(r"^\d+_", "", folder).replace("_", " ")
    return name.replace(" and ", " & ")


def parse_reading_list(text: str) -> list[dict]:
    """Parse one of research/*/Reading_List.md into [{title, role, links}]. Entries are numbered ("1. Authors - "Title", venue")
    followed by indented "- Role:" / "- Link(s):" lines; URLs may continue on the following indented lines."""
    out: list[dict] = []
    cur = None
    in_links = False
    for raw in text.splitlines():
        line = raw.rstrip()
        m = _re.match(r"^\s{0,2}(\d+)\.\s+(.*\S)\s*$", line)
        if m:
            cur = {"title": m.group(2), "role": "", "links": []}
            in_links = False
            out.append(cur)
            continue
        if cur is None:
            continue
        s = line.strip()
        if not s:
            in_links = False
            continue
        if s.startswith("- Role:"):
            cur["role"] = s[len("- Role:"):].strip()
            in_links = False
        elif s.startswith("- Link"):
            in_links = True
            cur["links"] += _re.findall(r"https?://[^\s)]+", s)
        elif in_links:
            cur["links"] += _re.findall(r"https?://[^\s)]+", s)
        elif s.startswith("#"):
            cur = None
    for e in out:
        e["links"] = list(dict.fromkeys(e["links"]))
        e["title"] = _re.sub(r"\s*\[IN REPO[^\]]*\]\s*", " ", e["title"]).strip()
    return out


@pages_bp.route("/api/library/papers")
def library_papers():
    root = _research_dir()
    cats: list[dict] = []
    used_pdfs: set[str] = set()
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        rl = folder / "Reading_List.md"
        entries = parse_reading_list(rl.read_text(encoding="utf-8", errors="replace")) if rl.exists() else []
        papers = []
        for i, e in enumerate(entries):
            pid = f"{folder.name}/{i + 1:02d}-{_slug(e['title'])[:48]}"
            pdf = root / folder.name / f"{pid.split('/')[1]}.pdf"
            if pdf.exists():
                used_pdfs.add(str(pdf))
            papers.append({"id": pid, "title": e["title"], "role": e["role"], "links": e["links"], "pdf": pdf.exists()})
        extra = [p for p in sorted(folder.rglob("*.pdf")) if str(p) not in used_pdfs]
        for p in extra:
            rel = p.relative_to(root).as_posix()
            papers.append({"id": rel[:-4], "title": p.stem.replace("_", " "), "role": "Your PDF", "links": [], "pdf": True})
        docs = [{"name": p.name, "path": p.relative_to(root).as_posix()} for p in sorted(folder.glob("*.md")) if p.name != "Reading_List.md"]
        cats.append({"id": folder.name, "name": _pretty_category(folder.name), "papers": papers, "docs": docs})
    loose = [p for p in sorted(root.glob("*.pdf"))]
    if loose:
        cats.append({"id": "_loose", "name": "Other PDFs", "papers": [{"id": p.stem, "title": p.stem.replace("_", " "), "role": "Your PDF", "links": [], "pdf": True} for p in loose], "docs": []})
    return jsonify({"categories": [c for c in cats if c["papers"] or c["docs"]]})


def _pdf_path(kind: str, ident: str):
    ident = (ident or "").strip().strip("/")
    if kind == "papers":
        root = _research_dir()
        p = (root / (ident + ".pdf")).resolve()
        if root.resolve() not in p.parents:
            return None
        return p
    if kind == "books":
        return _lib_dir("books") / (_slug(ident) + ".pdf")
    return None


@pages_bp.route("/library-files/<kind>/<path:ident>")
def library_file(kind, ident):
    from flask import abort, send_file
    p = _pdf_path(kind, ident[:-4] if ident.endswith(".pdf") else ident)
    if p is None or not p.exists():
        abort(404)
    return send_file(str(p), mimetype="application/pdf", conditional=True, max_age=0)


@pages_bp.route("/api/library/upload", methods=["POST"])
def library_upload():
    kind = request.form.get("kind", "")
    ident = request.form.get("id", "")
    f = request.files.get("file")
    if kind not in ("books", "papers") or not ident or f is None:
        return jsonify({"ok": False, "error": "Pick a PDF first."}), 400
    p = _pdf_path(kind, ident)
    if p is None:
        return jsonify({"ok": False, "error": "Bad item."}), 400
    data = f.read(90 * 1024 * 1024 + 1)
    if len(data) > 90 * 1024 * 1024:
        return jsonify({"ok": False, "error": "That PDF is over 90 MB."}), 400
    if not data.startswith(b"%PDF"):
        return jsonify({"ok": False, "error": "That file is not a PDF."}), 400
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return jsonify({"ok": True, "url": f"/library-files/{kind}/{ident}.pdf"})


@pages_bp.route("/api/library/delete", methods=["POST"])
def library_delete():
    data = request.get_json(silent=True) or {}
    p = _pdf_path(data.get("kind", ""), data.get("id", ""))
    if p is not None and p.exists():
        p.unlink()
    return jsonify({"ok": True})


@pages_bp.route("/api/library/books")
def library_books():
    d = _lib_dir("books")
    return jsonify({"pdfs": sorted(p.stem for p in d.glob("*.pdf"))})


# ---- focus sessions
def _focus_path() -> _Path:
    return _lib_dir() / "focus_sessions.json"


def _load_focus() -> list:
    try:
        d = _json.loads(_focus_path().read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception:  # noqa: BLE001
        return []


@pages_bp.route("/api/library/focus", methods=["GET", "POST"])
def library_focus():
    if request.method == "GET":
        return jsonify({"sessions": _load_focus()[-300:]})
    data = request.get_json(silent=True) or {}
    try:
        minutes = int(data.get("minutes", 0))
        done = float(data.get("done_minutes", minutes))
    except (TypeError, ValueError):
        return jsonify({"ok": False}), 400
    if minutes not in (10, 20, 30):
        return jsonify({"ok": False, "error": "Sessions are 10, 20 or 30 minutes."}), 400
    rows = _load_focus()
    rows.append({"at": _time.time(), "minutes": minutes, "done_minutes": round(max(0.0, min(done, minutes)), 1),
                 "completed": bool(data.get("completed")), "label": str(data.get("label", ""))[:60]})
    _focus_path().write_text(_json.dumps(rows[-1000:]), encoding="utf-8")
    return jsonify({"ok": True})


@pages_bp.route("/api/library/focus/clear", methods=["POST"])
def library_focus_clear():
    _focus_path().write_text("[]", encoding="utf-8")
    return jsonify({"ok": True})


# ---- ESV Bible (Crossway's API with the person's own free key; chapters they read are cached for offline reading)
ESV_API = "https://api.esv.org/v3/passage/text/"
ESV_COPYRIGHT = ("Scripture quotations are from the ESV® Bible (The Holy Bible, English Standard Version®), copyright © 2001 by Crossway, "
                 "a publishing ministry of Good News Publishers. Used by permission. All rights reserved.")
_ESV_CACHE_MAX = 60


def _esv_key() -> str:
    import os
    k = os.environ.get("ESV_API_KEY", "").strip()
    if k:
        return k
    try:
        return str(_json.loads((_lib_dir() / "esv_key.json").read_text(encoding="utf-8")).get("key", "")).strip()
    except Exception:  # noqa: BLE001
        return ""


@pages_bp.route("/api/bible/esv/status")
def esv_status():
    return jsonify({"configured": bool(_esv_key()), "copyright": ESV_COPYRIGHT})


@pages_bp.route("/api/bible/esv/key", methods=["POST"])
def esv_save_key():
    data = request.get_json(silent=True) or {}
    key = str(data.get("key", "")).strip()
    p = _lib_dir() / "esv_key.json"
    if not key:
        if p.exists():
            p.unlink()
        return jsonify({"ok": True, "configured": bool(_esv_key())})
    if not _re.fullmatch(r"[A-Za-z0-9]{20,80}", key):
        return jsonify({"ok": False, "error": "That does not look like an ESV API key (letters and numbers, about 40 characters)."}), 400
    p.write_text(_json.dumps({"key": key}), encoding="utf-8")
    return jsonify({"ok": True, "configured": True})


@pages_bp.route("/api/bible/esv")
def esv_passage():
    import urllib.parse
    import urllib.request
    q = (request.args.get("q") or "").strip()
    if not _re.fullmatch(r"[1-3]?\s?[A-Za-z ]{2,24}\s\d{1,3}(:\d{1,3}(-\d{1,3})?)?", q):
        return jsonify({"ok": False, "error": "Unrecognised passage."}), 400
    cache_dir = _lib_dir("esv_cache")
    cp = cache_dir / (_slug(q) + ".json")
    if cp.exists():
        try:
            d = _json.loads(cp.read_text(encoding="utf-8"))
            d["cached"] = True
            return jsonify(d)
        except Exception:  # noqa: BLE001
            pass
    key = _esv_key()
    if not key:
        return jsonify({"ok": False, "error": "no_key"}), 401
    params = urllib.parse.urlencode({"q": q, "include-passage-references": "false", "include-verse-numbers": "true", "include-footnotes": "false",
                                     "include-footnote-body": "false", "include-headings": "true", "include-short-copyright": "false",
                                     "include-passage-horizontal-lines": "false", "include-heading-horizontal-lines": "false", "indent-poetry": "false"})
    req = urllib.request.Request(f"{ESV_API}?{params}", headers={"Authorization": f"Token {key}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = _json.load(r)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Could not reach the ESV service ({exc}). The KJV works offline."}), 502
    text = "".join(data.get("passages") or []).strip()
    out = {"ok": True, "ref": data.get("canonical") or q, "text": text, "copyright": ESV_COPYRIGHT}
    try:
        files = sorted(cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for old in files[:-_ESV_CACHE_MAX + 1]:
            old.unlink()
        cp.write_text(_json.dumps(out), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return jsonify(out)


@pages_bp.route("/api/next-step")
def api_next_step():
    """Per-strategy 'what do I do next' plan (same resolver the Dashboard cards use).
    ?pick=<type>::<filename>  or  ?name=<display name>  (falls back to the tracked current strategy)."""
    from flask import jsonify, request
    from app.guidance import next_step
    from app.reports import strategy_state
    from app.scoring import champion_board
    rows = champion_board.list_board()
    pick = (request.args.get("pick") or "").strip()
    name = (request.args.get("name") or "").strip().lower()
    if not pick and not name:
        cur = strategy_state.get_current_strategy()
        if cur:
            name = str(cur.get("strategy_name") or "").strip().lower()
    row = None
    for r in rows:
        if pick and f"{r['strategy_type']}::{r['filename']}" == pick:
            row = r
        elif name and (name in (r["display_name"].lower(), (r.get("metadata_name") or "").lower())
                       or name in r["filename"].lower() or r["display_name"].lower() in name):
            row = r
        if row:
            break
    if row is None:
        return jsonify({"none": True})
    out = next_step.resolve(row, None)
    out["name"] = row["display_name"]
    out["stage_title"] = row["promotion"]["stage_title"]
    return jsonify(out)


# ---------------------------------------------------------------------------------------------- Morning Briefing
@pages_bp.route("/api/briefing")
def api_briefing():
    """Briefing payload for the floating assistant's Brief tab (saved results only; see app/briefing)."""
    from flask import jsonify
    from app import briefing
    try:
        return jsonify(briefing.get_briefing())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"enabled": True, "due": False, "sections": [], "empty": True, "empty_line": "The briefing could not be built just now.", "error": str(exc)[:120]})


@pages_bp.route("/api/briefing/settings", methods=["POST"])
def api_briefing_settings():
    from flask import jsonify, request
    from app import briefing
    data = request.get_json(silent=True) or request.form
    freq = str(data.get("frequency") or "").strip().lower()
    if freq not in briefing.FREQUENCIES:
        return jsonify({"ok": False, "error": "frequency must be daily, weekly, startup or off"}), 400
    return jsonify({"ok": True, "settings": briefing.set_frequency(freq)})


@pages_bp.route("/api/briefing/seen", methods=["POST"])
def api_briefing_seen():
    from flask import jsonify
    from app import briefing
    briefing.mark_popped()
    return jsonify({"ok": True})
