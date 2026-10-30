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
    from app import achievements
    from app.achievements import share
    model = evaluate()
    prof = achievements.profile(model)
    ranks = []
    for at, metal, label in achievements.RANK_METALS:
        top = next((a2 for a2, _m, _l in achievements.RANK_METALS if a2 > at), None)
        ranks.append({"metal": metal, "label": label, "from": at + 1, "to": (top if top else None),
                      "reached": model["level"]["level"] - 1 >= at})
    return render_template("achievements.html", active_page="achievements", model=model, rank=prof["rank"], badges=prof["badges"],
                           ranks=ranks, metals=share.METALS, who=_display_name())


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


# ------------------------------------------------------------------ Research profile (shown in Account settings only)
def _display_name() -> str:
    try:
        from app.accounts.settings import load_account_settings
        a = load_account_settings()
        return (a.display_name or a.username or "").strip() or "T58 Researcher"
    except Exception:  # noqa: BLE001
        return "T58 Researcher"


@pages_bp.route("/api/profile")
def profile_api():
    from app import achievements
    return jsonify(achievements.profile())


@pages_bp.route("/api/unlocks")
def unlocks_api():
    from app import achievements
    return jsonify(achievements.unlock_state(fresh=True))


@pages_bp.route("/api/profile/title", methods=["POST"])
def profile_title():
    """Equip a special title (only ones earned) -- stored with the synced preferences so it follows the account."""
    from app import achievements
    from app.persistence import ui_prefs
    tid = str((request.get_json(silent=True) or {}).get("id") or "")
    if tid:
        t = achievements.unlock_state()["titles"].get(tid)
        if not t or not t["unlocked"]:
            return jsonify({"ok": False, "error": "That title has not been earned yet."}), 400
    ui_prefs.update({"t58-title": tid or None})
    return jsonify({"ok": True})


@pages_bp.route("/api/profile/showcase", methods=["POST"])
def profile_showcase():
    """Pin up to three earned badges as the profile showcase."""
    import json as _json
    from app import achievements
    from app.persistence import ui_prefs
    ids = (request.get_json(silent=True) or {}).get("ids")
    ids = [str(i) for i in ids][:3] if isinstance(ids, list) else []
    got = {i["id"] for i in achievements.profile()["badges"]}
    bad = [i for i in ids if i not in got]
    if bad:
        return jsonify({"ok": False, "error": "You can only pin badges you have earned."}), 400
    ui_prefs.update({"t58-showcase": _json.dumps(ids) if ids else None})
    return jsonify({"ok": True, "ids": ids})


@pages_bp.route("/achievements/card/<aid>.svg")
def achievement_card(aid: str):
    from flask import Response
    from app import achievements
    from app.achievements import share
    prof = achievements.profile()
    badge = next((b for b in prof["badges"] if b["id"] == aid), None)
    if badge is None:
        return jsonify({"ok": False, "error": "Badge not earned yet."}), 404
    svg = share.achievement_card_svg(badge, _display_name(), prof["rank"]["metal_label"], prof["level"]["level"])
    resp = Response(svg, mimetype="image/svg+xml")
    if request.args.get("download"):
        resp.headers["Content-Disposition"] = f'attachment; filename="t58-badge-{aid}.svg"'
    return resp


@pages_bp.route("/achievements/medal/<metal>.svg")
def achievement_medal(metal: str):
    from flask import Response
    from app import achievements
    from app.achievements import share
    row = next((r for r in achievements.RANK_METALS if r[1] == metal), None)
    if row is None:
        return jsonify({"ok": False, "error": "Unknown medal."}), 404
    lvl = achievements.profile()["level"]["level"]
    if lvl - 1 < row[0]:
        return jsonify({"ok": False, "error": "Medal not reached yet."}), 404
    top = next((a for a, _m, _l in achievements.RANK_METALS if a > row[0]), None)
    rng = f"Levels {row[0] + 1}-{top}" if top else f"Level {row[0] + 1} and above"
    svg = share.medal_svg(metal, row[2], rng, _display_name())
    resp = Response(svg, mimetype="image/svg+xml")
    if request.args.get("download"):
        resp.headers["Content-Disposition"] = f'attachment; filename="t58-medal-{metal}.svg"'
    return resp


@pages_bp.route("/achievements/certificate.svg")
def achievement_certificate():
    from flask import Response
    from app import achievements
    from app.achievements import share
    prof = achievements.profile()
    counters = achievements.evaluate(persist=False)["counters"]
    svg = share.certificate_svg(_display_name(), prof["rank"], prof["level"], prof["title"], counters, prof["collection"],
                                prof["unlocked"], prof["total"], prof["showcase"])
    resp = Response(svg, mimetype="image/svg+xml")
    if request.args.get("download"):
        resp.headers["Content-Disposition"] = 'attachment; filename="t58-research-certificate.svg"'
    return resp


# ------------------------------------------------------------------ Notification inbox (bell icon beside the profile picture)
@pages_bp.route("/api/inbox")
def inbox_list():
    """Inbox messages (weekly digests). Writes this week's digest first if the Sunday digest is due."""
    from app import inbox
    try:
        inbox.maybe_generate_weekly()
    except Exception:  # noqa: BLE001 -- the bell must never break a page
        pass
    return jsonify(inbox.list_messages())


@pages_bp.route("/api/inbox/read", methods=["POST"])
def inbox_read():
    from app import inbox
    ids = (request.get_json(silent=True) or {}).get("ids")
    inbox.mark_read(ids if isinstance(ids, list) else None)
    return jsonify({"ok": True})


@pages_bp.route("/api/inbox/delete", methods=["POST"])
def inbox_delete():
    from app import inbox
    mid = str((request.get_json(silent=True) or {}).get("id") or "")
    if mid:
        inbox.delete_message(mid)
    return jsonify({"ok": True})


@pages_bp.route("/api/inbox/preview-digest")
def inbox_preview():
    """The digest for the last 7 days, without saving it (the 'Preview this week' button)."""
    from app import inbox
    return jsonify(inbox.build_weekly_digest())


# ------------------------------------------------------------------ Playground
PLAYGROUND_SECTIONS = {
    "fidgets": {"title": "Fidgets", "blurb": "Calm things to fiddle with between runs. A twisting cube with secret combos, a ring of liquid light, a field of drifting particles.",
                "tag": "Unwind", "accent": "#35e0b0", "toys": [["Infinity Cube", "cube"], ["Liquid Playground", "liquid"], ["Particle Field", "field"]]},
    "arcade": {"title": "Arcade", "blurb": "Quick games that fit in a coffee break. Chess against the computer, 2048 with a quant twist, and a polished block stacker.",
               "tag": "Play", "accent": "#f5b83d", "toys": [["Chess vs computer", "chess"], ["2048 Quant Edition", "2048"], ["Block Stacker", "tetris"]]},
    "puzzles": {"title": "Daily Puzzles", "blurb": "A fresh set every day, the same for everyone. Keep a streak going and sharpen your pattern recognition.",
                "tag": "Think", "accent": "#4d90ff", "toys": [["Infinite Maze", "maze"], ["Daily chess puzzle", "chess"], ["Chart analyzer", "chart"]]},
    "quantum": {"title": "Quantum Playground", "blurb": "Physics toys and tiny challenges: bend gravity, steer particle storms, fly a relaxed light tunnel.",
                "tag": "Explore", "accent": "#9b8cff", "toys": [["Gravity Well", "well"], ["Particle Storm", "storm"], ["Orbit Challenge", "orbit"], ["Signal Path", "signal"], ["Light Tunnel", "tunnel"]]},
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


_RESEARCH_SEEDED = False


def _seed_research(dest: _Path) -> None:
    """Copy the research papers shipped with the app into the folder the Library reads, without overwriting anything.

    A packaged build keeps its papers inside the bundle (sys._MEIPASS/research) or next to the executable's release
    folder; the Library reads <app data folder>/research, so on a customer's machine it came up with 0 papers.
    Files the person added or changed are never overwritten. Runs at most once per process."""
    global _RESEARCH_SEEDED
    if _RESEARCH_SEEDED:
        return
    _RESEARCH_SEEDED = True
    try:
        if any(dest.rglob("*.pdf")):          # already stocked (or deliberately curated): leave it alone
            return
    except OSError:
        pass
    import shutil
    import sys
    sources: list[_Path] = []
    mei = getattr(sys, "_MEIPASS", None)
    if mei:
        sources.append(_Path(mei) / "research")
    sources.append(_Path(__file__).resolve().parents[2] / "research")
    if getattr(sys, "frozen", False):
        sources.append(_Path(sys.executable).resolve().parent / "research")
    for src in sources:
        try:
            if not src.is_dir() or src.resolve() == dest.resolve():
                continue
            for f in src.rglob("*"):
                if not f.is_file():
                    continue
                target = dest / f.relative_to(src)
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
        except OSError:
            continue


def _research_dir() -> _Path:
    d = _base_dir() / "research"
    d.mkdir(parents=True, exist_ok=True)
    _seed_research(d)
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
        e["in_repo"] = "[IN REPO" in e["title"]
        e["links"] = list(dict.fromkeys(e["links"]))
        e["title"] = _re.sub(r"\s*\[IN REPO[^\]]*\]\s*", " ", e["title"]).strip()
    return out


# One line on why each bundled paper matters to a prop-firm backtester. Keyed by lower-cased file stem; unknown PDFs still list fine.
_PAPER_NOTES = {
    "the deflated sharpe ratio": ("Bailey, Lopez de Prado", "Corrects a Sharpe ratio for how many strategies you tried and for non-normal returns. The basis of the engine's search-bias check."),
    "the probability of backtest overfitting": ("Bailey, Borwein, Lopez de Prado, Zhu", "How to estimate the chance the best in-sample strategy is a fluke (CSCV)."),
    "backtest overfitting in financial markets": ("Survey", "Why optimised backtests look great and then fail live, and the usual remedies."),
    "determining optimal trading rules": ("Optimal trading rules", "Choosing profit-target and stop levels without overfitting them to one history."),
    "do simple technical trading rules survive realistic costs": ("Cost study", "Most simple rules vanish once spreads, slippage and commissions are charged."),
    "a novel approach to trading strategy parameter optimization": ("Parameter optimisation", "A way to pick parameters that stay stable rather than the single best in-sample peak."),
    "structural limits of ohlcv-based intraday signals in mnq futures": ("MNQ study", "How much edge OHLCV bars can honestly carry on micro Nasdaq futures."),
    "a validated volatility-volume-gap classifier for regime identification mnq data": ("MNQ study", "A regime classifier built on volatility, volume and gaps, validated on MNQ."),
    "a century of evidence on trend-following investing": ("Hurst, Ooi, Pedersen", "A hundred years of trend-following results across asset classes."),
    "momentum crashes": ("Daniel, Moskowitz", "When and why momentum strategies suffer sharp losses, and how to temper them."),
    "do momentum and reversal strategies work in commodity futures": ("Commodity futures", "Tests momentum and reversal rules on commodity futures."),
    "momentum strategies in futures markets and trend-following funds": ("Futures momentum", "How momentum works in futures and how trend-following funds apply it."),
    "trend following, risk parity and momentum in commodity futures": ("Commodity futures", "Links trend following, risk parity and momentum."),
    "a comparative study of breakout and mean-reversion strategies": ("Comparative study", "Breakout and mean-reversion rules compared on the same data."),
    "analysis of the bollinger bands and mean reversion trading strategy in sp 500": ("S&P 500 study", "Bollinger-band mean reversion tested on the S&P 500."),
    "design and development of mean reversion strategies on quantconnect platform": ("QuantConnect", "Building and testing mean-reversion strategies end to end."),
    "mean reversion trading strategies and the black swan events": ("Tail risk", "How mean reversion behaves when rare shocks arrive."),
    "101 formulaic alphas": ("Kakushadze", "101 real-world quantitative alpha formulas, a source of signal ideas."),
    "a hybrid ai-driven trading system integrating technical analysis, machine learning and financial sentiment": ("Hybrid system", "Combines technical, ML and sentiment inputs in one system."),
    "a reinforcement learning and genetic algorithm framework for intraday index futures trading": ("RL + GA", "Reinforcement learning with a genetic search on intraday index futures."),
    "adaptive limits of algorithmic cryptocurrency trading from a complex adaptive systems perspective": ("Crypto", "Why algorithmic edges decay as markets adapt."),
    "deep reinforcement learning on a multi-asset": ("Deep RL", "Deep reinforcement learning applied across several assets."),
    "deep reinforcement learning for trading": ("Deep RL", "A practical look at deep RL trading agents."),
    "expectancy-optimised algorithmic trading on gold spot markets": ("Gold", "Optimising for expectancy, not win rate, on gold."),
    "machine learning in futures markets": ("ML in futures", "What machine learning does and does not add on futures data."),
    "how order flow affects stock price": ("Order flow", "How buying and selling pressure moves prices."),
    "order flow and exchange rate dynamics in electronic brokerage system data": ("Order flow, FX", "Order flow and currency movements from brokerage data."),
    "order flow and exchange rate dynamics": ("Evans, Lyons", "The classic result that signed order flow explains exchange-rate moves."),
    "an actuarial analysis of proprietary trading firm challenges": ("Prop firm math", "Models prop-firm challenges as insurance-style bets: pass rates, fees and expected value."),
    "day trading adapting to change": ("Day trading", "How day-trading edges change over time and what to do about it."),
    "the kelly criterion and the stock market": ("Kelly", "Optimal bet sizing, and why fractional Kelly is safer when the edge is uncertain."),
}


def _paper_state_path() -> _Path:
    return _lib_dir() / "paper_state.json"


def _load_paper_state() -> dict:
    try:
        d = _json.loads(_paper_state_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _paper_year(stem: str) -> str:
    m = _re.search(r"\b(19|20)\d{2}\b", stem)
    return m.group(0) if m else ""


@pages_bp.route("/api/library/papers")
def library_papers():
    """Bundled PDFs (research/papers/<NN_Category>/*.pdf) first, then the link-only entries from the reading lists
    (research/<NN_...>/Reading_List.md), plus per-paper like/read state."""
    root = _research_dir()
    state = _load_paper_state()
    cats: list[dict] = []
    seen_pdf: set[str] = set()
    pdf_root = root / "papers"
    if pdf_root.is_dir():
        for folder in sorted(p for p in pdf_root.iterdir() if p.is_dir()):
            papers = []
            for f in sorted(folder.glob("*.pdf")):
                seen_pdf.add(str(f))
                key = _re.sub(r"\s*\(alt\)$", "", f.stem).strip().lower()
                au, why = _PAPER_NOTES.get(key, ("", ""))
                pid = f.relative_to(root).with_suffix("").as_posix()
                st = state.get(pid, {})
                papers.append({"id": pid, "title": f.stem, "authors": au, "role": why, "links": [], "pdf": True, "bundled": True,
                               "size_kb": round(f.stat().st_size / 1024), "liked": bool(st.get("liked")), "read": bool(st.get("read"))})
            if papers:
                cats.append({"id": folder.name, "name": _pretty_category(folder.name), "group": "library", "papers": papers, "docs": []})
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "papers"):
        rl = folder / "Reading_List.md"
        entries = parse_reading_list(rl.read_text(encoding="utf-8", errors="replace")) if rl.exists() else []
        papers = []
        for i, e in enumerate(entries):
            if e.get("in_repo"):
                continue  # already bundled as a PDF above
            pid = f"{folder.name}/{i + 1:02d}-{_slug(e['title'])[:48]}"
            pdf = root / folder.name / f"{pid.split('/')[1]}.pdf"
            m = _re.match(r"^(.*?)\s-\s(.*)$", e["title"])
            st = state.get(pid, {})
            papers.append({"id": pid, "title": (m.group(2) if m else e["title"]).strip(' "'), "authors": m.group(1) if m else "", "role": e["role"],
                           "links": e["links"], "pdf": pdf.exists(), "bundled": False, "liked": bool(st.get("liked")), "read": bool(st.get("read"))})
        extra = [p for p in sorted(folder.rglob("*.pdf")) if str(p) not in seen_pdf and not (p.name.startswith(tuple(f"{j + 1:02d}-" for j in range(60))))]
        for p in extra:
            rel = p.relative_to(root).with_suffix("").as_posix()
            st = state.get(rel, {})
            papers.append({"id": rel, "title": p.stem.replace("_", " "), "authors": "", "role": "Your PDF", "links": [], "pdf": True, "bundled": False,
                           "liked": bool(st.get("liked")), "read": bool(st.get("read"))})
        docs = [{"name": p.name, "path": p.relative_to(root).as_posix()} for p in sorted(folder.glob("*.md")) if p.name != "Reading_List.md"]
        if papers or docs:
            cats.append({"id": folder.name, "name": _pretty_category(folder.name), "group": "reading" if folder.name[:2] != "00" else "guides",
                         "papers": papers, "docs": docs})
    loose = [p for p in sorted(root.glob("*.pdf"))]
    if loose:
        cats.append({"id": "_loose", "name": "Other PDFs", "group": "library", "docs": [], "papers": [
            {"id": p.stem, "title": p.stem.replace("_", " "), "authors": "", "role": "Your PDF", "links": [], "pdf": True, "bundled": False,
             "liked": bool(state.get(p.stem, {}).get("liked")), "read": bool(state.get(p.stem, {}).get("read"))} for p in loose]})
    return jsonify({"categories": cats})


@pages_bp.route("/api/library/paper-state", methods=["POST"])
def library_paper_state():
    data = request.get_json(silent=True) or {}
    pid = str(data.get("id", ""))[:300]
    if not pid:
        return jsonify({"ok": False}), 400
    state = _load_paper_state()
    cur = state.get(pid, {})
    for k in ("liked", "read"):
        if k in data:
            cur[k] = bool(data[k])
    if cur.get("liked") or cur.get("read"):
        state[pid] = cur
    else:
        state.pop(pid, None)
    if len(state) > 5000:
        return jsonify({"ok": False, "error": "Too many saved items."}), 400
    _paper_state_path().write_text(_json.dumps(state), encoding="utf-8")
    return jsonify({"ok": True, "state": cur})


@pages_bp.route("/research-docs/<path:rel>")
def library_research_doc(rel):
    """Read-only view of the markdown guides (research/00_Start_Here/*.md) so they open inside the app."""
    from flask import abort, send_file
    root = _research_dir().resolve()
    p = (root / rel).resolve()
    if root not in p.parents or p.suffix.lower() != ".md" or not p.is_file():
        abort(404)
    return send_file(str(p), mimetype="text/plain; charset=utf-8", max_age=0)


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
