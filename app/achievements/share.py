"""Shareable achievement cards and research certificates, drawn as self-contained SVG (no fonts or images to fetch, so the file
looks the same wherever it is opened; the Account page also offers a PNG made from the same SVG in the browser).

Honesty rule: both only state things the app counted (badges earned, tests run, XP). Neither claims a strategy is profitable,
and the certificate says so on its face."""
from __future__ import annotations

import hashlib
import time
from datetime import datetime
from html import escape
from typing import Any

# metal -> (light, mid, dark, ink)
METALS = {
    "bronze": ("#f0b27a", "#c97c3b", "#7a4a1d", "#2b1a0a"),
    "silver": ("#f4f7fb", "#b8c2d0", "#6b7685", "#1c222b"),
    "gold": ("#ffe9a0", "#f2b934", "#9a6a07", "#2a1c02"),
    "platinum": ("#eaf6ff", "#9fd4f5", "#4a86b3", "#0a1e2e"),
    "obsidian": ("#ffe9a0", "#c9a24a", "#2a2418", "#f5e6b8"),
}


def _wrap(text: str, width: int, max_lines: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,; ") + "…"
    return lines


BG_DEF = '<radialGradient id="bg" cx=".25" cy=".1" r="1.1"><stop offset="0" stop-color="#16222c"/><stop offset=".6" stop-color="#0a1016"/><stop offset="1" stop-color="#05070a"/></radialGradient>'


def _defs(metal: str) -> str:
    l, m, d, _ = METALS[metal]
    return (f'<linearGradient id="mt-{metal}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{l}"/><stop offset=".45" stop-color="{m}"/>'
            f'<stop offset="1" stop-color="{d}"/></linearGradient>'
            f'<linearGradient id="mt2-{metal}" x1="1" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{l}"/><stop offset=".5" stop-color="{m}"/><stop offset="1" stop-color="{d}"/></linearGradient>'
            f'<radialGradient id="glow-{metal}" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="{m}" stop-opacity=".45"/><stop offset="1" stop-color="{m}" stop-opacity="0"/></radialGradient>')


def _medal(cx: int, cy: int, r: int, metal: str, rare: bool, glyph: str = "★") -> str:
    l, m, d, ink = METALS[metal]
    rays = ""
    if rare:
        import math
        for k in range(16):
            a = k * math.pi / 8
            rays += (f'<line x1="{cx + math.cos(a) * (r + 10):.1f}" y1="{cy + math.sin(a) * (r + 10):.1f}" '
                     f'x2="{cx + math.cos(a) * (r + (34 if k % 2 == 0 else 20)):.1f}" y2="{cy + math.sin(a) * (r + (34 if k % 2 == 0 else 20)):.1f}" '
                     f'stroke="url(#mt-{metal})" stroke-width="3" stroke-linecap="round" opacity=".85"/>')
    return (f'<circle cx="{cx}" cy="{cy}" r="{r + 70}" fill="url(#glow-{metal})"/>{rays}'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#mt-{metal})"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r - 12}" fill="none" stroke="{ink}" stroke-opacity=".35" stroke-width="3"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r - 22}" fill="url(#mt2-{metal})" opacity=".9"/>'
            f'<text x="{cx}" y="{cy + r * 0.28:.0f}" text-anchor="middle" font-size="{int(r * 0.85)}" fill="{ink}" fill-opacity=".85" '
            f'font-family="Georgia, serif" font-weight="700">{glyph}</text>')


def achievement_card_svg(badge: dict[str, Any], who: str, rank_label: str, level: int) -> str:
    metal = badge["metal"]
    l, m, d, ink = METALS[metal]
    when = datetime.fromtimestamp(badge["unlocked_at"]).strftime("%B %d, %Y") if badge.get("unlocked_at") else ""
    desc = _wrap(badge["description"], 46, 3)
    desc_svg = "".join(f'<text x="440" y="{330 + i * 38}" font-size="27" fill="#aebbc6">{escape(t)}</text>' for i, t in enumerate(desc))
    rare = "RARE · " if badge.get("rare") else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" font-family="Segoe UI, Helvetica, Arial, sans-serif">
<defs>{BG_DEF}{_defs(metal)}</defs>
<rect width="1200" height="630" rx="34" fill="url(#bg)"/>
<rect x="14" y="14" width="1172" height="602" rx="26" fill="none" stroke="url(#mt-{metal})" stroke-width="3" opacity=".8"/>
{_medal(220, 315, 120, metal, bool(badge.get("rare")))}
<text x="440" y="150" font-size="22" letter-spacing="5" fill="{m}" font-weight="700">{rare}{escape(badge['metal_label']).upper()} BADGE · {escape(badge['category']).upper()}</text>
<text x="440" y="250" font-size="62" fill="#ffffff" font-weight="800">{escape(badge['title'])}</text>
<rect x="440" y="272" width="190" height="5" rx="2.5" fill="url(#mt-{metal})"/>
{desc_svg}
<text x="440" y="500" font-size="26" fill="#ffffff" font-weight="700">{escape(who)}</text>
<text x="440" y="534" font-size="21" fill="#8fa1af">{escape(rank_label)} rank · level {level}{(' · earned ' + when) if when else ''}</text>
<text x="1160" y="582" text-anchor="end" font-size="19" letter-spacing="2" fill="#6f808d">T58 QUANT ALGO BACKTESTER</text>
</svg>'''


def certificate_svg(who: str, rank: dict[str, str], level: dict[str, Any], title: str, counters: dict[str, int], stats_badges: dict[str, int],
                    unlocked: int, total: int, showcase: list[dict[str, Any]]) -> str:
    metal = rank["metal"]
    l, m, d, ink = METALS[metal]
    today = datetime.fromtimestamp(time.time()).strftime("%B %d, %Y")
    cid = hashlib.sha256(f"{who}|{level['xp']}|{unlocked}|{today}".encode()).hexdigest()[:10].upper()
    rows = [("Backtests run", counters.get("runs", 0)), ("Strategies tested", counters.get("strategies", 0)),
            ("Instruments analysed", counters.get("instruments", 0)), ("Robustness checks", counters.get("validations", 0)),
            ("Strategies in library", counters.get("library", 0)), ("Lessons in the graveyard", counters.get("graveyard", 0))]
    cells = ""
    for i, (k, v) in enumerate(rows):
        x = 440 + (i % 3) * 360
        y = 722 + (i // 3) * 100
        cells += (f'<text x="{x}" y="{y}" text-anchor="middle" font-size="54" fill="#ffffff" font-weight="800">{v:,}</text>'
                  f'<text x="{x}" y="{y + 34}" text-anchor="middle" font-size="20" letter-spacing="2" fill="#8fa1af">{escape(k).upper()}</text>')
    shown = showcase[:3]
    sc = ""
    for i, b in enumerate(shown):
        cx = 800 + (i - (len(shown) - 1) / 2) * 340
        sc += (_medal(int(cx), 950, 42, b["metal"], bool(b.get("rare")), "★") +
               f'<text x="{int(cx)}" y="1022" text-anchor="middle" font-size="21" fill="#ffffff" font-weight="700">{escape(b["title"])}</text>')
    metals = sorted({metal} | {b["metal"] for b in shown})
    defs = BG_DEF + "".join(_defs(mm) for mm in metals)
    title_line = f'<text x="800" y="598" text-anchor="middle" font-size="24" letter-spacing="4" fill="{m}" font-weight="700">{escape(title.upper())}</text>' if title else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1131" viewBox="0 0 1600 1131" font-family="Georgia, 'Times New Roman', serif">
<defs>{defs}</defs>
<rect width="1600" height="1131" fill="url(#bg)"/>
<rect x="34" y="34" width="1532" height="1063" fill="none" stroke="url(#mt-{metal})" stroke-width="6"/>
<rect x="54" y="54" width="1492" height="1023" fill="none" stroke="{m}" stroke-opacity=".45" stroke-width="1.5"/>
<g transform="translate(0,-20)">{_medal(800, 200, 62, metal, True)}</g>
<text x="800" y="330" text-anchor="middle" font-size="26" letter-spacing="9" fill="{m}" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-weight="700">T58 QUANT ALGO BACKTESTER</text>
<text x="800" y="410" text-anchor="middle" font-size="68" fill="#ffffff" font-weight="700">Certificate of Research</text>
<text x="800" y="458" text-anchor="middle" font-size="24" fill="#8fa1af" font-style="italic">awarded in recognition of tested ideas, honest validation and steady work</text>
<text x="800" y="548" text-anchor="middle" font-size="60" fill="url(#mt-{metal})" font-weight="700">{escape(who)}</text>
{title_line}
<text x="800" y="648" text-anchor="middle" font-size="22" fill="#aebbc6" font-family="Segoe UI, Helvetica, Arial, sans-serif">{escape(rank['metal_label'])} rank · {escape(level['title'])} · level {level['level']} · {level['xp']:,} XP · {unlocked} of {total} badges</text>
<g font-family="Segoe UI, Helvetica, Arial, sans-serif">{cells}</g>
<g font-family="Segoe UI, Helvetica, Arial, sans-serif">{sc}</g>
<text x="120" y="1060" font-size="19" fill="#8fa1af" font-family="Segoe UI, Helvetica, Arial, sans-serif">Issued {today}</text>
<text x="120" y="1082" font-size="17" fill="#6f808d" font-family="Segoe UI, Helvetica, Arial, sans-serif">Certificate {cid}</text>
<text x="1480" y="1082" text-anchor="end" font-size="16" fill="#6f808d" font-family="Segoe UI, Helvetica, Arial, sans-serif">Counts tests run and checks completed in the app. Not a performance record or a financial credential.</text>
</svg>'''
