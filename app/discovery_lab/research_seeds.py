"""
Research seeds -- lets the Discovery Lab start from ideas in the Resource Library papers (and from the AI, when it is switched on)
instead of only blind random search.

How it stays honest: a seed is only a *starting point*. Every seed is evaluated exactly like any randomly drawn idea (train / validate /
sealed test / neighbour / double-cost / real-engine proof) and counts toward the "ideas tried" luck bar, so a paper cannot smuggle
in an unproven winner. Seeds just make sure the families the papers talk about get looked at first and bred from.

* Papers: keyword-scored against the papers in the top-level ``research/`` folder (no AI needed).
* AI (optional): when a language model is usable (local Ollama or a stored hosted key) it is shown short paper excerpts plus the list of
  strategy types and returns JSON picks. Anything outside the real parameter grids is discarded.
"""
from __future__ import annotations

import json
import re

# strategy type -> words/phrases that point at it in a paper
FAMILY_HINTS: dict[str, tuple[str, ...]] = {
    "ema_cross": ("moving average crossover", "ma crossover", "moving average", "trend following", "trend-following", "time series momentum"),
    "supertrend": ("supertrend", "atr trailing", "trailing stop"),
    "adx_di": ("adx", "directional movement", "trend strength"),
    "macd": ("macd",),
    "donchian": ("donchian", "channel breakout", "breakout", "turtle", "n-day high"),
    "keltner": ("keltner", "volatility channel", "atr channel"),
    "squeeze": ("squeeze", "volatility compression", "volatility contraction", "bollinger bandwidth"),
    "inside_bar": ("inside bar", "inside day"),
    "orb": ("opening range", "orb "),
    "rsi_rev": ("rsi", "relative strength index", "oversold", "short-term reversal"),
    "boll_rev": ("bollinger", "mean reversion", "mean-reversion"),
    "zscore_rev": ("z-score", "zscore", "standard deviations from", "statistical arbitrage", "pairs trading"),
    "stoch_rev": ("stochastic",),
    "cci_rev": ("commodity channel",),
    "willr": ("williams %r", "williams r"),
    "failed_break": ("false breakout", "failed breakout", "fade the breakout"),
    "ibs_rev": ("internal bar strength", "ibs", "close location", "close-to-range"),
    "streak_rev": ("consecutive", "streak", "down days in a row"),
    "low_rev": ("n-day low", "new lows", "capitulation"),
    "dist_ma_rev": ("distance from the moving average", "overextended", "stretched"),
    "body_bar": ("candlestick", "bar body", "momentum bar", "large bar"),
    "vol_expansion": ("volatility expansion", "volatility breakout", "range expansion", "volatility clustering"),
    "price_ma": ("price above", "200-day", "trend filter", "regime filter"),
    "nr_break": ("narrow range", "nr7", "nr4"),
    "tod_hold": ("time of day", "intraday seasonality", "overnight", "turn of the month", "calendar effect", "seasonality", "day-of-week", "weekend effect"),
    "gap_play": ("opening gap", "gap fill", "overnight gap", "gap"),
    "session_mom": ("intraday momentum", "first half-hour", "first 30 minutes", "last half-hour", "session momentum"),
    "pullback": ("pullback", "buy the dip", "retracement", "dip"),
    "ind_rule": ("volume", "order flow", "volatility regime", "regime", "indicator"),
    "momentum": ("momentum", "cross-sectional momentum", "rate of change"),
}

_cache: dict = {"sig": None, "scores": None}


def _paper_scores() -> tuple[dict, int, dict]:
    """(family -> weighted hit count, number of chunks, family -> best excerpt)."""
    from app.ai.research_library import build_index
    chunks, _stats = build_index()
    sig = (len(chunks), sum(len(c.text) for c in chunks[:50]))
    if _cache["sig"] == sig and _cache["scores"] is not None:
        return _cache["scores"]
    scores = {f: 0.0 for f in FAMILY_HINTS}
    best: dict = {}
    for ch in chunks:
        low = ch.text.lower()
        for fam, hints in FAMILY_HINTS.items():
            hit = sum(low.count(h) for h in hints)
            if hit:
                scores[fam] += min(hit, 4)
                if fam not in best or hit > best[fam][0]:
                    best[fam] = (hit, ch.text[:240].replace("\n", " "))
    out = (scores, len(chunks), {k: v[1] for k, v in best.items()})
    _cache["sig"], _cache["scores"] = sig, out
    return out


def _spread(grid: dict, categorical: set, k: int) -> list[dict]:
    """k parameter sets spread across a family's grid (centre first, then the corners) -- a deterministic, unbiased starting spread."""
    keys = list(grid)
    sets: list[dict] = []
    def pick(frac):
        return {a: grid[a][min(len(grid[a]) - 1, int(round(frac * (len(grid[a]) - 1))))] for a in keys}
    for frac in (0.5, 0.0, 1.0, 0.25, 0.75):
        p = pick(frac)
        if p not in sets:
            sets.append(p)
        if len(sets) >= k:
            break
    return sets


def _valid(fam: str, params: dict) -> bool:
    from app.discovery_lab import families as F
    from app.discovery_lab.engine import _valid_params
    f = F.FAMILIES.get(fam)
    if f is None or not isinstance(params, dict) or set(params) != set(f.grid):
        return False
    for k, v in params.items():
        if v not in f.grid[k]:
            return False
    return _valid_params(fam, params)


def _candidate(fam: str, params: dict, source: str, why: str) -> dict:
    return {"family": fam, "params": params, "flt": {}, "stop_atr": None, "target_r": 2.0, "hold": 48, "_seed_source": source, "_seed_why": why}


def paper_seeds(max_families: int = 6, per_family: int = 3) -> tuple[list[dict], list[str]]:
    """Seed candidates from the Resource Library papers + log lines. Empty list when there are no papers."""
    from app.discovery_lab import families as F
    try:
        scores, n_chunks, excerpts = _paper_scores()
    except Exception as exc:  # noqa: BLE001 -- a bad paper must never stop a search
        return [], [f"Resource Library papers couldn't be read ({type(exc).__name__}); searching without them."]
    if not n_chunks:
        return [], ["Resource Library: no papers found in the research folder, so there is nothing to seed from."]
    ranked = [(s, f) for f, s in scores.items() if s > 0 and f in F.FAMILIES]
    ranked.sort(reverse=True)
    top = ranked[:max_families]
    if not top:
        return [], [f"Resource Library: scanned {n_chunks:,} excerpts, none pointed at a strategy type this Lab can test."]
    out: list[dict] = []
    for s, fam in top:
        f = F.FAMILIES[fam]
        for p in _spread(f.grid, set(getattr(f, "categorical", ()) or ()), per_family):
            if _valid(fam, p):
                out.append(_candidate(fam, p, "papers", f"{int(s)} paper mentions"))
    names = ", ".join(f for _, f in top)
    return out, [f"Resource Library: read {n_chunks:,} excerpts from your papers; starting the search with {len(out)} ideas from the strategy types they discuss most ({names})."]


def ai_seeds(max_picks: int = 8) -> tuple[list[dict], list[str]]:
    """Seed candidates proposed by the AI (only when one is usable). Never raises; returns ([], [note]) when AI is off."""
    try:
        from app.ai.llm_client import NullClient, preferred_client
        client = preferred_client()
    except Exception:  # noqa: BLE001
        return [], []
    if isinstance(client, NullClient):
        return [], ["AI assist is off, so the AI isn't proposing starting ideas (turn it on in Settings to add them)."]
    from app.discovery_lab import families as F
    try:
        scores, n_chunks, excerpts = _paper_scores()
    except Exception:  # noqa: BLE001
        scores, n_chunks, excerpts = {}, 0, {}
    catalogue = {k: {a: list(b) for a, b in v.grid.items()} for k, v in F.FAMILIES.items() if k != "ind_rule"}
    ctx_lines = [f"- {fam}: {txt}" for fam, txt in list(excerpts.items())[:8]]
    prompt = (
        "You help seed a systematic futures strategy search. Pick up to %d starting ideas from the catalogue below. Prefer strategy types "
        "supported by the research excerpts. Reply with ONLY a JSON list like "
        '[{"family":"donchian","params":{"n":30,"trend":100},"why":"short reason"}]. '
        "Use only family names and parameter values that appear in the catalogue.\n\nCATALOGUE:\n%s\n\nRESEARCH EXCERPTS:\n%s\n"
        % (max_picks, json.dumps(catalogue)[:6000], "\n".join(ctx_lines) or "(none)"))
    try:
        raw = client.complete(prompt, system="You output strict JSON only.")
    except Exception as exc:  # noqa: BLE001
        return [], [f"The AI couldn't be reached ({str(exc)[:80]}); continuing without AI ideas."]
    m = re.search(r"\[.*\]", raw or "", re.S)
    try:
        picks = json.loads(m.group(0)) if m else []
    except Exception:  # noqa: BLE001
        picks = []
    out = []
    for p in picks[:max_picks]:
        if isinstance(p, dict) and _valid(str(p.get("family")), p.get("params")):
            out.append(_candidate(p["family"], p["params"], "ai", str(p.get("why", ""))[:120]))
    if not out:
        return [], ["The AI replied, but none of its ideas matched the strategy catalogue; continuing without them."]
    return out, [f"AI assist proposed {len(out)} starting ideas ({', '.join(sorted({c['family'] for c in out}))})."]


def build_seeds(use_papers: bool = True, use_ai: bool = True) -> tuple[list[dict], list[str]]:
    seeds: list[dict] = []
    notes: list[str] = []
    for enabled, fn in ((use_papers, paper_seeds), (use_ai, ai_seeds)):
        if not enabled:
            continue
        s, n = fn()
        seeds += s
        notes += n
    return seeds, notes
