"""Theaters: the handful of geopolitical / supply-shock regions that actually move futures, FX and energy.

Each theater has a keyword list (for headlines), an optional bounding box (for map events), and the instruments it
tends to move. A theater is a *lens*, not a prediction -- the Theater Heat score (see ``heat``) is a transparent,
uncalibrated activity measure and is labelled experimental in the UI.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Theater:
    id: str
    name: str
    short: str
    keywords: tuple[str, ...]
    instruments: tuple[str, ...]
    bbox: tuple[float, float, float, float] | None = None   # lon_min, lat_min, lon_max, lat_max
    label_at: tuple[float, float] | None = None             # lon, lat for the map label
    extra: dict = field(default_factory=dict)

    @property
    def area(self) -> float:
        if not self.bbox:
            return math.inf
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


CALIBRATION_DAYS = 4   # distinct days of stored headlines needed before the 14-day baseline is trusted

THEATERS: tuple[Theater, ...] = (
    Theater("mideast", "Middle East / Gulf / Hormuz", "Gulf",
            ("iran", "israel", "gaza", "hormuz", "houthi", "tanker", "hezbollah", "lebanon", "saudi", "persian gulf", "strike on", "missile"),
            ("CL", "BZ", "GC", "ES"), (44.0, 20.0, 63.0, 38.0), (53.5, 34.5)),
    Theater("ukraine", "Russia-Ukraine / Black Sea", "Ukraine",
            ("ukraine", "russia", "kyiv", "black sea", "grain corridor", "nord stream", "kremlin", "zelensky", "putin"),
            ("NG", "ZW", "EUR/USD", "GC"), (22.0, 43.0, 50.0, 56.0), (34.0, 49.5)),
    Theater("taiwan", "Taiwan Strait / South China Sea", "Taiwan",
            ("taiwan", "pla ", "blockade", "south china sea", "export controls", "tsmc", "strait", "semiconductor"),
            ("NQ", "SOX", "USD/JPY"), (108.0, 5.0, 126.0, 28.0), (118.0, 22.0)),
    Theater("redsea", "Red Sea / Suez / Shipping", "Red Sea",
            ("red sea", "suez", "bab el-mandeb", "port closure", "freight rates", "container", "shipping lane", "panama canal"),
            ("CL", "BZ"), (32.0, 12.0, 44.0, 30.0), (38.0, 8.0)),
    Theater("trade", "US Trade / Tariffs / Sanctions", "Trade",
            ("tariff", "section 301", "sanction", "ofac", "trade war", "export ban", "retaliat", "customs duty"),
            ("ES", "NQ", "DXY", "USD/CNH")),
    Theater("korea", "Korea / North Asia", "Korea",
            ("north korea", "missile launch", "dmz", "pyongyang", "kim jong", "south korea"),
            ("USD/JPY", "KOSPI"), (124.0, 33.0, 132.0, 43.0), (128.0, 38.0)),
    Theater("hazard", "Natural-hazard supply shocks", "Hazards",
            ("earthquake", "typhoon", "hurricane", "refinery fire", "volcano", "flood", "wildfire", "power outage", "blackout"),
            ("NQ", "CL", "NG")),
)
THEATER_BY_ID = {t.id: t for t in THEATERS}

# Places whose disruption matters to markets (fabs, refineries, chokepoints, ports): (lon, lat, radius_deg).
HAZARD_HUBS: tuple[tuple[float, float, float], ...] = (
    (121.0, 24.0, 4.0),     # Taiwan fabs
    (139.7, 35.7, 4.0),     # Japan / Tokyo industrial belt
    (127.0, 37.0, 3.0),     # Korea fabs
    (-95.0, 29.0, 4.0),     # US Gulf Coast refining
    (50.0, 27.0, 5.0),      # Persian Gulf oil
    (32.5, 30.0, 3.0),      # Suez
    (104.0, 1.3, 3.0),      # Singapore / Malacca
    (121.5, 31.2, 4.0),     # Shanghai / Yangtze delta
)


def theater_for_point(lon: float | None, lat: float | None) -> str | None:
    """Smallest theater bounding box containing the point (so the Red Sea wins over the wider Gulf box)."""
    if lon is None or lat is None:
        return None
    best: Theater | None = None
    for t in THEATERS:
        if not t.bbox:
            continue
        x0, y0, x1, y1 = t.bbox
        if x0 <= lon <= x1 and y0 <= lat <= y1 and (best is None or t.area < best.area):
            best = t
    return best.id if best else None


def near_hazard_hub(lon: float | None, lat: float | None) -> bool:
    if lon is None or lat is None:
        return False
    return any(abs(lon - x) <= r and abs(lat - y) <= r for x, y, r in HAZARD_HUBS)


@lru_cache(maxsize=512)
def _pattern(words: tuple[str, ...]) -> re.Pattern:
    """Word-boundary matcher: a leading boundary always (so "tariff" matches "tariffs"), a trailing one for short words
    (so "ppi" never matches inside "shipping" and "oil" never inside "boil")."""
    parts = []
    for w in words:
        w = re.escape(w.strip().lower())
        parts.append(r"\b" + w + (r"\b" if len(w.replace("\\", "")) <= 4 else ""))
    return re.compile("|".join(parts))


def has_any(normalized_text: str, words: tuple[str, ...]) -> bool:
    return bool(_pattern(tuple(words)).search(normalized_text))


def theaters_for_text(text: str) -> list[str]:
    low = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return [t.id for t in THEATERS if t.keywords and has_any(low, t.keywords)]


def heat(tier_weighted_24h: float, mean_14d: float, std_14d: float, hi_6h: int, hazard_24h: int, calibrating: bool = False) -> tuple[float, dict]:
    """Transparent v1 Theater Heat, 0..100 (uncalibrated -- shown with its components, labelled experimental).

        vol_z = (tier_weighted_hits_24h - mean_14d) / max(std_14d, 0.5)
        score = clamp(40 + 12*vol_z + 6*min(hi_6h, 5) + 5*min(hazard_24h, 3), 0, 100)
    """
    z = (tier_weighted_24h - mean_14d) / max(std_14d, 0.5)
    z = 0.0 if calibrating else max(-3.0, min(6.0, z))   # fewer than CALIBRATION_DAYS of history: no baseline yet, so no z-score
    hi = min(int(hi_6h), 5)
    hz = min(int(hazard_24h), 3)
    score = max(0.0, min(100.0, 40.0 + 12.0 * z + 6.0 * hi + 5.0 * hz))
    return round(score, 1), {"vol_z": round(z, 2), "hits_24h": round(tier_weighted_24h, 2), "hi_impact_6h": hi, "hazard_24h": hz, "calibrating": bool(calibrating)}
