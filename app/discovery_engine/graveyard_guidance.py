"""Use the strategy graveyard to steer the search -- not just to skip exact repeats.

The labs already drop a candidate whose coarse parameter signature died ``graveyard_min_attempts`` (8) times. That is a cliff: 7
deaths cost nothing, 8 is a ban. This module turns the same record into a smooth, auditable prior that BOTH labs read:

  ``GraveyardGuidance.from_rows(rows)``
      per-family   deaths, near-misses (net positive at doubled costs, enough trades), median net R
      per-signature deaths
  ``penalty_r(family, config)``     graduated fitness penalty in R for sitting in a neighbourhood that keeps dying
                                    (0 below ``free_deaths``; grows linearly; capped so it ranks, never bans)
  ``family_weight(family)``         sampling weight for allocating Stage-1 / GA budget across families: families with a history of
                                    near-misses get MORE budget than families that only ever produced clear losers
  ``reorder(candidates)``           stable sort of a candidate list so well-prior families are screened first

It is a PRIOR for where to spend compute, never evidence of an edge: nothing here lifts a candidate over any gate. Every use is
logged by the caller with ``summary()``.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from app.search.graveyard import param_signature


@dataclass
class FamilyRecord:
    family: str
    deaths: int = 0
    near_misses: int = 0
    avg_r: list = field(default_factory=list)

    @property
    def median_r(self) -> float | None:
        return statistics.median(self.avg_r) if self.avg_r else None


@dataclass
class GraveyardGuidance:
    families: dict = field(default_factory=dict)       # name -> FamilyRecord
    signatures: dict = field(default_factory=dict)     # signature -> deaths
    free_deaths: int = 2
    per_death_r: float = 0.004
    max_penalty_r: float = 0.04
    n_rows: int = 0

    @classmethod
    def from_rows(cls, rows: list[dict] | None, *, min_near_miss_trades: int = 100, free_deaths: int = 2,
                  per_death_r: float = 0.004, max_penalty_r: float = 0.04) -> "GraveyardGuidance":
        g = cls(free_deaths=free_deaths, per_death_r=per_death_r, max_penalty_r=max_penalty_r)
        for r in rows or []:
            fam = r.get("family") or "unknown"
            rec = g.families.setdefault(fam, FamilyRecord(fam))
            rec.deaths += 1
            ar = r.get("doubled_cost_avg_r")
            ar = r.get("avg_r_net") if ar is None else ar
            if isinstance(ar, (int, float)) and math.isfinite(ar):
                rec.avg_r.append(float(ar))
                if ar > 0 and int(r.get("n_trades") or 0) >= min_near_miss_trades:
                    rec.near_misses += 1
            sig = r.get("param_signature") or fam
            g.signatures[sig] = g.signatures.get(sig, 0) + 1
            g.n_rows += 1
        return g

    @classmethod
    def load(cls, instrument: str, timeframe: str = "unknown", **kw) -> "GraveyardGuidance":
        """Read the instrument+timeframe graveyard (empty guidance when it cannot be read or read-back is disabled)."""
        try:
            from app.search.graveyard import graveyard_path_for, load_graveyard, readback_enabled
            if not readback_enabled():
                return cls(**{k: v for k, v in kw.items() if k in ("free_deaths", "per_death_r", "max_penalty_r")})
            return cls.from_rows(load_graveyard(graveyard_path_for(instrument, timeframe)), **kw)
        except Exception:  # noqa: BLE001 -- advisory only
            return cls()

    # ---------------------------------------------------------------- queries
    def deaths_in_neighbourhood(self, family: str, config: dict | None) -> int:
        return int(self.signatures.get(param_signature(family, config), 0))

    def penalty_r(self, family: str, config: dict | None) -> float:
        extra = max(0, self.deaths_in_neighbourhood(family, config) - self.free_deaths)
        return float(min(self.max_penalty_r, extra * self.per_death_r))

    def family_weight(self, family: str) -> float:
        """(near_misses + 1) / (deaths + 2) -- the Laplace-smoothed share of a family's graveyard rows that were near-misses.
        An unseen family gets 0.5 (no information). Always in (0, 1)."""
        rec = self.families.get(family)
        if rec is None:
            return 0.5
        return float((rec.near_misses + 1.0) / (rec.deaths + 2.0))

    def reorder(self, candidates: list, family_of=lambda c: c) -> list:
        """Stable sort, best-prior family first. ``family_of(item)`` extracts the family name."""
        return sorted(candidates, key=lambda c: -self.family_weight(family_of(c)))

    def summary(self, top: int = 3) -> str:
        if not self.n_rows:
            return "Graveyard guidance: no rows for this instrument/timeframe -- no prior applied."
        fam = sorted(self.families.values(), key=lambda r: (-r.near_misses, r.deaths))
        best = "; ".join(f"{r.family} ({r.near_misses} near-miss/{r.deaths})" for r in fam[:top])
        worst = "; ".join(f"{r.family} ({r.deaths} deaths, 0 near-miss)" for r in sorted(
            (r for r in self.families.values() if r.near_misses == 0), key=lambda r: -r.deaths)[:top])
        return f"Graveyard guidance from {self.n_rows:,} row(s): most promising prior {best or 'n/a'}; most-failed {worst or 'n/a'}."
