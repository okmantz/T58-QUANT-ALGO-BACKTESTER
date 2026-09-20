"""
Auto Ensemble Builder -- turns a pile of already-scored candidates (Search
Lab's ResultsDB leaderboard, Evolution Lab's leaderboard, or the Strategy
Graveyard's survivors) into a basket of 3-5 WEAKLY-CORRELATED legs,
automatically, instead of asking Owen to hand-pick strategies for
app.ensemble.ensemble.run_ensemble_blend himself.

Why this exists: app.evolution.prop_fitness's PROP FITNESS formula (and
Search Lab's composite_score) already reward high pass/payout probability,
low drawdown, and OOS consistency -- but they score EVERY candidate in
isolation. Pushing one single strategy's own parameters harder and harder
to satisfy every gate at once (significance, robustness, OOS, PBO) is
fighting the multiple-testing correction, not the market. Several
imperfectly-correlated, individually-modest edges combined into one
account smooth the combined equity curve and raise the ACCOUNT's real
eval-pass probability far more reliably -- that's exactly what
app.portfolio.portfolio's correlation-aware combine step already proves
for multi-instrument portfolios; this module is the automatic version of
the same idea for multi-STRATEGY (see app.ensemble.ensemble's own
docstring), applied to whatever a search run already found.

Pipeline:
  1. Filter candidates -- must carry a runnable source_type + config/
     code_text, and (if `min_individual_score` is set) a composite_score
     at or above it. Anything else is rejected with a stated reason.
  2. Cap how many legs may come from the SAME classified family
     (app.search.family_diversity.enforce_family_diversity) -- this is
     the anti-"5 near-identical RSI variants" guard, applied BEFORE any
     correlation math runs.
  3. Backtest every survivor once (app.backtest.engine.run_backtest) to
     get its own daily-return series.
  4. Greedily grow a basket: start with the single highest-scoring
     survivor, then repeatedly add whichever remaining candidate has the
     LOWEST average correlation with everyone already in the basket,
     stopping once `max_legs` is reached or no remaining candidate keeps
     every pairwise correlation under `max_pairwise_correlation` (but
     never stopping before `min_legs`, if enough candidates exist at all
     -- past that floor, correlation quality wins over basket size).
  5. Run the chosen basket through app.ensemble.ensemble.run_ensemble_blend
     (which itself reuses app.portfolio.portfolio.run_portfolio_backtest
     UNCHANGED) to get the real, combined-account Monte Carlo eval-pass /
     first-payout probability -- not an estimate, the same number every
     other tool in this app reports for a portfolio.

This module never invents a new correlation metric or a new Monte Carlo
path -- every number in the result comes from code this app already
trusts elsewhere.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.ensemble.ensemble import EnsembleError, run_ensemble_blend
from app.monte_carlo.engine import MonteCarloConfig
from app.portfolio.portfolio import PortfolioConfig, PortfolioResult, _daily_returns, _rebuild_equity_curve
from app.prop.simulator import PropRules
from app.search.family_diversity import enforce_family_diversity
from app.strategy.base import Strategy, StrategyError
from app.strategy.family_taxonomy import classify_record
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy


class AutoEnsembleError(Exception):
    """Raised for a caller error (bad config), not for "not enough good
    candidates" -- that is a normal outcome, reported via
    AutoEnsembleResult.status, not an exception."""


DEFAULT_MIN_LEGS = 3
DEFAULT_MAX_LEGS = 5
DEFAULT_MAX_PER_FAMILY = 2
DEFAULT_MAX_PAIRWISE_CORRELATION = 0.35


def _leg_name(record: dict) -> str:
    family = record.get("family") or "unknown_family"
    cid = str(record.get("candidate_id") or record.get("id") or "?")
    return f"{family}_{cid[:8]}"


def strategy_from_record(record: dict, _tmp_paths: list[str] | None = None) -> Strategy:
    """Reconstructs a runnable Strategy directly from a candidate record's
    own `config` (manual) or `code_text` (python/pinescript/mql5) --
    deliberately independent of app.strategy.library_loader, since a
    Search Lab / Evolution Lab candidate is very often NOT (yet) saved to
    the Strategy Library. `_tmp_paths`, if given, collects any temp file
    this creates (only needed for Python, whose Strategy subclass reads
    its source from a real file path lazily on every `.generate()` call)
    so the caller can clean them up once the ensemble is fully built."""
    source_type = (record.get("source_type") or "manual").lower()
    if source_type == "manual":
        config = record.get("config")
        if not isinstance(config, dict):
            raise AutoEnsembleError(
                f"Candidate {record.get('candidate_id', '?')!r} is source_type='manual' but has no usable 'config'."
            )
        return ManualStrategy(config)

    code_text = record.get("code_text")
    if not code_text:
        raise AutoEnsembleError(
            f"Candidate {record.get('candidate_id', '?')!r} (source_type={source_type!r}) has no 'code_text'."
        )
    if source_type == "pinescript":
        return PineScriptStrategy(code_text)
    if source_type == "mql5":
        return MQL5Strategy(code_text)
    if source_type == "python":
        fd, path = tempfile.mkstemp(suffix=".py", prefix="t58_ensemble_leg_")
        with os.fdopen(fd, "w") as f:
            f.write(code_text)
        if _tmp_paths is not None:
            _tmp_paths.append(path)
        return PythonStrategy(path)
    raise AutoEnsembleError(f"Candidate {record.get('candidate_id', '?')!r} has unsupported source_type={source_type!r}.")


@dataclass
class _Leg:
    record: dict
    name: str
    family: str
    strategy: Strategy
    composite_score: float
    daily_returns: pd.Series


@dataclass
class RejectedCandidate:
    candidate_id: str
    reason: str


@dataclass
class AutoEnsembleResult:
    status: str  # "built" | "insufficient_candidates"
    basket_names: list[str] = field(default_factory=list)
    basket_families: list[str] = field(default_factory=list)
    rejected: list[RejectedCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    portfolio_result: PortfolioResult | None = None

    def to_summary_dict(self) -> dict:
        out = {
            "status": self.status,
            "basket_names": list(self.basket_names),
            "basket_families": list(self.basket_families),
            "rejected": [{"candidate_id": r.candidate_id, "reason": r.reason} for r in self.rejected],
            "notes": list(self.notes),
        }
        if self.portfolio_result is not None:
            out["portfolio"] = self.portfolio_result.to_summary_dict()
        return out


def _score_of(record: dict) -> float:
    for key in ("composite_score", "fitness", "quick_score"):
        val = record.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return float("-inf")


def _avg_abs_correlation(candidate: _Leg, basket: list[_Leg]) -> float:
    corrs = []
    for member in basket:
        aligned = pd.concat([candidate.daily_returns, member.daily_returns], axis=1).dropna()
        if len(aligned) < 5:
            # Too little overlap to trust a correlation estimate -- treat
            # as fully uncorrelated rather than silently NaN-ing out of
            # the running average (an unhelpful bias toward candidates
            # with the LEAST data overlap, which is the opposite of safe).
            corrs.append(0.0)
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        corrs.append(0.0 if pd.isna(corr) else abs(corr))
    return sum(corrs) / len(corrs) if corrs else 0.0


def build_diversified_ensemble(
    df: pd.DataFrame,
    records: list[dict],
    risk: RiskConfig,
    prop_rules: PropRules | None = None,
    mc_config: MonteCarloConfig | None = None,
    min_legs: int = DEFAULT_MIN_LEGS,
    max_legs: int = DEFAULT_MAX_LEGS,
    max_per_family: int = DEFAULT_MAX_PER_FAMILY,
    max_pairwise_correlation: float = DEFAULT_MAX_PAIRWISE_CORRELATION,
    min_individual_score: float | None = None,
    initial_balance: float = 100_000.0,
) -> AutoEnsembleResult:
    """See module docstring for the full pipeline. `records` is any list
    of plain dicts shaped like app.search.results_db's candidate rows
    (`candidate_id`/`family`/`source_type`/`config`/`code_text`/
    `composite_score` -- see app.search.budget_allocator.
    normalize_evolution_record for turning an Evolution Lab checkpoint
    dict into this same shape). All legs are backtested against the SAME
    `df` -- pull `records` from a single instrument/timeframe's own
    search results, not a pooled cross-instrument leaderboard (a leg
    needs real trade history on the instrument it's about to be combined
    on)."""
    if min_legs < 2:
        raise AutoEnsembleError("min_legs must be at least 2 (an ensemble needs >=2 legs by definition).")
    if max_legs < min_legs:
        raise AutoEnsembleError("max_legs must be >= min_legs.")

    rejected: list[RejectedCandidate] = []
    notes: list[str] = []
    tmp_paths: list[str] = []

    try:
        # -- Step 1: filter to runnable, sufficiently-scored candidates --
        usable: list[dict] = []
        for rec in records:
            cid = str(rec.get("candidate_id", "?"))
            if not (rec.get("config") or rec.get("code_text")):
                rejected.append(RejectedCandidate(cid, "no runnable config/code_text on this record"))
                continue
            score = _score_of(rec)
            if min_individual_score is not None and score < min_individual_score:
                rejected.append(RejectedCandidate(cid, f"composite score {score:.3f} below min_individual_score {min_individual_score:.3f}"))
                continue
            usable.append(rec)

        # -- Step 2: cap legs per family BEFORE spending any backtest time --
        kept, dropped = enforce_family_diversity(usable, max_per_family=max_per_family, score_key="composite_score")
        for rec in dropped:
            rejected.append(RejectedCandidate(
                str(rec.get("candidate_id", "?")),
                f"family {classify_record(rec)!r} already has {max_per_family} higher-scoring leg(s)",
            ))

        # -- Step 3: backtest every survivor once to get its own return series --
        legs: list[_Leg] = []
        for rec in kept:
            cid = str(rec.get("candidate_id", "?"))
            try:
                strategy = strategy_from_record(rec, _tmp_paths=tmp_paths)
                bt = run_backtest(df, strategy, risk)
            except (AutoEnsembleError, StrategyError) as exc:
                rejected.append(RejectedCandidate(cid, f"could not run: {exc}"))
                continue
            if not bt.trades:
                rejected.append(RejectedCandidate(cid, "produced zero trades on this dataset"))
                continue
            equity = _rebuild_equity_curve(bt.trades, initial_balance)
            returns = _daily_returns(equity)
            legs.append(_Leg(
                record=rec, name=_leg_name(rec), family=classify_record(rec),
                strategy=strategy, composite_score=_score_of(rec), daily_returns=returns,
            ))

        if len(legs) < min_legs:
            notes.append(
                f"Only {len(legs)} candidate(s) survived filtering/backtesting -- need at least {min_legs} "
                "diversified legs to build an ensemble. Run a wider search (see "
                "app.search.budget_allocator) or relax min_individual_score/max_per_family."
            )
            return AutoEnsembleResult(status="insufficient_candidates", rejected=rejected, notes=notes)

        # -- Step 4: greedy correlation-diversified basket construction --
        legs.sort(key=lambda l: l.composite_score, reverse=True)
        basket = [legs.pop(0)]
        while legs and len(basket) < max_legs:
            scored = [(candidate, _avg_abs_correlation(candidate, basket)) for candidate in legs]
            scored.sort(key=lambda pair: pair[1])
            best_candidate, best_corr = scored[0]
            if best_corr > max_pairwise_correlation and len(basket) >= min_legs:
                notes.append(
                    f"Stopped at {len(basket)} legs: the next-best candidate's average correlation "
                    f"with the basket ({best_corr:.2f}) exceeds max_pairwise_correlation "
                    f"({max_pairwise_correlation:.2f})."
                )
                break
            if best_corr > max_pairwise_correlation:
                notes.append(
                    f"Added a leg above max_pairwise_correlation (avg corr {best_corr:.2f}) to reach "
                    f"the min_legs floor of {min_legs} -- this basket is less diversified than requested."
                )
            legs.remove(best_candidate)
            basket.append(best_candidate)

        if len(basket) < min_legs:
            notes.append(f"Only able to build a {len(basket)}-leg basket; need at least {min_legs}.")
            return AutoEnsembleResult(status="insufficient_candidates", rejected=rejected, notes=notes)

        # -- Step 5: run the real, combined-account backtest + Monte Carlo --
        names = [leg.name for leg in basket]
        strategies = [leg.strategy for leg in basket]
        try:
            portfolio_result = run_ensemble_blend(
                df, strategies, risk, names=names,
                config=PortfolioConfig(initial_balance=initial_balance, prop_rules=prop_rules, mc_config=mc_config),
            )
        except EnsembleError as exc:
            raise AutoEnsembleError(f"Basket selection produced an invalid ensemble: {exc}") from exc

        return AutoEnsembleResult(
            status="built",
            basket_names=names,
            basket_families=[leg.family for leg in basket],
            rejected=rejected,
            notes=notes,
            portfolio_result=portfolio_result,
        )
    finally:
        for path in tmp_paths:
            try:
                os.remove(path)
            except OSError:
                pass
