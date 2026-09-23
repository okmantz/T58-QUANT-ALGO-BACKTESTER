"""
Alternate per-family optimizers for the Evolution Lab's "Optimizer mode
(per-family child proposals)" setting.

BACKGROUND / BUG THIS FIXES: EvolutionConfig.optimizer_mode has existed
since the web form's "Optimizer mode" dropdown (OPTIMIZER_MODES in
app.optimize.refinement) was wired into every OTHER consumer of that same
dropdown (Search Lab / Quick Optimize / Full Pipeline / Walk-Forward GA,
all in app.optimize.refinement) -- but this engine's own child-proposal
step never branched on it. Selecting "tpe" or "cma_es" here silently ran
the exact same plain-mutation/GP-surrogate path as "genetic", with zero
error and zero indication anything was wrong. This module is the fix:
TPEFamilyBank and CMAESFamilyBank below give EvolutionRunner a real,
working TPE / CMA-ES per-family child proposer.

Both classes implement the EXACT SAME interface as
app.evolution.surrogate.FamilySurrogateBank --
    .propose(family, n, rng, n_genes, pool_size=...) -> np.ndarray | None
    .observe(family, genome_norm, fitness) -> None
    .n_observations(family) -> int
-- so EvolutionRunner.__init__ only needs to choose WHICH bank to build
for self._surrogate based on cfg.optimizer_mode; every call site in
app/evolution/engine.py (_generate_population's propose() call,
_record_surrogate_observations' observe() call) is untouched, exactly as
every one of that dropdown's other consumers already promises: "same
generation-by-generation search per family -- only how each family's
next children are proposed differs."

Both work in the same normalized [0, 1]^d genome space
FamilySurrogateBank already uses. The tricky part neither optuna's nor
cma's native ask/tell API solves for free: propose() hands out genomes
BEFORE the resulting candidates are backtested, and observe() is only
called afterwards, for whichever of those candidates actually survive
pre-filter and reach full evaluation -- some proposed genomes may never
be told about at all (killed at pre-filter), and int-valued genes go
through apply_genome's rounding, so the genome_norm recomputed in
observe() is never bit-for-bit identical to what propose() handed out
(the same is already true of FamilySurrogateBank's own observations).
Both banks below account for this defensively: matching by NEAREST
pending genome rather than exact equality, and force-closing/discarding
stale pending state after a bounded number of rounds rather than ever
blocking or crashing a run over an optimizer library's internal
bookkeeping. An optimizer proposal must only ever make the search
smarter -- never required, never load-bearing, exactly like the GP
surrogate it stands in for. Any internal failure returns None from
propose() (falls back to plain mutation for that family, that round,
identical to a GP-surrogate cold start) rather than raising.
"""
from __future__ import annotations

import math

import numpy as np


class TPEFamilyBank:
    """Optuna TPE sampler, one Study per family, in normalized [0,1]^d
    genome space. Requires the optional `optuna` package -- raised at
    construction (not at first use) so a run that picks optimizer_mode
    ="tpe" fails fast, with a clear message, before spending any time on
    generation 0, instead of silently behaving like plain mutation the
    way this whole module exists to stop happening."""

    def __init__(self, seed: int | None = None, max_pending_per_family: int = 300):
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError(
                "optimizer_mode='tpe' requires the optuna package (pip install optuna) -- "
                "it isn't a hard dependency of the rest of this app, only of TPE search."
            ) from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._optuna = optuna
        self._seed = seed
        self._max_pending = max(int(max_pending_per_family), 1)
        self._studies: dict = {}
        # family -> list[(trial, genome_norm ndarray)], asked but not yet told
        self._pending: dict = {}

    def _study(self, family: str):
        study = self._studies.get(family)
        if study is None:
            sampler = self._optuna.samplers.TPESampler(seed=self._seed)
            study = self._optuna.create_study(direction="maximize", sampler=sampler)
            self._studies[family] = study
        return study

    def n_observations(self, family: str) -> int:
        study = self._studies.get(family)
        if study is None:
            return 0
        return sum(1 for t in study.trials if t.state.is_finished())

    def observe(self, family: str, genome_norm: np.ndarray, fitness: float) -> None:
        if not math.isfinite(fitness):
            return
        pending = self._pending.get(family)
        if not pending:
            # Nothing this bank proposed is waiting to be told about --
            # e.g. this candidate came from a random immigrant or plain
            # mutation slot, not from propose() below. Nothing to do.
            return
        best_i, best_d = None, None
        for i, (_trial, x) in enumerate(pending):
            d = float(np.sum((x - genome_norm) ** 2))
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        trial, _x = pending.pop(best_i)
        try:
            self._study(family).tell(trial, float(fitness))
        except Exception:  # noqa: BLE001 -- a stale/duplicate tell must never break a run
            pass

    def propose(self, family: str, n: int, rng: np.random.Generator, n_genes: int, pool_size: int = 300) -> np.ndarray | None:
        try:
            study = self._study(family)
            pending = self._pending.setdefault(family, [])
            out = []
            for _ in range(max(int(n), 0)):
                trial = study.ask()
                genome = np.array([trial.suggest_float(f"g{i}", 0.0, 1.0) for i in range(n_genes)])
                pending.append((trial, genome))
                out.append(genome)
            if not out:
                return None
            # Bound memory / orphaned trials: candidates that died before
            # ever reaching observe() (pre-filter rejects) would otherwise
            # accumulate here forever. Tell the oldest a deliberately poor
            # score once the pending queue gets long, so optuna's study
            # stays finite and those trials stop being "RUNNING" forever.
            if len(pending) > self._max_pending:
                stale = pending[: len(pending) - self._max_pending]
                del pending[: len(pending) - self._max_pending]
                for trial, _x in stale:
                    try:
                        study.tell(trial, -1e6)
                    except Exception:  # noqa: BLE001
                        pass
            return np.array(out)
        except Exception:  # noqa: BLE001 -- an optimizer hiccup must fall back to mutation, never crash the run
            return None


class CMAESFamilyBank:
    """CMA-ES, one CMAEvolutionStrategy per family, in normalized [0,1]^d
    genome space -- same normalize-by-span approach and the same
    isolated-RandomState "randn" reproducibility fix already used by
    app.optimize.refinement._run_cma_es_search (see that function's own
    comment for why plain `seed=` alone is not reproducible here).

    CMA-ES's ask()/tell() must be called in matched batches: propose()
    asks exactly `n` solutions and opens a "batch" for that family;
    observe() fills that batch in as matching candidates report back;
    once every solution in the batch has been told about (or the batch
    has gone stale -- see _STALE_ROUNDS_LIMIT), the batch closes with a
    single es.tell() call and the family becomes askable again. While a
    family's batch is still open, propose() returns None for it (falls
    back to plain mutation for that family that round) rather than
    double-asking the same CMA-ES instance out of turn."""

    _STALE_ROUNDS_LIMIT = 3  # propose() calls a still-open batch survives before being force-closed

    def __init__(self, seed: int | None = None):
        try:
            import cma
        except ImportError as exc:
            raise RuntimeError(
                "optimizer_mode='cma_es' requires the cma package (pip install cma) -- "
                "it isn't a hard dependency of the rest of this app, only of CMA-ES search."
            ) from exc
        self._cma = cma
        self._seed = seed
        self._es: dict = {}
        self._n_genes: dict = {}
        # family -> {"solutions": [...], "collected": {idx: penalty}, "stale_rounds": int}
        self._open_batch: dict = {}

    def _ensure_es(self, family: str, n_genes: int):
        es = self._es.get(family)
        if es is None or self._n_genes.get(family) != n_genes:
            rs = np.random.RandomState(self._seed if self._seed is not None else 0)
            es = self._cma.CMAEvolutionStrategy(
                [0.5] * n_genes, 0.3,
                {"bounds": [0.0, 1.0], "verbose": -9, "randn": rs.randn},
            )
            self._es[family] = es
            self._n_genes[family] = n_genes
            self._open_batch.pop(family, None)
        return es

    def n_observations(self, family: str) -> int:
        es = self._es.get(family)
        if es is None:
            return 0
        return int(getattr(es, "countiter", 0)) * int(getattr(es, "popsize", 0) or 0)

    def _close_batch(self, family: str) -> None:
        batch = self._open_batch.pop(family, None)
        if not batch:
            return
        solutions = batch["solutions"]
        collected = batch["collected"]
        if not collected:
            return  # nothing at all reported back -- discard the batch rather than telling cma zero information
        # Any solution nothing ever reported back for (killed at
        # pre-filter, or the batch was force-closed early) gets the
        # worst observed penalty in this batch rather than being left
        # out of the tell() call entirely -- cma requires one value per
        # asked solution.
        worst = max(collected.values())
        penalties = [collected.get(i, worst) for i in range(len(solutions))]
        try:
            self._es[family].tell(solutions, penalties)
        except Exception:  # noqa: BLE001
            pass

    def observe(self, family: str, genome_norm: np.ndarray, fitness: float) -> None:
        batch = self._open_batch.get(family)
        if not batch:
            return
        penalty = -float(fitness) if math.isfinite(fitness) else 1e12  # cma minimizes; non-finite -> large-but-finite
        solutions = batch["solutions"]
        collected = batch["collected"]
        best_i, best_d = None, None
        for i, sol in enumerate(solutions):
            if i in collected:
                continue
            d = float(np.sum((np.asarray(sol) - genome_norm) ** 2))
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        if best_i is None:
            return
        collected[best_i] = penalty
        if len(collected) >= len(solutions):
            self._close_batch(family)

    def propose(self, family: str, n: int, rng: np.random.Generator, n_genes: int, pool_size: int = 300) -> np.ndarray | None:
        n = max(int(n), 0)
        if n == 0:
            return None
        try:
            batch = self._open_batch.get(family)
            if batch is not None:
                batch["stale_rounds"] = batch.get("stale_rounds", 0) + 1
                if batch["stale_rounds"] >= self._STALE_ROUNDS_LIMIT:
                    # Enough candidates from this batch died before
                    # reaching observe() that it will never close on its
                    # own -- force it closed (padding missing entries
                    # with this batch's worst-seen penalty, or dropping
                    # it entirely if NOTHING from it ever reported back)
                    # so this family doesn't get stuck on plain mutation
                    # for the rest of the run.
                    self._close_batch(family)
                else:
                    return None  # still waiting on this batch -- fall back to mutation for this family this round
            es = self._ensure_es(family, n_genes)
            solutions = es.ask(number=n)
            self._open_batch[family] = {"solutions": solutions, "collected": {}, "stale_rounds": 0}
            return np.array([np.clip(np.asarray(s, dtype=float), 0.0, 1.0) for s in solutions])
        except Exception:  # noqa: BLE001 -- an optimizer hiccup must fall back to mutation, never crash the run
            self._open_batch.pop(family, None)
            return None


def build_family_bank(optimizer_mode: str, *, seed: int | None, min_observations: int, kappa: float):
    """Single entry point EvolutionRunner.__init__ uses to build whichever
    per-family proposer cfg.optimizer_mode asks for -- keeps the
    "genetic"/"tpe"/"cma_es" choice in one place rather than duplicated
    at every call site. Raises RuntimeError (with a clear, actionable
    message) for "tpe"/"cma_es" if the optional package isn't installed,
    same as every other OPTIMIZER_MODES consumer in
    app.optimize.refinement -- fails at run start, not silently later."""
    from app.evolution.surrogate import FamilySurrogateBank

    if optimizer_mode == "tpe":
        return TPEFamilyBank(seed=seed)
    if optimizer_mode == "cma_es":
        return CMAESFamilyBank(seed=seed)
    return FamilySurrogateBank(min_observations=min_observations, kappa=kappa)
