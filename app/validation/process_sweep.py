"""Process-pool evaluation of independent candidate backtests for the validation sweeps (2026-11-07, audit finding 2).

Walk-forward optimisation, the 1-D sensitivity sweeps, the 2-D heatmaps and parameter robustness all run many
INDEPENDENT backtests of one price frame. They used thread pools on the belief that ``run_backtest`` releases the GIL for a
meaningful share of its time. It does not: the bar loop is pure Python, so threads measured 0.92x of serial while processes
measured 1.91x on 2 cores (see tests/validation/test_process_sweep.py and scripts/bench_sweep_processes.py).

This module is the one place that ships those evaluations to worker processes, reusing the proven pattern of Search Lab /
the walk-forward GA (``app.orchestration.stage_pool``):

* the price frame is spilled to disk ONCE and each worker loads it itself, so nothing big is pickled per task and the
  Windows ``spawn`` build works exactly like fork would;
* tasks carry only a genome (a short list of numbers) and return one float;
* anything that goes wrong (no pool, a broken pool, a pickling failure) falls back to the SAME math run serially in this
  process, so a sweep can never get worse than "serial".

Small jobs stay serial: spawning and importing the app in a worker costs seconds, which only pays for itself when
``n_items * n_bars`` is large (override with ``T58_SWEEP_PROCESSES``: ``0`` = never, ``force`` = always, for tests).
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import threading
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd

ENV_FLAG = "T58_SWEEP_PROCESSES"
MIN_WORK_UNITS = 400_000        # n_items * n_bars below which spawning workers costs more than it saves
MIN_ITEMS = 2


# --------------------------------------------------------------------------------------------------------------------
# Strategy description that survives pickling (a live Strategy is not sent to workers)
# --------------------------------------------------------------------------------------------------------------------
class CodeStrategyShim:
    """Carries only what ``materialize_code_strategy`` reads: source_type plus file_path or code."""
    __slots__ = ("source_type", "file_path", "code")

    def __init__(self, source_type: str, file_path=None, code=None):
        self.source_type = source_type
        self.file_path = file_path
        self.code = code


@dataclass(frozen=True)
class StrategySpec:
    kind: str                          # "manual" or a code source type (python / pinescript / mql5)
    manual_config: Optional[dict] = None
    shim: Optional[CodeStrategyShim] = None


def spec_from_strategy(strategy) -> Optional[StrategySpec]:
    """None when the strategy type is not one the genome adapter can rebuild in a worker (callers then stay serial)."""
    st = getattr(strategy, "source_type", None)
    if st == "manual":
        cfg = getattr(strategy, "config", None)
        return StrategySpec("manual", manual_config=cfg) if isinstance(cfg, dict) else None
    try:
        from app.optimize.refinement import CODE_SOURCE_TYPES
    except Exception:  # noqa: BLE001
        return None
    if st in CODE_SOURCE_TYPES:
        return StrategySpec(st, shim=CodeStrategyShim(st, getattr(strategy, "file_path", None), getattr(strategy, "code", None)))
    return None


# --------------------------------------------------------------------------------------------------------------------
# Spill cache: one pickle of a frame serves every stage that sweeps it (a robustness run is ~10 sweeps over one frame)
# --------------------------------------------------------------------------------------------------------------------
_SPILL_LOCK = threading.Lock()
_SPILL: dict = {}                # fingerprint -> (_Spilled, directory); insertion-ordered, newest last
_SPILL_KEEP = 3


def _fingerprint(df: pd.DataFrame) -> tuple:
    try:
        cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        sums = tuple(float(df[c].to_numpy(dtype="float64", copy=False).sum()) for c in cols)
        ts = df["timestamp"] if "timestamp" in df.columns else df.index.to_series()
        return (id(df), len(df), tuple(df.columns), str(ts.iloc[0]) if len(df) else "", str(ts.iloc[-1]) if len(df) else "", sums)
    except Exception:  # noqa: BLE001
        return (id(df), len(df), None)


def _spill(df: pd.DataFrame):
    from app.orchestration.stage_pool import _Spilled

    key = _fingerprint(df)
    with _SPILL_LOCK:
        hit = _SPILL.get(key)
        if hit is not None and Path(hit[0].path).exists():
            return hit[0]
        d = Path(tempfile.mkdtemp(prefix="t58_sweep_"))
        p = d / "df.pkl"
        df.to_pickle(p)
        ref = _Spilled(str(p))
        _SPILL[key] = (ref, d)
        while len(_SPILL) > _SPILL_KEEP:
            _old_key = next(iter(_SPILL))
            _old_ref, _old_dir = _SPILL.pop(_old_key)
            shutil.rmtree(_old_dir, ignore_errors=True)
        return ref


def clear_spill_cache() -> None:
    with _SPILL_LOCK:
        for _ref, d in _SPILL.values():
            shutil.rmtree(d, ignore_errors=True)
        _SPILL.clear()


atexit.register(clear_spill_cache)


# --------------------------------------------------------------------------------------------------------------------
# Worker side (module level so a spawned process can import these)
# --------------------------------------------------------------------------------------------------------------------
_W: dict = {}


def _worker_init(df, spec: StrategySpec, genes: list, risk, prop_rules, mc_config, tmp_dir: str, extra: dict) -> None:
    _W.clear()
    _W.update(df=df, spec=spec, genes=genes, risk=risk, prop_rules=prop_rules, mc=mc_config, tmp_dir=Path(tmp_dir), extra=extra)


def _worker_build(genome: list):
    spec: StrategySpec = _W["spec"]
    if spec.kind == "manual":
        from app.optimize.parameter_space import apply_genome
        from app.strategy.manual import ManualStrategy
        return ManualStrategy(apply_genome(spec.manual_config, _W["genes"], genome))
    from app.optimize.code_parameter_space import materialize_code_strategy
    return materialize_code_strategy(spec.shim, _W["genes"], genome, _W["tmp_dir"])


def task_metric(genome: list, metric: str) -> float:
    """One sensitivity / heatmap cell: the named metric of the strategy built from ``genome``."""
    from app.validation.sensitivity import _metric_for
    return _metric_for(_W["df"], _worker_build(genome), _W["risk"], _W["prop_rules"], _W["mc"], metric)


def task_wfo_fitness(genome: list) -> float:
    """One walk-forward genome on its fold's train window: backtest + prop simulation + Monte Carlo + fitness."""
    from app.validation.walk_forward_opt import score_genome_on_window
    ex = _W["extra"]
    return score_genome_on_window(
        _W["df"], _worker_build(genome), _W["risk"], _W["prop_rules"], _W["mc"], ex["fitness_metric"], ex["mc_sims"],
    )


TASKS: dict = {"metric": task_metric, "wfo_fitness": task_wfo_fitness}


# --------------------------------------------------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------------------------------------------------
def _decide_parallel(n_items: int, n_bars: int) -> bool:
    flag = os.environ.get(ENV_FLAG, "").strip().lower()
    if flag in ("0", "false", "no", "off", "never"):
        return False
    if n_items < MIN_ITEMS:
        return False
    if flag in ("force", "always"):
        return True
    return n_items * max(int(n_bars), 0) >= MIN_WORK_UNITS


class GenomeEvaluator:
    """Evaluates many genomes of ONE frame/strategy, in worker processes when worthwhile, else serially.

    ``local_fn(genome) -> value`` is the serial implementation (and the fallback). ``task`` names the matching function in
    ``TASKS`` that the workers run; both must compute the same number. Results always come back in input order."""

    def __init__(
        self, *, df: pd.DataFrame, strategy, genes: list, risk, prop_rules, mc_config, local_fn: Callable[[list], Any],
        task: str, task_args: tuple = (), n_items: int, tmp_dir: Optional[Path] = None, extra: Optional[dict] = None,
        workers: Optional[int] = None,
    ) -> None:
        self._local = local_fn
        self._task = TASKS[task]
        self._task_args = tuple(task_args)
        self._handle = None
        self._own_tmp: Optional[Path] = None
        self.parallel = False
        self.workers = 1
        if not _decide_parallel(n_items, len(df)):
            return
        spec = spec_from_strategy(strategy)
        if spec is None:
            return
        try:
            from app.orchestration.resource_guard import safe_worker_count
            from app.orchestration.stage_pool import acquire_stage

            n_workers = safe_worker_count(df, requested=workers, max_candidates_in_flight=n_items)
            if os.environ.get(ENV_FLAG, "").strip().lower() in ("force", "always") and n_workers < 2:
                n_workers = 2
            if n_workers < 2:
                return
            tdir = Path(tmp_dir) if tmp_dir is not None else None
            if tdir is None:
                self._own_tmp = tdir = Path(tempfile.mkdtemp(prefix="t58_sweep_tmp_"))
            handle = acquire_stage(
                n_workers, _worker_init,
                (_spill(df), spec, genes, risk, prop_rules, mc_config, str(tdir), dict(extra or {})),
            )
            if handle is not None:
                self._handle, self.parallel, self.workers = handle, True, n_workers
        except Exception:  # noqa: BLE001 -- a sweep must never get worse than serial
            self._handle = None
            self.parallel = False

    def map(self, genomes: Sequence[list], on_result: Optional[Callable[[int, Any], None]] = None) -> list:
        genomes = [list(g) for g in genomes]
        out: list = [None] * len(genomes)
        done = [False] * len(genomes)
        if self._handle is not None and len(genomes) >= MIN_ITEMS:
            try:
                futs = {self._handle.submit(self._task, g, *self._task_args): i for i, g in enumerate(genomes)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    out[i] = fut.result()
                    done[i] = True
                    if on_result is not None:
                        on_result(i, out[i])
            except Exception:  # noqa: BLE001 -- broken pool / pickling problem: finish the rest serially, same math
                self._abandon_pool()
        for i, g in enumerate(genomes):
            if not done[i]:
                out[i] = self._local(g)
                done[i] = True
                if on_result is not None:
                    on_result(i, out[i])
        return out

    def _abandon_pool(self) -> None:
        h, self._handle, self.parallel = self._handle, None, False
        if h is not None:
            try:
                h.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        h, self._handle = self._handle, None
        if h is not None:
            try:
                h.shutdown()
            except Exception:  # noqa: BLE001
                pass
        if self._own_tmp is not None:
            shutil.rmtree(self._own_tmp, ignore_errors=True)
            self._own_tmp = None

    def __enter__(self) -> "GenomeEvaluator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
