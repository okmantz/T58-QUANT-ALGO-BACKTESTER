"""The Strategy Discovery Engine: ONE entry point for Search Lab, Evolution Lab, Forge, Speed Run and the hypothesis layer.

    result = DiscoveryEngine().run(EngineRequest(preset="search_lab", df=df, risk=risk, prop_rules=rules, ...))

What is shared (and therefore cannot drift between labs again):

  * ONE objective        edge_lb_r (day-block-bootstrap lower bound of mean R after costs, complexity-charged)
  * ONE entry floor      100+ trades, mean-R t >= 1, cost < 0.15 R (``app.discovery_engine.objective.DEFAULT_FLOOR``)
  * ONE trial count      the lifetime ledger, clustered to effective trials, read BEFORE the run books itself
  * ONE sealed slice     the last ``sealed_frac`` of the data is carved off BEFORE any backend sees a bar; it is read once, by the
                         promotion sequence, through a ``HoldoutBudget``
  * ONE promotion path   reproduction -> sequence-deflated Sharpe -> random-timing null gate -> pooled cross-market edge -> sealed slice.
                         Any failed step fails the champion (and, in loop mode, the run); a champion that passes still needs a forward test.
  * ONE rollout guard    a v13 family runs only when it is PRE-REGISTERED (kill condition on record) and its rollout batch has been
                         opened by the surrogate false-discovery gate (or it was requested explicitly by name)

A preset (``presets.PRESETS``) only picks the generator and the budget. Backends are injectable (``DiscoveryEngine(backends={...})``)
so the orchestration is testable without a 20-minute run; the defaults call the real labs.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from app.discovery_engine import objective as obj
from app.discovery_engine import provenance as prov
from app.discovery_engine.holdout import HoldoutBudget, split_three_way
from app.discovery_engine.presets import Preset, apply_overrides, get_preset
from app.discovery_engine.promotion import PromotionConfig, PromotionResult, run_promotion_sequence


@dataclass
class EngineRequest:
    preset: str
    df: pd.DataFrame
    risk: object
    prop_rules: object = None
    instrument: str = "unknown"
    timeframe: str = "unknown"
    families: object = None                     # None -> the preset's choice; "all" | "ready" | list[str]
    seed: int = 42
    sealed_frac: float = 0.10                   # carved off before any backend runs; read once by the promotion sequence
    frames: dict | None = None                  # {SYMBOL: DataFrame} for the pooled cross-market test
    risks: dict | None = None                   # {SYMBOL: RiskConfig}
    work_dir: str | Path | None = None
    overrides: dict = field(default_factory=dict)
    promotion: PromotionConfig | None = None
    max_promote: int | None = None
    max_candidates: int | None = None            # override the preset's candidate budget
    use_memory: bool = True                      # searched-variant ledger: skip known variants, breed from past leaders, record this run
    register_retests: bool = True                # record champions / near-misses for the automatic out-of-sample retest
    promote: bool = True
    progress_cb: Callable | None = None
    cancel_event: threading.Event | None = None


@dataclass
class BackendOutput:
    native: object = None
    champions: list = field(default_factory=list)   # [{"spec", "family", "candidate_id", "edge_lb_r", "provenance"?, "trade_r"?}]
    pool_matrix: np.ndarray | None = None           # T x N daily results of the whole evaluated pool (Sharpe spread for the DSR)
    streams: list = field(default_factory=list)     # sample daily P&L Series for effective-trial clustering
    evaluations: int = 0                            # this run's evaluation count
    promoted_inside: bool = False                   # the backend already ran the mandatory sequence itself
    notes: list = field(default_factory=list)


@dataclass
class EngineResult:
    run_id: str
    preset: str
    backend: str
    objective: str
    families_run: list
    gated_out: dict
    n_evaluations: int
    trial_inputs: dict
    champions: list
    promotions: list                               # [PromotionResult.to_dict()] aligned with ``champions``
    provenance: dict
    notes: list
    elapsed_seconds: float
    native: object = None

    @property
    def passed(self) -> list:
        return [c for c, p in zip(self.champions, self.promotions) if p.get("passed")]

    @property
    def forward_test_candidates(self) -> list:
        """Champions that cleared every mandatory step. ALL of them still need a forward test."""
        return self.passed

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "preset": self.preset, "backend": self.backend, "objective": self.objective,
                "families_run": self.families_run, "gated_out": self.gated_out, "n_evaluations": self.n_evaluations,
                "trial_inputs": self.trial_inputs, "champions": [{k: v for k, v in c.items() if k != "trades"} for c in self.champions],
                "promotions": self.promotions, "provenance": self.provenance, "notes": self.notes,
                "elapsed_seconds": self.elapsed_seconds, "n_forward_test_candidates": len(self.passed)}


# ---------------------------------------------------------------------------------------------------------- family guard
def resolve_families(requested, preset: Preset, *, store=None) -> tuple[list, dict]:
    """(families to run, {family: reason it was gated out}).

    ``requested`` None -> the preset's choice. "all" -> ``active_family_names()`` (v13 batches closed until the surrogate gate opens
    them). "ready" -> only families whose status is ready. An explicit list is honoured, except that a v13 family must be
    pre-registered: a family whose kill condition is not on record does not run, ever."""
    from app.search import strategy_space as ss
    from app.discovery.hypothesis import is_preregistered
    want = preset.families if requested is None else requested
    gated: dict = {}
    if isinstance(want, str):
        names = list(ss.active_family_names())
        if want == "ready":
            from app.search.families_v13 import V13_META
            names = [n for n in names if V13_META.get(n) is None or V13_META[n].status == "ready"]
    else:
        names = [n for n in want if n in ss.FAMILIES]
        for n in want:
            if n not in ss.FAMILIES:
                gated[n] = "unknown family"
    out = []
    for n in names:
        try:
            ok = is_preregistered(n, store)
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            out.append(n)
        else:
            gated[n] = "not pre-registered: write its kill condition first (preregister_v13_families)"
    return out, gated


# ------------------------------------------------------------------------------------------------------------ the engine
class DiscoveryEngine:
    def __init__(self, backends: dict | None = None, steps_factory: Callable | None = None):
        self.backends = {**default_backends(), **(backends or {})}
        self._steps_factory = steps_factory

    # -------------------------------------------------------------------------------------------------------- promotion
    def _steps(self, **kw):
        if self._steps_factory is not None:
            return self._steps_factory(**kw)
        from app.discovery_engine.engine_steps import build_steps
        return build_steps(**kw)

    def promote(self, champions: list, *, dev_df, sealed_df, risk, frames=None, risks=None, pool_matrix=None, n_trials: int = 1,
                cfg: PromotionConfig | None = None, budget: HoldoutBudget | None = None, loop: bool = False) -> list:
        cfg = cfg or PromotionConfig()
        budget = budget or HoldoutBudget()
        steps = self._steps(dev_df=dev_df, sealed_df=sealed_df, risk=risk, frames=frames, risks=risks, pool_matrix=pool_matrix,
                            n_trials=n_trials, cfg=cfg, holdout_budget=budget)
        out = []
        for c in champions:
            res = run_promotion_sequence(c, steps, cfg, family=c.get("family"))
            out.append(res)
            if loop and not res.passed:
                break
        return out

    # ------------------------------------------------------------------------------------------------------------- run
    def run(self, req: EngineRequest) -> EngineResult:
        t0 = time.time()
        preset = get_preset(req.preset)
        log = req.progress_cb or (lambda m: None)
        run_id = uuid.uuid4().hex[:12]
        notes: list = []
        families, gated = resolve_families(req.families, preset)
        if gated:
            notes.append(f"{len(gated)} famil{'y' if len(gated) == 1 else 'ies'} not run: " + "; ".join(f"{k} ({v})" for k, v in list(gated.items())[:6]))
        if not families:
            raise ValueError("No families to run: every requested family was gated out. " + (notes[0] if notes else ""))

        dev_gate, _gate_empty, sealed = split_three_way(req.df, gate_frac=0.0, sealed_frac=req.sealed_frac)
        dev_gate = dev_gate if len(dev_gate) else req.df
        log(f"[{preset.label}] {len(families)} families, {len(dev_gate):,} bars searchable, {len(sealed):,} bars sealed.")
        backend = self.backends.get(preset.backend)
        if backend is None:
            raise KeyError(f"No backend registered for '{preset.backend}'")
        out: BackendOutput = backend(req, preset, families, dev_gate)
        notes.extend(out.notes)

        # trial count: ledger read BEFORE this run is booked by the backend (they book at the end of a stage), clustered to effective trials
        from app.discovery_engine.trial_accounting import trial_inputs
        data_fp = prov.data_fingerprint(dev_gate)
        names = list(families)
        ti = trial_inputs(max(1, out.evaluations), names=names, data_fp=data_fp, streams=out.streams)

        # champions: entry floor + provenance (the facade measures edge on the same dev slice the sequence reproduces on)
        champs = self._prepare_champions(out.champions, dev_gate, req, preset)
        k = req.max_promote if req.max_promote is not None else preset.top_n_promote
        champs = champs[: max(0, int(k))]
        promotions: list = []
        if out.promoted_inside:
            promotions = [c.get("promotion") or {} for c in champs]
        elif req.promote and champs:
            cfg = req.promotion or PromotionConfig()
            budget = HoldoutBudget()
            res = self.promote(champs, dev_df=dev_gate, sealed_df=sealed, risk=req.risk, frames=req.frames, risks=req.risks,
                               pool_matrix=out.pool_matrix, n_trials=ti.n_tests, cfg=cfg, budget=budget, loop=preset.loop)
            promotions = [r.to_dict() if hasattr(r, "to_dict") else r for r in res]
            # a loop-mode run FAILS when its stop-candidate fails any mandatory step
            if preset.loop and promotions and not promotions[0].get("passed"):
                notes.append(f"loop mode: the stop-candidate FAILED '{promotions[0].get('failed_step')}' -- the run is failed, not 'won'.")
            promotions += [{"passed": False, "failed_step": "not_promoted", "failed_reason": "loop run stopped at the first failed candidate"}] * (len(champs) - len(promotions))
        else:
            promotions = [{"passed": False, "failed_step": "not_promoted", "failed_reason": "promotion disabled for this request"} for _ in champs]
        run_prov = prov.capture(seed=req.seed, df=dev_gate, config={"preset": preset.name, "families": families, "overrides": req.overrides},
                                instrument=req.instrument, timeframe=req.timeframe).to_dict()
        result = EngineResult(run_id=run_id, preset=preset.name, backend=preset.backend, objective=obj.EDGE_OBJECTIVE, families_run=families,
                              gated_out=gated, n_evaluations=int(out.evaluations), trial_inputs=ti.to_dict(), champions=champs, promotions=promotions,
                              provenance=run_prov, notes=notes, elapsed_seconds=time.time() - t0, native=out.native)
        led, scope = _engine_ledger(req)
        if led is not None and out.champions:
            try:
                from app.search.ledger_search import engine_key
                for c, pr in zip(champs, promotions):
                    if c.get("candidate_id") and c.get("family"):
                        led.mark(scope, engine_key(req.timeframe, c["family"], c["candidate_id"]), "winner" if (pr or {}).get("passed") else "near_miss")
                led.flush()
            except Exception:  # noqa: BLE001
                pass
        if req.register_retests:
            try:                                    # champions and near-misses are re-tested on newer data later; the lab is ranked by that record
                from app.discovery_engine.retest import register_run
                register_run(result, req.df, instrument=req.instrument)
            except Exception:  # noqa: BLE001 -- bookkeeping never fails a run
                pass
        return result

    # --------------------------------------------------------------------------------------------------- champion prep
    def _prepare_champions(self, champions: list, dev_df, req: EngineRequest, preset: Preset) -> list:
        from app.discovery_engine.engine_steps import edge_record, evaluate_spec
        ranked = []
        for c in champions or []:
            c = dict(c)
            spec = c.get("spec")
            if not spec:
                continue
            trades = c.get("trades")
            if trades is None:
                trades = evaluate_spec(spec, dev_df, req.risk, 1.0)
            sc = obj.score_edge(trades, spec)
            stats = {"total_trades": sc.n_trades, "profit_factor": getattr(sc, "profit_factor", 0.0), "mean_r": sc.mean_r, "t_stat": sc.t_stat}
            rec = edge_record(trades, spec)
            c["edge_lb_r"] = rec["edge_lb_r"]
            c["edge"] = rec
            if c.get("provenance") is None:
                c["provenance"] = prov.capture(seed=req.seed, df=dev_df, config={"preset": preset.name, "overrides": req.overrides}, spec=spec,
                                               instrument=req.instrument, timeframe=req.timeframe, edge=rec).to_dict()
            c["floor_reasons"] = []
            if sc.n_trades < obj.DEFAULT_FLOOR.min_trades:
                c["floor_reasons"].append(f"{sc.n_trades} trades < {obj.DEFAULT_FLOOR.min_trades}")
            if np.isfinite(sc.t_stat) and sc.t_stat < obj.DEFAULT_FLOOR.min_t_stat:
                c["floor_reasons"].append(f"mean-R t {sc.t_stat:.2f} < {obj.DEFAULT_FLOOR.min_t_stat:g}")
            ranked.append(c)
        # the ONE ranking: edge_lb_r (None sorts last); candidates under the floor stay listed but cannot be promoted
        ranked.sort(key=lambda c: (-1e9 if c["edge_lb_r"] is None else c["edge_lb_r"]), reverse=True)
        eligible = [c for c in ranked if not c["floor_reasons"]]
        for c in ranked:
            if c["floor_reasons"]:
                c.setdefault("promotion", {"passed": False, "failed_step": "entry_floor", "failed_reason": "; ".join(c["floor_reasons"])})
        return eligible if eligible else ranked[:1]


# ------------------------------------------------------------------------------------------------------ default backends
def _work_dir(req: EngineRequest) -> Path:
    import tempfile
    d = Path(req.work_dir) if req.work_dir else Path(tempfile.mkdtemp(prefix="t58_engine_"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _engine_ledger(req: EngineRequest):
    """(ledger, scope) when the request wants search memory and it can be opened, else (None, '')."""
    if not getattr(req, "use_memory", True):
        return None, ""
    try:
        from app.search.ledger_search import scope_for
        from app.search.variant_ledger import get_ledger
        return get_ledger(), scope_for(req.instrument)
    except Exception:  # noqa: BLE001 -- memory is a bonus
        return None, ""


def _search_backend(req: EngineRequest, preset: Preset, families: list, df) -> BackendOutput:
    from app.search.batch_runner import SearchStageConfig, run_search
    from app.search.strategy_space import generate_search_space
    wd = _work_dir(req)
    want = int(req.max_candidates or preset.max_candidates)
    led, scope = _engine_ledger(req)
    notes: list = []
    seed = req.seed
    draw = want
    if led is not None:
        led.begin_run(source="engine:" + preset.name, scope=scope, effort=preset.name, seed=req.seed)
        # a different sample every run (same seed + same grid would re-draw the same strategies), over-drawn so known variants can be dropped
        seed = int(req.seed) + 7919 * led.run_count(scope)
        draw = min(want * 4, want + 20000)
    space = generate_search_space(mode="family", family="all", only_families=list(families), max_candidates=draw,
                                  seed=seed, has_pair_data=bool(req.overrides.get("has_pair_data")),
                                  has_calendar_data=bool(req.overrides.get("has_calendar_data")))
    if led is not None:
        try:
            from app.search.ledger_search import fresh_space
            info = fresh_space(space, led, scope, req.timeframe, want)
        except Exception as exc:  # noqa: BLE001
            led.finish_run()
            raise
        if not space.candidates:
            led.finish_run()
            raise ValueError(f"Every variant these strategy types can produce on {scope} has already been searched ({led.count_known(scope):,} on record). "
                             "Pick other strategy types, or tick 'Start fresh' to repeat the search.")
        notes.append(f"Search memory: skipped {info['skipped_known']:,} variants already tested on {scope}"
                     + (f"; added {info['warm_neighbours']} neighbours of the best past variants" if info["warm_neighbours"] else "")
                     + f"; testing {info['kept']:,} new ones.")
    cfg = apply_overrides(SearchStageConfig(random_seed=req.seed), preset, {k: v for k, v in req.overrides.items() if not k.startswith("has_")})
    try:
        summary = run_search(df, req.risk, req.prop_rules, space, cfg, db_path=str(wd / "search.db"), instrument=req.instrument,
                             timeframe=req.timeframe, progress_cb=req.progress_cb, cancel_event=req.cancel_event)
    finally:
        if led is not None:
            try:
                from app.search.ledger_search import record_outcomes
                rid = None
                try:
                    from app.search.results_db import ResultsDB
                    with ResultsDB(wd / "search.db") as _db:
                        runs = _db.list_runs(limit=1)
                        rid = runs[0]["run_id"] if runs else None
                except Exception:  # noqa: BLE001
                    rid = None
                n = record_outcomes(led, scope, req.timeframe, wd / "search.db", rid, space) if rid else 0
                led.finish_run(tested=n)
            except Exception:  # noqa: BLE001
                pass
    return BackendOutput(native=summary, champions=_champions_from_db(wd / "search.db", summary.run_id), evaluations=int(summary.total_candidates),
                         notes=notes)


def _champions_from_db(db_path, run_id, top_n: int = 25) -> list:
    from app.search.results_db import ResultsDB
    db = ResultsDB(db_path)
    out = []
    for row in db.leaderboard(run_id, stage="stage3", top_n=top_n, only_passed=False):
        st = row.get("source_type") or "manual"
        spec = ({"source_type": "manual", "config": row.get("config") or row.get("config_json") or {}} if st == "manual"
                else {"source_type": st, "code_text": row.get("code_text"), "code_extension": row.get("code_extension")})
        out.append({"spec": spec, "family": row.get("family"), "candidate_id": row.get("candidate_id")})
    return out


def _evolution_backend(req: EngineRequest, preset: Preset, families: list, df) -> BackendOutput:
    from app.evolution.engine import EvolutionConfig, EvolutionRunner
    wd = _work_dir(req)
    cfg = apply_overrides(EvolutionConfig(), preset, {"knowledge_graph_path": str(wd / "kg.jsonl"), "checkpoint_path": str(wd / "checkpoint.json"),
                                                        "tested_log_path": str(wd / "tested.jsonl"), "save_to_library": False, **req.overrides})
    runner = EvolutionRunner(df, req.risk, req.prop_rules, cfg, progress_cb=req.progress_cb)
    runner._run_loop()           # synchronous: the engine owns the thread model
    champs = [{"spec": r.spec, "family": (r.meta or {}).get("family"), "candidate_id": r.candidate_id, "provenance": r.provenance}
              for r in runner.leaderboard]
    return BackendOutput(native=runner, champions=champs, evaluations=int(getattr(runner._run_trials, "total", 0)), promoted_inside=False)


def _forge_backend(req: EngineRequest, preset: Preset, families: list, df) -> BackendOutput:
    from app.orchestration.forge import ForgeConfig, run_forge
    wd = _work_dir(req)
    cfg = apply_overrides(ForgeConfig(random_seed=req.seed), preset, {"exclude_families": set(_all_families()) - set(families), **req.overrides})
    res = run_forge(df, req.risk, req.prop_rules, cfg, db_path=str(wd / "forge.db"), instrument=req.instrument, timeframe=req.timeframe,
                    progress_cb=req.progress_cb, cancel_event=req.cancel_event)
    return BackendOutput(native=res, champions=_champions_from_db(wd / "forge.db", res.run_id), evaluations=int(cfg.n_hypotheses))


def _speed_backend(req: EngineRequest, preset: Preset, families: list, df) -> BackendOutput:
    from app.orchestration.speed_run import SpeedRunConfig, run_speed_run
    wd = _work_dir(req)
    cfg = apply_overrides(SpeedRunConfig(), preset, {"exclude_families": set(_all_families()) - set(families), **req.overrides})
    res = run_speed_run(df, req.risk, req.prop_rules, wd, cfg, progress_cb=req.progress_cb, instrument=req.instrument, cancel_event=req.cancel_event)
    s = res.search_summary
    return BackendOutput(native=res, champions=_champions_from_db(s.db_path, s.run_id), evaluations=int(s.total_candidates))


def _discovery_backend(req: EngineRequest, preset: Preset, families: list, df) -> BackendOutput:
    # hypothesis-first: each pre-registered family is a one-family Search run; the kill condition is on record before it runs
    return _search_backend(req, preset, families, df)


def _all_families() -> list:
    from app.search.strategy_space import FAMILIES
    return list(FAMILIES)


def default_backends() -> dict:
    return {"search": _search_backend, "evolution": _evolution_backend, "forge": _forge_backend, "speed_run": _speed_backend,
            "discovery": _discovery_backend}
