"""Strategy Discovery Engine behaviour for ``EvolutionRunner`` (mixin).

Kept out of engine.py so the 3,000-line runner keeps its shape: everything here is additive and reads/writes only attributes the
runner already owns plus the ones ``_init_discovery`` creates.

  pool P&L          every full-evaluated candidate's daily R is streamed to a compressed per-run file
  pool tests        White's Reality Check / Hansen SPA over the whole pool + PBO on a stratified sample of it
  deflation         every new elite gets a Deflated Sharpe verdict on the lifetime ledger, clustered to effective trials
  provenance        seed, data fingerprint, git hash and full config stored with every elite
  graveyard prior   graduated neighbourhood charge carried into the fitness
  structure         depth cap on every child; pruning pass on a champion before promotion
  promotion         mandatory deflated -> null gate -> pooled -> sealed sequence; the run FAILS when any step fails
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd


class DiscoveryMixin:
    # ------------------------------------------------------------------------------------------------------- setup
    def _init_discovery(self) -> None:
        from app.discovery_engine.holdout import HoldoutBudget
        from app.discovery_engine.provenance import data_fingerprint
        from app.discovery_engine.trial_accounting import TrialCounter

        self.promotion_frames = None        # {SYMBOL: DataFrame} -- the cross-market universe for the pooled test
        self.promotion_risks = None         # {SYMBOL: RiskConfig}
        self.run_failed = False
        self.run_failed_reason = ""
        self.promotion_result = None
        self._holdout_budget = HoldoutBudget(max_looks=int(self.cfg.holdout_max_looks))
        self._pool_store = None
        self._guidance = None
        self._run_trials = TrialCounter()
        self._pool_report = None
        self._data_fp = data_fingerprint(self.df) if self.df is not None else "empty"
        self._lifetime_prior = None
        self._ledger_run_key = None

    def discovery_checkpoint_extra(self) -> dict:
        return {"objective": self.cfg.objective, "holdout_budget": self._holdout_budget.to_dict(), "data_fp": self._data_fp,
                "run_trials": self._run_trials.by_family()}

    def _restore_discovery_extra(self, extra: dict | None) -> None:
        from app.discovery_engine.holdout import HoldoutBudget
        extra = extra or {}
        if extra.get("holdout_budget"):
            self._holdout_budget = HoldoutBudget.from_dict(extra["holdout_budget"])
            self._holdout_budget.max_looks = int(self.cfg.holdout_max_looks)

    # --------------------------------------------------------------------------------------------------- pool P&L
    def _pool_store_path(self) -> Path:
        if self.cfg.pool_pnl_path:
            return Path(self.cfg.pool_pnl_path)
        return Path(self.cfg.checkpoint_path).with_name("evolution_pool_pnl")

    def _ensure_pool_store(self):
        if self._pool_store is None and self.cfg.capture_pool_pnl:
            from app.discovery_engine.pnl_store import PnlStore
            self._pool_store = PnlStore(self._pool_store_path(), max_candidates=int(self.cfg.pool_max_candidates))
        return self._pool_store

    def _capture_pool(self, evaluated: list, gen: int) -> None:
        """Stream the daily R of every full-evaluated candidate (controls excluded) to the per-run file."""
        store = self._ensure_pool_store()
        if store is None:
            return
        try:
            for r in evaluated:
                if (r.meta or {}).get("control") or not r.trades:
                    continue
                store.add_trades(f"g{gen}:{r.candidate_id}", (r.meta or {}).get("family", "?"), r.trades, "r")
            store.flush()
        except Exception as exc:  # noqa: BLE001 -- capture is best-effort
            self._log(f"  Pool P&L capture skipped: {exc}")

    def pool_tests(self, force: bool = False) -> dict | None:
        """Reality Check / SPA over the whole pool + PBO on a stratified sample. Cached on ``self._pool_report``."""
        if self._pool_store is None:
            return self._pool_report
        try:
            from app.discovery_engine.pnl_store import PnlStore
            from app.discovery_engine.pool_stats import pool_report
            series, fams = PnlStore.load_series(self._pool_store.path)
            if len(series) < 2:
                return self._pool_report
            rep = pool_report(series, strata=fams, alpha=self.cfg.pool_alpha, pbo_sample=self.cfg.pool_pbo_sample, seed=self.cfg.random_seed)
            self._pool_report = rep
            fw, pb = rep["family_wise"], rep["pbo"]
            self._log(
                f"  POOL TESTS over {rep['n_candidates']:,} evaluated candidate(s): Reality Check p="
                f"{'n/a' if fw.get('rc_p') is None else format(fw['rc_p'], '.3f')}, SPA p="
                f"{'n/a' if fw.get('spa_p') is None else format(fw['spa_p'], '.3f')} -> "
                f"{'best edge survives the whole-pool correction' if fw.get('passed') else 'best edge is NOT distinguishable from luck across the pool'}; "
                f"PBO={'n/a' if pb.get('pbo') is None else format(pb['pbo'], '.2f')} ({pb.get('note', '')}).")
            return rep
        except Exception as exc:  # noqa: BLE001
            self._log(f"  Pool tests skipped: {exc}")
            return self._pool_report

    # --------------------------------------------------------------------------------------------- trial accounting
    def lifetime_prior(self) -> int:
        """Ledger trials for this run's families on THIS data window, read once (before this run books itself)."""
        if self._lifetime_prior is None:
            try:
                from app.discovery_engine.trial_accounting import lifetime_trials
                from app.search.strategy_space import list_families
                fams = self.cfg.families or list(list_families())
                self._lifetime_prior = int(lifetime_trials(["family:" + f for f in fams], data_fp=self._data_fp))
            except Exception:  # noqa: BLE001
                self._lifetime_prior = 0
        return int(self._lifetime_prior)

    def _attach_dsr(self, recs: list) -> None:
        """Deflated Sharpe on each new elite: this run's evaluations + the lifetime ledger, clustered into effective trials."""
        if not self.cfg.lifetime_deflation or not recs:
            return
        try:
            from app.discovery_engine.pool_stats import daily_series_from_trades
            from app.discovery_engine.trial_accounting import effective_trials
            from app.search.robustness import deflated_sharpe_gate
            raw = max(1, int(self._run_trials.total) + self.lifetime_prior())
            sample = []
            if self._pool_store is not None:
                from app.discovery_engine.pnl_store import PnlStore
                ser, _f = PnlStore.load_series(self._pool_store.path)
                sample = list(ser.values())
                if len(sample) > 120:
                    import random
                    sample = random.Random(7).sample(sample, 120)
            eff = effective_trials(raw, sample, self.cfg.dsr_cluster_corr)
            n_trials = int(eff["effective_trials"])
            pool_sharpes = []
            for s in sample:
                a = s.to_numpy(dtype=float)
                pool_sharpes.append(float(a.mean() / a.std(ddof=1)) if a.size > 2 and a.std(ddof=1) > 0 else 0.0)
            for r in recs:
                s = daily_series_from_trades(r.trades, "r") if r.trades else pd.Series(dtype=float)
                if len(s) < 30:
                    r.dsr = {"passed": False, "reason": f"only {len(s)} active days -- cannot deflate", "n_trials": n_trials}
                    continue
                cal = pd.bdate_range(s.index.min(), s.index.max())
                a = s.reindex(cal).fillna(0.0).to_numpy(dtype=float)
                sd = a.std(ddof=1)
                sr = float(a.mean() / sd) if sd > 0 else 0.0
                mu, sg = float(a.mean()), float(a.std(ddof=0))
                sk = float(np.mean(((a - mu) / sg) ** 3)) if sg > 0 else 0.0
                ku = float(np.mean(((a - mu) / sg) ** 4)) if sg > 0 else 3.0
                trial_sr = pool_sharpes if len(pool_sharpes) >= 2 else [sr]
                g = deflated_sharpe_gate(sr, trial_sr, n_trials, int(a.size), returns_skew=sk, returns_kurtosis=ku)
                r.dsr = {"passed": bool(g.passed), "psr": g.probabilistic_sharpe, "n_trials": g.n_trials, "raw_trials": raw,
                         "ledger_prior": self.lifetime_prior(), "clusters": eff["n_clusters"], "reason": g.reason}
        except Exception as exc:  # noqa: BLE001
            self._log(f"  Deflated Sharpe on new elites skipped: {exc}")

    # ---------------------------------------------------------------------------------------------------- provenance
    def _attach_provenance(self, recs: list) -> None:
        try:
            from app.discovery_engine import provenance as pv
            from app.discovery_engine.engine_steps import edge_record
            for r in recs:
                if r.provenance:
                    continue
                r.provenance = pv.capture(
                    seed=self.cfg.random_seed, df=self.df, config=dataclasses.asdict(self.cfg), spec=r.spec,
                    instrument=self.cfg.instrument, timeframe=self.cfg.timeframe, edge=edge_record(r.trades, r.spec),
                ).to_dict()
        except Exception as exc:  # noqa: BLE001
            self._log(f"  Provenance capture skipped: {exc}")

    # ----------------------------------------------------------------------------------------------- graveyard prior
    def _apply_graveyard_guidance(self, population: list) -> list:
        """Stamp ``meta['graveyard_penalty_r']`` on every real child (graduated, capped; the hard skip happens elsewhere)."""
        if not self.cfg.use_graveyard_guidance or not population:
            return population
        try:
            if self._guidance is None:
                from app.discovery_engine.graveyard_guidance import GraveyardGuidance
                self._guidance = GraveyardGuidance.load(self.cfg.instrument, self.cfg.timeframe)
                self._log("  " + self._guidance.summary())
            if not self._guidance.n_rows:
                return population
            n = 0
            for _cid, spec, meta in population:
                if meta is None or meta.get("control"):
                    continue
                pen = self._guidance.penalty_r(meta.get("family", "?"), (spec or {}).get("config") if isinstance(spec, dict) else None)
                if pen > 0:
                    meta["graveyard_penalty_r"] = pen
                    n += 1
            if n:
                self._log(f"  GRAVEYARD GUIDANCE: {n} child(ren) charged for sitting in neighbourhoods that keep dying.")
        except Exception as exc:  # noqa: BLE001
            self._log(f"  Graveyard guidance skipped: {exc}")
        return population

    # ---------------------------------------------------------------------------------------------------- structure
    def _cap_population_structure(self, population: list) -> list:
        """Depth cap: no child may carry more than the configured number of entry / exit conditions per side."""
        try:
            from app.discovery_engine.simplify import StructureCap, cap_structure
            cap = StructureCap(int(self.cfg.max_entry_conditions_per_side), int(self.cfg.max_exit_conditions_per_side))
            out, trimmed = [], 0
            for cid, spec, meta in population:
                if isinstance(spec, dict) and spec.get("source_type", "manual") == "manual" and not (meta or {}).get("control"):
                    new, removed = cap_structure(spec, cap)
                    trimmed += 1 if removed else 0
                    out.append((cid, new, meta))
                else:
                    out.append((cid, spec, meta))
            if trimmed:
                self._log(f"  STRUCTURE CAP: trimmed {trimmed} child(ren) to <= {cap.max_entry_conditions_per_side} entry / "
                          f"{cap.max_exit_conditions_per_side} exit condition(s) per side.")
            return out
        except Exception:  # noqa: BLE001
            return population

    def prune_record(self, rec) -> dict:
        """Pruning pass: drop each condition in turn while the edge lower bound does not fall (> 0.005R). Updates ``rec.spec``."""
        from app.backtest.engine import run_backtest
        from app.discovery_engine.objective import edge_lb_r
        from app.discovery_engine.simplify import prune
        from app.search.strategy_space import build_strategy_from_spec

        def score(spec):
            try:
                bt = run_backtest(self.df, build_strategy_from_spec(spec), self.risk, lookahead_check="skip")
                return edge_lb_r(bt.trades) if bt.trades else float("-inf")
            except Exception:  # noqa: BLE001
                return float("-inf")

        res = prune(rec.spec, score)
        if res.removed:
            rec.spec = res.spec
            rec.provenance = None          # the spec changed -> a new provenance record, captured when re-run
            self._log(f"  PRUNING {rec.candidate_id}: removed {len(res.removed)} condition(s) with no edge cost "
                      f"({res.score_before:+.4f} -> {res.score_after:+.4f}R).")
        return res.to_dict()

    # ------------------------------------------------------------------------------------------------------ promotion
    def _promotion_steps(self):
        from app.discovery_engine.engine_steps import build_steps
        from app.discovery_engine.promotion import PromotionConfig
        from app.discovery_engine.trial_accounting import effective_trials

        matrix = None
        n_trials = max(1, int(self._run_trials.total) + self.lifetime_prior())
        if self._pool_store is not None:
            try:
                from app.discovery_engine.pnl_store import PnlStore
                matrix, _ids, _f, _c = PnlStore.load(self._pool_store.path)
                ser, _ = PnlStore.load_series(self._pool_store.path)
                sample = list(ser.values())[:120]
                n_trials = int(effective_trials(n_trials, sample, self.cfg.dsr_cluster_corr)["effective_trials"])
            except Exception:  # noqa: BLE001
                matrix = None
        return build_steps(
            dev_df=self.df, sealed_df=self.sealed_final_df, risk=self.risk, frames=self.promotion_frames, risks=self.promotion_risks,
            pool_matrix=matrix, n_trials=n_trials, cfg=PromotionConfig(), holdout_budget=self._holdout_budget,
        )

    def run_promotion(self, candidate=None, prune: bool | None = None):
        """Run the mandatory sequence for ``candidate`` (default: the leaderboard champion). Returns the ``PromotionResult``; sets
        ``self.run_failed`` when it fails. The candidate's ``promotion`` field carries the verdicts."""
        from app.discovery_engine.promotion import run_promotion_sequence
        rec = candidate if candidate is not None else (self.leaderboard[0] if self.leaderboard else None)
        if rec is None:
            raise ValueError("No champion to promote: the leaderboard is empty.")
        if (self.cfg.prune_champions if prune is None else prune) and rec.spec.get("source_type", "manual") == "manual":
            self.prune_record(rec)
        if not rec.trades:      # a resumed leaderboard row carries no raw trades
            from app.discovery_engine.engine_steps import evaluate_spec
            rec.trades = evaluate_spec(rec.spec, self.df, self.risk, 1.0)
        if not rec.provenance:
            self._attach_provenance([rec])
        self.pool_tests()
        res = run_promotion_sequence({"spec": rec.spec, "provenance": rec.provenance, "family": (rec.meta or {}).get("family")},
                                     self._promotion_steps(), family=(rec.meta or {}).get("family"))
        rec.promotion = res.to_dict()
        self.promotion_result = res
        for line in res.render().splitlines():
            self._log("  " + line)
        if not res.passed:
            self.run_failed = True
            self.run_failed_reason = f"promotion failed at '{res.failed_step}': {res.failed_reason}"
        return res

    def _finalize_target_hit(self, rec) -> None:
        """Loop-mode stop: the stop is kept, the champion is not -- unless the whole sequence passes. Otherwise the run FAILS."""
        if not self.cfg.mandatory_promotion_on_stop:
            return
        try:
            res = self.run_promotion(rec)
        except Exception as exc:  # noqa: BLE001 -- no evidence is not a pass
            self.run_failed = True
            self.run_failed_reason = f"promotion sequence could not run: {type(exc).__name__}: {exc}"
            self._log("  PROMOTION FAILED: " + self.run_failed_reason)
            return
        if not res.passed:
            self._log(f"  RUN FAILED: the '{self.cfg.target_metric}' target was reached but the mandatory promotion sequence did not "
                      f"pass ({res.failed_step}). Nothing is promoted; the candidate stays a near-miss, not a forward-test candidate.")
        else:
            self._log("  PROMOTION PASSED: every mandatory step cleared. Forward test is MANDATORY before real money.")
