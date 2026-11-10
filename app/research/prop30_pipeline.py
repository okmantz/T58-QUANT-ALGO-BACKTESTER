"""End-to-end 30-day prop evaluation research (Phases 2-6), reproducible and deliberately unable to manufacture a pass.

Stages: feasibility frontier -> opportunity scan (DISCOVERY data only) -> exit lab on the best survivor's FROZEN entries ->
portfolio of survivors (shared risk) -> full-history ledger of the frozen top candidate -> validation gate (independence audit,
walk-forward, block robustness, holdout, forward registry). Everything is deterministic given the data and the arguments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.discovery.opportunity_scanner import (EXIT_GRID, INSTRUMENTS, build_ledger, cell_entries, prepare_bars, scan)
from app.portfolio.prop_portfolio import allocate_weights
from app.prop.feasibility import frontier
from app.prop.presets import get_preset
from app.prop.thirty_day import ThirtyDayConfig, TradeLedger, evaluate_thirty_day
from app.research.exit_lab import ExitPolicy, FrozenEntry, default_policy_grid, run_exit_lab
from app.validation.prop30_validation import ObservedRange, Partitions, validate_prop30


@dataclass
class Prop30Config:
    preset: str = "lucid_50k"
    discovery: tuple = ("2020-01-01", "2023-01-01")
    development: tuple = ("2023-01-01", "2024-07-01")
    holdout: tuple | None = ("2024-09-01", "2025-09-01")
    assume_untouched_after: str | None = None          # without this the holdout is treated as previously observed (see run)
    risk_usd: float = 500.0
    max_contracts: int = 4
    top_k: int = 3
    minutes: int = 5
    seed: int = 7


def _observed(cfg: Prop30Config, data_end) -> list:
    """The repo's earlier experiments (incl. the NQ GAD run) and the bundled sealed window read data from this very dataset, and the
    repo cannot prove which dates were never read. Unless the caller asserts a cut-off after which data is fresh, everything after
    development is treated as observed, so no holdout carved out of the bundled data can be called independent."""
    if cfg.assume_untouched_after:
        return [ObservedRange(cfg.discovery[0], cfg.assume_untouched_after, "caller-declared previously observed span")]
    return [ObservedRange(cfg.development[1], str(pd.Timestamp(data_end).date()),
                          "bundled data after development: read by earlier experiments / sealed window (cannot be proven unread)")]


def run(frames: dict, cfg: Prop30Config | None = None) -> dict:
    cfg = cfg or Prop30Config()
    rules = get_preset(cfg.preset).to_prop_rules()
    tcfg = ThirtyDayConfig()
    out: dict = {"config": cfg.__dict__, "rules": {"account": rules.account_size, "target_pct": rules.evaluation_profit_target_pct,
                                                   "max_dd_pct": rules.max_drawdown_pct, "daily_loss_pct": rules.daily_loss_limit_pct}}
    out["feasibility_frontier"] = frontier(rules)
    scan_rep = scan(frames, rules, discovery_start=cfg.discovery[0], discovery_end=cfg.discovery[1], minutes=cfg.minutes,
                    risk_usd=cfg.risk_usd, max_contracts=cfg.max_contracts, cfg=tcfg)
    out["scan"] = {"n_trials": scan_rep.n_trials, "notes": scan_rep.notes, "skipped_instruments": scan_rep.skipped_instruments,
                   "top": [r.to_dict() for r in scan_rep.top(15)]}
    survivors = [r for r in scan_rep.results if r.result is not None and r.net_expectancy_r > 0][: cfg.top_k]
    out["survivors"] = [r.to_dict() for r in survivors]
    if not survivors:
        out["verdict"] = "NO_SURVIVOR"
        out["conclusion"] = ("No scanned cell had positive net expectancy at enough frequency on discovery data. The current data cannot "
                             "establish a 30-day pass strategy; nothing further was validated.")
        return out

    # ---- exit lab on the best survivor's frozen entries (discovery window)
    best = survivors[0]
    raw = frames[best.cell.instrument]
    bars, atr, nb = prepare_bars(raw, start=cfg.discovery[0], end=cfg.discovery[1], minutes=cfg.minutes)
    ents = cell_entries(best.cell, nb, atr)
    def _lab_ledger(pol, es):
        led, _, rs = build_ledger(bars, atr, es, pol, best.cell.instrument, risk_usd=cfg.risk_usd, max_contracts=cfg.max_contracts, label=pol.name)
        return led, (sum(rs) / len(rs) if rs else 0.0)

    lab = run_exit_lab(bars, ents, rules, policies=default_policy_grid(), risk_usd=cfg.risk_usd, instrument=best.cell.instrument, cfg=tcfg,
                       data_start=bars["timestamp"].iloc[0], data_end=bars["timestamp"].iloc[-1], ledger_fn=_lab_ledger)
    out["exit_lab"] = {"cell": best.cell.key, **lab.to_dict()}

    # ---- portfolio of survivors on discovery data
    ledgers = {}
    for r in survivors:
        b2, a2, nb2 = prepare_bars(frames[r.cell.instrument], start=cfg.discovery[0], end=cfg.discovery[1], minutes=cfg.minutes)
        pol = next(p for p in EXIT_GRID if p.name == r.cell.exit_name)
        led, _, _ = build_ledger(b2, a2, cell_entries(r.cell, nb2, a2), pol, r.cell.instrument, risk_usd=cfg.risk_usd,
                                 max_contracts=cfg.max_contracts, label=r.cell.key)
        ledgers[r.cell.key] = led
    if len(ledgers) > 1:
        pr = allocate_weights(ledgers, rules, grid=(0.0, 0.5, 1.0), data_start=cfg.discovery[0], data_end=cfg.discovery[1], cfg=tcfg)
        out["portfolio"] = pr.to_dict()
    else:
        out["portfolio"] = {"notes": ["Only one survivor: no portfolio to build."]}

    # ---- full-history ledger of the frozen top candidate, then the validation gate
    _last = pd.Timestamp(frames[best.cell.instrument]["timestamp"].iloc[-1])
    full_end = _last.tz_convert("UTC").tz_localize(None) if _last.tzinfo is not None else _last
    bF, aF, nbF = prepare_bars(raw, start=cfg.discovery[0], end=None, minutes=cfg.minutes)
    pol = next(p for p in EXIT_GRID if p.name == best.cell.exit_name)
    ledF, _, _ = build_ledger(bF, aF, cell_entries(best.cell, nbF, aF), pol, best.cell.instrument, risk_usd=cfg.risk_usd,
                              max_contracts=cfg.max_contracts, label=best.cell.key)
    parts = Partitions(cfg.discovery, cfg.development, cfg.holdout, selection_cutoff=cfg.development[1])
    val = validate_prop30(ledF, rules, parts, config_id=best.cell.key, observed_ranges=_observed(cfg, full_end), cfg=tcfg)
    out["validation"] = val.to_dict()
    out["verdict"] = val.verdict
    out["conclusion"] = (
        "The top discovery candidate " + ("passes every statistical gate but still needs forward confirmation." if val.verdict.startswith("ESTABLISHED")
                                          else "does NOT establish a 30-day pass: " + "; ".join(val.reasons[:3])))
    return out


def to_markdown(res: dict) -> str:
    L = ["# 30-Day Prop Evaluation Research Report", "", f"**Verdict:** `{res.get('verdict')}`", "", res.get("conclusion", ""), ""]
    L += ["## Rules simulated", "", "```", json.dumps(res.get("rules", {}), indent=1), "```", ""]
    L += ["## Feasibility frontier (i.i.d. upper bound, best risk size)", "", "| Net edge R/trade | Trades/30d | Best risk % | P(pass<=30d) |", "|---|---|---|---|"]
    for r in res.get("feasibility_frontier", []):
        L.append(f"| {r['expectancy_r']} | {r['trades_per_30d']} | {r['best_risk_pct']} | {r['p_pass_30'] * 100:.0f}% |")
    sc = res.get("scan", {})
    L += ["", f"## Opportunity scan (discovery data only, {sc.get('n_trials', 0)} cells tried)", "", *[f"- {n}" for n in sc.get("notes", [])]]
    for k, v in sc.get("skipped_instruments", {}).items():
        L.append(f"- {k}: skipped ({v})")
    L += ["", "| Cell | Trades/30d | Net R | Feasibility UB | P(30d) | 95% lower |", "|---|---|---|---|---|---|"]
    for r in sc.get("top", [])[:10]:
        p = "n/a" if r["p_pass_30"] is None else f"{r['p_pass_30'] * 100:.1f}%"
        lb = "n/a" if r["lower_bound"] is None else f"{r['lower_bound'] * 100:.1f}%"
        L.append(f"| {r['cell']} | {r['trades_per_30d']} | {r['net_expectancy_r']} | {r['feasibility_upper_bound']} | {p} | {lb} |")
    ex = res.get("exit_lab")
    if ex:
        L += ["", f"## Exit lab (frozen entries of {ex['cell']}; {ex['n_trials']} policies tried)", "", "| Policy | Mean R | P(30d) | 95% lower |", "|---|---|---|---|"]
        for c in ex["ranked"][:8]:
            L.append(f"| {c['policy']} | {c['mean_r']:.3f} | {c['p_pass_30'] * 100:.1f}% | {c['lower_bound'] * 100:.1f}% |")
    pf = res.get("portfolio", {})
    if pf.get("ranked"):
        L += ["", f"## Portfolio (in-sample, {pf['n_trials']} weight vectors)", "", f"Diversification ratio {pf['diversification_ratio']:.2f}", ""]
        for c in pf["ranked"][:3]:
            L.append(f"- weights {c['weights']} -> P(30d) {c['p_pass_30'] * 100:.1f}%, lower {c['lower_bound'] * 100:.1f}%, bust {c['p_bust_30'] * 100:.1f}%")
    v = res.get("validation")
    if v:
        L += ["", "## Validation gate", "", f"Verdict: **{v['verdict']}**", ""]
        L += [f"- {r}" for r in v["reasons"]]
        for k in ("discovery", "development", "holdout"):
            if v.get(k):
                d = v[k]
                L.append(f"- {k}: P(30d) {d['p_pass_30'] * 100:.1f}% (95% lower {d['lower_bound'] * 100:.1f}%, {d['n_windows']} windows, ~{d['n_independent']:.0f} independent)")
    L += ["", "## Honesty notes", "",
          "- Pass probability is measured on real calendar time from purchase, using the existing prop rule engine's floating drawdown / daily loss.",
          "- Eventual (chain) Monte Carlo pass is never used as a 30-day number.",
          "- No holdout inside the bundled data can be certified untouched; a forward test is required before any success claim."]
    return "\n".join(L) + "\n"
