"""
Deployment-page glue for the Evaluation Guardian and the Automated Payout Optimization presets.

The Deploy page posts a few extra fields; ``apply`` turns them into the ``guardian`` / ``optimizer`` settings on a ``LiveExecutionConfig``.
Nothing here can loosen a limit: the Guardian only adds brakes on top of the existing guardrails, and the optimizer's risk is clamped to
the hard limits in ``app.prop.payout_optimizer`` whatever preset is chosen.

    payout_preset         conservative | balanced | accelerated            (default balanced)
    payout_stage          auto | evaluation | newly_funded | approaching_payout | post_first_payout | scaling
    guardian_on           "0" turns the Guardian off (default on)
    enforce_news          "0" turns the economic-calendar news restriction off (default on)
    optimizer_sizes_risk  "1" lets the optimizer set per-trade risk (default off: your configured risk stays)
    firm_preset           a preset key (e.g. lucid_50k): use that firm's actual rules instead of the form's limits
"""
from __future__ import annotations

from app.guardian.engine import GuardianRules
from app.prop import payout_optimizer as po


def parse(form) -> dict:
    stage = str(form.get("payout_stage", "auto") or "auto")
    return {"preset": po.preset_name(form.get("payout_preset")), "stage": stage if stage in po.STAGES else "auto",
            "guardian_on": str(form.get("guardian_on", "1")) != "0", "enforce_news": str(form.get("enforce_news", "1")) != "0",
            "sizes_risk": str(form.get("optimizer_sizes_risk", "0")) == "1", "firm_preset": str(form.get("firm_preset", "") or "")}


def build_rules(opts: dict, *, account_size: float, daily_loss_pct: float | None, max_dd_pct: float, profit_target_pct: float,
                stage: str) -> GuardianRules:
    ev = "evaluation" if stage == "evaluation" else "funded"
    if opts["firm_preset"]:
        r = GuardianRules.from_preset(opts["firm_preset"], stage=ev, news_restricted=opts["enforce_news"])
        r.account_size = r.account_size or account_size
        return r
    return GuardianRules(account_size=account_size,
                         daily_loss_limit=(daily_loss_pct / 100.0 * account_size) if daily_loss_pct else None,
                         max_drawdown=max_dd_pct / 100.0 * account_size if max_dd_pct else None, drawdown_type="trailing",
                         profit_target=(profit_target_pct / 100.0 * account_size) if (ev == "evaluation" and profit_target_pct) else None,
                         news_restricted=opts["enforce_news"], stage=ev, source="form")


def apply(form, cfg, *, account_size: float, daily_loss_pct: float | None, max_dd_pct: float, profit_target_pct: float) -> list[str]:
    """Set ``cfg.guardian`` / ``cfg.guardian_preset`` / ``cfg.optimizer`` from the form. Returns log lines to show the user."""
    opts = parse(form)
    stage = opts["stage"] if opts["stage"] != "auto" else "evaluation"
    cfg.guardian_preset = opts["preset"]
    cfg.optimizer = {"stage": stage, "preset": opts["preset"], "sizes_risk": opts["sizes_risk"]}
    lines = [f"[info] Payout optimizer: {po.PRESETS[opts['preset']]['label']} preset, stage '{po.STAGE_INFO[stage]['label']}' "
             f"({po.STAGE_INFO[stage]['objective']}). {'Sets per-trade risk' if opts['sizes_risk'] else 'Risk stays as configured'}; "
             "hard limits are never overridden."]
    if opts["guardian_on"]:
        cfg.guardian = build_rules(opts, account_size=account_size, daily_loss_pct=daily_loss_pct, max_dd_pct=max_dd_pct,
                                   profit_target_pct=profit_target_pct, stage=stage)
        lines.append("[info] Evaluation Guardian on: daily loss, drawdown, target, consistency, contracts, exposure"
                     + (", news windows" if opts["enforce_news"] else "") + " and trading hours are watched against the broker's own balance.")
    else:
        cfg.guardian = None
        lines.append("[warn] Evaluation Guardian is OFF for this session.")
    return lines
