"""Detail view for ONE graveyard row: friendly labels, headline stats and an on-demand equity curve.

Graveyard rows are tiny (one JSON line each). The equity curve is NOT stored -- it is rebuilt on demand by re-running the
real backtest engine on the dataset + costs + exact parameters the row recorded. That is only possible when the row
recorded them (the bundled starter rows do: dataset label, date window, costs and exact params live in `notes`). For a row
that did not, the page says so plainly instead of drawing a made-up curve.

Everything shown is either read straight from the row or measured by re-running the engine; nothing is estimated here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
_MAX_POINTS = 160

STAGE_LABELS = {
    "cpcv": "Failed the held-out (CPCV) test",
    "stress": "Failed the stress test",
    "not_ranked": "Did not make the leaderboard",
    "prefilter": "Cut by the quick pre-filter",
}
FAILURE_LABELS = {
    "no edge after costs": "No edge after trading costs",
    "out-of-sample collapse": "Worked in-sample, failed on new data",
    "too few trades": "Too few trades to trust",
    "drawdown breach": "Drawdown too deep",
    "marginal edge": "Edge too thin after costs",
}


def family_title(name: str) -> str:
    return str(name or "?").replace("_", " ").strip().capitalize()


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, str(stage or "Rejected").replace("_", " ").capitalize())


def failure_label(text: str | None) -> str:
    if not text:
        return ""
    return FAILURE_LABELS.get(text, text[:1].upper() + text[1:])


def file_label(f: dict) -> str:
    """'EURUSD 4h  (212 rejected)' from a list_graveyard_files() entry -- instead of 'eurusd-eurusd240-csv / 4h'."""
    inst = str(f.get("instrument") or "unknown")
    toks = [t for t in inst.split("-") if t and t != "csv"]
    first = toks[0] if toks else "unknown"
    rest = [t for t in toks[1:] if not (t.startswith(first) and re.search(r"\d+$", t)) and t != first]
    name = first.upper() + (f" ({' '.join(rest).title()})" if rest else "")
    return f"{name} \u00b7 {f.get('timeframe', '')}".strip(" \u00b7")


def friendly_settings(signature: str | None, limit: int = 5) -> str:
    """'family|bb_period=30.0,ema_trend=50.0' -> 'bb period 30 \u00b7 ema trend 50' (what a person would read as the settings)."""
    if not signature or "|" not in signature:
        return ""
    out = []
    for part in signature.split("|", 1)[1].split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            fv = float(v)
            v = str(int(fv)) if fv == int(fv) else f"{fv:g}"
        except ValueError:
            pass
        out.append(f"{k.replace('_', ' ')} {v}")
    more = f" +{len(out) - limit} more" if len(out) > limit else ""
    return " \u00b7 ".join(out[:limit]) + more


def find_row(rows: list[dict], candidate_id: str) -> dict | None:
    for r in rows:
        if r.get("candidate_id") == candidate_id:
            return r
    return None


def seed_context(row: dict) -> dict[str, Any] | None:
    """Dataset label, window, costs and exact params parsed from a row's notes, or None if they weren't recorded."""
    notes = [str(n) for n in (row.get("notes") or [])]
    if not notes:
        return None
    head = next((n for n in notes if n.startswith("starter seed") and "measured with" in n), "")
    m = re.search(r"on (.+?) \((\w+)\), (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", head)
    mi = re.search(r"in-sample (\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2}), held-out (\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})", head)
    c = next((n for n in notes if n.startswith("costs:")), "")
    mc = re.search(r"spread ([\d.]+)p, slippage ([\d.]+)p, \$([\d.]+)/trade; ([\d.]+)% risk on \$([\d,]+)", c)
    p = next((n for n in notes if n.startswith("params:")), "")
    if not (m and mc and p):
        return None
    try:
        params = json.loads(p[len("params:"):].strip())
    except json.JSONDecodeError:
        return None
    return {
        "dataset": m.group(1), "timeframe": m.group(2), "start": m.group(3), "end": m.group(4),
        "split": mi.group(2) if mi else None,
        "spread": float(mc.group(1)), "slippage": float(mc.group(2)), "commission": float(mc.group(3)),
        "risk_pct": float(mc.group(4)), "account": float(mc.group(5).replace(",", "")), "params": params,
    }


def _num(v: Any, nd: int = 2) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return round(f, nd)


def summary(row: dict) -> dict[str, Any]:
    """Plain-language facts that come straight from the stored row (no recomputation)."""
    ctx = seed_context(row)
    facts = [
        ("Eval pass chance", _num(row.get("prop_sim_pass_pct"), 0), "%", "Chance of passing a prop-firm evaluation, from the Monte Carlo run when this strategy was tested."),
        ("Held-out pass chance", _num(row.get("cpcv_oos_pass_pct"), 0), "%", "The honest estimate: pass chance on data the strategy was never tuned on."),
        ("Neighbour robustness", _num(row.get("neighbor_robustness_pct"), 0), "%", "How many nearby parameter settings also worked. Low = it only worked at one lucky setting."),
        ("Profit factor", _num(row.get("profit_factor")), "", "Gross profit ÷ gross loss after costs. Below 1.0 loses money."),
        ("Avg R / trade", _num(row.get("avg_r_net")), "R", "Average result per trade in units of the amount risked."),
        ("Trades", row.get("n_trades"), "", "Number of trades behind the numbers."),
    ]
    return {
        "candidate_id": row.get("candidate_id"),
        "family": row.get("family"), "family_title": family_title(row.get("family")),
        "stage": stage_label(row.get("stage_died")), "reason": row.get("reason") or "",
        "primary_failure": failure_label(row.get("primary_failure")), "secondary_failure": failure_label(row.get("secondary_failure")),
        "strength": row.get("strength"), "weakness": row.get("weakness"), "suggestion": row.get("suggested_mutation"),
        "oos": row.get("oos_result"), "signature": row.get("param_signature"),
        "facts": [{"label": a, "value": b, "unit": c, "hint": d} for a, b, c, d in facts if b is not None],
        "can_rebuild": bool(ctx) and bool(_dataset_path(ctx)),
        "context": ({k: ctx[k] for k in ("dataset", "timeframe", "start", "end", "spread", "slippage", "commission", "risk_pct", "account")} if ctx else None),
        "params": (ctx or {}).get("params"),
    }


def _dataset_path(ctx: dict) -> Path | None:
    p = (RAW_DIR / ctx["dataset"]).resolve()
    try:
        p.relative_to(RAW_DIR.resolve())          # the label comes from a data file; never let it escape data/raw
    except ValueError:
        return None
    return p if p.is_file() else None


def _downsample(xs: list, n: int = _MAX_POINTS) -> list:
    if len(xs) <= n:
        return xs
    step = (len(xs) - 1) / (n - 1)
    return [xs[round(i * step)] for i in range(n)]


def rebuild_curve(row: dict) -> dict[str, Any]:
    """Re-run the engine for this row and return {ok, points, stats, split} -- or {ok: False, message} when it can't."""
    ctx = seed_context(row)
    if not ctx:
        return {"ok": False, "message": "This rejection didn't record its exact settings, so its equity curve can't be rebuilt. The reason and scores above are what was saved."}
    path = _dataset_path(ctx)
    if not path:
        return {"ok": False, "message": f"The dataset it was measured on ({ctx['dataset']}) isn't on this computer, so the curve can't be redrawn."}
    try:
        import warnings

        import pandas as pd

        from app.backtest.engine import run_backtest
        from app.backtest.risk import RiskConfig
        from app.data.importer import import_csv
        from app.search.strategy_space import FAMILIES, build_strategy_from_spec

        warnings.filterwarnings("ignore")
        fam = FAMILIES.get(row.get("family"))
        if fam is None:
            return {"ok": False, "message": "That strategy family isn't available in this version of the app."}
        df = import_csv(str(path)).dataframe
        ts = pd.to_datetime(df["timestamp"])
        keep = (ts >= pd.Timestamp(ctx["start"])) & (ts <= pd.Timestamp(ctx["end"]) + pd.Timedelta(days=1))
        df = df.loc[keep].reset_index(drop=True)
        if len(df) < 100:
            return {"ok": False, "message": "Not enough data in the recorded window to redraw the curve."}
        pip = 0.01 if "JPY" in ctx["dataset"].upper() else (0.1 if "XAU" in ctx["dataset"].upper() else 0.0001)
        risk = RiskConfig(initial_balance=ctx["account"], risk_value=ctx["risk_pct"], pip_size=pip,
                          spread_pips=ctx["spread"], slippage_pips=ctx["slippage"], commission_per_trade=ctx["commission"])
        strat = build_strategy_from_spec({"source_type": "manual", "config": fam.build(ctx["params"])})
        res = run_backtest(df, strat, risk, lookahead_check="skip")
        st = res.statistics
        eq = res.equity_curve
        pts = [[str(t)[:10], round(float(v), 2)] for t, v in zip(eq["timestamp"], eq["equity"])]
        pts = _downsample(pts)
        stats = None
        if st is not None:
            stats = {
                "return_pct": _num(st.return_pct, 1), "net_profit": _num(st.net_profit, 0), "sharpe": _num(st.sharpe_ratio),
                "profit_factor": _num(st.profit_factor), "win_rate": _num(st.win_rate, 1), "max_drawdown_pct": _num(st.max_drawdown_pct, 1),
                "trades": int(st.total_trades), "expectancy": _num(st.expectancy, 2), "avg_r": _num(st.average_r, 3),
                "max_losing_streak": int(st.max_losing_streak),
            }
        extra: dict[str, Any] = {}
        try:   # eval-pass / payout chances, measured against a standard 100k evaluation so rows are comparable
            from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
            from app.prop.presets import get_preset

            if len(res.trades) >= 15:
                preset = get_preset("ftmo_100k")
                mc = run_monte_carlo(res.trades, preset.to_prop_rules(), MonteCarloConfig(n_simulations=400))
                extra = {"eval_pass_pct": _num(mc.per_attempt_pass_probability, 0), "payout_pct": _num(mc.per_attempt_payout_probability, 0),
                         "rules_name": "FTMO $100k Challenge"}
        except Exception:   # noqa: BLE001 -- the curve and stats are still useful without the Monte Carlo extras
            extra = {}
        return {"ok": True, "points": pts, "stats": stats, "split": ctx.get("split"), "account": ctx["account"], **extra,
                "note": f"Redrawn just now from {ctx['dataset']} ({ctx['start']} to {ctx['end']}) with the saved settings and costs over the whole window. (Totals can differ slightly from the in-sample / held-out split run that rejected it.)"}
    except Exception as exc:   # noqa: BLE001 -- a detail panel must degrade to a message, never a 500
        return {"ok": False, "message": f"Couldn't redraw the curve ({type(exc).__name__}). The saved reason and scores above are still correct."}
