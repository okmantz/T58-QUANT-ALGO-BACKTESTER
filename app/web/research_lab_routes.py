"""Research Labs web routes (``/research-labs``): the seven research engines.

Thin wrappers: parse the form, load data, call ``app.research_labs.<lab>.run``, return the experiment record as JSON.
The page (templates/research_lab.html) renders the record generically -- headline, warnings, every table, assumptions,
data fingerprint and spec hash -- so what the engine assumed always travels with its answer.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from flask import Blueprint, abort, jsonify, render_template, request

from app.research_labs import common as C

research_lab_bp = Blueprint("research_labs", __name__, url_prefix="/research-labs")

_DATASETS = {"label": "Market data in the app", "help": "Pick one or more datasets already loaded in the app, or upload CSV files."}

LABS: dict[str, dict] = {
    "volume-profile": {
        "title": "Volume Profile & Market Profile Lab", "badge": "Structure", "module": "volume_profile", "markets": 1,
        "summary": "Volume by price, POC, VAH/VAL, HVN/LVN and TPO letters, then tests whether liquidity-sweep setups behave differently at value-area edges and low-volume nodes.",
        "why": "Price accepted vs rejected at high-volume areas changes how sweeps resolve.",
        "data": "Granular volume bars (1-5 min) or trades; TPO is built from timestamped bars.",
        "fields": [("n_bins", "Price bins", "number", 60), ("value_area_pct", "Value area %", "number", 70), ("horizon", "Outcome horizon (bars)", "number", 12),
                   ("sweep_lookback", "Sweep lookback (bars)", "number", 20), ("tpo_minutes", "TPO period (minutes)", "number", 30), ("oos_fraction", "Out-of-sample fraction", "number", 0.3)],
    },
    "order-flow": {
        "title": "Footprint & Order-Flow Lab", "badge": "Flow", "module": "order_flow", "markets": 1, "trades_upload": True,
        "summary": "Bid/ask volume, diagonal imbalances, bar delta, CVD and absorption; tests whether entries after sweeps do better when aggression fails to move price.",
        "why": "Failed aggression (absorption) is the order-flow footprint of a real sweep reversal.",
        "data": "Trade-level data with aggressor side is best. OHLCV alone gives only a labelled PROXY -- it cannot reconstruct true bid/ask flow.",
        "fields": [("horizon", "Outcome horizon (bars)", "number", 12), ("sweep_lookback", "Sweep lookback (bars)", "number", 20), ("bar_minutes", "Bar minutes (trades mode)", "number", 5),
                   ("tick_size", "Tick size (trades mode)", "number", 0.25), ("imbalance_ratio", "Diagonal imbalance ratio", "number", 3), ("oos_fraction", "Out-of-sample fraction", "number", 0.3)],
    },
    "order-book": {
        "title": "Liquidity & Order-Book Lab", "badge": "Microstructure", "module": "order_book_lab", "markets": 0, "book_upload": True,
        "summary": "Resting depth, queue imbalance, replenishment, cancellations and price movement after large market orders; does visible liquidity mean rejection, continuation or nothing?",
        "why": "Tests whether displayed liquidity predicts anything before you build on it.",
        "data": "Needs Level 2 snapshots (MBO even better). Resting liquidity is never inferred from candles -- without L2 this lab stops and says so. A demo mode runs on clearly-labelled SYNTHETIC data.",
        "fields": [("levels", "Book levels", "number", 5), ("horizon", "Outcome horizon (events)", "number", 20), ("large_quantile", "'Large order' quantile", "number", 0.98),
                   ("oos_fraction", "Out-of-sample fraction", "number", 0.3), ("demo", "Run on synthetic demo data", "checkbox", False)],
    },
    "lead-lag": {
        "title": "Cross-Asset Lead-Lag Analyzer", "badge": "Cross-asset", "module": "lead_lag", "markets": 2,
        "summary": "A lead-lag matrix with correlation, lagged predictive performance, stability across time blocks and statistical uncertainty -- not just a heatmap.",
        "why": "ES, NQ, GC, yields, the dollar and crude move together; which one leads, at what horizon, and does it survive out-of-sample?",
        "data": "Two or more instruments (any timeframe; resampled to a common bucket).",
        "fields": [("freq", "Bucket", "text", "15min"), ("max_lag", "Max lag (buckets)", "number", 8), ("cost_bps", "Round-trip cost (bps)", "number", 1.0),
                   ("blocks", "Stability blocks", "number", 6), ("oos_fraction", "Out-of-sample fraction", "number", 0.3)],
    },
    "stat-arb": {
        "title": "Statistical Arbitrage & Spread Lab", "badge": "Stat Arb", "module": "stat_arb", "markets": 2, "exact_markets": 2,
        "summary": "Cointegration, half-life, rolling hedge ratio and spread mean reversion, plus a stability monitor that separates a stretched spread from a structurally broken one.",
        "why": "Cheap-looking spreads that have structurally broken are how pairs traders lose.",
        "data": "Exactly two instruments: A and B (for a calendar spread: A = front, B = back).",
        "fields": [("mode", "Mode (pair / calendar)", "text", "pair"), ("freq", "Bucket", "text", "1h"), ("hedge_window", "Hedge-ratio window", "number", 250),
                   ("z_entry", "Entry |z|", "number", 2.0), ("z_exit", "Exit |z|", "number", 0.5), ("z_stop", "Stop |z|", "number", 4.0),
                   ("cost_bps", "Round-trip cost (bps)", "number", 2.0), ("oos_fraction", "Out-of-sample fraction", "number", 0.3)],
    },
    "event-study": {
        "title": "Economic Event Study Lab", "badge": "Events", "module": "event_study", "markets": 1, "events_upload": True,
        "summary": "CPI, PCE, NFP, FOMC, auctions: pre-event volatility, post-event expansion, adverse excursion, and event days vs matched non-event sessions; initial reaction vs continuation.",
        "why": "Know whether your strategy should trade, shrink or sit out around releases.",
        "data": "Bars plus an events CSV with a 'timestamp' column (and optional 'event'), in the SAME timezone as the bars. Or generate the NFP schedule.",
        "fields": [("pre_bars", "Pre-event bars", "number", 12), ("post_bars", "Post-event bars", "number", 12), ("reaction_bars", "Initial-reaction bars", "number", 3),
                   ("cont_bars", "Continuation bars", "number", 12), ("controls_per_event", "Controls per event", "number", 3), ("nfp", "Generate NFP schedule (first Fridays 08:30) instead of uploading", "checkbox", False)],
    },
    "hypothesis": {
        "title": "Alpha Discovery & Hypothesis Lab", "badge": "Hypothesis", "module": "hypothesis_lab", "markets": 1, "spec_editor": True,
        "summary": "Pre-registered experiments: hypothesis, sample, feature, baseline, outcome and acceptance criteria are fixed before running; reports effect size, CI, sample sizes, multiple-testing adjustment and independent validation.",
        "why": "Stops you from fooling yourself: every variant you try raises the bar for the next.",
        "data": "One OHLCV dataset (500+ bars). The spec is JSON and its hash is stored with the result.",
        "fields": [],
    },
}


def _slug_or_404(slug: str) -> dict:
    lab = LABS.get(slug)
    if lab is None:
        abort(404)
    return lab


def _stored_datasets() -> list[str]:
    try:
        from app.data.storage import list_stored_datasets

        return [d.name for d in list_stored_datasets()]
    except Exception:  # noqa: BLE001
        return []


def _import_path(path: str) -> pd.DataFrame:
    from app.data.importer import import_csv

    res = import_csv(path)
    if res.dataframe is None:
        raise C.ResearchLabError("Could not read the data: " + "; ".join(i.message for i in res.issues if i.level == "error"))
    return res.dataframe


def _load_markets() -> dict[str, pd.DataFrame]:
    from app.data.importer import import_csv_bytes
    from app.data.storage import get_raw_data_dir

    out: dict[str, pd.DataFrame] = {}
    raw = get_raw_data_dir().resolve()
    for name in request.form.getlist("datasets"):
        name = name.strip()
        if not name:
            continue
        path = (raw / name).resolve()
        if raw not in path.parents:
            raise C.ResearchLabError(f"Dataset '{name}' is not inside the app's data folder.")
        out[Path(name).stem] = _import_path(str(path))
    for up in request.files.getlist("uploads"):
        if not up or not up.filename:
            continue
        res = import_csv_bytes(up.read(), filename=up.filename)
        if res.dataframe is None:
            raise C.ResearchLabError(f"Could not read '{up.filename}' as OHLCV data.")
        out[Path(up.filename).stem] = res.dataframe
    return out


def _num(form, key, default, cast=float):
    raw = (form.get(key) or "").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise C.ResearchLabError(f"'{key}' must be a number.") from exc


def _read_table(field: str) -> pd.DataFrame | None:
    up = request.files.get(field)
    if up is None or not up.filename:
        return None
    return pd.read_csv(io.BytesIO(up.read()))


@research_lab_bp.route("/")
def hub():
    labs = [{"slug": s, **{k: v for k, v in cfg.items() if k in ("title", "badge", "summary", "why", "data")}} for s, cfg in LABS.items()]
    return render_template("research_labs.html", labs=labs, recent=C.list_experiments(limit=12), active_page="research_labs")


@research_lab_bp.route("/<slug>")
def lab_page(slug: str):
    lab = _slug_or_404(slug)
    cfg = {k: v for k, v in lab.items() if k != "module"}
    spec = None
    if slug == "hypothesis":
        from app.research_labs.hypothesis_lab import default_spec

        spec = json.dumps(default_spec(), indent=2)
    mod = __import__(f"app.research_labs.{lab['module']}", fromlist=["LAB"])
    return render_template("research_lab.html", slug=slug, lab=cfg, datasets=_stored_datasets(), default_spec=spec,
                           recent=C.list_experiments(lab=mod.LAB, limit=8), active_page="research_labs")


@research_lab_bp.route("/<slug>/run", methods=["POST"])
def run_lab(slug: str):
    lab = _slug_or_404(slug)
    form = request.form
    seed = _num(form, "seed", C.DEFAULT_SEED, int)
    try:
        mod = __import__(f"app.research_labs.{lab['module']}", fromlist=["run"])
        if slug == "order-book":
            book, trades = _read_table("book"), _read_table("trades")
            rec = mod.run(book, trades, label="uploaded book", levels=_num(form, "levels", 5, int), horizon=_num(form, "horizon", 20, int),
                          large_quantile=_num(form, "large_quantile", 0.98), oos_fraction=_num(form, "oos_fraction", 0.3),
                          demo=form.get("demo") in ("on", "true", "1"), seed=seed)
        else:
            markets = _load_markets()
            need = lab.get("markets", 1)
            if len(markets) < need:
                raise C.ResearchLabError(f"This lab needs {'two or more' if need == 2 and slug == 'lead-lag' else str(need)} dataset(s); got {len(markets)}.")
            first_name, first = next(iter(markets.items()))
            if slug == "volume-profile":
                rec = mod.run(first, label=first_name, n_bins=_num(form, "n_bins", 60, int), value_area_pct=_num(form, "value_area_pct", 70.0),
                              horizon=_num(form, "horizon", 12, int), sweep_lookback=_num(form, "sweep_lookback", 20, int),
                              oos_fraction=_num(form, "oos_fraction", 0.3), tpo_minutes=_num(form, "tpo_minutes", 30, int), seed=seed)
            elif slug == "order-flow":
                rec = mod.run(first, trades=_read_table("trades"), label=first_name, horizon=_num(form, "horizon", 12, int),
                              sweep_lookback=_num(form, "sweep_lookback", 20, int), oos_fraction=_num(form, "oos_fraction", 0.3),
                              bar_minutes=_num(form, "bar_minutes", 5, int), tick_size=_num(form, "tick_size", 0.25),
                              imbalance_ratio=_num(form, "imbalance_ratio", 3.0), seed=seed)
            elif slug == "lead-lag":
                rec = mod.run(markets, freq=(form.get("freq") or "15min").strip(), max_lag=_num(form, "max_lag", 8, int),
                              oos_fraction=_num(form, "oos_fraction", 0.3), cost_bps=_num(form, "cost_bps", 1.0),
                              blocks=_num(form, "blocks", 6, int), seed=seed)
            elif slug == "stat-arb":
                items = list(markets.items())[:2]
                mode = (form.get("mode") or "pair").strip().lower()
                if mode not in ("pair", "calendar"):
                    raise C.ResearchLabError("Mode must be 'pair' or 'calendar'.")
                rec = mod.run(dict(items), mode=mode, freq=(form.get("freq") or "1h").strip(), hedge_window=_num(form, "hedge_window", 250, int),
                              z_entry=_num(form, "z_entry", 2.0), z_exit=_num(form, "z_exit", 0.5), z_stop=_num(form, "z_stop", 4.0),
                              cost_bps=_num(form, "cost_bps", 2.0), oos_fraction=_num(form, "oos_fraction", 0.3), seed=seed)
            elif slug == "event-study":
                if form.get("nfp") in ("on", "true", "1"):
                    d = C.prepare_ohlcv(first)
                    events = mod.nfp_schedule(str(d["timestamp"].iloc[0].date()), str(d["timestamp"].iloc[-1].date()))
                else:
                    events = _read_table("events")
                    if events is None:
                        raise C.ResearchLabError("Upload an events CSV (column 'timestamp', optional 'event') or tick 'Generate NFP schedule'.")
                rec = mod.run(first, events, label=first_name, pre_bars=_num(form, "pre_bars", 12, int), post_bars=_num(form, "post_bars", 12, int),
                              reaction_bars=_num(form, "reaction_bars", 3, int), cont_bars=_num(form, "cont_bars", 12, int),
                              controls_per_event=_num(form, "controls_per_event", 3, int), seed=seed)
            else:  # hypothesis
                try:
                    spec = json.loads(form.get("spec") or "")
                except ValueError as exc:
                    raise C.ResearchLabError(f"The spec is not valid JSON: {exc}") from exc
                rec = mod.run(first, spec, label=first_name, seed=seed)
        C.release_memory()
        return jsonify({"ok": True, "record": rec.to_dict()})
    except C.ResearchLabError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except (ValueError, KeyError) as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"The lab hit an unexpected error: {type(exc).__name__}: {exc}"}), 500


@research_lab_bp.route("/api/experiments")
def api_experiments():
    return jsonify({"experiments": C.list_experiments(lab=request.args.get("lab") or None, limit=int(request.args.get("limit", 30)))})


@research_lab_bp.route("/api/experiment/<lab>/<exp_id>")
def api_experiment(lab: str, exp_id: str):
    if not exp_id.isalnum() or not lab.replace("_", "").isalnum():
        abort(404)
    rec = C.load_experiment(lab, exp_id)
    if rec is None:
        abort(404)
    return jsonify(rec)


@research_lab_bp.route("/api/spec-template")
def api_spec_template():
    from app.research_labs.hypothesis_lab import default_spec

    return jsonify(default_spec())


@research_lab_bp.route("/api/journal-tags", methods=["POST"])
def api_journal_tags():
    """Turn a SURVIVING finding of a saved experiment into Trading Journal tags (group "Research findings")."""
    from app.research_labs.journal_bridge import GROUP_NAME, tags_from_record
    from app.web import journal_store

    body = request.get_json(silent=True) or {}
    lab, exp_id = str(body.get("lab") or ""), str(body.get("id") or "")
    if not lab.replace("_", "").isalnum() or not exp_id.isalnum():
        return jsonify({"ok": False, "error": "Unknown experiment."}), 400
    rec = C.load_experiment(lab, exp_id)
    if rec is None:
        return jsonify({"ok": False, "error": "Experiment not found."}), 404
    tags = tags_from_record(rec)
    if not tags:
        return jsonify({"ok": True, "added": [], "message": "No finding in this experiment survived multiple-testing correction and out-of-sample validation, so there is nothing to tag."})
    return jsonify({"ok": True, "added": journal_store.add_tags(GROUP_NAME, tags), "tags": tags, "group": GROUP_NAME})
