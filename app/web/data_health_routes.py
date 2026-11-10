"""Data-health review: plain-words problem cards for one stored file, each with a one-click fix that saves a cleaned COPY."""
from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import quote

from flask import Blueprint, redirect, render_template, request, url_for

from app.data.health_cards import FIXES, apply_fix, build_cards
from app.data.importer import import_csv
from app.data.storage import resolve_stored_dataset, store_dataset_bytes

data_health_bp = Blueprint("data_health", __name__)


def _load(name: str):
    path = resolve_stored_dataset(name)
    if path is None:
        return None, "That file is not in your data folder."
    result = import_csv(path)
    if not result.is_valid:
        return None, "Could not read the file: " + "; ".join(result.errors)
    return result.dataframe, None


@data_health_bp.route("/data-center/review")
def data_review():
    name = (request.args.get("name") or "").strip()
    df, error = _load(name)
    cards = build_cards(df, PurePosixPath(name).name) if df is not None else []
    return render_template("data_review.html", active_page="data_center", name=name, cards=cards, error=error,
                           bars=(len(df) if df is not None else 0), notice=request.args.get("notice"))


@data_health_bp.route("/data-center/fix", methods=["POST"])
def data_fix():
    name = (request.form.get("name") or "").strip()
    fix = (request.form.get("fix") or "").strip()
    review = lambda msg: redirect(url_for("data_health.data_review", name=name, notice=msg))  # noqa: E731
    if fix not in FIXES:
        return review("Unknown fix.")
    df, error = _load(name)
    if df is None:
        return review(error)
    cleaned, summary = apply_fix(df, fix)
    stem = PurePosixPath(name)
    out_rel = str(stem.with_name(f"{stem.stem}_{fix}.csv"))
    store_dataset_bytes(cleaned.to_csv(index=False).encode("utf-8"), out_rel)
    return review(f"{summary} Saved as a new file ({PurePosixPath(out_rel).name}); your original is untouched.")
