"""Objective mode for every optimizer / pipeline form: PROP FIRM (default) or NET PROFIT (live account).

``objective_mode`` is a single radio group (``_objective_mode_field.html``) included in every form that has a fitness metric. This module
installs one ``before_request`` hook that reads it and, for ``net_profit``, rewrites the posted ``fitness_metric`` to a net-profit metric
before any route sees the form -- so the ~20 routes that read ``form.get("fitness_metric", ...)`` all honour the mode without each one
being edited, and no route can accidentally keep a prop-firm metric (which would put the prop rules back into a live-account test).

Prop-firm testing is the default everywhere: a form that does not post ``objective_mode`` (an old bookmark, a script) behaves exactly as
before, with the prop-lifecycle metric as the default fitness.
"""
from __future__ import annotations

from flask import g, request
from werkzeug.datastructures import ImmutableMultiDict

from app.optimize.refinement import fitness_for_objective_mode
from app.prop.lifecycle import DEFAULT_OBJECTIVE_MODE, OBJECTIVE_MODE_LABELS, OBJECTIVE_NET_PROFIT, normalize_objective_mode

_FORM_TYPES = ("application/x-www-form-urlencoded", "multipart/form-data")


def apply_objective_mode_to_form() -> None:
    """before_request: honour ``objective_mode`` on POSTed forms (see module docstring). Never raises."""
    try:
        if request.method != "POST" or request.mimetype not in _FORM_TYPES:
            return
        form = request.form
        if "objective_mode" not in form:
            return
        mode = normalize_objective_mode(form.get("objective_mode"))
        g.objective_mode = mode
        if mode != OBJECTIVE_NET_PROFIT:
            return
        rewritten = form.copy()
        rewritten["fitness_metric"] = fitness_for_objective_mode(mode, form.get("fitness_metric"))
        rewritten["objective_mode"] = mode
        request.form = ImmutableMultiDict(rewritten)
    except Exception:  # noqa: BLE001 -- a hook that fails must never take a page down
        return


def current_objective_mode(form=None) -> str:
    """The mode of the current request (form value, else g, else the prop-firm default)."""
    try:
        src = form if form is not None else request.form
        if src is not None and "objective_mode" in src:
            return normalize_objective_mode(src.get("objective_mode"))
    except Exception:  # noqa: BLE001
        pass
    return getattr(g, "objective_mode", DEFAULT_OBJECTIVE_MODE)


def install(app) -> None:
    app.before_request(apply_objective_mode_to_form)
    app.jinja_env.globals["objective_modes"] = OBJECTIVE_MODE_LABELS
    app.jinja_env.globals["default_objective_mode"] = DEFAULT_OBJECTIVE_MODE
