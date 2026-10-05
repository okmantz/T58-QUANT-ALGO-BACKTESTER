"""
Crash reporting (v7, worker D, 2026-10-05) -- Sentry opt-in wiring.

Design rules (hard):
  1. DISABLED BY DEFAULT. Nothing is ever sent unless the user has
     explicitly opted in -- via the Settings checkbox ("Send anonymous
     crash reports") or the first-run prompt. See is_opted_in() /
     set_opt_in().
  2. sentry-sdk is an OPTIONAL dependency. The import is lazy and guarded:
     a machine without it installed simply never initializes, and
     everything here degrades to a no-op instead of raising.
  3. Never breaks startup. init_crash_reporting_if_opted_in() catches
     everything; app.main calls it inside its own try/except as well.
  4. No PII. send_default_pii=False, no breadcrumbs containing market
     data or strategy internals beyond the exception itself.

Build-time DSN: read from the T58_SENTRY_DSN environment variable or
from app/_build_config.py's SENTRY_DSN (written by the release workflow --
see sentry_dsn()). Sentry DSNs are public client keys by design (Sentry's
own docs: safe to embed in shipped clients) -- but a missing DSN means
"not configured", which also disables everything. Owen's exact setup
steps are in docs/buyer/SENTRY_SETUP.md.

Privacy disclosure: enabling this sends crash tracebacks to Sentry;
that is disclosed in docs/buyer/PRIVACY_POLICY_TEMPLATE.md.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

TELEMETRY_SETTINGS_FILENAME = "telemetry_settings.json"

#: Environment variable carrying the Sentry DSN (injected at build time).
SENTRY_DSN_ENV_VAR = "T58_SENTRY_DSN"


@dataclass
class TelemetrySettings:
    """Local-only crash-reporting preferences. Stored in the same app
    config directory as the other JSON settings (see
    app.data.storage.get_app_base_dir); never leaves the machine."""
    crash_reporting_opt_in: bool = False


def _settings_path() -> Path:
    from app.data.storage import get_app_base_dir
    return get_app_base_dir() / TELEMETRY_SETTINGS_FILENAME


def load_telemetry_settings() -> TelemetrySettings:
    path = _settings_path()
    if not path.exists():
        return TelemetrySettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TelemetrySettings(
            crash_reporting_opt_in=bool(data.get("crash_reporting_opt_in", False)),
        )
    except Exception:  # noqa: BLE001 -- corrupt settings must never crash the app
        return TelemetrySettings()


def save_telemetry_settings(settings: TelemetrySettings) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


def is_opted_in() -> bool:
    """Whether the user has explicitly opted in to crash reporting."""
    return load_telemetry_settings().crash_reporting_opt_in


def set_opt_in(opted_in: bool) -> TelemetrySettings:
    """Record the user's opt-in/opt-out choice. Returns the saved settings.

    Wire this to the Settings-tab checkbox and to the first-run prompt's
    Yes/No buttons (see docs/buyer/SENTRY_SETUP.md for the exact UI
    insertion points).
    """
    settings = TelemetrySettings(crash_reporting_opt_in=bool(opted_in))
    save_telemetry_settings(settings)
    return settings


def sentry_dsn() -> str | None:
    """The configured Sentry DSN, or None when not configured.

    Precedence: T58_SENTRY_DSN env var (dev runs / CI) >
    app/_build_config.py's SENTRY_DSN (baked in at release-build time --
    see docs/buyer/SENTRY_SETUP.md step 2) > None. An empty/missing DSN
    disables crash reporting entirely -- opting in without a DSN is a
    silent no-op, never an error.
    """
    dsn = os.environ.get(SENTRY_DSN_ENV_VAR, "").strip()
    if dsn:
        return dsn
    try:
        from app import _build_config  # written by the release workflow; absent in dev
        dsn = str(getattr(_build_config, "SENTRY_DSN", "") or "").strip()
        return dsn or None
    except Exception:  # noqa: BLE001 -- no build config is the normal dev case
        return None


def sentry_available() -> bool:
    """Whether sentry_sdk is importable on this machine."""
    try:
        import sentry_sdk  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def init_crash_reporting_if_opted_in() -> bool:
    """Initialize Sentry ONLY if the user opted in AND a DSN is configured
    AND sentry_sdk is installed. Returns True when Sentry was actually
    initialized. Never raises -- any failure degrades to "no crash
    reporting" rather than breaking startup.
    """
    try:
        if not is_opted_in():
            return False
        dsn = sentry_dsn()
        if not dsn:
            log.info("Crash reporting opted in but no %s configured -- disabled.",
                     SENTRY_DSN_ENV_VAR)
            return False
        import sentry_sdk
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0.0,   # crash reports only, no performance tracing
            send_default_pii=False,    # never attach user/machine PII
            attach_stacktrace=True,
        )
        log.info("Crash reporting initialized (Sentry, user opted in).")
        return True
    except Exception as exc:  # noqa: BLE001 -- telemetry must never break startup
        log.warning("Crash reporting init failed (continuing without it): %s", exc)
        return False


def capture_exception_safe(exc: BaseException) -> None:
    """Report one exception to Sentry if (and only if) crash reporting is
    active. Safe to call anywhere -- no-ops when disabled/unconfigured.
    Never raises."""
    try:
        import sentry_sdk
        if sentry_sdk.Hub.current.client is not None:
            sentry_sdk.capture_exception(exc)
    except Exception:  # noqa: BLE001
        pass
