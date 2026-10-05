# Sentry crash-reporting setup (Owen's steps)

Crash reporting is **code-complete and OFF by default** — nothing sends
until a buyer opts in AND a DSN is configured. These are the exact steps
to make it live. Nothing here costs money (Sentry free tier).

## What the code does (already built)

- `app/telemetry.py` — opt-in state (`telemetry_settings.json`, default
  **off**), lazy/guarded `sentry_sdk` import, `init_crash_reporting_if_opted_in()`.
- `app/main.py` — one clearly-marked hook at the top of `main()`.
- `sentry-sdk` is intentionally NOT a hard dependency in
  `config/requirements.txt` (lazy import instead), so CI machines without
  it are unaffected.

## Step 1 — Create the Sentry account & project (15 min)

1. Go to https://sentry.io and sign up (free tier).
2. **Projects → Create Project** → platform **Python** → name it
   `t58-backtester` (any name works).
3. Open the project → **Settings → Client Keys (DSN)** → copy the DSN.
   It looks like `https://<key>@o<org>.ingest.sentry.io/<project>`.
   (Sentry DSNs are public client keys by design — safe to bake into the
   shipped app. They can only *submit* crash events, never read data.)

## Step 2 — Bake the DSN into release builds (5 min)

The app reads the DSN from the `T58_SENTRY_DSN` environment variable at
startup (same pattern as the license master-key hash). For buyer builds,
inject it at build time:

**[OWEN: DECIDE]** — pick one:

- **Option A (recommended):** add a GitHub Actions secret named
  `T58_SENTRY_DSN`, then in `build-exe.yml` / `build-web-exe.yml` add it
  to the PyInstaller build step's environment:
  ```yaml
  env:
    T58_SENTRY_DSN: ${{ secrets.T58_SENTRY_DSN }}
  ```
  and freeze it in via a tiny build step that writes it into the bundle
  (PyInstaller can't read Actions env at *buyer* runtime — the value must
  be baked in; e.g. write it to `app/_build_config.py` during the
  workflow and have `telemetry.sentry_dsn()` read that file as a
  fallback). [Worker C owns build-exe.yml/build-web-exe.yml — hand them
  this snippet.]
- **Option B:** for your own dev builds, just
  `set T58_SENTRY_DSN=<dsn>` (Windows) / `export T58_SENTRY_DSN=<dsn>`
  before launching — no code change needed.

Until EITHER is done, opting in is a silent no-op (logged, never an
error).

## Step 3 — Add sentry-sdk to the built app (5 min)

Because the import is lazy/guarded, the app *runs* without it — but no
reports can send until it's installed:

1. Add to `config/requirements.txt`:
   ```
   # Crash reporting (optional at runtime -- app.telemetry guards the
   # import; only needed so release builds can actually send reports).
   sentry-sdk>=2.0.0
   ```
2. The exe workflows install from `config/requirements.txt`, so release
   builds pick it up automatically.

## Step 4 — Wire the two UI surfaces (30–60 min, UI owner)

The backend functions exist; the visible surfaces need wiring in the UI
code (desktop `main_window.py` Settings tab + web settings page):

1. **Settings checkbox** — "Send anonymous crash reports to help fix
   bugs (off by default)":
   ```python
   from app.telemetry import is_opted_in, set_opt_in
   # checkbox initial state:
   checkbox.set(is_opted_in())
   # on toggle:
   set_opt_in(checkbox.is_checked())
   ```
   After a user opts in at runtime, call
   `init_crash_reporting_if_opted_in()` once more so it takes effect
   without a restart.
2. **First-run prompt** — a Yes/No dialog on first launch:
   ```python
   from app.telemetry import set_opt_in, init_crash_reporting_if_opted_in
   answer = ask_yes_no(
       "Help improve T58?",
       "Send anonymous crash reports if the app ever crashes?\n\n"
       "This sends only the error traceback, app version, and OS version "
       "— never your strategies, data, or personal info. You can change "
       "this anytime in Settings → Privacy.")
   set_opt_in(answer)
   if answer:
       init_crash_reporting_if_opted_in()
   ```
3. Show first-run prompt only when no `telemetry_settings.json` exists
   yet (i.e. the user has never answered).

## Step 5 — Verify (5 min)

1. Set `T58_SENTRY_DSN` to your DSN, launch the app, opt in.
2. In a Python console against the same environment:
   ```python
   from app.telemetry import init_crash_reporting_if_opted_in, capture_exception_safe
   assert init_crash_reporting_if_opted_in() is True
   try:
       1/0
   except ZeroDivisionError as e:
       capture_exception_safe(e)
   ```
3. Check Sentry → Issues: the ZeroDivisionError appears within a minute.

## Privacy disclosure

Crash reporting is disclosed in
`docs/buyer/PRIVACY_POLICY_TEMPLATE.md` (section 3) — keep that template
in sync with whatever you actually ship.
