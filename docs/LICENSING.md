# Licensing — how activation works

The app has three ways to activate. They can coexist; the app picks whichever applies, in the order below.

| Mode | Who it's for | Needs internet? | Needs a server you run? |
|---|---|---|---|
| **Master key** | You (the owner), testing, support | No — fully offline | No |
| **Google Sheet license keys** (primary, for customers) | Buyers | Yes, once; then a 3-day offline grace between checks | No — a Google Sheet + Apps Script |
| **License server** (optional / legacy) | Only if you outgrow the sheet | Yes | Yes (`license_server/`) |

## 1. Master key (still supported)

The master key is **not** gone. A SHA-256 hash of your key is compiled into every build
(`_DEFAULT_MASTER_KEY_HASH` in `app/licensing/client.py`), so your key activates with zero configuration, offline, on any
machine — from source, in a CI-built `.exe`, anywhere. The key itself is never stored in the repo; only its hash is.

- Always shows as plan **Lifetime**.
- To use a different key: compute `SHA-256(KEY.strip().upper())` and either
  - set the `T58_MASTER_LICENSE_KEY_HASH` environment variable (instant, no rebuild), or
  - bake it into `app/licensing/build_config.py` as `MASTER_LICENSE_KEY_HASH` (see `build_config.py.example`).
- Precedence: env var > `build_config.py` > compiled-in default.
- Rotating the key only affects builds that carry the new hash; already-shipped builds keep honouring the hash they were built with.

## 2. Google Sheet license keys (primary customer flow)

Customers' keys live in a Google Sheet that you edit by hand; a small Apps Script web app (`license_sheet/Code.gs`) answers
"is this key valid?" for exactly one key at a time — the sheet itself is never exposed.

- Setup walk-through: [`license_sheet/SETUP.md`](../license_sheet/SETUP.md). After editing `Code.gs`, redeploy
  (Deploy > Manage deployments > Edit > New version).
- Point the app at it with the `T58_LICENSE_SHEET_URL` environment variable, or bake `LICENSE_SHEET_URL` into
  `app/licensing/build_config.py` at build time.
- The Account page shows the sheet's PLAN column for the activated key (Pro Monthly, Pro Annual, Lifetime, Trial,
  Complimentary). Users cannot change it.
- **Revoking** a key = changing its row in the sheet; the app notices on its next check.
- **Offline grace:** if the sheet can't be reached (no internet, Apps Script busy), the last successful check is honoured for
  **3 days** (`OFFLINE_GRACE_DAYS`). After that the app asks the user to reconnect.

## 3. License server (optional / legacy)

`license_server/` is a small Flask + SQLite service with an admin CLI. It is **not needed** if you use the sheet. If you do
use it, set `T58_LICENSE_SERVER_URL` (or bake it in). See `license_server/README.md` and `DEPLOY.md`.
There is also an allowlist mode (`T58_LICENSE_ALLOWLIST_URL`) for a plain hosted list of keys.

## Precedence summary

Remote mode: **allowlist URL > sheet URL > license-server URL**. The master key is checked independently and always works
offline, regardless of which remote mode is configured.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "License key invalid" | Paste the key exactly (keys are uppercase, no spaces). |
| "no license server configured" | Set `T58_LICENSE_SHEET_URL` (or bake it in), or use the master key. |
| "license sheet is busy or not set up" | Apps Script is cold/throttled or not deployed as a web app — retry; check `SETUP.md`. |
| Master key rejected | The build/env hash doesn't match the key you typed; check `T58_MASTER_LICENSE_KEY_HASH`. |
