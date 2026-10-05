# Deploying the T58 License Server — exact steps for Owen

This gets the license server (`license_server/`, the Flask app that
activates/validates every customer copy of the backtester) running on
Render in about 20 minutes, then wires Whop (payments) and email (key
delivery) into it, then points the shipped `.exe` at it.

**Cost:** $0 on Render's free tier while testing. Read the disk note in
step 2 before you take your first paying customer.

**Nothing here spends money, creates accounts on your behalf, or touches
your creditors.** You do each step below yourself; this doc is the
checklist.

---

## Step 0 — What you'll end up with

```
Customer buys on Whop
        │
        ▼
Whop webhook ──► your Render server ──► auto-creates license + emails the key
        │                                        │
        │                                        ▼
        │                              customer pastes key into the app
        │                                        │
        └──────── (server validates) ◄───────────┘
```

---

## Step 1 — Create a Render account and connect GitHub

1. Go to **https://render.com** and sign up (the free account is fine).
2. During onboarding, connect your GitHub account and give Render access
   to the **`T58-QUANT-ALGO-BACKTESTER`** repository.

## Step 2 — Deploy from the blueprint

1. In the Render dashboard: **New → Blueprint**.
2. Select the `T58-QUANT-ALGO-BACKTESTER` repo. Render detects
   `render.yaml` at the repo root automatically.
3. Review the plan (service name `t58-license-server`, Python, gunicorn,
   health check on `/health`) and click **Apply**.
4. Wait for the first build to finish (~2–3 minutes). The service gets a
   public URL like:

   ```
   https://t58-license-server.onrender.com
   ```

   **Write this URL down** — you need it in steps 4, 5, and 7.

> **Disk / free-tier honesty note.** The blueprint asks for a 1 GB
> persistent disk so the SQLite license database (`licenses.db`) survives
> redeploys. Render only attaches disks to **paid** instance types
> (Starter, ~$7/mo — verify current pricing). On the free tier there is
> no persistent disk: `licenses.db` lives on ephemeral disk and is
> **wiped on every redeploy/restart**, taking your license list with it.
>
> - Testing with zero customers: free tier is fine.
> - First paying customer: switch the service to Starter ($7/mo) in
>   Render (**service → Settings → Instance Type**). The disk then works
>   as configured and nothing else changes.
> - If you insist on staying free with real customers: after every sale,
>   export a backup — see step 8 — and accept that a redeploy wipes the
>   live DB until you restore it.

## Step 3 — Set the environment variables on Render

In the Render dashboard: open `t58-license-server` → **Environment** →
**Add Environment Variable**. Set these:

| Variable | Value | Notes |
|---|---|---|
| `T58_LICENSE_ADMIN_TOKEN` | a long random string | Generate: `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Gates every `/admin/*` endpoint — treat like a password. |
| `T58_LICENSE_DB_PATH` | *(pre-filled by the blueprint)* | `/opt/render/project/src/license_server/data/licenses.db` |
| `SMTP_HOST` | `smtp.gmail.com` | See the Gmail app-password walkthrough below. |
| `SMTP_PORT` | `587` | Pre-filled. |
| `SMTP_USER` | your Gmail address | The address the key emails come *from*. |
| `SMTP_PASS` | your Gmail **app password** | NOT your normal Gmail password — see below. |
| `SMTP_FROM` | `T58 Trading <you@gmail.com>` | Optional; defaults to `SMTP_USER`. |
| `T58_WHOP_WEBHOOK_SECRET` | *(leave empty for now)* | You paste this in step 5 after creating the Whop webhook. |

Click **Save Changes** — Render redeploys with the new env vars.

### Gmail app password (5 minutes, one time)

Google blocks plain-password SMTP logins, so you need an app password:

1. Go to **https://myaccount.google.com** → **Security**.
2. Turn on **2-Step Verification** if it isn't already.
3. **Security → 2-Step Verification → App passwords** (near the bottom).
4. Create one named `T58 license server`. Google shows a 16-character
   password — paste it as `SMTP_PASS` on Render. You can't see it again,
   so copy it now.

## Step 4 — Create the Whop product

1. Go to your Whop dashboard: **https://whop.com/dashboard**
   (you already sell through Whop — use the same account).
2. Create a product for the backtester (e.g. **"T58 Quant Algo
   Backtester — Monthly License"**), priced however you want, with
   whatever billing interval you want. The server doesn't care about the
   price — it only listens for membership events.
3. Note the product's name — the server records it as the license `plan`.

## Step 5 — Wire the Whop webhook (auto-revoke + auto-issue)

1. In the Whop dashboard: **Developer → Webhooks → Create Webhook**.
2. **URL:** `https://t58-license-server.onrender.com/webhook/whop`
   (your real Render URL from step 2 + `/webhook/whop`).
3. Subscribe to at least these two events:
   - `membership.went_valid`
   - `membership.went_invalid`
4. Whop shows you a signing secret starting with `whsec_...` — copy it.
5. Back on Render (**Environment** for `t58-license-server`), set
   `T58_WHOP_WEBHOOK_SECRET` to that secret and **Save Changes**.

What this buys you (already implemented in `license_server/app.py`):
- `membership.went_invalid` → the matching license is auto-**revoked**.
- `membership.went_valid` for a brand-new membership → a license key is
  auto-**created and emailed** to the buyer via your SMTP config, so a
  3 AM sale fulfills itself. If the email fails, the key is logged loudly
  on the server and you deliver it by hand with `admin_cli.py`.

Test it: buy your own product once (or use a 100%-off promo), and check
that a key email arrives and that canceling revokes it.

## Step 6 — Create your first license (and your own master key)

From your own machine, with the admin token:

```bash
cd license_server
export T58_LICENSE_SERVER_URL=https://t58-license-server.onrender.com
export T58_LICENSE_ADMIN_TOKEN=<the token from step 3>

# A license for a customer (emails them the key automatically):
python admin_cli.py create customer@email.com --plan monthly --days 30

# List / inspect / revoke / extend:
python admin_cli.py list
python admin_cli.py revoke T58-XXXX-XXXX-XXXX-XXXX
```

**Your own access — the master key (no server needed):** pick a long
random key phrase once, e.g.:

```bash
python -c "import secrets; print('T58-' + secrets.token_hex(8).upper())"
```

Compute its SHA-256 hash **exactly** the way the app checks it
(uppercased, whitespace stripped):

```bash
python -c "import hashlib; print(hashlib.sha256('T58-YOUR-KEY-HERE'.strip().upper().encode()).hexdigest())"
```

Save that 64-hex-character hash — it goes into the *build* in step 7,
never into the repo. Entering the key phrase in the app's activation
window gives you permanent offline activation in any build carrying the
hash.

## Step 7 — Point the shipped app at your server (build time)

The `.exe` must be built **knowing** your server URL and your master-key
hash. Both are baked in at build time by the GitHub Actions workflows
(`build-exe.yml` / `build-web-exe.yml`) from repository secrets — they
never live in the repo source.

1. GitHub repo → **Settings → Secrets and variables → Actions →
   New repository secret**. Add:
   - `T58_LICENSE_SERVER_URL` = `https://t58-license-server.onrender.com`
     (no trailing slash)
   - `T58_MASTER_LICENSE_KEY_HASH` = the 64-hex hash from step 6
2. Push a `v*` tag (e.g. `git tag v1.0.0 && git push origin v1.0.0`).
   The workflow writes `app/licensing/build_config.py` from those two
   secrets right before PyInstaller runs, so the built `.exe` carries
   your URL and your master-key hash — and nothing else does.
3. Local/dev builds: the same two values can be supplied as plain
   environment variables (`T58_LICENSE_SERVER_URL`,
   `T58_MASTER_LICENSE_KEY_HASH`) when you run from source — env vars
   take precedence over the baked-in values, which is also how you point
   a dev build at a staging server without rebuilding.

**If a release build has no server URL configured at all**, the app now
fails LOUDLY at startup with a clear "no license server configured"
message instead of dying mysteriously against a dead placeholder
domain. (The master-key activation path still works fully offline.)

**Rotate per release if you want:** the master-key hash is per-build —
old builds keep honoring their old hash until they're replaced, so
rotating is "new hash → new secrets → rebuild → re-ship".

## Step 8 — Backups (2 minutes, do this)

The license database is a single SQLite file. Back it up:

```bash
# From your machine, with the admin token set (step 6):
python admin_cli.py list > t58-licenses-backup-$(date +%F).json
```

Do this after every sale while you're on Render's free tier (step 2's
disk note), and weekly once you're on Starter. If the DB is ever wiped,
re-create the keys from the backup with `admin_cli.py create`.

## Troubleshooting

- **"Couldn't reach the license server" in the app** — the Render free
  tier **sleeps after 15 minutes idle**; the first activation attempt can
  take ~60 seconds to wake it. Retry once. (Another reason to move to
  Starter once you're selling.)
- **`/health` returns 404** — you're hitting the wrong URL; the blueprint
  serves the app at the domain root.
- **Webhook 401s in Whop's delivery log** — `T58_WHOP_WEBHOOK_SECRET`
  doesn't match the secret Whop shows for that webhook. Re-paste it.
- **Key email never arrives** — check Render logs for
  `delivery email FAILED`; almost always `SMTP_PASS` (must be the app
  password, not your Gmail password) or `SMTP_USER`.
- **You locked yourself out of admin** — the admin token only lives on
  Render (**Environment**) and wherever you saved it. Rotate it there if
  lost; the app's customers are unaffected.
