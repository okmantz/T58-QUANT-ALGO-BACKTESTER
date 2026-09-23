# T58 License Server

A small, self-hosted Flask app that is the only thing standing between a
copied ZIP of the T58 Quant Algo Backtester and someone actually being
able to use it. You run this yourself — no third-party licensing
service, no per-seat fees to anyone but Whop (which you're already using
for payment). SQLite storage, no managed database needed.

## What this is (and isn't)

- **Is:** an activation/validation server. The desktop app calls
  `/activate` once and `/validate` periodically; you manage licenses
  with `admin_cli.py`; Whop tells this server when someone's
  subscription lapses so it can auto-revoke.
- **Isn't:** a payment processor, a customer portal, or an email sender.
  Whop handles checkout and billing. You (or a small script) handle
  telling a *new* customer their license key after Whop notifies you of
  a sale — see "New customer flow" below for why that one step stays
  manual.

## 1. Deploy it somewhere free and always-on

Any of these work with zero cost for a low-volume product like this
(check current free-tier limits before committing — they change):

- **Render** (Web Service, free tier): connect this `license_server/`
  folder as its own repo (or a subfolder with a build command that `cd`s
  into it), start command `gunicorn -w 2 -b 0.0.0.0:$PORT app:app` (add
  `gunicorn` to `requirements.txt` if you go this route — the built-in
  `python app.py` dev server is fine for testing but says so itself:
  "do not use in production").
- **Fly.io** (free allowance): `fly launch` in this folder, same start
  command as above.
- **PythonAnywhere** (free tier): upload this folder, point a web app at
  `app.py`'s `app` object via their WSGI config.
- **Your own always-on machine** (a home server, a spare VPS you already
  pay for, etc.): `python app.py` behind a reverse proxy (Caddy/nginx)
  for HTTPS. HTTPS matters here — license keys and the admin token
  travel over these requests.

Whichever you pick, you need:
1. The service reachable at a public HTTPS URL.
2. Two environment variables set on it:
   - `T58_LICENSE_ADMIN_TOKEN` — a long random string you generate once
     (`python -c "import secrets; print(secrets.token_urlsafe(32))"`).
     This gates every `/admin/*` endpoint. Keep it secret — anyone with
     it can create/revoke licenses.
   - `T58_WHOP_WEBHOOK_SECRET` — see step 3 below (skip for now if
     you're not wiring up Whop yet).
3. SQLite needs a writable, *persistent* disk at whatever
   `T58_LICENSE_DB_PATH` points to (default: `licenses.db` next to
   `app.py`). Some free-tier hosts wipe local disk on redeploy — check
   whether your host offers a small persistent volume, and point
   `T58_LICENSE_DB_PATH` at it if so; otherwise your license list is
   only as durable as that host's local disk.

## 2. Point the desktop app at your server

In `app/licensing/client.py`, either:
- edit `DEFAULT_SERVER_URL` to your deployed URL before you build the
  .exe, or
- set the `T58_LICENSE_SERVER_URL` environment variable on whatever
  machine builds it.

Every activation/validation call from every customer's copy of the app
goes to this one URL.

## 3. New customer flow

When someone subscribes via Whop:

```
python admin_cli.py create their@email.com --plan monthly --days 30 \
  --whop-membership-id mem_xxxxx
```

(`--days 30` gives a fixed expiry as a safety net even if a webhook is
ever missed; omit it to rely entirely on the Whop webhook for
access control. `--whop-membership-id` is optional but strongly
recommended — it's what lets the webhook in step 4 auto-revoke this
exact license later without guessing by email.)

Then send them the printed `license_key` (Whop's own post-purchase
"thank you" page/email, or a Whop "hosted experience" content block, are
both reasonable places to paste it — wiring either of those up is a
Whop-dashboard task, not something this server does for you). This one
step — turning "Whop told me someone paid" into "here is their key" —
is the one place this system still expects a human (or your own small
script watching for `payment.succeeded`, if you want to automate
issuance too; the pattern is identical to the revoke webhook below).

## 4. Wire up the Whop webhook (auto-revoke on cancellation)

This is the part that actually answers "if they stop paying, I need a
way to rescind access" without you doing anything by hand.

1. In your Whop dashboard: **Developer → Webhooks → Create Webhook**.
2. URL: `https://your-license-server.example.com/webhook/whop`
3. Subscribe to at least: `membership.went_valid`, `membership.went_invalid`.
4. Whop shows you a signing secret (`whsec_...`) for this webhook — set
   it as `T58_WHOP_WEBHOOK_SECRET` on your license server and redeploy.

That's it. Whop signs every webhook with the open
[Standard Webhooks](https://github.com/standard-webhooks/standard-webhooks)
spec (this is a real, documented scheme — not something guessed at
here), and `app.py`'s `_verify_standard_webhook()` checks it properly
before trusting anything in the payload. On `membership.went_invalid`,
the matching license (by `whop_membership_id` if you set one, else by
email) is automatically set to `revoked`. On `membership.went_valid` for
a license that was previously revoked/expired/suspended, it's
automatically reactivated — so a lapsed customer who resubscribes just
starts working again the next time the app validates, no admin action
needed.

## Admin CLI reference

```
export T58_LICENSE_SERVER_URL=https://your-license-server.example.com
export T58_LICENSE_ADMIN_TOKEN=the-token-you-set-on-the-server

python admin_cli.py create alice@example.com --plan monthly --days 30
python admin_cli.py list
python admin_cli.py show T58-XXXX-XXXX-XXXX-XXXX
python admin_cli.py revoke T58-XXXX-XXXX-XXXX-XXXX
python admin_cli.py suspend T58-XXXX-XXXX-XXXX-XXXX
python admin_cli.py reactivate T58-XXXX-XXXX-XXXX-XXXX
python admin_cli.py extend T58-XXXX-XXXX-XXXX-XXXX --days 30
```

License states: `active`, `expired` (past its own `expires_at`, if it
has one), `revoked` (you or a webhook pulled access), `suspended` (a
softer, reversible pause — same effect as revoked from the customer's
side, but reads differently in your own records).

## How the desktop app uses this

See `app/licensing/` in the main repo — `client.py` (HTTP calls + local
encrypted state, mirroring how this app already stores Alpaca/Ollama
credentials) and `gate.py` (the Tkinter activation screen shown before
the app launches). That package's own docstrings explain the client
side in full; this file only covers the server you're running.

Key behaviors worth knowing about on the client side:
- **One license, one device.** The first successful activate binds the
  license to that machine. A second machine with the same key gets a
  clear "already active elsewhere" message, not a silent failure.
  "Deactivate this device" (in the app, or `client.deactivate()`) frees
  the binding so the license can move to a new machine.
- **Offline grace period (3 days).** If the license server can't be
  reached at all, a machine that validated successfully within the last
  3 days keeps working. A license the server had already marked
  revoked/expired the last time it *was* reachable does not get this
  grace — only a genuine network outage is forgiven, not an actual
  access problem.
- **No API keys or secrets ship in the client.** The desktop app only
  ever sends its email/license key/device fingerprint to your server's
  public endpoints — there is nothing in the shipped .exe for anyone to
  extract that would let them create or forge a license themselves.

## Local testing

```
cd license_server
pip install -r requirements.txt
T58_LICENSE_ADMIN_TOKEN=dev-token python app.py
```

Then, in another terminal:
```
export T58_LICENSE_SERVER_URL=http://localhost:5055
export T58_LICENSE_ADMIN_TOKEN=dev-token
python admin_cli.py create you@example.com --plan test --days 7
```
and point the desktop app at `http://localhost:5055` via
`T58_LICENSE_SERVER_URL` to activate against your local server before
you've deployed anything.
