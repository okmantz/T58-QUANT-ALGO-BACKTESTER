# T58 License Control Center -- setup (about 10 minutes, one time)

Your whole licensing system is one **private** Google Sheet plus a small script
attached to it. No server, no hosting bill, nothing to maintain.

## 1. Make the sheet private
Open your Google Sheet -> **Share** -> General access -> **Restricted** (only you).
The sheet will hold every license key and customer email, so it must never be
"Anyone with the link". The app never reads the sheet directly (step 4 explains how it
checks keys without exposing it).

## 2. Add the script and build everything
1. In the sheet: **Extensions -> Apps Script**.
2. Delete the placeholder code, paste the whole of `license_sheet/Code.gs`, click **Save** (disk icon).
3. In the function dropdown at the top choose **setup**, click **Run**.
4. Google asks for permission. Click **Review permissions** -> choose your account ->
   **Advanced** -> **Go to (project name) (unsafe)** -> **Allow**.
   (That warning appears for every script you write yourself; it is your own code.)
5. Go back to the sheet. You now have three tabs: **Dashboard**, **Licenses** (500 keys, status
   "Unused") and **Guide**, plus a **T58 License** menu next to Help.

Re-running `setup` any time is safe: it restyles and refreshes but never touches existing keys or customers.

## 3. Deploy the web app (what the T58 app talks to)
1. In Apps Script: **Deploy -> New deployment** -> gear icon -> **Web app**.
2. Description: `T58 license`. **Execute as: Me**. **Who has access: Anyone**.
   ("Anyone" only lets the app *ask a question*; the script answers about one key at a time and
   never returns the sheet. The sheet itself stays private.)
3. **Deploy**, then copy the **Web app URL** (ends in `/exec`).
4. Quick check: paste `<that URL>?action=validate&license_key=T58-TEST&device_id=x` into a browser.
   You should see `{"ok":false,"error":"not_found"}`. If you see a Google sign-in page instead,
   "Who has access" isn't set to Anyone.

> Later, if you edit the script: **Deploy -> Manage deployments -> pencil -> Version: New version -> Deploy**.
> Saving alone does not update the live endpoint.

## 4. Point the app at it
In your GitHub repo: **Settings -> Secrets and variables -> Actions -> New repository secret**
- Name: `T58_LICENSE_SHEET_URL`
- Value: the `/exec` URL

Rebuild (re-run the build workflows). That URL gets baked into the shipped app. It is safe for
that to be extractable: with it, someone can only ask "is key X valid?" -- keys are random 100-bit
values, so they can't be guessed, and nothing else is ever returned.

## 5. Running it day to day
- **Sell:** pick an *Unused* key on the Licenses tab, send it to the customer (Whop message or
  email), and fill in CUSTOMER, EMAIL, PLAN, ISSUED. With EMAIL filled in, only that email can
  activate the key; leave it blank and the first activation fills it in.
- **Trial or monthly plan:** put a date in EXPIRES (works through the end of that day).
- **Revoke / suspend:** click the key's row -> **T58 License** menu -> Revoke / Suspend (or just
  change STATUS). The customer is locked out the next time the app checks (each launch; at the
  latest after the 3-day offline grace period).
- **Customer got a new computer:** click the row -> **T58 License -> Release device**.
- **Need more keys:** **T58 License -> Generate more keys**.
- **Sell / give a key to someone:** click a key row (or click nothing and the next unused key is used) -> **T58 License -> Assign key to a customer & email it...**. Type their name, email and plan. The key is locked to that email and they receive an email with the key and three steps; they can sign in straight away and start on the default settings. Google asks you once to allow the script to send mail. **Re-send key email** repeats it.
- **Your own owner key:** put `owenkmantz@gmail.com` in the EMAIL column of your key and set PLAN to Lifetime. The app unlocks everything for that email + key once the sheet confirms it (it never unlocks on the email alone).
- Status meanings: Unused (never activated) - Active - Inactive (device released) - Suspended (temporary) - Revoked.

## Whop note
Whop's Software app issues its **own** license keys to buyers. Those keys are not in this sheet and
won't work in the app. Either remove the Software app from the product (use a download link in
the product description) or tell buyers their key comes from you, and send them one from the sheet.

## Safety nets built in
- If Google is slow or the script errors, the app treats it as "can't reach" -- customers keep working
  through the 3-day offline grace period instead of being locked out by your mistake.
- A revoked, suspended or expired key is never forgiven by the grace period.
- A released device can't quietly re-claim a key; only an explicit activation binds a device.


## Plan in the app
The app's Account page shows the PLAN column of this sheet for the activated key (Pro Monthly, Pro Annual, Lifetime, Trial, Complimentary) and users cannot change it. After editing `Code.gs`, redeploy: Deploy > Manage deployments > Edit > New version. The master key always shows Lifetime.

## No URL baked into a build?
If a build was made without the `T58_LICENSE_SHEET_URL` secret, the Activate window shows a **License connection (owner setup)** box. Paste the `/exec` URL there once; the app remembers it (`license_sheet_url.txt`). Email matching is forgiving (case, spaces, several addresses in one cell), but you must redeploy `Code.gs` as a **New version** after pulling this update.
