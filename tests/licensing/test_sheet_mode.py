"""Google Sheet licensing: the real client talking to the real license_sheet/Code.gs.

license_sheet/sim_server.js loads Code.gs and serves it over HTTP (including the
302 redirect Google's /exec URL does), keeping the sheet rows in memory. So these
tests cover the client's request format, redirect handling, JSON handling and the
actual license rules end to end. Skipped automatically when Node isn't installed.
FAKE keys/emails only.
"""
import json
import shutil
import subprocess
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.licensing import client

SIM = Path(__file__).resolve().parents[2] / "license_sheet" / "sim_server.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")

KEY = "T58-AAAAA-BBBBB-CCCCC-DDDDD"
KEY2 = "T58-EEEEE-FFFFF-GGGGG-HHHHH"
EMAIL = "jane@example.com"
COLS = 11  # key,status,customer,email,plan,issued,expires,device,activated,seen,notes


def _row(key, status="Unused", email="", expires="", device="", customer="", plan=""):
    r = [""] * COLS
    r[0], r[1], r[2], r[3], r[4], r[6], r[7] = key, status, customer, email, plan, expires, device
    return r


class Sim:
    def __init__(self):
        self.proc = subprocess.Popen(["node", str(SIM)], stdout=subprocess.PIPE, text=True)
        line = self.proc.stdout.readline().strip()
        assert line.startswith("PORT "), line
        self.base = f"http://127.0.0.1:{line.split()[1]}"
        self.url = self.base + "/exec"

    def admin(self, **msg):
        req = urllib.request.Request(self.base + "/__admin", data=json.dumps(msg).encode(), method="POST")
        urllib.request.urlopen(req, timeout=5).read()

    def rows(self):
        return json.loads(urllib.request.urlopen(self.base + "/__dump", timeout=5).read())

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(client, "_try_keyring", lambda: None)
    for var in ("T58_LICENSE_ALLOWLIST_URL", "T58_LICENSE_SERVER_URL", "T58_LICENSE_SHEET_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(client, "device_id", lambda: "pc-one")
    client._session_state = None
    client._session_only_active = False
    yield
    client._session_state = None
    client._session_only_active = False


@pytest.fixture
def sim(monkeypatch):
    s = Sim()
    s.admin(rows=[_row(KEY), _row(KEY2)])
    monkeypatch.setenv("T58_LICENSE_SHEET_URL", s.url)
    yield s
    s.stop()


def test_activate_then_validate_and_sheet_is_updated(sim):
    ok, msg = client.activate(EMAIL, KEY.lower())
    assert ok, msg
    assert client.validate() == (True, "Active.")
    row = sim.rows()[0]
    assert row[1] == "Active" and row[7] == "pc-one" and row[3] == EMAIL
    assert row[8] and row[9]  # activated + last seen stamped
    assert sim.rows()[1][1] == "Unused"  # the other key is untouched


def test_unknown_key_rejected(sim):
    ok, msg = client.activate(EMAIL, "T58-ZZZZZ-ZZZZZ-ZZZZZ-ZZZZZ")
    assert not ok and "wasn't found" in msg


def test_email_on_file_must_match(sim):
    sim.admin(rows=[_row(KEY, email="someone-else@example.com")])
    ok, msg = client.activate(EMAIL, KEY)
    assert not ok and "different email" in msg


def test_second_computer_refused_until_device_released(sim, monkeypatch):
    assert client.activate(EMAIL, KEY)[0]
    client.clear_state()
    monkeypatch.setattr(client, "device_id", lambda: "pc-two")
    ok, msg = client.activate(EMAIL, KEY)
    assert not ok and "different device" in msg
    # release the binding in the sheet (what the "Release device" menu does)
    rows = sim.rows()
    rows[0][1], rows[0][7] = "Inactive", ""
    sim.admin(rows=rows)
    assert client.activate(EMAIL, KEY)[0]
    assert sim.rows()[0][7] == "pc-two"


def test_released_old_pc_cannot_reclaim_via_validate(sim, monkeypatch):
    assert client.activate(EMAIL, KEY)[0]  # pc-one is activated and keeps its local state
    rows = sim.rows()
    rows[0][1], rows[0][7] = "Inactive", ""
    sim.admin(rows=rows)
    ok, msg = client.validate()
    assert not ok and "isn't activated" in msg
    assert sim.rows()[0][7] == ""  # not silently re-bound


def test_revoking_locks_the_app(sim):
    assert client.activate(EMAIL, KEY)[0]
    rows = sim.rows()
    rows[0][1] = "Revoked"
    sim.admin(rows=rows)
    ok, msg = client.validate()
    assert not ok and "revoked" in msg.lower()
    # revoked is never forgiven by the offline grace period, even if the sheet then goes dark
    sim.admin(mode="http500")
    assert not client.validate()[0]


def test_suspended_then_restored(sim):
    assert client.activate(EMAIL, KEY)[0]
    rows = sim.rows()
    rows[0][1] = "Suspended"
    sim.admin(rows=rows)
    assert not client.validate()[0]
    rows[0][1] = "Active"
    sim.admin(rows=rows)
    state = client.load_state()
    assert client.activate(EMAIL, KEY)[0], "reactivation after suspension should work"


def test_expiry_date(sim):
    yesterday = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    nextmonth = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
    sim.admin(rows=[_row(KEY, expires=yesterday), _row(KEY2, expires=nextmonth)])
    ok, msg = client.activate(EMAIL, KEY)
    assert not ok and "expired" in msg.lower()
    assert client.activate(EMAIL, KEY2)[0]
    assert client.load_state().expires_at is not None


def test_deactivate_frees_the_device(sim):
    assert client.activate(EMAIL, KEY)[0]
    ok, _ = client.deactivate()
    assert ok and client.load_state().license_key == ""
    row = sim.rows()[0]
    assert row[7] == "" and row[1] == "Inactive"


@pytest.mark.parametrize("mode", ["busy", "html", "http500"])
def test_sheet_problems_are_not_a_verdict_on_the_customer(sim, mode):
    assert client.activate(EMAIL, KEY)[0]
    sim.admin(mode=mode)  # transient error / Google sign-in page / HTTP error
    ok, msg = client.validate()
    assert ok and "Offline" in msg
    assert client.load_state().status == "active"


def test_grace_period_runs_out_then_locks(sim):
    assert client.activate(EMAIL, KEY)[0]
    sim.admin(mode="http500")
    state = client.load_state()
    state.last_validated_at = (datetime.now(timezone.utc) - timedelta(days=client.OFFLINE_GRACE_DAYS + 1)).isoformat()
    client.save_state(state)
    ok, msg = client.validate()
    assert not ok and "grace period" in msg


def test_cannot_activate_while_sheet_unreachable(sim):
    sim.admin(mode="html")
    ok, msg = client.activate(EMAIL, KEY)
    assert not ok and "Couldn't reach" in msg


def test_sheet_url_counts_as_configured(monkeypatch):
    assert client.check_license_server_configured()[0] is False
    monkeypatch.setenv("T58_LICENSE_SHEET_URL", "https://script.google.com/macros/s/X/exec")
    assert client.check_license_server_configured()[0] is True


def test_url_with_existing_query_string_still_works(sim, monkeypatch):
    monkeypatch.setenv("T58_LICENSE_SHEET_URL", sim.url + "?v=1")
    assert client.activate(EMAIL, KEY)[0]


# --- plan: the Account page's subscription type comes from the sheet's PLAN column ---------------
@pytest.mark.parametrize("plan", ["Pro Monthly", "Pro Annual", "Lifetime", "Trial", "Complimentary"])
def test_plan_from_sheet_is_saved_with_the_license(sim, plan):
    sim.admin(rows=[_row(KEY, plan=plan)])
    assert client.activate(EMAIL, KEY)[0]
    state = client.load_state()
    assert state.plan == plan and client.current_plan(state) == plan


def test_plan_change_in_sheet_shows_up_at_next_check_in(sim):
    sim.admin(rows=[_row(KEY, plan="Pro Monthly")])
    assert client.activate(EMAIL, KEY)[0]
    rows = sim.rows()
    rows[0][4] = "Lifetime"
    sim.admin(rows=rows)
    assert client.validate()[0]
    assert client.load_state().plan == "Lifetime"


def test_plan_spelling_is_normalised_but_unknown_plans_are_kept_as_typed(sim):
    sim.admin(rows=[_row(KEY, plan="pro_monthly"), _row(KEY2, plan="Founders Edition")])
    assert client.activate(EMAIL, KEY)[0]
    assert client.load_state().plan == "Pro Monthly"
    client.clear_state()
    assert client.activate(EMAIL, KEY2)[0]
    assert client.load_state().plan == "Founders Edition"


def test_blank_plan_in_sheet_is_blank_in_state(sim):
    assert client.activate(EMAIL, KEY)[0]
    assert client.load_state().plan == "" and client.current_plan() == ""


def test_refresh_license_info_pulls_a_missing_plan(sim, monkeypatch):
    sim.admin(rows=[_row(KEY, plan="Pro Annual")])
    assert client.activate(EMAIL, KEY)[0]
    state = client.load_state()
    state.plan = ""  # a license activated before plans existed
    client.save_state(state)
    monkeypatch.setattr(client, "_last_refresh_monotonic", 0.0)
    assert client.refresh_license_info() is True
    assert client.load_state().plan == "Pro Annual"


def test_refresh_license_info_is_throttled_and_never_raises(sim, monkeypatch):
    sim.admin(rows=[_row(KEY, plan="Pro Annual")])
    assert client.activate(EMAIL, KEY)[0]
    monkeypatch.setattr(client, "_last_refresh_monotonic", 0.0)
    state = client.load_state()
    state.plan = ""
    client.save_state(state)
    assert client.refresh_license_info() is True
    state = client.load_state()
    state.plan = ""
    client.save_state(state)
    assert client.refresh_license_info() is False  # inside the 10-minute window
    sim.admin(mode="http500")
    assert client.refresh_license_info(force=True) is True  # an unreachable sheet is swallowed, not raised


def test_master_key_is_always_lifetime(monkeypatch):
    import hashlib

    monkeypatch.setenv("T58_MASTER_LICENSE_KEY_HASH", hashlib.sha256(b"T58-MASTER-KEY").hexdigest())
    assert client.activate(EMAIL, "t58-master-key")[0]
    state = client.load_state()
    assert state.plan == "Lifetime" and client.current_plan(state) == "Lifetime"
    state.plan = ""  # even an old master activation that never stored a plan reads as Lifetime
    assert client.current_plan(state) == "Lifetime"
    assert client.refresh_license_info(force=True) is False  # the master key never touches the network


def test_normalize_plan_unit():
    n = client.normalize_plan
    assert n("Pro Monthly") == "Pro Monthly" and n("monthly") == "Pro Monthly" and n("PRO-ANNUAL") == "Pro Annual"
    assert n("  lifetime ") == "Lifetime" and n("trial") == "Trial" and n("comp") == "Complimentary"
    assert n("") == "" and n(None) == "" and n("Something New") == "Something New"


def test_connection_test_reports_healthy_sheet(sim):
    ok, msg = client.test_sheet_connection()
    assert ok and "Connected" in msg
    assert sim.rows()[0][1] == "Unused"  # the probe never changes the sheet


def test_connection_test_rejects_bad_urls():
    assert not client.test_sheet_connection("")[0]
    assert not client.test_sheet_connection("not a url")[0]
    ok, msg = client.test_sheet_connection("https://script.google.com/macros/s/ABC/dev")
    assert not ok and "/exec" in msg


def test_connection_test_unreachable():
    ok, msg = client.test_sheet_connection("http://127.0.0.1:1/exec", timeout=2)
    assert not ok


def test_not_linked_message_when_no_sheet_url():
    ok, msg = client.activate(EMAIL, KEY)
    assert not ok and "isn't linked" in msg


def test_saved_url_can_be_replaced(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "_sheet_url_files", lambda: [tmp_path / "a.txt", tmp_path / "b.txt"])
    assert client.save_sheet_url("https://script.google.com/macros/s/OLD/exec")[0]
    assert client.save_sheet_url(' "https://script.google.com/macros/s/NEW/exec" ')[0]
    assert client.resolve_sheet_url().endswith("/NEW/exec")
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()
