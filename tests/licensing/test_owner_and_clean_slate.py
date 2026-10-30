"""Owner account (email + key + sheet-verified), master-key clean slate every time, default strategy library."""
import json
from pathlib import Path

import pytest

OWNER_EMAIL = "owenkmantz@gmail.com"
OWNER_KEY = "T58-UAXLS-HES9D-JHT7Q-EHLNG"


@pytest.fixture
def lic(monkeypatch, tmp_path):
    from app.licensing import client
    monkeypatch.setattr(client, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(client, "_try_keyring", lambda: None)
    client._session_state = None
    client._session_only_active = False
    client._reset_master_cache()
    yield client
    client._session_state = None
    client._reset_master_cache()


def _state(client, email, key, status="active", validated=True):
    from datetime import datetime, timezone
    client.save_state(client.LicenseState(email=email, license_key=key, device_id="d", status=status, plan="Lifetime",
                                          last_validated_at=datetime.now(timezone.utc).isoformat() if validated else None))
    client._reset_master_cache()


def test_owner_needs_email_key_and_active_status(lic):
    _state(lic, OWNER_EMAIL, OWNER_KEY)
    assert lic.owner_account_active() and lic.full_access_active()
    assert lic.current_plan() == "Lifetime"


def test_owner_email_alone_is_nothing(lic):
    _state(lic, OWNER_EMAIL, "T58-AAAAA-BBBBB-CCCCC-DDDDD")
    assert not lic.owner_account_active() and not lic.full_access_active()


def test_owner_key_with_other_email_is_nothing(lic):
    _state(lic, "someone@else.com", OWNER_KEY)
    assert not lic.owner_account_active()


def test_owner_revoked_or_unverified_loses_access(lic):
    _state(lic, OWNER_EMAIL, OWNER_KEY, status="revoked")
    assert not lic.owner_account_active()
    _state(lic, OWNER_EMAIL, OWNER_KEY, validated=False)
    assert not lic.owner_account_active()


def test_obsidian_gold_theme_free_for_everyone():
    from app import achievements
    spec = achievements.THEME_UNLOCKS["obsidian-gold"]
    assert achievements._requirement(spec, {}, 1)["unlocked"] is True
    assert "arctic-glass" not in achievements.THEME_UNLOCKS and "midnight-azure" in achievements.THEME_UNLOCKS
    assert achievements._requirement(achievements.THEME_UNLOCKS["midnight-azure"], {}, 1)["unlocked"] is False


def test_default_library_reset_keeps_only_repo_strategies(monkeypatch, tmp_path):
    from app.strategy import default_library as dl
    monkeypatch.setattr(dl, "_lib_dir", lambda: tmp_path / "strategies")
    py = tmp_path / "strategies" / "python"
    py.mkdir(parents=True)
    keep = dl.DEFAULT_STRATEGY_FILES["python"][0]
    (py / keep).write_text("# default", encoding="utf-8")
    (py / (keep + ".meta.json")).write_text("{}", encoding="utf-8")        # sidecar: default strategy starts un-run
    (py / "mine.py").write_text("# user", encoding="utf-8")
    (py / "mine.py.meta.json").write_text("{}", encoding="utf-8")
    backup = tmp_path / "bk"
    assert dl.reset_to_defaults(backup) == 3
    assert sorted(p.name for p in py.iterdir()) == [keep]
    assert (backup / "python" / "mine.py").exists()                        # moved, never deleted


def test_repo_defaults_match_committed_files():
    from app.strategy import default_library as dl
    root = Path(__file__).resolve().parents[2] / "strategies" / "python"
    on_disk = {p.name for p in root.glob("*.py")}
    assert set(dl.DEFAULT_STRATEGY_FILES["python"]) <= on_disk


def test_master_key_fresh_start_runs_every_time(monkeypatch, tmp_path):
    from app.licensing import fresh_start
    from app.data import storage
    monkeypatch.setenv("T58_FORCE_FRESH_START", "1")
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(storage, "user_data_dir", lambda: tmp_path / "ud")
    (tmp_path / "account_settings.json").write_text("{}", encoding="utf-8")
    assert fresh_start.run_if_needed() is True
    assert fresh_start.run_if_needed() is False                            # normal calls: once
    (tmp_path / "account_settings.json").write_text('{"name":"x"}', encoding="utf-8")
    assert fresh_start.run_if_needed(force=True) is True                   # typing the master key again: blank slate again
    assert not (tmp_path / "account_settings.json").exists()
