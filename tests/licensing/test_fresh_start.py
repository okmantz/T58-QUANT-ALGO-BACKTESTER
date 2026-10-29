"""Owner clean slate: runs once for the master key, moves (never deletes) saved state, then never again."""
import json

import pytest

from app.licensing import fresh_start


@pytest.fixture
def env(tmp_path, monkeypatch):
    base, ud = tmp_path / "app", tmp_path / "ud"
    (base / "reports" / "pbo").mkdir(parents=True)
    (base / "reports" / "pbo" / "r.json").write_text("{}")
    (base / "data" / "config").mkdir(parents=True)
    (base / "data" / "config" / "keys.json").write_text("{}")
    (base / "data" / "raw").mkdir(parents=True)
    (base / "data" / "raw" / "x.csv").write_text("a")
    (base / "strategies").mkdir()
    (base / "strategies" / "s.py").write_text("x=1")
    (base / "data" / "run_history.json").write_text("[1]")
    (base / "data" / "achievements.json").write_text("{}")
    import app.data.storage as storage
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: base)
    monkeypatch.setattr(storage, "user_data_dir", lambda: ud)
    monkeypatch.delenv("T58_NO_FRESH_START", raising=False)
    monkeypatch.setenv("T58_FORCE_FRESH_START", "1")
    monkeypatch.setenv("T58_VAULT_DIR", str(tmp_path / "vault"))
    return base, ud


def test_runs_once_and_keeps_keys_data_and_strategies(env):
    base, ud = env
    assert fresh_start.needed() and fresh_start.epoch() == ""
    assert fresh_start.run_if_needed() is True
    assert not (base / "data" / "run_history.json").exists()
    assert not (base / "data" / "achievements.json").exists()
    assert (base / "reports").is_dir() and not any((base / "reports").iterdir())
    assert (base / "data" / "config" / "keys.json").exists()
    assert (base / "data" / "raw" / "x.csv").exists() and (base / "strategies" / "s.py").exists()
    marker = json.loads((ud / fresh_start.MARKER_NAME).read_text())
    assert marker["epoch"] and fresh_start.epoch() == marker["epoch"]
    backup = ud.glob("fresh_start_backup_*")
    assert any((b / "reports" / "pbo" / "r.json").exists() for b in backup)
    # saves normally afterwards: a second call does nothing
    (base / "data" / "run_history.json").write_text("[2]")
    assert fresh_start.run_if_needed() is False
    assert (base / "data" / "run_history.json").read_text() == "[2]"


def test_disabled_under_pytest_by_default(env, monkeypatch):
    monkeypatch.delenv("T58_FORCE_FRESH_START")
    assert fresh_start.run_if_needed() is False
    assert fresh_start.needed()
