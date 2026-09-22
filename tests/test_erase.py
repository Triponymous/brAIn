"""braind erase: lists every daemon store first, deletes only with --yes, never while the daemon runs."""
import os
from pathlib import Path

import pytest

import server.braind as braind
import server.control as control

STORES = ["brain.sqlite", "brain.sqlite-wal", "brain.sqlite-shm", "brain.sqlite.tmp",
          "episodes.db", "episodes.db-wal", "experience.db", "experience.db-shm",
          "grants.sqlite", "consent.json",
          "backups/brain-2026-09-20.sqlite", "backups/brain-2026-09-21.sqlite"]


def _daemon_dir(d: Path) -> list[Path]:
    for name in STORES:
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"personal")
    return [d / name for name in STORES]


@pytest.fixture
def stopped(monkeypatch):
    monkeypatch.setattr(braind, "_daemon_running", lambda port: False)


def test_lists_every_store_and_deletes_nothing_without_yes(tmp_path, stopped, capsys):
    files = _daemon_dir(tmp_path)
    (tmp_path / "notes.txt").write_text("not the daemon's")
    (tmp_path / "backups" / "other-2026-09-21.sqlite").write_text("another checkpoint's backup")
    assert sorted(braind.erase_plan(tmp_path / "brain.sqlite")) == sorted(files)
    assert braind.main(["erase", "--checkpoint", str(tmp_path / "brain.sqlite")]) == 0
    assert all(f.exists() for f in files)
    assert "--yes" in capsys.readouterr().out


def test_yes_deletes_the_stores_and_nothing_else(tmp_path, stopped):
    files = _daemon_dir(tmp_path)
    (tmp_path / "notes.txt").write_text("not the daemon's")
    assert braind.main(["erase", "--checkpoint", str(tmp_path / "brain.sqlite"), "--yes"]) == 0
    assert not any(f.exists() for f in files)
    assert (tmp_path / "notes.txt").exists() and not (tmp_path / "backups").exists()


def test_refuses_while_the_daemon_runs(tmp_path, monkeypatch, capsys):
    files = _daemon_dir(tmp_path)
    monkeypatch.setattr(braind, "_daemon_running", lambda port: True)
    assert braind.erase(tmp_path / "brain.sqlite", yes=True) == 1
    assert all(f.exists() for f in files)
    assert "Stop it" in capsys.readouterr().out


def test_default_directory_also_covers_pidfile_and_service_logs(tmp_path, stopped, monkeypatch):
    monkeypatch.setattr(braind, "_PROJ", tmp_path)
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "braind.sqlite").write_text("x")
    (ck / "braind.pid").write_text("123")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "brain.out.log").write_text("log")
    plan = braind.erase_plan(ck / "braind.sqlite")
    assert plan[0] == ck / "braind.pid"      # removed first: the control server must not respawn mid-erase
    assert tmp_path / "logs" / "brain.out.log" in plan


def test_running_check_follows_the_pidfile(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "_PIDFILE", tmp_path / "braind.pid")
    (tmp_path / "braind.pid").write_text(str(os.getpid()))   # a live process
    assert braind._daemon_running(port=9) is True
    (tmp_path / "braind.pid").unlink()
    assert braind._daemon_running(port=9) is False             # nothing listens on the discard port
