"""braind erase: lists every daemon store first, deletes only with --yes, never while a daemon uses them."""
from pathlib import Path

import server.braind as braind

STORES = ["brain.sqlite", "brain.sqlite-wal", "brain.sqlite-shm", "brain.sqlite.tmp",
          "episodes.db", "episodes.db-wal", "experience.db", "experience.db-shm",
          "grants.sqlite", "consent.json", "consent-mock.json",
          "backups/brain-2026-09-20.sqlite", "backups/brain-2026-09-21.sqlite"]
NOT_OURS = ["notes.txt", "backups/brain-exp-2026-09-21.sqlite",   # another checkpoint's backup, same prefix
            "backups/other-2026-09-21.sqlite"]


def _daemon_dir(d: Path) -> list[Path]:
    for name in STORES + NOT_OURS:
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"personal")
    return [d / name for name in STORES]


def test_lists_every_store_and_deletes_nothing_without_yes(tmp_path, capsys):
    files = _daemon_dir(tmp_path)
    assert sorted(braind.erase_plan(tmp_path / "brain.sqlite")) == sorted(files)
    assert braind.main(["erase", "--checkpoint", str(tmp_path / "brain.sqlite")]) == 0
    assert all(f.exists() for f in files)
    assert "--yes" in capsys.readouterr().out
    assert not (tmp_path / "brain.sqlite.lock").exists()          # a look leaves nothing behind


def test_yes_deletes_the_stores_and_nothing_else(tmp_path):
    files = _daemon_dir(tmp_path)
    assert braind.main(["erase", "--checkpoint", str(tmp_path / "brain.sqlite"), "--yes"]) == 0
    assert not any(f.exists() for f in files)
    assert all((tmp_path / name).exists() for name in NOT_OURS)
    assert not (tmp_path / "brain.sqlite.lock").exists()


def test_refuses_while_a_daemon_holds_the_checkpoint_on_any_port(tmp_path, capsys):
    files = _daemon_dir(tmp_path)
    daemon = braind.lock_checkpoint(tmp_path / "brain.sqlite")    # what a running daemon holds
    assert braind.erase(tmp_path / "brain.sqlite", yes=True) == 1
    assert all(f.exists() for f in files) and "Stop it first" in capsys.readouterr().out
    daemon.close()                                                 # the daemon exits
    assert braind.erase(tmp_path / "brain.sqlite", yes=True) == 0
    assert not any(f.exists() for f in files)


def test_one_daemon_per_checkpoint(tmp_path):
    first = braind.lock_checkpoint(tmp_path / "brain.sqlite")
    assert braind.lock_checkpoint(tmp_path / "brain.sqlite") is None
    first.close()
    again = braind.lock_checkpoint(tmp_path / "brain.sqlite")
    assert again is not None
    again.close()


def test_default_directory_covers_pidfile_and_logs_and_an_older_daemon(tmp_path, monkeypatch):
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
    # A daemon started before the lock existed holds none; its pidfile or port still counts.
    monkeypatch.setattr(braind, "_default_daemon_up", lambda port: True)
    assert braind.erase(ck / "braind.sqlite", yes=True) == 1 and (ck / "braind.sqlite").exists()
    monkeypatch.setattr(braind, "_default_daemon_up", lambda port: False)
    assert braind.erase(ck / "braind.sqlite", yes=True) == 0 and not (ck / "braind.sqlite").exists()
