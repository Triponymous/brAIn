"""Brain persistence tests.

A Brain saved to SQLite must reload to a state that produces identical
outputs on the same input. This includes:
- All synapse weights
- All STDP eligibility traces (apre, apost)
- All region membrane potentials
- Working memory last_spikes
- Modulator levels
- Tick count
"""
import asyncio
import tempfile
from pathlib import Path
import pytest
import torch
from brain.core import Brain
from brain.persistence import save_brain, load_brain, backup_checkpoint


def test_save_load_round_trip_default():
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    # Run a few ticks to populate non-zero state
    torch.manual_seed(123)
    for _ in range(20):
        brain.tick(torch.rand(8) * 2.0, reward=0.1)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "brain.sqlite"
        save_brain(brain, path)
        loaded = load_brain(path)

    # Check tick count
    assert loaded.tick_count == brain.tick_count
    # Check modulators
    for name in ["DA", "NE", "ACh", "5HT"]:
        assert abs(loaded.modulators.level(name) - brain.modulators.level(name)) < 1e-6
    # Check all synapse weights identical
    for key in brain.synapses:
        assert torch.allclose(loaded.synapses[key].weights, brain.synapses[key].weights)
    # Check eligibility traces identical
    for key in brain.synapses:
        assert torch.allclose(loaded.synapses[key].apre, brain.synapses[key].apre)
        assert torch.allclose(loaded.synapses[key].apost, brain.synapses[key].apost)
    # Check region membranes identical
    for key in brain.regions:
        assert torch.allclose(loaded.regions[key].membrane, brain.regions[key].membrane)
    # Check WM last_spikes
    assert torch.allclose(loaded.regions["wm"].last_spikes, brain.regions["wm"].last_spikes)


def test_save_load_then_tick_matches_unsaved():
    """Brain saved, loaded, then ticked once should match the original ticked once."""
    torch.manual_seed(7)
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    for _ in range(10):
        brain.tick(torch.rand(8) * 2.0)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "brain.sqlite"
        save_brain(brain, path)

        # Continue brain A in place
        next_input = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.0, 2.5, 1.0, 0.5])
        out_a = brain.tick(next_input)

        # Load fresh brain B and tick same input
        brain_b = load_brain(path)
        out_b = brain_b.tick(next_input)

    for key in out_a:
        assert torch.allclose(out_a[key], out_b[key]), f"Divergence in {key}"
    assert brain.tick_count == brain_b.tick_count


def test_save_creates_file():
    brain = Brain(num_sensory=4)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "brain.sqlite"
        assert not path.exists()
        save_brain(brain, path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_felt_state_round_trip():
    """The learned emotional self-model survives save/load."""
    from bridge.felt_state import FeltState
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    fs = FeltState()
    fs.label("flow", [0.028, 0.027, 0.058, 0.035, 0.060, 0.069])
    brain.felt_state = fs

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "brain.sqlite"
        save_brain(brain, path)
        loaded = load_brain(path)

    assert loaded.felt_state.recognize([0.030, 0.025, 0.055, 0.037, 0.058, 0.071])[0] == "flow"
    assert loaded.felt_state.known_labels() == ["flow"]


# ── Atomic save ──────────────────────────────────────────────────────────────
# The checkpoint is the individual. A save must never leave it torn or missing.

def _small_brain(ticks: int) -> Brain:
    brain = Brain(num_sensory=8, num_concept=4, num_wm=4)
    for _ in range(ticks):
        brain.tick(torch.rand(8) * 2.0)
    return brain


def test_save_leaves_no_tmp_file(tmp_path):
    path = tmp_path / "brain.sqlite"
    save_brain(_small_brain(5), path)
    assert path.exists()
    assert not path.with_name("brain.sqlite.tmp").exists()


def test_save_over_existing_checkpoint_yields_new_state(tmp_path):
    path = tmp_path / "brain.sqlite"
    save_brain(_small_brain(5), path)
    save_brain(_small_brain(20), path)
    assert load_brain(path).tick_count == 20
    assert not path.with_name("brain.sqlite.tmp").exists()


def test_crash_mid_save_keeps_previous_checkpoint(tmp_path, monkeypatch):
    """A failure while writing must not destroy the checkpoint on disk.

    The old implementation deleted the checkpoint before writing the new one,
    so any crash in between lost the organism. Simulate a crash partway
    through serialisation and check the previous checkpoint still loads.
    """
    import brain.persistence as persistence

    path = tmp_path / "brain.sqlite"
    save_brain(_small_brain(5), path)

    def boom(_t):
        raise RuntimeError("simulated crash mid-save")

    monkeypatch.setattr(persistence, "_tensor_to_blob", boom)
    with pytest.raises(RuntimeError):
        save_brain(_small_brain(20), path)

    assert load_brain(path).tick_count == 5
    assert not path.with_name("brain.sqlite.tmp").exists()


# ── Daily backup ─────────────────────────────────────────────────────────────

def test_backup_checkpoint_copies_loadable_brain(tmp_path):
    path = tmp_path / "braind.sqlite"
    save_brain(_small_brain(7), path)

    dest = backup_checkpoint(path)

    assert dest is not None
    assert dest.parent == tmp_path / "backups"
    assert dest.name.startswith("braind-") and dest.suffix == ".sqlite"
    assert load_brain(dest).tick_count == 7


def test_backup_checkpoint_is_one_copy_per_day(tmp_path):
    path = tmp_path / "braind.sqlite"
    save_brain(_small_brain(3), path)
    first = backup_checkpoint(path)
    save_brain(_small_brain(9), path)
    second = backup_checkpoint(path)

    assert first == second
    assert len(list((tmp_path / "backups").glob("braind-*.sqlite"))) == 1
    assert load_brain(second).tick_count == 9  # same-day copy is refreshed


def test_backup_checkpoint_prunes_to_keep_newest(tmp_path):
    path = tmp_path / "braind.sqlite"
    save_brain(_small_brain(3), path)
    bdir = tmp_path / "backups"
    bdir.mkdir()
    for day in ("2026-01-01", "2026-01-02", "2026-01-03"):
        (bdir / f"braind-{day}.sqlite").write_bytes(b"old")

    backup_checkpoint(path, keep=2)

    names = sorted(p.name for p in bdir.glob("braind-*.sqlite"))
    assert len(names) == 2
    assert "braind-2026-01-01.sqlite" not in names
    assert "braind-2026-01-02.sqlite" not in names
    assert names[0] == "braind-2026-01-03.sqlite"  # today's copy is the newest


def test_backup_checkpoint_without_checkpoint_is_noop(tmp_path):
    assert backup_checkpoint(tmp_path / "missing.sqlite") is None
    assert not (tmp_path / "backups").exists()


# ── Persistence loop ─────────────────────────────────────────────────────────

async def test_persistence_loop_saves_and_backs_up_once_per_day(tmp_path):
    from server.main import persistence_loop

    brain = _small_brain(4)
    ckpt = tmp_path / "braind.sqlite"
    task = asyncio.create_task(persistence_loop(brain, str(ckpt), period_sec=0.02))
    try:
        for _ in range(200):
            await asyncio.sleep(0.02)
            if ckpt.exists() and list((tmp_path / "backups").glob("braind-*.sqlite")):
                break
        await asyncio.sleep(0.1)  # let several more save cycles run
    finally:
        task.cancel()
        await task  # the loop swallows CancelledError and returns

    assert ckpt.exists()
    assert not ckpt.with_name("braind.sqlite.tmp").exists()
    backups = list((tmp_path / "backups").glob("braind-*.sqlite"))
    assert len(backups) == 1
    assert load_brain(backups[0]).tick_count == 4
