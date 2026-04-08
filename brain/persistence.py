"""SQLite persistence for the Brain.

Schema (single SQLite file):
    meta(key TEXT PRIMARY KEY, value TEXT)        — tick_count, brain config JSON
    modulators(name TEXT PRIMARY KEY, level REAL)
    region_state(name TEXT PRIMARY KEY, membrane BLOB, last_spikes BLOB)
    synapse_state(name TEXT PRIMARY KEY, weights BLOB, apre BLOB, apost BLOB)

Tensors are serialized as raw float32 byte buffers via torch.save → BytesIO.

We choose torch.save (not numpy) because it preserves dtype/shape exactly
and round-trips through .load() into a fresh tensor without manual reshape.
"""
from __future__ import annotations
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

import torch

from brain.core import Brain


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS modulators (
    name TEXT PRIMARY KEY,
    level REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS region_state (
    name TEXT PRIMARY KEY,
    membrane BLOB NOT NULL,
    last_spikes BLOB
);
CREATE TABLE IF NOT EXISTS synapse_state (
    name TEXT PRIMARY KEY,
    weights BLOB NOT NULL,
    apre BLOB NOT NULL,
    apost BLOB NOT NULL
);
"""


def _tensor_to_blob(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _blob_to_tensor(b: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(b), weights_only=True)


def _brain_config(brain: Brain) -> dict[str, Any]:
    return {
        "num_sensory": brain.regions["sensory"].num_neurons,
        "num_feature": brain.regions["feature"].num_neurons,
        "num_association": brain.regions["association"].num_neurons,
        "num_concept": brain.regions["concept"].num_neurons,
        "num_wm": brain.regions["wm"].num_neurons,
        "num_motor": brain.regions["motor"].num_neurons,
        "num_meta": brain.regions["meta"].num_neurons,
        "concept_k": brain.regions["concept"].k,
        # Numerical / learning kwargs — sourced from a representative region/synapse.
        # All regions share tau_mem/threshold; all STDP synapses share a_plus/a_minus.
        "tau_mem": brain.regions["sensory"].tau_mem,
        "threshold": brain.regions["sensory"].threshold,
        "a_plus": brain.synapses["sensory_feature"].a_plus,
        "a_minus": brain.synapses["sensory_feature"].a_minus,
        # w_init / w_init_jitter only affect __init__; the saved weight tensors
        # already capture the full state, so reload value is irrelevant. We
        # store w_init=0.0, w_init_jitter=0.0 to skip the wasted jitter step.
        "w_init": 0.0,
        "w_init_jitter": 0.0,
    }


def save_brain(brain: Brain, path: Path) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)
        # Meta
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            ("tick_count", str(brain.tick_count)),
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            ("config", json.dumps(_brain_config(brain))),
        )
        # Modulators
        for name, level in brain.modulators.snapshot().items():
            conn.execute(
                "INSERT INTO modulators(name, level) VALUES (?, ?)",
                (name, level),
            )
        # Regions
        for name, region in brain.regions.items():
            membrane = _tensor_to_blob(region.membrane)
            last_spikes = None
            if hasattr(region, "last_spikes"):
                last_spikes = _tensor_to_blob(region.last_spikes)
            conn.execute(
                "INSERT INTO region_state(name, membrane, last_spikes) VALUES (?, ?, ?)",
                (name, membrane, last_spikes),
            )
        # Synapses
        for name, syn in brain.synapses.items():
            conn.execute(
                "INSERT INTO synapse_state(name, weights, apre, apost) VALUES (?, ?, ?, ?)",
                (
                    name,
                    _tensor_to_blob(syn.weights),
                    _tensor_to_blob(syn.apre),
                    _tensor_to_blob(syn.apost),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def load_brain(path: Path) -> Brain:
    path = Path(path)
    conn = sqlite3.connect(str(path))
    try:
        # Read config
        row = conn.execute("SELECT value FROM meta WHERE key = 'config'").fetchone()
        config = json.loads(row[0])
        brain = Brain(**config)

        # Tick count
        row = conn.execute("SELECT value FROM meta WHERE key = 'tick_count'").fetchone()
        brain.tick_count = int(row[0])

        # Modulators
        for name, level in conn.execute("SELECT name, level FROM modulators"):
            brain.modulators._levels[name] = level

        # Regions
        for name, membrane_blob, last_spikes_blob in conn.execute(
            "SELECT name, membrane, last_spikes FROM region_state"
        ):
            region = brain.regions[name]
            region.membrane = _blob_to_tensor(membrane_blob)
            if last_spikes_blob is not None and hasattr(region, "last_spikes"):
                region.last_spikes = _blob_to_tensor(last_spikes_blob)

        # Synapses
        for name, weights_blob, apre_blob, apost_blob in conn.execute(
            "SELECT name, weights, apre, apost FROM synapse_state"
        ):
            syn = brain.synapses[name]
            syn.weights = _blob_to_tensor(weights_blob)
            syn.apre = _blob_to_tensor(apre_blob)
            syn.apost = _blob_to_tensor(apost_blob)

        return brain
    finally:
        conn.close()
