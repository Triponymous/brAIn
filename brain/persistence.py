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
CREATE TABLE IF NOT EXISTS wta_state (
    name TEXT PRIMARY KEY,
    thresholds BLOB NOT NULL,
    firing_rate BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS concept_tracker (
    cluster_id INTEGER PRIMARY KEY,
    centroid BLOB NOT NULL,
    label TEXT,
    count INTEGER NOT NULL,
    last_seen INTEGER NOT NULL
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
        "num_concept": brain.regions["concept"].num_neurons,
        "num_wm": brain.regions["wm"].num_neurons,
        "concept_k": brain.regions["concept"].k,
        "tau_mem": brain.regions["sensory"].tau_mem,
        "threshold": brain.regions["sensory"].threshold,
        "a_plus": brain.synapses["sensory_concept"].a_plus,
        "a_minus": brain.synapses["sensory_concept"].a_minus,
        "w_init": 0.0,
        "w_init_std": 0.0,
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
        # WTA adaptive thresholds (intrinsic plasticity state)
        for name, region in brain.regions.items():
            if hasattr(region, "thresholds"):  # WTALayer
                conn.execute(
                    "INSERT INTO wta_state(name, thresholds, firing_rate) VALUES (?, ?, ?)",
                    (
                        name,
                        _tensor_to_blob(region.thresholds),
                        _tensor_to_blob(region._firing_rate),
                    ),
                )
        # Expansion layer weights (fixed random — must be saved for deterministic replay)
        if hasattr(brain, '_expansion_weights'):
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('expansion_weights', ?)",
                (_tensor_to_blob(brain._expansion_weights),),
            )

        # ConceptTracker clusters (stable concept IDs + labels)
        if hasattr(brain, 'concept_tracker'):
            ct = brain.concept_tracker
            for cid in sorted(ct._clusters):
                c = ct._clusters[cid]
                conn.execute(
                    "INSERT INTO concept_tracker(cluster_id, centroid, label, count, last_seen) VALUES (?, ?, ?, ?, ?)",
                    (cid, _tensor_to_blob(c.centroid), c.label, c.count, c.last_seen),
                )

        # Personality state (from BrainInterpreter)
        if hasattr(brain, '_interpreter') and brain._interpreter:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('personality', ?)",
                (json.dumps(brain._interpreter.save_personality()),),
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

        # Regions — skip regions that no longer exist (Feature, Association, Motor, Meta)
        for name, membrane_blob, last_spikes_blob in conn.execute(
            "SELECT name, membrane, last_spikes FROM region_state"
        ):
            region = brain.regions.get(name)
            if region is None:
                continue  # old checkpoint has regions we removed
            region.membrane = _blob_to_tensor(membrane_blob)
            if last_spikes_blob is not None and hasattr(region, "last_spikes"):
                region.last_spikes = _blob_to_tensor(last_spikes_blob)

        # Synapses — skip synapses that no longer exist
        for name, weights_blob, apre_blob, apost_blob in conn.execute(
            "SELECT name, weights, apre, apost FROM synapse_state"
        ):
            syn = brain.synapses.get(name)
            if syn is None:
                continue  # old checkpoint has synapses we removed
            syn.weights = _blob_to_tensor(weights_blob)
            syn.apre = _blob_to_tensor(apre_blob)
            syn.apost = _blob_to_tensor(apost_blob)

        # WTA adaptive thresholds (may not exist in older checkpoints)
        try:
            for name, thresholds_blob, firing_rate_blob in conn.execute(
                "SELECT name, thresholds, firing_rate FROM wta_state"
            ):
                region = brain.regions.get(name)
                if region and hasattr(region, "thresholds"):
                    region.thresholds = _blob_to_tensor(thresholds_blob)
                    region._firing_rate = _blob_to_tensor(firing_rate_blob)
        except sqlite3.OperationalError:
            pass  # Old checkpoint without wta_state table — use defaults

        # Expansion layer weights
        try:
            row = conn.execute("SELECT value FROM meta WHERE key='expansion_weights'").fetchone()
            if row and hasattr(brain, '_expansion_weights'):
                brain._expansion_weights = _blob_to_tensor(row[0])
        except (sqlite3.OperationalError, Exception):
            pass

        # ConceptTracker clusters (may not exist in older checkpoints)
        try:
            from brain.concept_tracker import _Cluster
            rows = conn.execute(
                "SELECT cluster_id, centroid, label, count, last_seen FROM concept_tracker ORDER BY cluster_id"
            ).fetchall()
            if rows and hasattr(brain, 'concept_tracker'):
                ct = brain.concept_tracker
                ct._clusters = {}
                for cluster_id, centroid_blob, label, count, last_seen in rows:
                    ct._clusters[cluster_id] = _Cluster(
                        centroid=_blob_to_tensor(centroid_blob),
                        label=label,
                        count=count,
                        last_seen=last_seen,
                        protected=label is not None,
                    )
                ct._next_id = max(ct._clusters.keys(), default=-1) + 1
        except sqlite3.OperationalError:
            pass  # Old checkpoint — use empty tracker

        # Personality state (will be loaded by BrainInterpreter later)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key='personality'").fetchone()
            if row:
                brain._personality_state = json.loads(row[0])
        except (sqlite3.OperationalError, Exception):
            pass

        # ── Post-load sanity checks ──
        # WM synapses may be saturated from old checkpoints (pre-scaling fix).
        # If mean weight > 0.7, reset to fresh initialization.
        for syn_name in ("concept_wm", "wm_concept"):
            syn = brain.synapses.get(syn_name)
            if syn is not None:
                mean_w = float(syn.weights.mean().item())
                if mean_w > 0.7:
                    print(f"[persistence] {syn_name} saturated (mean={mean_w:.3f}), resetting to fresh weights")
                    import torch
                    shape = syn.weights.shape
                    if syn_name == "wm_concept":
                        syn.weights = torch.clamp(
                            torch.randn(shape) * 0.05 + 0.1,
                            syn.w_min, syn.w_max,
                        )
                    else:
                        syn.weights = torch.clamp(
                            torch.randn(shape) * 0.15 + 0.3,
                            syn.w_min, syn.w_max,
                        )
                    syn._target_w_sum = syn.weights.sum(dim=1, keepdim=True).mean().reshape(1)
                    syn.apre = torch.zeros(syn.num_pre)
                    syn.apost = torch.zeros(syn.num_post)

        return brain
    finally:
        conn.close()
