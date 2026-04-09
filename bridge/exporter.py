"""BrainStateExporter — produces structured LLM-readable snapshots of the brain.

Includes auto-correlation: tracks which sensor states are active when each
concept fires, enabling automatic label suggestions like
"C27 fires 85% when App=VSCode and typing is high → suggested: coding".
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any

import torch

from brain.core import Brain


class BrainStateExporter:
    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self._labels: dict[int, str] = {}
        num_concepts = brain.regions["concept"].num_neurons
        self._spike_counts = torch.zeros(num_concepts)
        self._decay = 0.995

        # ── Auto-correlation tracker ──
        # For each concept, track how often each sensor state co-occurs with a spike
        # Structure: concept_id → { "app:VSCode": count, "keys:high": count, ... }
        self._concept_correlations: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._concept_total_spikes: dict[int, float] = defaultdict(float)
        self._corr_decay = 0.999  # slower decay for correlations (long-term memory)

    def record_spikes_with_context(self, concept_spikes: torch.Tensor, sensor_snapshot: dict[str, Any]) -> None:
        """Call each tick: accumulate spikes AND track sensor co-occurrence."""
        self._spike_counts = self._spike_counts * self._decay + concept_spikes.detach()

        # Build sensor context tags from current sensor state
        tags = _sensor_to_tags(sensor_snapshot)

        # For each concept that spiked, increment its correlation with current tags
        spike_indices = (concept_spikes > 0).nonzero(as_tuple=True)[0].tolist()
        for idx in spike_indices:
            self._concept_total_spikes[idx] += 1
            for tag in tags:
                self._concept_correlations[idx][tag] += 1

        # Slow decay on all correlations (prevents stale data from dominating)
        if len(spike_indices) > 0 and self.brain.tick_count % 1000 == 0:
            for cid in self._concept_correlations:
                self._concept_total_spikes[cid] *= self._corr_decay
                for tag in self._concept_correlations[cid]:
                    self._concept_correlations[cid][tag] *= self._corr_decay

    def get_concept_profile(self, concept_id: int, top_n: int = 5) -> dict[str, Any]:
        """Get the sensor correlation profile for a concept.

        Returns something like:
        { "total_spikes": 234, "correlations": [
            {"tag": "app:VSCode", "pct": 82},
            {"tag": "keys:active", "pct": 71},
            {"tag": "idle:short", "pct": 90},
        ], "suggested_label": "coding in VSCode" }
        """
        total = self._concept_total_spikes.get(concept_id, 0)
        if total < 5:
            return {"total_spikes": total, "correlations": [], "suggested_label": None}

        corr = self._concept_correlations.get(concept_id, {})
        ranked = sorted(corr.items(), key=lambda x: x[1], reverse=True)[:top_n]
        correlations = [
            {"tag": tag, "pct": round(count / total * 100)}
            for tag, count in ranked if count / total > 0.2
        ]

        # Auto-suggest a label from the top correlations
        suggested = _suggest_label(correlations)

        return {
            "total_spikes": round(total),
            "correlations": correlations,
            "suggested_label": suggested,
        }

    def record_spikes(self, concept_spikes: torch.Tensor) -> None:
        """Legacy: record without context."""
        self._spike_counts = self._spike_counts * self._decay + concept_spikes.detach()

    def set_label(self, concept_id: int, label: str) -> None:
        self._labels[concept_id] = label

    def get_label(self, concept_id: int) -> str | None:
        return self._labels.get(concept_id)

    def all_labels(self) -> dict[int, str]:
        return dict(self._labels)

    def snapshot(self, sensor_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        brain = self.brain
        concept_layer = brain.regions["concept"]
        num_concepts = concept_layer.num_neurons
        spike_accum = brain.concept_spike_accum

        active_concepts = []
        for i in range(num_concepts):
            activation = float(spike_accum[i].item())
            entry: dict[str, Any] = {
                "id": i,
                "activation": round(activation, 4),
            }
            label = self._labels.get(i)
            if label is not None:
                entry["label"] = label
            # Add auto-correlation profile for active concepts
            if activation > 0.01:
                profile = self.get_concept_profile(i, top_n=3)
                if profile["correlations"]:
                    entry["profile"] = profile["correlations"]
                if profile["suggested_label"] and label is None:
                    entry["suggested_label"] = profile["suggested_label"]
            active_concepts.append(entry)

        active_concepts.sort(key=lambda x: x["activation"], reverse=True)
        active_concepts = active_concepts[:12]

        return {
            "tick_count": brain.tick_count,
            "modulators": brain.modulators.snapshot(),
            "active_concepts": active_concepts,
            "sensor_summary": sensor_summary or {},
        }


def _sensor_to_tags(snap: dict[str, Any]) -> list[str]:
    """Convert a sensor bus snapshot into discrete tags for correlation tracking."""
    tags = []
    if "active_app" in snap:
        app = snap["active_app"].get("name", "")
        if app:
            tags.append(f"app:{app}")

    if "keystroke_rate" in snap:
        keys = snap["keystroke_rate"].get("count", 0)
        if keys > 10:
            tags.append("keys:heavy")
        elif keys > 3:
            tags.append("keys:active")
        else:
            tags.append("keys:idle")

    if "mouse_rate" in snap:
        mouse = snap["mouse_rate"].get("count", 0)
        if mouse > 10:
            tags.append("mouse:active")
        else:
            tags.append("mouse:idle")

    if "idle" in snap:
        idle = snap["idle"].get("seconds", 0)
        if idle > 300:
            tags.append("idle:away")
        elif idle > 30:
            tags.append("idle:paused")
        else:
            tags.append("idle:present")

    if "mic" in snap:
        rms = snap["mic"].get("rms", 0)
        if rms > 0.05:
            tags.append("mic:loud")
        elif rms > 0.01:
            tags.append("mic:ambient")
        else:
            tags.append("mic:silent")

    return tags


def _suggest_label(correlations: list[dict]) -> str | None:
    """Generate a human-readable label suggestion from top correlations."""
    if not correlations:
        return None
    parts = []
    for c in correlations[:3]:
        tag = c["tag"]
        pct = c["pct"]
        if pct < 30:
            continue
        if tag.startswith("app:"):
            parts.append(tag.split(":")[1])
        elif tag == "keys:heavy":
            parts.append("typing")
        elif tag == "keys:active":
            parts.append("light typing")
        elif tag == "mouse:active":
            parts.append("mouse active")
        elif tag == "idle:away":
            parts.append("away")
        elif tag == "idle:present":
            parts.append("at desk")
        elif tag == "mic:loud":
            parts.append("talking/noise")
        elif tag == "mic:ambient":
            parts.append("ambient sound")
    if not parts:
        return None
    return " + ".join(parts)
