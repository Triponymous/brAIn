"""BrainStateExporter — produces structured LLM-readable snapshots of the brain.

Consumed by the LLM bridge system prompt and by the memory tools. Updated
~1 Hz in production but callable on-demand via snapshot().

Concept labels are stored in-memory and persisted via the label_concept
memory tool (Phase 3b) or directly by the dashboard concept-labeling UI.
"""
from __future__ import annotations
from typing import Any

import torch

from brain.core import Brain


class BrainStateExporter:
    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self._labels: dict[int, str] = {}

    def set_label(self, concept_id: int, label: str) -> None:
        self._labels[concept_id] = label

    def get_label(self, concept_id: int) -> str | None:
        return self._labels.get(concept_id)

    def all_labels(self) -> dict[int, str]:
        return dict(self._labels)

    def snapshot(self, sensor_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        """Produce a structured snapshot of the brain state."""
        brain = self.brain

        # Active concepts: concept layer membrane values as proxy for "activation"
        concept_layer = brain.regions["concept"]
        membrane = concept_layer.membrane
        num_concepts = concept_layer.num_neurons

        active_concepts = []
        for i in range(num_concepts):
            activation = float(membrane[i].item())
            entry: dict[str, Any] = {
                "id": i,
                "activation": round(activation, 4),
            }
            label = self._labels.get(i)
            if label is not None:
                entry["label"] = label
            active_concepts.append(entry)

        # Sort by activation descending, keep top 12
        active_concepts.sort(key=lambda x: x["activation"], reverse=True)
        active_concepts = active_concepts[:12]

        return {
            "tick_count": brain.tick_count,
            "modulators": brain.modulators.snapshot(),
            "active_concepts": active_concepts,
            "sensor_summary": sensor_summary or {},
        }
