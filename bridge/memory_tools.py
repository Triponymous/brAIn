"""Memory tools — functions the LLM calls via tool-use to query the brain.

Each tool is a method on MemoryTools. tool_definitions() returns the
schema in Anthropic/OpenAI tool-use format so the LLM knows what's available.
execute(name, args) dispatches a tool call by name.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING

import torch

from brain.core import Brain
from bridge.exporter import BrainStateExporter

if TYPE_CHECKING:
    from bridge.episode_log import EpisodeLogger


class MemoryTools:
    def __init__(self, brain: Brain, exporter: BrainStateExporter, episode_logger: "EpisodeLogger | None" = None) -> None:
        self.brain = brain
        self.exporter = exporter
        self.episode_logger = episode_logger

    def current_state(self) -> dict[str, Any]:
        """Full brain state snapshot."""
        return self.exporter.snapshot()

    def query_concepts(self, limit: int = 10, min_activation: float = 0.0) -> list[dict]:
        """Top-N active concept neurons."""
        snap = self.exporter.snapshot()
        concepts = snap["active_concepts"]
        filtered = [c for c in concepts if c["activation"] >= min_activation]
        return filtered[:limit]

    def label_concept(self, concept_id: int, label: str) -> dict[str, str]:
        """Assign a human-readable label to a concept neuron."""
        self.exporter.set_label(concept_id, label)
        return {"status": "ok", "concept_id": concept_id, "label": label}

    def recall_associations(self, concept_id: int) -> list[dict]:
        """Find concepts that are strongly connected to the given concept.

        Reads the association->concept synapse weights to find which other
        concept neurons share strong incoming connections with concept_id.
        """
        syn = self.brain.synapses.get("association_concept")
        if syn is None:
            return []
        weights = syn.weights  # shape (num_concept, num_association)
        target_weights = weights[concept_id]  # shape (num_association,)
        # Cosine similarity between this concept's weight vector and all others
        norms = torch.norm(weights, dim=1)
        target_norm = torch.norm(target_weights)
        if target_norm < 1e-8:
            return []
        similarities = (weights @ target_weights) / (norms * target_norm + 1e-8)
        similarities[concept_id] = -1  # exclude self
        top_k = min(5, len(similarities))
        values, indices = torch.topk(similarities, top_k)
        result = []
        for idx, sim in zip(indices.tolist(), values.tolist()):
            if sim > 0.1:
                entry = {"concept_id": idx, "similarity": round(sim, 4)}
                label = self.exporter.get_label(idx)
                if label:
                    entry["label"] = label
                result.append(entry)
        return result

    def episode_search(self, time_range: float = 24.0) -> dict[str, Any]:
        """Search historical brain episodes from the last N hours."""
        if self.episode_logger is None:
            return {"error": "Episode logger not available"}
        return self.episode_logger.daily_summary(since_hours=time_range)

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        """Dispatch a tool call by name."""
        dispatch = {
            "current_state": lambda: self.current_state(),
            "query_concepts": lambda: self.query_concepts(**args),
            "label_concept": lambda: self.label_concept(**args),
            "recall_associations": lambda: self.recall_associations(**args),
            "episode_search": lambda: self.episode_search(**args),
        }
        fn = dispatch.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        return fn()

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool schemas for the LLM system prompt."""
        return [
            {
                "name": "current_state",
                "description": "Get the full current brain state snapshot including modulators, active concepts, and sensor summary.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "query_concepts",
                "description": "Get the top-N most active concept neurons. Optionally filter by minimum activation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10},
                        "min_activation": {"type": "number", "default": 0.0},
                    },
                },
            },
            {
                "name": "label_concept",
                "description": "Assign a human-readable label to a concept neuron. Use when the user tells you what a concept represents.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "concept_id": {"type": "integer"},
                        "label": {"type": "string"},
                    },
                    "required": ["concept_id", "label"],
                },
            },
            {
                "name": "recall_associations",
                "description": "Find concepts that are strongly connected to a given concept via shared synaptic weights.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "concept_id": {"type": "integer"},
                    },
                    "required": ["concept_id"],
                },
            },
            {
                "name": "episode_search",
                "description": "Search historical brain activity. Returns a summary of what happened over a time range: top concepts, average modulators, most-used apps, and total ticks.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "time_range": {
                            "type": "number",
                            "default": 24.0,
                            "description": "Number of hours to look back (e.g. 1.0 for last hour, 24.0 for last day, 168.0 for last week).",
                        },
                    },
                },
            },
        ]
