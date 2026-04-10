"""ConceptTracker — assigns stable IDs to recurring sensory patterns.

The SNN's WTA concept layer rotates winners due to IP/lateral inhibition,
making individual neuron IDs unstable. But the EXPANSION layer output is
stable — the same sensory pattern always produces the same expansion spike
signature (fixed random weights, no learning).

This module clusters expansion signatures into stable "concept groups" and
assigns persistent IDs. The LLM sees these stable IDs, not raw neuron IDs.

Algorithm:
1. Every N ticks, snapshot the expansion spike pattern (binary 500-dim vector)
2. Compare to known cluster centroids via cosine similarity
3. If similarity > threshold → assign to existing cluster
4. If not → create new cluster
5. Clusters decay slowly if not seen for a long time
"""
from __future__ import annotations
from typing import Any

import torch


class ConceptTracker:
    def __init__(
        self,
        expansion_dim: int = 500,
        similarity_threshold: float = 0.6,  # Jaccard similarity to match (typing vs zoom Jaccard ~0.43)
        max_clusters: int = 50,
        snapshot_interval: int = 200,  # snapshot every 200 ticks (2 seconds) — fast reaction
    ) -> None:
        self.expansion_dim = expansion_dim
        self.similarity_threshold = similarity_threshold
        self.max_clusters = max_clusters
        self.snapshot_interval = snapshot_interval

        # Cluster centroids: running average of expansion spike patterns
        self.centroids: list[torch.Tensor] = []
        self.cluster_labels: list[str | None] = []
        self.cluster_counts: list[int] = []
        self.cluster_last_seen: list[int] = []

        # Current state
        self._tick_counter = 0
        self._current_cluster: int = -1
        self._accum = torch.zeros(expansion_dim)
        self._accum_count = 0

    def tick(self, expansion_spikes: torch.Tensor, tick_count: int) -> None:
        """Call every brain tick with the expansion layer output."""
        self._accum += expansion_spikes.detach()
        self._accum_count += 1
        self._tick_counter += 1

        if self._tick_counter >= self.snapshot_interval and self._accum_count > 0:
            # Create binary signature: which neurons fired > 30% of the time
            signature = (self._accum / self._accum_count > 0.3).float()
            self._assign_cluster(signature, tick_count)
            self._accum = torch.zeros(self.expansion_dim)
            self._accum_count = 0
            self._tick_counter = 0

    def _assign_cluster(self, signature: torch.Tensor, tick_count: int) -> None:
        """Assign the signature to the best matching cluster, or create new."""
        if signature.sum() < 3:
            self._current_cluster = -1
            return  # too few active neurons, skip

        best_sim = 0.0
        best_idx = -1

        for i, centroid in enumerate(self.centroids):
            sim = self._cosine_sim(signature, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.similarity_threshold and best_idx >= 0:
            # Match existing cluster — update centroid (running average)
            alpha = 0.1  # blend new signature into centroid
            self.centroids[best_idx] = (
                self.centroids[best_idx] * (1 - alpha) + signature * alpha
            )
            self.cluster_counts[best_idx] += 1
            self.cluster_last_seen[best_idx] = tick_count
            self._current_cluster = best_idx
        elif len(self.centroids) < self.max_clusters:
            # New cluster
            self.centroids.append(signature.clone())
            self.cluster_labels.append(None)
            self.cluster_counts.append(1)
            self.cluster_last_seen.append(tick_count)
            self._current_cluster = len(self.centroids) - 1
        else:
            # At capacity — replace least-seen cluster
            min_idx = min(range(len(self.cluster_counts)),
                          key=lambda i: self.cluster_counts[i])
            self.centroids[min_idx] = signature.clone()
            self.cluster_labels[min_idx] = None
            self.cluster_counts[min_idx] = 1
            self.cluster_last_seen[min_idx] = tick_count
            self._current_cluster = min_idx

    def _cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Jaccard similarity between two binary vectors.
        Better than cosine for sparse binary patterns."""
        intersection = (a * b).sum()
        union = ((a + b) > 0).float().sum()
        if union < 1e-8:
            return 0.0
        return float(intersection / union)

    @property
    def current_cluster_id(self) -> int:
        return self._current_cluster

    @property
    def current_cluster_label(self) -> str | None:
        if 0 <= self._current_cluster < len(self.cluster_labels):
            return self.cluster_labels[self._current_cluster]
        return None

    def set_label(self, cluster_id: int, label: str) -> None:
        if 0 <= cluster_id < len(self.cluster_labels):
            self.cluster_labels[cluster_id] = label

    def snapshot(self) -> dict[str, Any]:
        """Return current state for LLM/dashboard."""
        clusters = []
        for i in range(len(self.centroids)):
            clusters.append({
                "id": i,
                "label": self.cluster_labels[i],
                "count": self.cluster_counts[i],
                "last_seen": self.cluster_last_seen[i],
                "active": i == self._current_cluster,
            })
        # Sort by count (most seen first)
        clusters.sort(key=lambda c: -c["count"])
        return {
            "current_cluster": self._current_cluster,
            "current_label": self.current_cluster_label,
            "num_clusters": len(self.centroids),
            "clusters": clusters[:20],  # top 20
        }
