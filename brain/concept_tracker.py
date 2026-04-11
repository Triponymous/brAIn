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
        similarity_threshold: float = 0.35, # Jaccard similarity — balance between too many and too few clusters
        max_clusters: int = 15,  # max distinct patterns — old weak ones get replaced
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
            # Periodically merge similar clusters (every 50 assignments)
            total_assigns = sum(self.cluster_counts)
            if total_assigns % 50 == 0:
                self._merge_similar_clusters()
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
            alpha = 0.2  # blend new signature into centroid (fast adaptation)
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

    def _merge_similar_clusters(self) -> None:
        """Merge clusters whose centroids are too similar (Jaccard > 0.5).
        Keeps the one with more counts, transfers the label if any."""
        if len(self.centroids) < 2:
            return
        merged = True
        while merged:
            merged = False
            for i in range(len(self.centroids)):
                for j in range(i + 1, len(self.centroids)):
                    sim = self._cosine_sim(self.centroids[i], self.centroids[j])
                    if sim > 0.7:  # very similar → merge (0.7 = high overlap)
                        # Keep the one with more observations
                        keep, drop = (i, j) if self.cluster_counts[i] >= self.cluster_counts[j] else (j, i)
                        # Transfer counts
                        self.cluster_counts[keep] += self.cluster_counts[drop]
                        # Keep label from either
                        if not self.cluster_labels[keep] and self.cluster_labels[drop]:
                            self.cluster_labels[keep] = self.cluster_labels[drop]
                        # Blend centroids
                        self.centroids[keep] = (self.centroids[keep] + self.centroids[drop]) / 2
                        # Remove the weaker
                        self.centroids.pop(drop)
                        self.cluster_labels.pop(drop)
                        self.cluster_counts.pop(drop)
                        self.cluster_last_seen.pop(drop)
                        # Fix current_cluster reference
                        if self._current_cluster == drop:
                            self._current_cluster = keep
                        elif self._current_cluster > drop:
                            self._current_cluster -= 1
                        merged = True
                        break
                if merged:
                    break

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

    def get_transition(self) -> dict[str, Any] | None:
        """Returns info about the last cluster transition, if any.
        Used by the proactive engine to ask 'what is this new pattern?'"""
        if not hasattr(self, '_prev_cluster'):
            self._prev_cluster = -1
        if self._current_cluster != self._prev_cluster:
            old = self._prev_cluster
            self._prev_cluster = self._current_cluster
            if old >= 0:  # don't trigger on first assignment
                return {
                    "from_cluster": old,
                    "from_label": self.cluster_labels[old] if old < len(self.cluster_labels) else None,
                    "to_cluster": self._current_cluster,
                    "to_label": self.current_cluster_label,
                    "is_new": self._current_cluster >= 0 and (
                        self._current_cluster >= len(self.cluster_counts) or
                        self.cluster_counts[self._current_cluster] <= 1
                    ),
                }
        return None

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
        clusters.sort(key=lambda c: -c["count"])
        return {
            "current_cluster": self._current_cluster,
            "current_label": self.current_cluster_label,
            "num_clusters": len(self.centroids),
            "clusters": clusters[:20],
        }
