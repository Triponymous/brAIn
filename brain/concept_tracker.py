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

Stability guarantees:
- Cluster IDs are stable integers that never shift when clusters are
  merged or removed (dict-based, not list-based).
- Labeled clusters are protected: they are never replaced when at capacity
  unless ALL clusters are labeled, and their centroids drift more slowly.
"""
from __future__ import annotations
from typing import Any

import torch


class _Cluster:
    """Internal representation of a single cluster."""
    __slots__ = ("centroid", "label", "count", "last_seen", "protected")

    def __init__(
        self,
        centroid: torch.Tensor,
        label: str | None = None,
        count: int = 1,
        last_seen: int = 0,
        protected: bool = False,
    ) -> None:
        self.centroid = centroid
        self.label = label
        self.count = count
        self.last_seen = last_seen
        self.protected = protected


class ConceptTracker:
    def __init__(
        self,
        input_dim: int = 200,  # sensory dim — NOT expansion. Tracker uses full sensory spikes.
        similarity_threshold: float = 0.40,  # higher = fewer, broader clusters (0.25 was too granular)
        max_clusters: int = 8,              # fewer clusters = more meaningful patterns
        snapshot_interval: int = 50,
        # Legacy name accepted for backward compat with old checkpoints
        expansion_dim: int | None = None,
    ) -> None:
        self.input_dim = expansion_dim if expansion_dim is not None else input_dim
        self.similarity_threshold = similarity_threshold
        self.max_clusters = max_clusters
        self.snapshot_interval = snapshot_interval

        # Stable ID → _Cluster mapping.  IDs never shift on merge/remove.
        self._clusters: dict[int, _Cluster] = {}
        self._next_id: int = 0

        # Current state
        self._tick_counter = 0
        self._current_cluster: int = -1
        self._accum = torch.zeros(self.input_dim)
        self._accum_count = 0

    # ------------------------------------------------------------------
    # Legacy list-style accessors (read-only) for backward compat with
    # persistence, benchmarks, and any code that still indexes by position.
    # ------------------------------------------------------------------

    @property
    def centroids(self) -> list[torch.Tensor]:
        """Return centroids in stable-ID order (for persistence compat)."""
        return [self._clusters[cid].centroid for cid in sorted(self._clusters)]

    @centroids.setter
    def centroids(self, value: list[torch.Tensor]) -> None:
        """Bulk-set from persistence loader (list ordered by cluster_id)."""
        # Called during load_brain — rebuild _clusters dict from parallel lists.
        # The caller sets centroids first, then labels/counts/last_seen.
        self._clusters = {}
        for i, c in enumerate(value):
            self._clusters[i] = _Cluster(centroid=c, label=None, count=1, last_seen=0)
        self._next_id = max(self._clusters.keys(), default=-1) + 1

    @property
    def cluster_labels(self) -> _LabelAccessor:
        return _LabelAccessor(self)

    @cluster_labels.setter
    def cluster_labels(self, value: list[str | None]) -> None:
        """Bulk-set labels from persistence loader."""
        ids = sorted(self._clusters)
        for idx, label in zip(ids, value):
            self._clusters[idx].label = label
            self._clusters[idx].protected = label is not None

    @property
    def cluster_counts(self) -> _CountAccessor:
        return _CountAccessor(self)

    @cluster_counts.setter
    def cluster_counts(self, value: list[int]) -> None:
        ids = sorted(self._clusters)
        for idx, count in zip(ids, value):
            self._clusters[idx].count = count

    @property
    def cluster_last_seen(self) -> _LastSeenAccessor:
        return _LastSeenAccessor(self)

    @cluster_last_seen.setter
    def cluster_last_seen(self, value: list[int]) -> None:
        ids = sorted(self._clusters)
        for idx, ls in zip(ids, value):
            self._clusters[idx].last_seen = ls

    # ------------------------------------------------------------------

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
            total_assigns = sum(c.count for c in self._clusters.values())
            if total_assigns % 50 == 0:
                self._merge_similar_clusters()
            self._accum = torch.zeros(self.input_dim)
            self._accum_count = 0
            self._tick_counter = 0

    def _assign_cluster(self, signature: torch.Tensor, tick_count: int) -> None:
        """Assign the signature to the best matching cluster, or create new."""
        active_neurons = int(signature.sum().item())
        if active_neurons < 2:
            # Too sparse — but keep current cluster sticky (don't reset to -1)
            # Only go to -1 if we've NEVER had a cluster
            self._last_debug = {"reason": "too_few_neurons", "active": active_neurons}
            return  # keep _current_cluster as-is (sticky)

        best_sim = 0.0
        best_id = -1
        all_sims: list[tuple[int, float, str | None]] = []

        for cid, cluster in self._clusters.items():
            sim = self._jaccard_sim(signature, cluster.centroid)
            all_sims.append((cid, sim, cluster.label))
            if sim > best_sim:
                best_sim = sim
                best_id = cid

        # Debug info for dashboard / logging
        all_sims.sort(key=lambda x: -x[1])
        self._last_debug = {
            "active_neurons": active_neurons,
            "best_sim": round(best_sim, 3),
            "best_idx": best_id,
            "best_label": self._clusters[best_id].label if best_id in self._clusters else None,
            "threshold": self.similarity_threshold,
            "matched": best_sim >= self.similarity_threshold,
            "top_matches": [(i, round(s, 3), lbl) for i, s, lbl in all_sims[:5]],
        }

        if best_sim >= self.similarity_threshold and best_id >= 0:
            # Match existing cluster — update centroid with ADAPTIVE alpha.
            # Young clusters (low count) adapt fast, mature clusters stay stable.
            # Protected (labeled) clusters drift VERY slowly (alpha capped at 0.01).
            cluster = self._clusters[best_id]
            count = cluster.count
            alpha = max(0.02, 0.2 / (1 + count / 50))  # 0.2→0.02 over ~500 observations
            if cluster.protected:
                alpha = min(alpha, 0.01)  # labeled clusters: max drift = 0.01
            cluster.centroid = cluster.centroid * (1 - alpha) + signature * alpha
            cluster.count += 1
            cluster.last_seen = tick_count
            self._current_cluster = best_id
        elif len(self._clusters) < self.max_clusters:
            # New cluster — assign next stable ID
            new_id = self._next_id
            self._next_id += 1
            self._clusters[new_id] = _Cluster(
                centroid=signature.clone(), count=1, last_seen=tick_count
            )
            self._current_cluster = new_id
        else:
            # At capacity — replace least-seen UNLABELED cluster first.
            # NEVER replace a labeled (protected) cluster if any unlabeled ones exist.
            unlabeled = [cid for cid, c in self._clusters.items() if not c.protected]
            if unlabeled:
                victim_id = min(unlabeled, key=lambda cid: self._clusters[cid].count)
            else:
                # ALL clusters are labeled — reluctantly replace the one with fewest counts
                victim_id = min(self._clusters, key=lambda cid: self._clusters[cid].count)

            # Remove victim and insert with a NEW stable ID
            del self._clusters[victim_id]
            new_id = self._next_id
            self._next_id += 1
            self._clusters[new_id] = _Cluster(
                centroid=signature.clone(), count=1, last_seen=tick_count
            )
            self._current_cluster = new_id

    def _merge_similar_clusters(self) -> None:
        """Merge clusters whose centroids are too similar (Jaccard > 0.7).
        Keeps the one with more counts, transfers the label if any.
        Uses stable IDs — no index shifting."""
        if len(self._clusters) < 2:
            return
        merged = True
        while merged:
            merged = False
            ids = sorted(self._clusters)
            for idx_i in range(len(ids)):
                for idx_j in range(idx_i + 1, len(ids)):
                    cid_i, cid_j = ids[idx_i], ids[idx_j]
                    # One of them may have been removed in a previous iteration
                    if cid_i not in self._clusters or cid_j not in self._clusters:
                        continue
                    ci = self._clusters[cid_i]
                    cj = self._clusters[cid_j]
                    sim = self._jaccard_sim(ci.centroid, cj.centroid)
                    if sim > 0.5:  # similar enough → merge (prevents cluster explosion)
                        # Keep the one with more observations
                        if ci.count >= cj.count:
                            keep_id, drop_id = cid_i, cid_j
                        else:
                            keep_id, drop_id = cid_j, cid_i
                        keeper = self._clusters[keep_id]
                        dropper = self._clusters[drop_id]
                        # Transfer counts
                        keeper.count += dropper.count
                        # Transfer label: prefer keeper's label, but take dropper's if keeper has none
                        if not keeper.label and dropper.label:
                            keeper.label = dropper.label
                            keeper.protected = True
                        # Blend centroids
                        keeper.centroid = (keeper.centroid + dropper.centroid) / 2
                        # Preserve protected status from either side
                        if dropper.protected:
                            keeper.protected = True
                        # Remove the weaker (stable IDs — no shifting!)
                        del self._clusters[drop_id]
                        # Fix current_cluster reference
                        if self._current_cluster == drop_id:
                            self._current_cluster = keep_id
                        merged = True
                        break
                if merged:
                    break

    def _jaccard_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
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
        if self._current_cluster in self._clusters:
            return self._clusters[self._current_cluster].label
        return None

    def set_label(self, cluster_id: int, label: str) -> None:
        if cluster_id in self._clusters:
            self._clusters[cluster_id].label = label
            self._clusters[cluster_id].protected = True

    def get_transition(self) -> dict[str, Any] | None:
        """Returns info about the last cluster transition, if any.
        Used by the proactive engine to ask 'what is this new pattern?'"""
        if not hasattr(self, '_prev_cluster'):
            self._prev_cluster = -1
        if self._current_cluster != self._prev_cluster:
            old = self._prev_cluster
            self._prev_cluster = self._current_cluster
            if old >= 0:  # don't trigger on first assignment
                old_label = self._clusters[old].label if old in self._clusters else None
                return {
                    "from_cluster": old,
                    "from_label": old_label,
                    "to_cluster": self._current_cluster,
                    "to_label": self.current_cluster_label,
                    "is_new": self._current_cluster >= 0 and (
                        self._current_cluster not in self._clusters or
                        self._clusters[self._current_cluster].count <= 1
                    ),
                }
        return None

    def snapshot(self) -> dict[str, Any]:
        """Return current state for LLM/dashboard."""
        clusters = []
        for cid in sorted(self._clusters):
            c = self._clusters[cid]
            clusters.append({
                "id": cid,
                "label": c.label,
                "count": c.count,
                "last_seen": c.last_seen,
                "active": cid == self._current_cluster,
            })
        clusters.sort(key=lambda c: -c["count"])
        debug = getattr(self, '_last_debug', {})
        return {
            "current_cluster": self._current_cluster,
            "current_label": self.current_cluster_label,
            "num_clusters": len(self._clusters),
            "clusters": clusters[:20],
            "debug": debug,
        }


# ------------------------------------------------------------------
# Accessor helpers — let legacy code use tracker.cluster_labels[i]
# and tracker.cluster_counts[i] with stable IDs as keys.
# ------------------------------------------------------------------

class _LabelAccessor:
    """Provides __getitem__/__setitem__/__len__ so old code like
    ``tracker.cluster_labels[cluster_id]`` keeps working."""
    def __init__(self, tracker: ConceptTracker) -> None:
        self._t = tracker

    def __getitem__(self, cid: int) -> str | None:
        c = self._t._clusters.get(cid)
        return c.label if c else None

    def __setitem__(self, cid: int, value: str | None) -> None:
        if cid in self._t._clusters:
            self._t._clusters[cid].label = value
            self._t._clusters[cid].protected = value is not None

    def __len__(self) -> int:
        return len(self._t._clusters)

    def append(self, value: str | None) -> None:
        """Used by persistence loader — no-op; labels set via bulk setter."""
        pass


class _CountAccessor:
    def __init__(self, tracker: ConceptTracker) -> None:
        self._t = tracker

    def __getitem__(self, cid: int) -> int:
        c = self._t._clusters.get(cid)
        return c.count if c else 0

    def __len__(self) -> int:
        return len(self._t._clusters)

    def append(self, value: int) -> None:
        pass


class _LastSeenAccessor:
    def __init__(self, tracker: ConceptTracker) -> None:
        self._t = tracker

    def __getitem__(self, cid: int) -> int:
        c = self._t._clusters.get(cid)
        return c.last_seen if c else 0

    def __len__(self) -> int:
        return len(self._t._clusters)

    def append(self, value: int) -> None:
        pass
