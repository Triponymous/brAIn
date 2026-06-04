"""FeltState — the pet's learned emotional self-model.

Maps a modulator SIGNATURE (the 180s trend) to Leon-given labels, learned from
sparse corrections. Blank slate: starts empty, earns labels one correction at a
time. Personalization core — the pet learns Leon's OWN states (e.g. "flow"),
which the generic 6-emotion classifier cannot represent.
"""
from __future__ import annotations

# Fixed-order trend dims that form a signature (a contract — don't reorder).
SIGNATURE_KEYS = ("avg_DA", "avg_NE", "avg_ACh", "avg_5HT", "peak_DA", "peak_NE")

_SCALE = 0.07              # modulators live ~0-0.07 -> normalize each dim to ~[0,1]
_DEFAULT_THRESHOLD = 0.35  # max normalized distance still counted as "same state"
_EWMA = 0.3                # weight for sharpening a prototype per correction
_CLUSTER_PENALTY = 0.4     # behavior axis: a concept-cluster mismatch ADDS this to the
                           # affect distance (soft, not a hard gate). > threshold so a pure
                           # cluster mismatch separates two same-affect states; an unknown
                           # cluster (-1/None) skips it -> graceful affect-only fallback.


def signature_from_trend(trend: dict) -> list[float]:
    return [float(trend.get(k, 0.0)) for k in SIGNATURE_KEYS]


def _distance(a: list[float], b: list[float]) -> float:
    return (sum(((x - y) / _SCALE) ** 2 for x, y in zip(a, b)) ** 0.5) / (len(a) ** 0.5)


class FeltState:
    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._prototypes: dict[str, dict] = {}
        self._threshold = threshold

    def label(self, name: str, sig: list[float], cluster: int | None = None) -> None:
        proto = self._prototypes.get(name)
        if proto is None:
            self._prototypes[name] = {"centroid": list(sig), "count": 1, "cluster": cluster}
        else:
            c = proto["centroid"]
            proto["centroid"] = [o * (1 - _EWMA) + n * _EWMA for o, n in zip(c, sig)]
            proto["count"] += 1
            if proto.get("cluster") is None and cluster is not None and cluster >= 0:
                proto["cluster"] = cluster

    def recognize(self, sig: list[float], cluster: int | None = None) -> tuple[str | None, float]:
        best, best_d = None, float("inf")
        for name, proto in self._prototypes.items():
            d = _distance(sig, proto["centroid"])
            pc = proto.get("cluster")
            if (cluster is not None and cluster >= 0
                    and pc is not None and pc >= 0 and pc != cluster):
                d += _CLUSTER_PENALTY          # behavior mismatch -> soft separation
            if d < best_d:
                best, best_d = name, d
        if best is None or best_d > self._threshold:
            return None, 0.0
        return best, round(1.0 - best_d / self._threshold, 3)

    def known_labels(self) -> list[str]:
        return sorted(self._prototypes)

    def to_dict(self) -> dict:
        return {"threshold": self._threshold, "prototypes": self._prototypes}

    @classmethod
    def from_dict(cls, data: dict) -> "FeltState":
        fs = cls(threshold=data.get("threshold", _DEFAULT_THRESHOLD))
        fs._prototypes = {
            k: {"centroid": list(v["centroid"]), "count": int(v["count"]),
                "cluster": v.get("cluster")}
            for k, v in data.get("prototypes", {}).items()
        }
        return fs


class FeltStateWatcher:
    """Decides WHEN to ask Leon to label a state — only on a SUSTAINED unknown
    stretch, rate-limited. Active learning: query at the moment of uncertainty
    (a state the pet doesn't recognize), not constantly. Fed the recognized
    label each tick; pure + testable (timestamps injected, no wall-clock here).
    """

    def __init__(self, hold_seconds: float = 25.0, cooldown_seconds: float = 180.0) -> None:
        self._hold = hold_seconds
        self._cooldown = cooldown_seconds
        self._unknown_since: float | None = None
        self._last_ask: float | None = None

    def observe(self, label: str | None, now: float) -> None:
        if label is None:                      # pet does not recognize the current state
            if self._unknown_since is None:
                self._unknown_since = now
        else:                                  # recognized → not a moment to ask
            self._unknown_since = None

    def should_ask(self, now: float) -> bool:
        if self._unknown_since is None:
            return False
        if now - self._unknown_since < self._hold:
            return False
        if self._last_ask is not None and now - self._last_ask < self._cooldown:
            return False
        self._last_ask = now
        self._unknown_since = None             # don't re-ask about the same stretch immediately
        return True
