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


def signature_from_trend(trend: dict) -> list[float]:
    return [float(trend.get(k, 0.0)) for k in SIGNATURE_KEYS]


def _distance(a: list[float], b: list[float]) -> float:
    return (sum(((x - y) / _SCALE) ** 2 for x, y in zip(a, b)) ** 0.5) / (len(a) ** 0.5)


class FeltState:
    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._prototypes: dict[str, dict] = {}
        self._threshold = threshold

    def label(self, name: str, sig: list[float]) -> None:
        proto = self._prototypes.get(name)
        if proto is None:
            self._prototypes[name] = {"centroid": list(sig), "count": 1}
        else:
            c = proto["centroid"]
            proto["centroid"] = [o * (1 - _EWMA) + n * _EWMA for o, n in zip(c, sig)]
            proto["count"] += 1

    def recognize(self, sig: list[float]) -> tuple[str | None, float]:
        best, best_d = None, float("inf")
        for name, proto in self._prototypes.items():
            d = _distance(sig, proto["centroid"])
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
            k: {"centroid": list(v["centroid"]), "count": int(v["count"])}
            for k, v in data.get("prototypes", {}).items()
        }
        return fs
