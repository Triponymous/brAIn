"""ExperienceLog — what happened between the organism and its human.

One row per event: who acted (the pet, the human, or the LLM on its behalf),
what they did, the state the organism was in, and — filled in later — what
followed. That tuple (state, action, consequence, response) is the raw
material for everything the thesis needs: the proof that it is learning
THIS person, the reward for a learned behaviour policy, and the numbers
behind any later decision to fine-tune.

Facts only, no judgements: reward is computed by whoever reads the log, so
the reward function can change without losing history. No content: never a
chat message, never a keystroke — kinds, labels the human chose, app names,
numbers. The same privacy line as the rest of the organism.
"""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from bridge.felt_state import SIGNATURE_KEYS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    tick INTEGER,
    actor TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    felt_label TEXT,
    felt_conf REAL,
    cluster INTEGER,
    sleep INTEGER,
    signature TEXT,
    after_signature TEXT,
    after_ts REAL,
    response_kind TEXT,
    response_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_experiences_ts ON experiences(ts);
"""

_COLS = ("id", "ts", "tick", "actor", "kind", "payload", "felt_label", "felt_conf",
         "cluster", "sleep", "signature", "after_signature", "after_ts",
         "response_kind", "response_id")


def state_of(brain: Any) -> dict[str, Any]:
    """The organism's state right now, as the log records it with every event."""
    sig = getattr(brain, "_last_signature", None)
    fs = getattr(brain, "felt_state", None)
    cluster = getattr(brain, "_last_concept_cluster", None)
    label, conf = (None, 0.0)
    if fs is not None and sig:
        label, conf = fs.recognize(sig, cluster)
    return {
        "tick": brain.tick_count,
        "felt_label": label,
        "felt_conf": conf,
        "cluster": int(cluster) if cluster is not None else None,
        "sleep": bool(brain.sleep_mode),
        "signature": list(sig) if sig else None,
    }


class ExperienceLog:
    def __init__(self, path: Path, settle_after: float = 120.0) -> None:
        self.path = Path(path)
        # How long after an action its consequence is read. 120 s: long enough
        # for the 180 s modulator trend to move, short enough to still be about
        # this action rather than the next thing that happened.
        self.settle_after = settle_after
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(self, actor: str, kind: str, payload: dict | None = None, *,
               state: dict | None = None, now: float | None = None) -> int:
        """Log an event in the given state; returns its id."""
        state = state or {}
        sig = state.get("signature")
        cur = self._conn.execute(
            "INSERT INTO experiences(ts, tick, actor, kind, payload, felt_label, felt_conf, "
            "cluster, sleep, signature) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time() if now is None else now, state.get("tick"), actor, kind,
             json.dumps(payload or {}), state.get("felt_label"), state.get("felt_conf"),
             state.get("cluster"), int(bool(state.get("sleep"))) if "sleep" in state else None,
             json.dumps(sig) if sig else None),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def respond(self, event_id: int, response_kind: str, response_id: int | None = None) -> None:
        """Mark a pet event with how the human responded (answered, dismissed)."""
        self._conn.execute(
            "UPDATE experiences SET response_kind = ?, response_id = ? WHERE id = ?",
            (response_kind, response_id, event_id),
        )
        self._conn.commit()

    def settle(self, now: float, signature: list[float] | None) -> int:
        """Write the consequence of events that are settle_after seconds old.

        The consequence is the signature NOW. An event older than twice the
        window (the daemon was down in between) gets no signature, only a
        settle timestamp: today's mood is not what followed that action.
        Returns the number of rows settled.
        """
        due = now - self.settle_after
        stale = now - 2 * self.settle_after
        n = 0
        if signature:
            n += self._conn.execute(
                "UPDATE experiences SET after_signature = ?, after_ts = ? "
                "WHERE after_ts IS NULL AND ts <= ? AND ts > ?",
                (json.dumps(list(signature)), now, due, stale),
            ).rowcount
        n += self._conn.execute(
            "UPDATE experiences SET after_ts = ? WHERE after_ts IS NULL AND ts <= ?",
            (now, stale),
        ).rowcount
        self._conn.commit()
        return n

    def recent(self, limit: int = 50, since: float | None = None) -> list[dict[str, Any]]:
        """Most recent events first."""
        if since is None:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLS)} FROM experiences ORDER BY ts DESC, id DESC LIMIT ?",
                (limit,)).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLS)} FROM experiences WHERE ts >= ? "
                "ORDER BY ts DESC, id DESC LIMIT ?", (since, limit)).fetchall()
        return [self._row(r) for r in rows]

    def summary(self, since_hours: float = 168.0, now: float | None = None) -> dict[str, Any]:
        """Counts per actor and kind, how asks were answered, and the mean
        change of the signature after each kind of pet action."""
        now = time.time() if now is None else now
        rows = [self._row(r) for r in self._conn.execute(
            f"SELECT {', '.join(_COLS)} FROM experiences WHERE ts >= ? ORDER BY ts",
            (now - since_hours * 3600,)).fetchall()]

        counts: dict[str, dict[str, int]] = {}
        for r in rows:
            per = counts.setdefault(r["actor"], {})
            per[r["kind"]] = per.get(r["kind"], 0) + 1

        asks = [r for r in rows if r["actor"] == "pet" and r["kind"] == "ask_label"]
        answered = sum(1 for r in asks if r["response_kind"] == "answered")
        dismissed = sum(1 for r in asks if r["response_kind"] == "dismissed")

        consequence: dict[str, dict[str, Any]] = {}
        for r in rows:
            if r["actor"] != "pet" or not r["signature"] or not r["after_signature"]:
                continue
            c = consequence.setdefault(r["kind"], {"n": 0, "delta": {k: 0.0 for k in SIGNATURE_KEYS}})
            c["n"] += 1
            for k, after, before in zip(SIGNATURE_KEYS, r["after_signature"], r["signature"]):
                c["delta"][k] += after - before
        for c in consequence.values():
            c["delta"] = {k: round(v / c["n"], 5) for k, v in c["delta"].items()}

        return {
            "since_hours": since_hours,
            "events": len(rows),
            "counts": counts,
            "asks": {"asked": len(asks), "answered": answered, "dismissed": dismissed,
                     "unanswered": len(asks) - answered - dismissed,
                     "answer_rate": round(answered / len(asks), 3) if asks else None},
            "consequence": consequence,
        }

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row(r: tuple) -> dict[str, Any]:
        d = dict(zip(_COLS, r))
        d["payload"] = json.loads(d["payload"])
        d["signature"] = json.loads(d["signature"]) if d["signature"] else None
        d["after_signature"] = json.loads(d["after_signature"]) if d["after_signature"] else None
        d["sleep"] = bool(d["sleep"]) if d["sleep"] is not None else None
        return d
