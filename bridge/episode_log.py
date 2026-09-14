"""Episode Logger — SQLite-backed periodic brain state snapshots.

Stores snapshots of brain state every ~10s so the LLM can answer
questions about historical activity: "what happened today?",
"how was this week?", etc.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class EpisodeLogger:
    def __init__(self, path: Path, retention_days: float = 90.0) -> None:
        self.path = Path(path)
        self.retention_days = retention_days
        self._log_count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS episodes(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                tick INTEGER,
                modulators TEXT,
                active_concepts TEXT,
                sensor_summary TEXT,
                sleep_mode INTEGER
            )"""
        )
        # Every reader and the prune below filter on timestamp.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp)"
        )
        self._conn.commit()
        self.prune()

    def prune(self) -> int:
        """Delete episodes older than retention_days; returns rows removed.

        A row lands every ~10 s for the daemon's whole life, ~3 M rows a year
        with no cap. No reader looks back further than the HabitMiner's 7-day
        window, so the default 90 days keeps everything anyone queries.
        """
        cutoff = time.time() - self.retention_days * 86400
        cur = self._conn.execute("DELETE FROM episodes WHERE timestamp < ?", (cutoff,))
        self._conn.commit()
        return cur.rowcount

    def log(
        self,
        tick: int,
        modulators: dict[str, Any],
        active_concepts: list[dict],
        sensor_summary: dict[str, Any],
        sleep_mode: bool,
    ) -> None:
        """Insert a brain state snapshot row."""
        self._conn.execute(
            "INSERT INTO episodes(timestamp, tick, modulators, active_concepts, sensor_summary, sleep_mode) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                tick,
                json.dumps(modulators),
                json.dumps(active_concepts),
                json.dumps(sensor_summary),
                int(sleep_mode),
            ),
        )
        self._conn.commit()
        self._log_count += 1
        if self._log_count % 1000 == 0:  # ~every 3 h at one row per 10 s
            self.prune()

    def query(self, last_n: int = 10, since_timestamp: float | None = None) -> list[dict[str, Any]]:
        """Return recent episodes as list of dicts, most recent first."""
        if since_timestamp is not None:
            cursor = self._conn.execute(
                "SELECT id, timestamp, tick, modulators, active_concepts, sensor_summary, sleep_mode "
                "FROM episodes WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                (since_timestamp, last_n),
            )
        else:
            cursor = self._conn.execute(
                "SELECT id, timestamp, tick, modulators, active_concepts, sensor_summary, sleep_mode "
                "FROM episodes ORDER BY timestamp DESC LIMIT ?",
                (last_n,),
            )
        rows = cursor.fetchall()
        result = []
        for row in rows:
            result.append({
                "id": row[0],
                "timestamp": row[1],
                "tick": row[2],
                "modulators": json.loads(row[3]),
                "active_concepts": json.loads(row[4]),
                "sensor_summary": json.loads(row[5]),
                "sleep_mode": bool(row[6]),
            })
        return result

    def daily_summary(self, since_hours: float = 24.0) -> dict[str, Any]:
        """Aggregate episodes from the last N hours into a summary."""
        cutoff = time.time() - since_hours * 3600
        episodes = self.query(last_n=10000, since_timestamp=cutoff)

        if not episodes:
            return {
                "total_ticks": 0,
                "top_concepts": [],
                "avg_modulators": {},
                "top_apps": [],
                "episode_count": 0,
            }

        # Total ticks: difference between max and min tick
        ticks = [e["tick"] for e in episodes]
        total_ticks = max(ticks) - min(ticks) if len(ticks) > 1 else ticks[0]

        # Top concepts: count how often each concept label/id appears as active
        concept_counter: Counter = Counter()
        for ep in episodes:
            for c in ep["active_concepts"]:
                label = c.get("label", f"C{c.get('id', '?')}")
                if c.get("activation", 0) > 0.01:
                    concept_counter[label] += 1
        top_concepts = [{"name": name, "count": count} for name, count in concept_counter.most_common(10)]

        # Average modulators
        mod_sums: dict[str, float] = defaultdict(float)
        mod_count = 0
        for ep in episodes:
            mods = ep["modulators"]
            if mods:
                mod_count += 1
                for k, v in mods.items():
                    if isinstance(v, (int, float)):
                        mod_sums[k] += v
        avg_modulators = {k: round(v / mod_count, 4) for k, v in mod_sums.items()} if mod_count > 0 else {}

        # Top apps from sensor summaries
        app_counter: Counter = Counter()
        for ep in episodes:
            sensor = ep["sensor_summary"]
            app = sensor.get("app")
            if app:
                app_counter[app] += 1
        top_apps = [{"name": name, "count": count} for name, count in app_counter.most_common(10)]

        return {
            "total_ticks": total_ticks,
            "top_concepts": top_concepts,
            "avg_modulators": avg_modulators,
            "top_apps": top_apps,
            "episode_count": len(episodes),
        }

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
