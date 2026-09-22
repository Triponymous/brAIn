"""NarrativeBuilder -- creates temporal stories from episode data.

Reads the EpisodeLogger and produces short German summaries
of what happened today, yesterday, or over a time window.
Used by BrainInterpreter to give the LLM temporal context.
"""
from __future__ import annotations

import datetime
import time
from typing import Any

from bridge.episode_log import EpisodeLogger


class NarrativeBuilder:
    def __init__(self, episode_logger: EpisodeLogger) -> None:
        self.logger = episode_logger

    def today_summary(self) -> str:
        """Narrative of today's activity so far, grouped by hour."""
        midnight = _local_midnight_ts(offset_days=0)
        episodes = self.logger.query(last_n=5000, since_timestamp=midnight)
        if not episodes:
            return "Heute noch nicht viel passiert."
        return self._build_narrative(episodes)

    def yesterday_summary(self) -> str:
        """Narrative of yesterday's activity, grouped by hour."""
        yesterday_start = _local_midnight_ts(offset_days=1)
        yesterday_end = _local_midnight_ts(offset_days=0)
        episodes = self.logger.query(last_n=5000, since_timestamp=yesterday_start)
        # Filter out anything from today (query returns >= since_timestamp)
        episodes = [e for e in episodes if e["timestamp"] < yesterday_end]
        if not episodes:
            return "Gestern keine Daten."
        return self._build_narrative(episodes)

    def _build_narrative(self, episodes: list[dict[str, Any]]) -> str:
        """Convert episodes into a short German narrative grouped by hour.

        Format: "10h: intensiv gearbeitet in VSCode | 11h: ruhig in Chrome | ..."
        """
        # Group by hour of day
        hourly: dict[int, list[dict[str, Any]]] = {}
        for ep in episodes:
            hour = datetime.datetime.fromtimestamp(ep["timestamp"]).hour
            hourly.setdefault(hour, []).append(ep)

        parts: list[str] = []
        for hour in sorted(hourly.keys()):
            eps = hourly[hour]
            # Determine dominant app
            # Only what was observed: an hour without the app or keyboard source
            # says nothing about either (it used to read as "ruhig in ?").
            apps = [a for ep in eps if (a := ep.get("sensor_summary", {}).get("app"))]
            top_app = max(set(apps), key=apps.count) if apps else None
            keys = [k for ep in eps if (k := ep.get("sensor_summary", {}).get("keys")) is not None]
            activity = None
            if keys:
                avg_keys = sum(keys) / len(keys)
                if avg_keys > 15:
                    activity = "intensiv gearbeitet"
                elif avg_keys > 3:
                    activity = "leicht aktiv"
                else:
                    activity = "ruhig"
            said = " ".join(x for x in (activity, f"in {top_app}" if top_app else None) if x)
            if said:
                parts.append(f"{hour}h: {said}")

        # Cap at 8 hourly blocks for conciseness
        return " | ".join(parts[:8]) if parts else "Keine Aktivitaet."


def _local_midnight_ts(offset_days: int = 0) -> float:
    """Return the UNIX timestamp of local midnight, optionally N days ago.

    offset_days=0 -> today's midnight (start of today)
    offset_days=1 -> yesterday's midnight
    """
    today = datetime.date.today()
    target = today - datetime.timedelta(days=offset_days)
    midnight = datetime.datetime.combine(target, datetime.time.min)
    return midnight.timestamp()
