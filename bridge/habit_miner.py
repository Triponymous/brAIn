"""HabitMiner — discovers recurring time-of-day and day-of-week patterns.

Queries the episode log for multi-day history and extracts:
- Hourly app usage patterns ("9am -> Slack, 10am -> VSCode")
- Weekly rhythms ("Mondays are meeting-heavy")
- Recurring concept clusters at specific times
"""
from __future__ import annotations
import json
import time
import datetime
from collections import Counter, defaultdict
from typing import Any

from bridge.episode_log import EpisodeLogger


class HabitMiner:
    def __init__(self, episode_logger: EpisodeLogger, lookback_days: int = 7) -> None:
        self.logger = episode_logger
        self.lookback_days = lookback_days
        self._cache: dict[str, Any] = {}
        self._cache_ts: float = 0
        self._cache_ttl: float = 300.0  # refresh every 5 min

    def _get_episodes(self) -> list[dict]:
        cutoff = time.time() - self.lookback_days * 86400
        return self.logger.query(last_n=100000, since_timestamp=cutoff)

    def hourly_profile(self) -> dict[int, dict[str, Any]]:
        """Activity profile per hour of day (0-23)."""
        episodes = self._get_episodes()
        hourly: dict[int, list[dict]] = defaultdict(list)

        for ep in episodes:
            ts = ep.get("timestamp", 0)
            hour = datetime.datetime.fromtimestamp(ts).hour
            hourly[hour].append(ep)

        profile = {}
        for hour in range(24):
            eps = hourly.get(hour, [])
            if not eps:
                profile[hour] = {"top_app": None, "avg_activity": 0, "episodes": 0}
                continue

            apps = Counter()
            for ep in eps:
                app = ep.get("sensor_summary", {}).get("app")
                if app:
                    apps[app] += 1

            top_app = apps.most_common(1)[0][0] if apps else None
            # Only episodes that observed the keyboard: an unshared one is not zero typing.
            keys = [k for ep in eps if (k := ep.get("sensor_summary", {}).get("keys")) is not None]
            avg_keys = round(sum(keys) / len(keys), 1) if keys else None

            profile[hour] = {
                "top_app": top_app,
                "avg_activity": avg_keys,
                "episodes": len(eps),
                "app_distribution": dict(apps.most_common(5)),
            }
        return profile

    def mine_habits(self) -> list[dict[str, Any]]:
        """Find recurring patterns: same app at same hour across multiple days."""
        episodes = self._get_episodes()
        # Group by (day_of_week, hour) -> app
        pattern_counts: dict[tuple[int, int, str], int] = Counter()
        day_counts: dict[tuple[int, int], int] = Counter()

        for ep in episodes:
            ts = ep.get("timestamp", 0)
            dt = datetime.datetime.fromtimestamp(ts)
            dow = dt.weekday()  # 0=Monday
            hour = dt.hour
            app = ep.get("sensor_summary", {}).get("app")
            if app:
                pattern_counts[(dow, hour, app)] += 1
                day_counts[(dow, hour)] += 1

        habits = []
        for (dow, hour, app), count in pattern_counts.most_common(50):
            total = day_counts[(dow, hour)]
            if total < 3:
                continue  # need at least 3 occurrences
            confidence = count / total
            if confidence > 0.5:  # app dominates >50% of that time slot
                habits.append({
                    "day_of_week": dow,
                    "day_name": ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
                                 "Freitag", "Samstag", "Sonntag"][dow],
                    "hour": hour,
                    "app": app,
                    "confidence": round(confidence, 2),
                    "occurrences": count,
                })

        habits.sort(key=lambda h: -h["confidence"])
        return habits[:20]

    def current_habits(self) -> list[dict[str, Any]]:
        """Cached version of mine_habits for real-time use."""
        now = time.time()
        if now - self._cache_ts > self._cache_ttl:
            self._cache["habits"] = self.mine_habits()
            self._cache["hourly"] = self.hourly_profile()
            self._cache_ts = now
        return self._cache.get("habits", [])

    def current_hour_context(self) -> dict[str, Any]:
        """What usually happens right now?"""
        hour = datetime.datetime.now().hour
        dow = datetime.datetime.now().weekday()
        habits = self.current_habits()
        matching = [h for h in habits if h["hour"] == hour]
        today_matching = [h for h in matching if h["day_of_week"] == dow]
        return {
            "hour": hour,
            "day_of_week": dow,
            "usual_habits": today_matching or matching[:3],
        }
