"""AnomalyDetector -- detects deviations from learned habits.

Compares current sensor state against the HabitMiner's baseline
for this hour/day and flags anything unusual:
- missing_habit: expected app not in use
- unusual_quiet: normally active but today silent
"""
from __future__ import annotations

import datetime
from typing import Any

from bridge.habit_miner import HabitMiner


class AnomalyDetector:
    def __init__(self, habit_miner: HabitMiner) -> None:
        self.habit_miner = habit_miner

    def check(self, current_sensor: dict[str, Any]) -> list[dict[str, Any]]:
        """Compare current state to what's usual at this time.

        Returns a list of anomaly dicts with keys:
            type: "missing_habit" | "unusual_quiet"
            description: German-language explanation
            severity: "low" | "medium" | "high"
        """
        now = datetime.datetime.now()
        hour = now.hour
        dow = now.weekday()

        habits = self.habit_miner.current_habits()
        anomalies: list[dict[str, Any]] = []

        # --- Check: expected app at this hour ---
        expected_at_hour = [
            h for h in habits
            if h["hour"] == hour and h["day_of_week"] == dow
        ]
        # Without the app source (or the keyboard below) there is nothing to compare:
        # "usually VSCode, today not" would be a claim about data never observed.
        current_app = current_sensor.get("app")

        for habit in expected_at_hour:
            if current_app is not None and habit["app"] != current_app and habit["confidence"] > 0.6:
                anomalies.append({
                    "type": "missing_habit",
                    "description": (
                        f"Normalerweise bist du um {hour}h in {habit['app']} "
                        f"({habit['day_name']}), aber heute nicht."
                    ),
                    "severity": "medium",
                    "expected": habit["app"],
                    "actual": current_app,
                })

        # --- Check: unusual inactivity ---
        hourly = self.habit_miner.hourly_profile()
        usual = hourly.get(hour, {})
        usual_activity = usual.get("avg_activity") or 0
        current_keys = current_sensor.get("keys")

        if current_keys is not None and usual_activity > 10 and current_keys < 2:
            anomalies.append({
                "type": "unusual_quiet",
                "description": (
                    f"Normalerweise tippst du um {hour}h viel "
                    f"(avg {usual_activity:.0f} keys), aber heute ist es still."
                ),
                "severity": "low",
            })

        return anomalies
