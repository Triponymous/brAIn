"""TimeTonicSensor — slow biological-clock oscillators.

Provides eight scalar values per sample (4 day-cycle phases, 4 week-cycle
phases). The four phases per cycle are sin/cos at the fundamental frequency
and at twice the frequency, giving the brain enough basis vectors to
disambiguate "morning" from "afternoon" from "early morning" from "late
morning" without us defining any of those concepts explicitly.

The brain only needs to learn that "this combination of 4 numbers correlates
with X happening" — and the sin/cos basis is dense enough.
"""
from __future__ import annotations
import math
from datetime import datetime
from typing import Any

from adapters.base import Sensor


class TimeTonicSensor(Sensor):
    name = "time_tonic"
    rate_hz = 1.0

    def _encode(self, now: datetime) -> dict[str, list[float]]:
        # Day phase: seconds since midnight / 86400 in [0, 1)
        seconds_today = now.hour * 3600 + now.minute * 60 + now.second
        day_frac = seconds_today / 86400.0
        day_angle = day_frac * 2 * math.pi
        day_phase = [
            math.cos(day_angle),
            math.sin(day_angle),
            math.cos(2 * day_angle),
            math.sin(2 * day_angle),
        ]
        # Week phase: weekday(0=Mon..6=Sun) + day fraction, normalized to [0, 1)
        week_frac = (now.weekday() + day_frac) / 7.0
        week_angle = week_frac * 2 * math.pi
        week_phase = [
            math.cos(week_angle),
            math.sin(week_angle),
            math.cos(2 * week_angle),
            math.sin(2 * week_angle),
        ]
        return {"day_phase": day_phase, "week_phase": week_phase}

    async def sample(self) -> dict[str, list[float]]:
        return self._encode(datetime.now())
