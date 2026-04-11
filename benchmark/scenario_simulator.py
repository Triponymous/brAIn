"""Realistic Desktop Activity Simulator.

Generates sensor snapshots that match real human behavior patterns.
Each scenario models a specific activity with realistic timing,
transitions, and noise. Used for benchmarking and validation.

Scenarios:
- calm_coding: steady typing, low mouse, VSCode, quiet
- stressed_coding: fast erratic typing, rapid window switches, high variability
- focus_flow: deep concentration, steady typing, no interruptions
- zoom_call: voices in mic, no typing, Zoom window
- browsing: lots of mouse, minimal typing, Chrome
- idle_away: nothing happening, high idle
- music_listening: Spotify, mic picks up music, occasional mouse
- returning: transition from idle to active (Leon comes back)
"""
import random
import math
from typing import Any


def _base_snap(app: str, keys: int, mouse: int, idle: float,
               mic_mel: list[float], mic_rms: float,
               variability: float = 0.0, burst: float = 0.0,
               switch_rate: float = 0.0) -> dict[str, Any]:
    """Build a sensor snapshot with realistic noise."""
    # Add natural jitter to all values
    keys = max(0, keys + random.randint(-2, 2))
    mouse = max(0, mouse + random.randint(-3, 3))
    idle = max(0, idle + random.uniform(-0.2, 0.2))
    mic_rms = max(0, mic_rms * random.uniform(0.8, 1.2))
    mel = [max(0, v * random.uniform(0.7, 1.3)) for v in mic_mel]

    return {
        "active_app": {"name": app, "switch_rate": switch_rate},
        "keystroke_rate": {"count": keys, "variability": variability, "burst": burst},
        "mouse_rate": {"count": mouse, "variability": random.uniform(0, 0.3)},
        "idle": {"seconds": idle},
        "mic": {"mel": mel, "rms": mic_rms},
        "time_tonic": _time_tonic(),
    }


def _time_tonic(hour: float | None = None) -> dict:
    """Generate time tonic for a specific hour (0-24)."""
    if hour is None:
        import datetime
        now = datetime.datetime.now()
        hour = now.hour + now.minute / 60.0
    day_frac = hour / 24.0
    day_angle = day_frac * 2 * math.pi
    return {
        "day_phase": [math.cos(day_angle), math.sin(day_angle),
                      math.cos(2 * day_angle), math.sin(2 * day_angle)],
        "week_phase": [0.5, 0.5, 0.5, 0.5],  # simplified
    }


# ── Quiet mic profiles ──
_SILENCE_MEL = [0.001] * 32
_KEYBOARD_MEL = [0.01] * 10 + [0.05] * 12 + [0.01] * 10  # mid-range click sounds
_VOICE_MEL = [0.08] * 8 + [0.12] * 8 + [0.06] * 8 + [0.03] * 8  # voice spectrum
_MUSIC_MEL = [0.03] * 4 + [0.08] * 8 + [0.15] * 8 + [0.08] * 8 + [0.03] * 4


def calm_coding() -> dict[str, Any]:
    """Steady coding in VSCode. Consistent typing rhythm, quiet."""
    return _base_snap(
        app="VSCode", keys=random.randint(12, 22), mouse=random.randint(1, 5),
        idle=random.uniform(0.1, 0.8), mic_mel=_KEYBOARD_MEL,
        mic_rms=random.uniform(0.01, 0.03), variability=random.uniform(0.1, 0.3),
    )


def stressed_coding() -> dict[str, Any]:
    """Hektisches Tippen, schnelle Fensterwechsel, hohe Variabilität."""
    return _base_snap(
        app=random.choice(["VSCode", "Terminal", "Chrome", "Slack"]),
        keys=random.randint(25, 45), mouse=random.randint(15, 40),
        idle=random.uniform(0.0, 0.3), mic_mel=_KEYBOARD_MEL,
        mic_rms=random.uniform(0.02, 0.06),
        variability=random.uniform(1.0, 2.0), burst=random.choice([0.0, 1.0]),
        switch_rate=random.uniform(8, 15),  # frantic switching
    )


def focus_flow() -> dict[str, Any]:
    """Deep focus: steady, rhythmic typing, no interruptions."""
    return _base_snap(
        app="VSCode", keys=random.randint(15, 25), mouse=random.randint(0, 3),
        idle=random.uniform(0.1, 0.5), mic_mel=_SILENCE_MEL,
        mic_rms=random.uniform(0.001, 0.005), variability=random.uniform(0.05, 0.15),
    )


def zoom_call() -> dict[str, Any]:
    """In a call: voices in mic, no typing, Zoom window."""
    return _base_snap(
        app="Zoom", keys=0, mouse=random.randint(0, 2),
        idle=random.uniform(0.5, 3.0), mic_mel=_VOICE_MEL,
        mic_rms=random.uniform(0.05, 0.15),
    )


def browsing() -> dict[str, Any]:
    """Browsing: lots of mouse, minimal typing, Chrome."""
    return _base_snap(
        app="Chrome", keys=random.randint(0, 3), mouse=random.randint(20, 50),
        idle=random.uniform(0.2, 1.5), mic_mel=_SILENCE_MEL,
        mic_rms=random.uniform(0.001, 0.005), variability=random.uniform(0.0, 0.2),
    )


def idle_away() -> dict[str, Any]:
    """User is away: nothing happening."""
    return _base_snap(
        app="Finder", keys=0, mouse=0,
        idle=random.uniform(60, 600), mic_mel=_SILENCE_MEL,
        mic_rms=random.uniform(0.0005, 0.002),
    )


def music_listening() -> dict[str, Any]:
    """Listening to music on Spotify, occasional browsing."""
    return _base_snap(
        app="Spotify", keys=0, mouse=random.randint(0, 5),
        idle=random.uniform(1.0, 10.0), mic_mel=_MUSIC_MEL,
        mic_rms=random.uniform(0.03, 0.08),
    )


def returning_from_break() -> dict[str, Any]:
    """Leon just came back — mouse moving, app switching."""
    return _base_snap(
        app=random.choice(["Chrome", "VSCode", "Mail"]),
        keys=random.randint(0, 5), mouse=random.randint(10, 30),
        idle=random.uniform(0.1, 1.0), mic_mel=_SILENCE_MEL,
        mic_rms=random.uniform(0.001, 0.01),
        switch_rate=random.uniform(3, 8),
    )


# ── Full day simulation ──
WORKDAY_SCHEDULE = [
    # (start_tick, duration_ticks, scenario_name, scenario_fn)
    (0,      5000,  "morning_mail",    browsing),          # 0:00 - checking mail
    (5000,   15000, "coding_morning",  calm_coding),       # 0:50 - morning coding
    (20000,  3000,  "coffee_break",    idle_away),          # 3:20 - coffee break
    (23000,  12000, "coding_focus",    focus_flow),         # 3:50 - deep focus
    (35000,  5000,  "standup_call",    zoom_call),          # 5:50 - standup
    (40000,  3000,  "browsing",        browsing),           # 6:40 - browsing after call
    (43000,  15000, "afternoon_code",  calm_coding),        # 7:10 - afternoon coding
    (58000,  5000,  "stress_sprint",   stressed_coding),    # 9:40 - deadline stress
    (63000,  3000,  "winding_down",    music_listening),    # 10:30 - winding down
    (66000,  4000,  "end_of_day",      idle_away),          # 11:00 - leaving
]


def get_workday_pattern(tick: int) -> tuple[str, dict[str, Any]]:
    """Get the scenario for a given tick in a simulated workday."""
    for start, duration, name, fn in WORKDAY_SCHEDULE:
        if start <= tick < start + duration:
            return name, fn()
    # Default: idle
    return "idle", idle_away()


ALL_SCENARIOS = {
    "calm_coding": calm_coding,
    "stressed_coding": stressed_coding,
    "focus_flow": focus_flow,
    "zoom_call": zoom_call,
    "browsing": browsing,
    "idle_away": idle_away,
    "music_listening": music_listening,
    "returning": returning_from_break,
}
