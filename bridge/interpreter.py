"""BrainInterpreter — the pet's 'consciousness layer'.

Orchestrates all interpretation modules and produces a unified
understanding of the current situation for the LLM and proactive engine.
"""
from __future__ import annotations
from typing import Any

from brain.core import Brain
from bridge.episode_log import EpisodeLogger
from bridge.state_detector import StateDetector
from bridge.habit_miner import HabitMiner
from bridge.personality import PersonalityTracker
from bridge.narrative import NarrativeBuilder
from bridge.anomaly import AnomalyDetector
from bridge.synapse_explainer import SynapseExplainer


class BrainInterpreter:
    def __init__(self, brain: Brain, episode_logger: EpisodeLogger) -> None:
        self.brain = brain
        self.state_detector = StateDetector()
        self.habit_miner = HabitMiner(episode_logger)
        self.personality = PersonalityTracker()
        self.narrative = NarrativeBuilder(episode_logger)
        self.anomaly = AnomalyDetector(self.habit_miner)
        self.synapse_explainer = SynapseExplainer(brain)

        # Load personality from brain if available
        if hasattr(brain, '_personality_state'):
            self.personality.load_state(brain._personality_state)

    def tick(self) -> None:
        """Call every ~100 ticks (1 second) to update rolling state."""
        self.state_detector.update(self.brain)
        # Update personality every 100 calls (~100 seconds)
        if self.state_detector._snapshot_count % 100 == 0:
            self.personality.update(self.brain.modulators.snapshot())

    def interpret(self) -> dict[str, Any]:
        """Full interpretation for LLM system prompt."""
        sd = getattr(self.brain, '_last_sensor_display', {})
        return {
            "states": self.state_detector.detect(),
            "personality": self.personality.snapshot(),
            "habits": self.habit_miner.current_hour_context(),
            "anomalies": self.anomaly.check(sd),
            "narrative": self.narrative.today_summary(),
            "explanations": self.synapse_explainer.explain(),
        }

    def format_for_prompt(self) -> str:
        """Render interpretation as text for system prompt injection."""
        data = self.interpret()
        lines = ["=== MEIN BEWUSSTSEIN (was ich VERSTEHE) ==="]

        # States
        states = data["states"]
        if states["flow"]:
            lines.append(f"Zustand: Im FLOW seit {states['flow_duration_min']}min in {states['flow_app']}")
        elif states["stress"]:
            lines.append("Zustand: STRESS erkannt — Leon ist hektisch")
        elif states["meeting"]:
            lines.append(f"Zustand: Im MEETING seit {states['meeting_duration_min']}min")
        elif states["needs_break"]:
            lines.append(f"Zustand: Leon arbeitet seit {states['active_minutes']}min ohne Pause")
        else:
            lines.append("Zustand: Normal")

        # Personality
        p = data["personality"]
        traits = p["traits"]
        trait_strs = []
        if traits["curiosity"] > 0.3:
            trait_strs.append("neugierig")
        if traits["patience"] > 0.3:
            trait_strs.append("geduldig")
        if traits["anxiety"] > 0.3:
            trait_strs.append("nervoes")
        if traits["attentiveness"] > 0.3:
            trait_strs.append("aufmerksam")
        if trait_strs:
            lines.append(f"Persoenlichkeit: {', '.join(trait_strs)}")

        # Habits
        habits = data["habits"]
        if habits.get("usual_habits"):
            for h in habits["usual_habits"][:2]:
                lines.append(f"Gewohnheit: Um {h['hour']}h {h['day_name']} -> meistens {h['app']}")

        # Anomalies
        for a in data["anomalies"][:2]:
            lines.append(f"Anomalie: {a['description']}")

        # Narrative (short)
        narrative = data["narrative"]
        if narrative and narrative != "Heute noch nicht viel passiert.":
            lines.append(f"Heute bisher: {narrative[:100]}")

        # Modulator explanations
        expl = data["explanations"].get("modulator_causes", {})
        important = {k: v for k, v in expl.items()
                     if "Hoch" in v or "Niedrig" in v}
        if important:
            lines.append("Warum ich mich so fuehle: " +
                         "; ".join(f"{k}={v}" for k, v in important.items()))

        return "\n".join(lines)

    def save_personality(self) -> dict[str, Any]:
        """For persistence — save personality state."""
        return self.personality.save_state()

    def load_personality(self, data: dict[str, Any]) -> None:
        """For persistence — load personality state."""
        self.personality.load_state(data)
