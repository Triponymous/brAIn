"""SNN State Narrator — translates brain internals into natural language.

The LLM should understand the SNN the way a neuroscientist would explain it
to a child: not numbers, but stories.

Instead of: "DA=0.006, NE=0.012, concept_cluster=3, wm_active=12"
The LLM sees: "Ich bin entspannt und aufmerksam. Mein Gehirn erkennt ein
vertrautes Muster — Leon tippt in Claude Code, das kenne ich gut (45x gesehen).
Im Kurzzeitgedaechtnis halte ich noch die Chrome-Session von vorhin."
"""
from __future__ import annotations
from typing import Any

from brain.core import Brain


class SNNNarrator:
    """Produces natural-language descriptions of the SNN's internal state."""

    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    def narrate_full_state(self) -> str:
        """Complete narrative of what the brain currently 'thinks'."""
        parts = []
        parts.append(self._narrate_modulators())
        parts.append(self._narrate_current_concept())
        parts.append(self._narrate_wm())
        parts.append(self._narrate_learning())
        return "\n".join(p for p in parts if p)

    def _narrate_modulators(self) -> str:
        """Why do I feel this way?"""
        mods = self.brain.modulators.snapshot()
        da = mods.get("DA", 0)
        ne = mods.get("NE", 0)
        ach = mods.get("ACh", 0)
        sht = mods.get("5HT", 0)

        parts = []

        # DA = Novelty/Interest
        if da > 0.05:
            parts.append("Etwas Neues ist passiert — meine Neugier ist geweckt")
        elif da > 0.02:
            parts.append("Leicht interessiert — es passiert was")
        elif da < 0.005:
            parts.append("Nichts Neues — alles wie immer")

        # NE = Arousal/Surprise
        if ne > 0.10:
            parts.append("Ich bin aufgeschreckt — etwas hat sich ploetzlich veraendert!")
        elif ne > 0.05:
            parts.append("Aufmerksam — irgendwas hat sich getan")

        # 5HT = Contentment
        if sht > 0.04:
            parts.append("Ich fuehle mich wohl — alles laeuft ruhig")
        elif sht < 0.01:
            parts.append("Etwas unruhig — die Stimmung ist nicht ganz entspannt")

        # ACh = Focus
        if ach > 0.05:
            parts.append("Mein Fokus ist hoch — ich beobachte genau")

        if not parts:
            return "Emotional ausgeglichen — weder aufgeregt noch gelangweilt."
        return "Emotionen: " + ". ".join(parts) + "."

    def _narrate_current_concept(self) -> str:
        """What pattern am I recognizing?"""
        ct = self.brain.concept_tracker
        snap = ct.snapshot()
        current = snap.get("current_cluster", -1)
        label = snap.get("current_label")
        debug = snap.get("debug", {})

        if current < 0:
            return "Ich erkenne gerade kein klares Muster."

        cluster = ct._clusters.get(current)
        if not cluster:
            return "Ich sehe ein Muster, aber es ist mir noch unbekannt."

        confidence = debug.get("best_sim", 0)
        count = cluster.count

        if label:
            if confidence > 0.6:
                return f"Ich erkenne '{label}' — sehr sicher (Aehnlichkeit {confidence:.0%}, {count}x gesehen)."
            else:
                return f"Das sieht nach '{label}' aus, aber ich bin nicht ganz sicher ({confidence:.0%})."
        else:
            if count > 50:
                return f"Ein haeufiges Muster ohne Namen ({count}x gesehen) — ich sollte Leon fragen was das ist."
            elif count > 10:
                return f"Ein Muster das ich schon {count}x gesehen habe, aber noch nicht kenne."
            else:
                return "Ein neues Muster — ich beobachte es noch."

    def _narrate_wm(self) -> str:
        """What am I holding in working memory?"""
        wm = self.brain.regions.get("wm")
        if wm is None:
            return ""

        active = int(wm.last_spikes.sum().item())
        if active == 0:
            return "Mein Kurzzeitgedaechtnis ist leer."
        elif active > 15:
            return f"Ich halte viel im Kurzzeitgedaechtnis ({active} aktive Erinnerungen)."
        elif active > 5:
            return f"Einige Erinnerungen aktiv ({active} Slots) — ich denke noch an was vorher war."
        else:
            return f"Wenig im Kopf ({active} Slots) — ziemlich aufgeraeumt gerade."

    def _narrate_learning(self) -> str:
        """How much have I learned?"""
        syn = self.brain.synapses.get("sensory_concept")
        if syn is None:
            return ""

        weights = syn.weights
        mean_w = float(weights.mean().item())
        std_w = float(weights.std().item())

        # Count specialized neurons (std of their weight row > threshold)
        row_stds = weights.std(dim=1)
        specialized = int((row_stds > 0.1).sum().item())
        total = weights.shape[0]
        pct = specialized / total * 100

        if pct > 80:
            return f"Mein Gehirn ist gut trainiert — {specialized}/{total} Neuronen haben sich spezialisiert."
        elif pct > 30:
            return f"Ich lerne noch — {specialized}/{total} Neuronen spezialisiert ({pct:.0f}%)."
        elif self.brain.tick_count < 10000:
            return "Ich bin noch ganz frisch — mein Gehirn lernt gerade die ersten Muster."
        else:
            return f"Meine Neuronen sind noch nicht sehr spezialisiert ({pct:.0f}%) — ich brauche mehr Erfahrung."
