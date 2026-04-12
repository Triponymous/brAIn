"""SynapseExplainer — translates SNN internals for LLM consumption.

Answers: WHY is NE high? WHAT does concept #5 respond to?
WHICH concepts are associated?
"""
from __future__ import annotations
import torch
from typing import Any
from brain.core import Brain


class SynapseExplainer:
    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    def explain_modulators(self) -> dict[str, str]:
        """Human-readable explanation of current modulator levels."""
        mods = self.brain.modulators.snapshot()
        explanations = {}
        da = mods.get("DA", 0)
        ne = mods.get("NE", 0)
        ach = mods.get("ACh", 0)
        sht = mods.get("5HT", 0)

        if ne > 0.08:
            explanations["NE"] = "Hoch — etwas Ueberraschendes oder Stressiges ist passiert"
        elif ne > 0.03:
            explanations["NE"] = "Leicht erhoeht — aufmerksam"
        else:
            explanations["NE"] = "Niedrig — entspannt"

        if da > 0.05:
            explanations["DA"] = "Hoch — etwas Neues passiert, ich lerne schneller"
        elif da > 0.01:
            explanations["DA"] = "Normal — stetige Wahrnehmung"
        else:
            explanations["DA"] = "Niedrig — nichts Neues, langweilig"

        if sht > 0.04:
            explanations["5HT"] = "Hoch — zufrieden, ruhige Arbeit"
        elif sht > 0.01:
            explanations["5HT"] = "Normal"
        else:
            explanations["5HT"] = "Niedrig — unruhig oder gestresst"

        if ach > 0.05:
            explanations["ACh"] = "Hoch — sehr fokussiert"
        else:
            explanations["ACh"] = "Normal"

        return explanations

    def concept_profile(self, cluster_id: int) -> dict[str, Any]:
        """What sensors typically trigger this concept cluster?"""
        syn = self.brain.synapses.get("sensory_concept")
        if syn is None:
            return {"error": "no synapse"}

        # Get the weight profile for neurons in this cluster
        # (approximation: use the concept tracker centroid)
        ct = self.brain.concept_tracker
        cluster = ct._clusters.get(cluster_id)
        if cluster is None:
            return {"error": f"cluster {cluster_id} not found"}

        # Top active dimensions in the centroid
        centroid = cluster.centroid
        top_dims = torch.topk(centroid, min(10, int(centroid.sum().item()))).indices.tolist()

        # Map dimensions to sensor names
        dim_names = self._dim_to_sensor(top_dims)
        return {
            "cluster_id": cluster_id,
            "label": cluster.label,
            "active_sensors": dim_names,
            "count": cluster.count,
        }

    def _dim_to_sensor(self, dims: list[int]) -> list[str]:
        """Map encoding dimensions to human-readable sensor names."""
        names = []
        for d in dims:
            if 0 <= d < 40:
                names.append(f"App-ID (dim {d})")
            elif 40 <= d < 60:
                names.append(f"Background-App (dim {d})")
            elif 60 <= d < 76:
                names.append(f"Tastatur-Bin (dim {d})")
            elif 76 <= d < 80:
                names.append(f"Tipp-Rhythmus (dim {d})")
            elif 80 <= d < 96:
                names.append(f"Maus-Bin (dim {d})")
            elif 96 <= d < 100:
                names.append(f"Maus-Rhythmus (dim {d})")
            elif 100 <= d < 108:
                names.append(f"Idle-Bin (dim {d})")
            elif 108 <= d < 112:
                names.append(f"Pausen-Typ (dim {d})")
            elif 112 <= d < 144:
                names.append(f"Mic-Mel (dim {d})")
            elif 144 <= d < 148:
                names.append(f"Mic-RMS (dim {d})")
            elif 148 <= d < 156:
                names.append(f"Tageszeit (dim {d})")
            elif 156 <= d < 160:
                names.append(f"App-Switch (dim {d})")
            elif 160 <= d < 164:
                names.append(f"Aktivitaets-Level (dim {d})")
            else:
                names.append(f"Reserve (dim {d})")
        return names

    def explain(self) -> dict[str, Any]:
        """Full explanation snapshot for LLM."""
        return {
            "modulator_causes": self.explain_modulators(),
        }
