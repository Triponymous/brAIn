"""Tests for SynapseExplainer — translates SNN internals for LLM."""
import pytest
import torch
from unittest.mock import MagicMock, PropertyMock
from bridge.synapse_explainer import SynapseExplainer


# ── helpers ──────────────────────────────────────────────────────────

def _mock_brain(da=0.02, ne=0.01, ach=0.03, sht=0.03):
    """Create a MagicMock Brain with configurable modulator levels."""
    brain = MagicMock()
    brain.modulators.snapshot.return_value = {
        "DA": da, "NE": ne, "ACh": ach, "5HT": sht,
    }
    brain.synapses = {"sensory_concept": MagicMock()}
    return brain


def _add_cluster(brain, cluster_id, centroid_dims, label=None, count=10):
    """Add a fake _Cluster to brain.concept_tracker._clusters.

    Uses a real dict and a SimpleNamespace (not MagicMock) for the cluster
    so that torch.topk receives an actual Tensor, not a MagicMock proxy.
    """
    centroid = torch.zeros(200)
    for d in centroid_dims:
        centroid[d] = 1.0

    # SimpleNamespace gives us real attribute access without MagicMock
    # intercepting .centroid and returning another mock.
    from types import SimpleNamespace
    cluster = SimpleNamespace(centroid=centroid, label=label, count=count)

    # Ensure _clusters is a real dict (MagicMock.getattr always returns
    # another MagicMock, never the default, so we check type explicitly).
    if not isinstance(getattr(brain.concept_tracker, '_clusters', None), dict):
        brain.concept_tracker._clusters = {}
    brain.concept_tracker._clusters[cluster_id] = cluster


# ── explain_modulators ───────────────────────────────────────────────

class TestExplainModulators:
    def test_high_ne(self):
        """NE > 0.08 triggers 'Ueberraschendes' explanation."""
        brain = _mock_brain(ne=0.15)
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert "Ueberraschendes" in expl["NE"]

    def test_medium_ne(self):
        """NE between 0.03 and 0.08 triggers 'aufmerksam'."""
        brain = _mock_brain(ne=0.05)
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert "aufmerksam" in expl["NE"]

    def test_low_ne(self):
        """NE < 0.03 triggers 'entspannt'."""
        brain = _mock_brain(ne=0.01)
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert "entspannt" in expl["NE"]

    def test_high_da(self):
        """DA > 0.05 triggers 'Neues' explanation."""
        brain = _mock_brain(da=0.10)
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert "Neues" in expl["DA"]

    def test_normal_da(self):
        """DA between 0.01 and 0.05 triggers 'stetige Wahrnehmung'."""
        brain = _mock_brain(da=0.03)
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert "stetige" in expl["DA"]

    def test_low_da(self):
        """DA < 0.01 triggers 'langweilig'."""
        brain = _mock_brain(da=0.005)
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert "langweilig" in expl["DA"]

    def test_high_sht(self):
        """5HT > 0.04 triggers 'zufrieden'."""
        brain = _mock_brain(sht=0.06)
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert "zufrieden" in expl["5HT"]

    def test_normal_sht(self):
        """5HT between 0.01 and 0.04 triggers 'Normal'."""
        brain = _mock_brain(sht=0.02)
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert expl["5HT"] == "Normal"

    def test_low_sht(self):
        """5HT < 0.01 triggers 'unruhig oder gestresst'."""
        brain = _mock_brain(sht=0.005)
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert "gestresst" in expl["5HT"]

    def test_high_ach(self):
        """ACh > 0.05 triggers 'fokussiert'."""
        brain = _mock_brain(ach=0.08)
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert "fokussiert" in expl["ACh"]

    def test_normal_ach(self):
        """ACh <= 0.05 triggers 'Normal'."""
        brain = _mock_brain(ach=0.03)
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert expl["ACh"] == "Normal"

    def test_all_four_modulators_present(self):
        """All four modulators must always be present in the output."""
        brain = _mock_brain()
        se = SynapseExplainer(brain)
        expl = se.explain_modulators()
        assert set(expl.keys()) == {"NE", "DA", "5HT", "ACh"}


# ── concept_profile ──────────────────────────────────────────────────

class TestConceptProfile:
    def test_valid_cluster(self):
        """concept_profile returns active sensors for a known cluster."""
        brain = _mock_brain()
        _add_cluster(brain, 3, [5, 12, 65, 82], label="VSCode-coding", count=42)
        se = SynapseExplainer(brain)
        result = se.concept_profile(3)
        assert result["cluster_id"] == 3
        assert result["label"] == "VSCode-coding"
        assert result["count"] == 42
        assert len(result["active_sensors"]) == 4

    def test_cluster_sensor_mapping(self):
        """Verify dimension-to-sensor name mapping is correct."""
        brain = _mock_brain()
        # Activate one dim from each category
        dims = [5, 45, 70, 77, 85, 97, 103, 109, 120, 145]
        _add_cluster(brain, 0, dims, label="mixed")
        se = SynapseExplainer(brain)
        result = se.concept_profile(0)
        sensors = result["active_sensors"]
        assert len(sensors) == 10
        # Check categories are present
        categories = [s.split(" (")[0] for s in sensors]
        assert "App-ID" in categories
        assert "Background-App" in categories
        assert "Tastatur-Bin" in categories
        assert "Tipp-Rhythmus" in categories
        assert "Maus-Bin" in categories
        assert "Maus-Rhythmus" in categories
        assert "Idle-Bin" in categories
        assert "Pausen-Typ" in categories
        assert "Mic-Mel" in categories
        assert "Mic-RMS" in categories

    def test_missing_cluster(self):
        """concept_profile returns error for nonexistent cluster."""
        brain = _mock_brain()
        brain.concept_tracker._clusters = {}
        se = SynapseExplainer(brain)
        result = se.concept_profile(999)
        assert "error" in result

    def test_no_synapse(self):
        """concept_profile returns error if sensory_concept synapse missing."""
        brain = _mock_brain()
        brain.synapses = {}
        se = SynapseExplainer(brain)
        result = se.concept_profile(0)
        assert result == {"error": "no synapse"}

    def test_unlabeled_cluster(self):
        """Cluster without label returns None for label field."""
        brain = _mock_brain()
        _add_cluster(brain, 7, [10, 20, 30], label=None, count=5)
        se = SynapseExplainer(brain)
        result = se.concept_profile(7)
        assert result["label"] is None


# ── _dim_to_sensor ───────────────────────────────────────────────────

class TestDimToSensor:
    def test_all_ranges(self):
        """Each dimension range maps to the correct category name."""
        brain = _mock_brain()
        se = SynapseExplainer(brain)

        cases = [
            (0, "App-ID"),
            (39, "App-ID"),
            (40, "Background-App"),
            (59, "Background-App"),
            (60, "Tastatur-Bin"),
            (75, "Tastatur-Bin"),
            (76, "Tipp-Rhythmus"),
            (79, "Tipp-Rhythmus"),
            (80, "Maus-Bin"),
            (95, "Maus-Bin"),
            (96, "Maus-Rhythmus"),
            (99, "Maus-Rhythmus"),
            (100, "Idle-Bin"),
            (107, "Idle-Bin"),
            (108, "Pausen-Typ"),
            (111, "Pausen-Typ"),
            (112, "Mic-Mel"),
            (143, "Mic-Mel"),
            (144, "Mic-RMS"),
            (147, "Mic-RMS"),
            (148, "Tageszeit"),
            (155, "Tageszeit"),
            (156, "App-Switch"),
            (159, "App-Switch"),
            (160, "Aktivitaets-Level"),
            (163, "Aktivitaets-Level"),
            (164, "Reserve"),
            (199, "Reserve"),
        ]
        for dim, expected_prefix in cases:
            result = se._dim_to_sensor([dim])
            assert len(result) == 1, f"dim {dim} returned {len(result)} items"
            assert result[0].startswith(expected_prefix), (
                f"dim {dim}: expected '{expected_prefix}', got '{result[0]}'"
            )

    def test_empty_list(self):
        """Empty input returns empty list."""
        brain = _mock_brain()
        se = SynapseExplainer(brain)
        assert se._dim_to_sensor([]) == []

    def test_includes_dim_number(self):
        """Each result includes the dimension number in parentheses."""
        brain = _mock_brain()
        se = SynapseExplainer(brain)
        result = se._dim_to_sensor([65])
        assert "dim 65" in result[0]


# ── explain ──────────────────────────────────────────────────────────

class TestExplain:
    def test_returns_modulator_causes(self):
        """explain() returns dict with modulator_causes key."""
        brain = _mock_brain(da=0.10, ne=0.15)
        se = SynapseExplainer(brain)
        result = se.explain()
        assert "modulator_causes" in result
        assert "NE" in result["modulator_causes"]
        assert "DA" in result["modulator_causes"]

    def test_explain_structure(self):
        """explain() result has exactly the expected keys."""
        brain = _mock_brain()
        se = SynapseExplainer(brain)
        result = se.explain()
        assert set(result.keys()) == {"modulator_causes"}
