"""Tests for SNNNarrator — verifies German natural-language narration of SNN state.

Uses MagicMock for Brain so tests run without torch/CUDA dependencies in the
actual SNN. Each narration method is tested independently with controlled mock
data, then narrate_full_state() is tested for integration.
"""
from __future__ import annotations

import torch
from unittest.mock import MagicMock, PropertyMock

from bridge.snn_narrator import SNNNarrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain(
    *,
    modulators: dict[str, float] | None = None,
    current_cluster: int = -1,
    current_label: str | None = None,
    clusters: dict | None = None,
    debug: dict | None = None,
    wm_active: int = 0,
    weights: torch.Tensor | None = None,
    tick_count: int = 50_000,
) -> MagicMock:
    """Build a MagicMock Brain with the attributes SNNNarrator reads."""
    brain = MagicMock()

    # -- modulators --
    mods = modulators or {"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.01}
    brain.modulators.snapshot.return_value = mods

    # -- concept_tracker --
    # _clusters must be a real dict (not a MagicMock) so .get() works properly
    if clusters is None:
        clusters = {}
    brain.concept_tracker._clusters = clusters

    snap = {
        "current_cluster": current_cluster,
        "current_label": current_label,
        "debug": debug or {},
    }
    brain.concept_tracker.snapshot.return_value = snap

    # -- regions["wm"] --
    wm_mock = MagicMock()
    wm_spikes = torch.zeros(100)
    if wm_active > 0:
        wm_spikes[:wm_active] = 1.0
    wm_mock.last_spikes = wm_spikes
    brain.regions = {"wm": wm_mock}

    # -- synapses["sensory_concept"] --
    syn_mock = MagicMock()
    if weights is None:
        # Default: well-trained weights with high row-std (>0.1 for most rows)
        weights = torch.randn(200, 500) * 0.2 + 0.5
    syn_mock.weights = weights
    brain.synapses = {"sensory_concept": syn_mock}

    # -- tick_count --
    brain.tick_count = tick_count

    return brain


def _make_cluster(
    label: str | None = None,
    count: int = 1,
    protected: bool = False,
) -> MagicMock:
    """Create a mock _Cluster object with the fields SNNNarrator reads."""
    cluster = MagicMock()
    cluster.label = label
    cluster.count = count
    cluster.protected = protected
    cluster.centroid = torch.zeros(200)
    return cluster


# ===========================================================================
# _narrate_modulators
# ===========================================================================

class TestNarrateModulators:
    """Modulator narration: German sentences, no raw numbers."""

    def test_high_da_curiosity(self):
        brain = _make_brain(modulators={"DA": 0.06, "NE": 0.01, "ACh": 0.01, "5HT": 0.005})
        narrator = SNNNarrator(brain)
        result = narrator._narrate_modulators()
        assert "Neugier" in result
        assert "Neues" in result
        # Must NOT contain raw numbers
        assert "DA=" not in result
        assert "0.06" not in result

    def test_medium_da_interest(self):
        brain = _make_brain(modulators={"DA": 0.03, "NE": 0.01, "ACh": 0.01, "5HT": 0.02})
        narrator = SNNNarrator(brain)
        result = narrator._narrate_modulators()
        assert "interessiert" in result

    def test_low_da_nothing_new(self):
        brain = _make_brain(modulators={"DA": 0.002, "NE": 0.01, "ACh": 0.01, "5HT": 0.02})
        narrator = SNNNarrator(brain)
        result = narrator._narrate_modulators()
        assert "Nichts Neues" in result

    def test_high_ne_startled(self):
        brain = _make_brain(modulators={"DA": 0.01, "NE": 0.15, "ACh": 0.01, "5HT": 0.02})
        narrator = SNNNarrator(brain)
        result = narrator._narrate_modulators()
        assert "aufgeschreckt" in result

    def test_medium_ne_attentive(self):
        brain = _make_brain(modulators={"DA": 0.01, "NE": 0.07, "ACh": 0.01, "5HT": 0.02})
        narrator = SNNNarrator(brain)
        result = narrator._narrate_modulators()
        assert "Aufmerksam" in result

    def test_high_sht_content(self):
        brain = _make_brain(modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.06})
        narrator = SNNNarrator(brain)
        result = narrator._narrate_modulators()
        assert "wohl" in result

    def test_low_sht_uneasy(self):
        brain = _make_brain(modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.005})
        narrator = SNNNarrator(brain)
        result = narrator._narrate_modulators()
        assert "unruhig" in result

    def test_high_ach_focused(self):
        brain = _make_brain(modulators={"DA": 0.01, "NE": 0.01, "ACh": 0.08, "5HT": 0.02})
        narrator = SNNNarrator(brain)
        result = narrator._narrate_modulators()
        assert "Fokus" in result

    def test_balanced_modulators_fallback(self):
        """When all modulators are in mid-range, get the balanced message."""
        brain = _make_brain(modulators={"DA": 0.01, "NE": 0.02, "ACh": 0.02, "5HT": 0.02})
        narrator = SNNNarrator(brain)
        result = narrator._narrate_modulators()
        assert "ausgeglichen" in result

    def test_multiple_modulators_combined(self):
        """High DA + high NE produces a sentence covering both."""
        brain = _make_brain(modulators={"DA": 0.06, "NE": 0.12, "ACh": 0.01, "5HT": 0.005})
        narrator = SNNNarrator(brain)
        result = narrator._narrate_modulators()
        assert "Neugier" in result
        assert "aufgeschreckt" in result
        assert result.startswith("Emotionen:")


# ===========================================================================
# _narrate_current_concept
# ===========================================================================

class TestNarrateCurrentConcept:
    """Concept narration: label, confidence, count in natural language."""

    def test_no_current_cluster(self):
        brain = _make_brain(current_cluster=-1)
        narrator = SNNNarrator(brain)
        result = narrator._narrate_current_concept()
        assert "kein klares Muster" in result

    def test_labeled_high_confidence(self):
        cluster = _make_cluster(label="Coding", count=45)
        brain = _make_brain(
            current_cluster=3,
            current_label="Coding",
            clusters={3: cluster},
            debug={"best_sim": 0.85},
        )
        narrator = SNNNarrator(brain)
        result = narrator._narrate_current_concept()
        assert "Coding" in result
        assert "sehr sicher" in result
        assert "85%" in result
        assert "45x gesehen" in result

    def test_labeled_low_confidence(self):
        cluster = _make_cluster(label="Browsing", count=20)
        brain = _make_brain(
            current_cluster=1,
            current_label="Browsing",
            clusters={1: cluster},
            debug={"best_sim": 0.45},
        )
        narrator = SNNNarrator(brain)
        result = narrator._narrate_current_concept()
        assert "Browsing" in result
        assert "nicht ganz sicher" in result

    def test_unlabeled_frequent(self):
        cluster = _make_cluster(label=None, count=80)
        brain = _make_brain(
            current_cluster=2,
            current_label=None,
            clusters={2: cluster},
            debug={"best_sim": 0.7},
        )
        narrator = SNNNarrator(brain)
        result = narrator._narrate_current_concept()
        assert "ohne Namen" in result
        assert "80x gesehen" in result
        assert "Leon fragen" in result

    def test_unlabeled_medium_count(self):
        cluster = _make_cluster(label=None, count=25)
        brain = _make_brain(
            current_cluster=4,
            current_label=None,
            clusters={4: cluster},
            debug={"best_sim": 0.5},
        )
        narrator = SNNNarrator(brain)
        result = narrator._narrate_current_concept()
        assert "25x gesehen" in result
        assert "noch nicht kenne" in result

    def test_unlabeled_new_pattern(self):
        cluster = _make_cluster(label=None, count=3)
        brain = _make_brain(
            current_cluster=5,
            current_label=None,
            clusters={5: cluster},
            debug={"best_sim": 0.4},
        )
        narrator = SNNNarrator(brain)
        result = narrator._narrate_current_concept()
        assert "neues Muster" in result

    def test_cluster_missing_from_dict(self):
        """current_cluster points to an ID not in _clusters dict."""
        brain = _make_brain(
            current_cluster=99,
            current_label=None,
            clusters={},  # empty dict — cluster 99 does not exist
            debug={},
        )
        narrator = SNNNarrator(brain)
        result = narrator._narrate_current_concept()
        assert "unbekannt" in result

    def test_clusters_is_dict_not_list(self):
        """Verify _clusters is accessed as a dict with .get(), not as a list."""
        cluster = _make_cluster(label="Meeting", count=10)
        # Use non-sequential IDs to prove dict-based access
        clusters = {7: cluster, 42: _make_cluster(count=5)}
        brain = _make_brain(
            current_cluster=7,
            current_label="Meeting",
            clusters=clusters,
            debug={"best_sim": 0.75},
        )
        narrator = SNNNarrator(brain)
        result = narrator._narrate_current_concept()
        assert "Meeting" in result
        assert "sehr sicher" in result


# ===========================================================================
# _narrate_wm
# ===========================================================================

class TestNarrateWM:
    """Working memory narration: active slot count in natural language."""

    def test_wm_empty(self):
        brain = _make_brain(wm_active=0)
        narrator = SNNNarrator(brain)
        result = narrator._narrate_wm()
        assert "leer" in result

    def test_wm_few_slots(self):
        brain = _make_brain(wm_active=3)
        narrator = SNNNarrator(brain)
        result = narrator._narrate_wm()
        assert "3 Slots" in result
        assert "aufgeraeumt" in result

    def test_wm_medium_slots(self):
        brain = _make_brain(wm_active=8)
        narrator = SNNNarrator(brain)
        result = narrator._narrate_wm()
        assert "8 Slots" in result
        assert "Erinnerungen aktiv" in result

    def test_wm_many_slots(self):
        brain = _make_brain(wm_active=20)
        narrator = SNNNarrator(brain)
        result = narrator._narrate_wm()
        assert "20" in result
        assert "viel" in result

    def test_wm_missing(self):
        """If 'wm' region does not exist, return empty string."""
        brain = _make_brain(wm_active=5)
        brain.regions = {}  # no wm region
        narrator = SNNNarrator(brain)
        result = narrator._narrate_wm()
        assert result == ""


# ===========================================================================
# _narrate_learning
# ===========================================================================

class TestNarrateLearning:
    """Learning narration: specialization percentage in natural language."""

    def test_well_trained(self):
        """>80% specialized neurons."""
        # Create weights where most rows have std > 0.1
        weights = torch.randn(200, 500) * 0.3 + 0.5
        brain = _make_brain(weights=weights, tick_count=100_000)
        narrator = SNNNarrator(brain)
        result = narrator._narrate_learning()
        row_stds = weights.std(dim=1)
        specialized = int((row_stds > 0.1).sum().item())
        pct = specialized / 200 * 100
        if pct > 80:
            assert "gut trainiert" in result
            assert f"{specialized}/200" in result

    def test_still_learning(self):
        """30-80% specialized."""
        # Mix: half rows with high std, half with low std
        weights = torch.zeros(200, 500)
        weights[:100] = torch.randn(100, 500) * 0.3 + 0.5  # high std
        weights[100:] = torch.ones(100, 500) * 0.5 + torch.randn(100, 500) * 0.02  # low std
        brain = _make_brain(weights=weights, tick_count=100_000)
        narrator = SNNNarrator(brain)
        result = narrator._narrate_learning()
        row_stds = weights.std(dim=1)
        specialized = int((row_stds > 0.1).sum().item())
        pct = specialized / 200 * 100
        if 30 < pct <= 80:
            assert "lerne noch" in result
            assert f"{specialized}/200" in result

    def test_fresh_brain(self):
        """Low specialization + low tick count."""
        weights = torch.ones(200, 500) * 0.5 + torch.randn(200, 500) * 0.01
        brain = _make_brain(weights=weights, tick_count=5000)
        narrator = SNNNarrator(brain)
        result = narrator._narrate_learning()
        assert "ganz frisch" in result

    def test_low_specialization_experienced(self):
        """Low specialization but high tick count."""
        weights = torch.ones(200, 500) * 0.5 + torch.randn(200, 500) * 0.01
        brain = _make_brain(weights=weights, tick_count=50_000)
        narrator = SNNNarrator(brain)
        result = narrator._narrate_learning()
        assert "nicht sehr spezialisiert" in result or "mehr Erfahrung" in result

    def test_synapse_missing(self):
        """If sensory_concept synapse does not exist, return empty string."""
        brain = _make_brain()
        brain.synapses = {}  # no synapses
        narrator = SNNNarrator(brain)
        result = narrator._narrate_learning()
        assert result == ""


# ===========================================================================
# narrate_full_state
# ===========================================================================

class TestNarrateFullState:
    """Integration: all four parts combined into one narrative."""

    def test_combines_all_parts(self):
        cluster = _make_cluster(label="Coding", count=45)
        brain = _make_brain(
            modulators={"DA": 0.06, "NE": 0.01, "ACh": 0.01, "5HT": 0.005},
            current_cluster=3,
            current_label="Coding",
            clusters={3: cluster},
            debug={"best_sim": 0.85},
            wm_active=8,
            tick_count=100_000,
        )
        narrator = SNNNarrator(brain)
        result = narrator.narrate_full_state()

        # Should contain pieces from each sub-narration
        assert "Neugier" in result       # modulators
        assert "Coding" in result        # concept
        assert "Erinnerungen" in result  # wm
        # learning part is present (depends on random weights)
        lines = result.strip().split("\n")
        assert len(lines) >= 3  # at least modulator + concept + wm

    def test_empty_parts_filtered(self):
        """Parts that return '' are excluded from the output."""
        brain = _make_brain(
            modulators={"DA": 0.01, "NE": 0.02, "ACh": 0.02, "5HT": 0.02},
            wm_active=5,
        )
        # Remove synapse to make _narrate_learning() return ""
        brain.synapses = {}
        # Remove wm to make _narrate_wm() return ""
        brain.regions = {}
        narrator = SNNNarrator(brain)
        result = narrator.narrate_full_state()
        # Should still have modulators and concept, no blank lines
        assert "\n\n" not in result

    def test_no_raw_numbers_in_output(self):
        """Full narration must not contain raw modulator assignments."""
        cluster = _make_cluster(label="Browsing", count=30)
        brain = _make_brain(
            modulators={"DA": 0.03, "NE": 0.07, "ACh": 0.06, "5HT": 0.05},
            current_cluster=1,
            current_label="Browsing",
            clusters={1: cluster},
            debug={"best_sim": 0.7},
            wm_active=10,
        )
        narrator = SNNNarrator(brain)
        result = narrator.narrate_full_state()
        assert "DA=" not in result
        assert "NE=" not in result
        assert "ACh=" not in result
        assert "5HT=" not in result
