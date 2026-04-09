"""Tests for the wish detector — matches concept labels to catalog entries."""
import pytest
from capabilities.wish_detector import match_concepts_to_catalog, WishCandidate
from capabilities.catalog import get_catalog


def test_match_finds_web_search_for_googlen():
    catalog = get_catalog()
    concepts = [{"id": 12, "label": "googlen", "activation": 0.8, "times_active_24h": 15}]
    matches = match_concepts_to_catalog(concepts, catalog, granted=set(), cooldown=set())
    assert len(matches) == 1
    assert matches[0].tool_name == "web_search"
    assert matches[0].concept_label == "googlen"


def test_match_finds_shell_for_terminal():
    catalog = get_catalog()
    concepts = [{"id": 5, "label": "terminal-arbeit", "activation": 0.5, "times_active_24h": 20}]
    matches = match_concepts_to_catalog(concepts, catalog, granted=set(), cooldown=set())
    assert any(m.tool_name == "shell" for m in matches)


def test_match_skips_already_granted():
    catalog = get_catalog()
    concepts = [{"id": 12, "label": "googlen", "activation": 0.8, "times_active_24h": 15}]
    matches = match_concepts_to_catalog(concepts, catalog, granted={"web_search"}, cooldown=set())
    assert len(matches) == 0


def test_match_skips_cooldown():
    catalog = get_catalog()
    concepts = [{"id": 12, "label": "googlen", "activation": 0.8, "times_active_24h": 15}]
    matches = match_concepts_to_catalog(concepts, catalog, granted=set(), cooldown={"web_search"})
    assert len(matches) == 0


def test_match_skips_low_activity():
    catalog = get_catalog()
    concepts = [{"id": 12, "label": "googlen", "activation": 0.8, "times_active_24h": 3}]
    matches = match_concepts_to_catalog(concepts, catalog, granted=set(), cooldown=set())
    assert len(matches) == 0


def test_match_skips_unlabeled():
    catalog = get_catalog()
    concepts = [{"id": 12, "label": None, "activation": 0.8, "times_active_24h": 50}]
    matches = match_concepts_to_catalog(concepts, catalog, granted=set(), cooldown=set())
    assert len(matches) == 0
