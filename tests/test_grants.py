"""Tests for capability grant persistence."""
import tempfile
from pathlib import Path
import pytest
from capabilities.grants import GrantStore


def test_grant_store_construction():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        assert store.list_grants() == {}


def test_grant_and_list():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        store.grant("web_search", concept_label="googlen", reason="User approved")
        grants = store.list_grants()
        assert "web_search" in grants
        assert grants["web_search"]["concept_label"] == "googlen"


def test_grant_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.sqlite"
        store1 = GrantStore(path)
        store1.grant("shell", concept_label="terminal")
        store2 = GrantStore(path)
        assert "shell" in store2.list_grants()


def test_is_granted():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        assert store.is_granted("web_search") is False
        store.grant("web_search")
        assert store.is_granted("web_search") is True


def test_deny_with_cooldown():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        store.deny("web_search", cooldown_days=7)
        assert store.is_in_cooldown("web_search") is True


def test_record_wish():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        wish_id = store.record_wish("web_search", concept_id=12, concept_label="googlen", reason="High activity")
        assert wish_id > 0
        wishes = store.list_wishes()
        assert len(wishes) == 1
        assert wishes[0]["tool_name"] == "web_search"
        assert wishes[0]["status"] == "pending"
