"""Wish detector — matches stable concept labels to catalog trigger keywords.

Runs periodically (every 5 min in production). For each labeled concept with
high activity, checks if any catalog entry's trigger keywords match the label.
If so, generates a WishCandidate.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from capabilities.catalog import CatalogEntry


@dataclass
class WishCandidate:
    tool_name: str
    concept_id: int
    concept_label: str
    match_keyword: str
    activity_count: int


def match_concepts_to_catalog(
    concepts: list[dict[str, Any]],
    catalog: dict[str, CatalogEntry],
    granted: set[str],
    cooldown: set[str],
    min_activity: int = 10,
) -> list[WishCandidate]:
    """Match labeled, active concepts against catalog trigger keywords."""
    candidates = []
    for concept in concepts:
        label = concept.get("label")
        if not label:
            continue
        activity = concept.get("times_active_24h", 0)
        if activity < min_activity:
            continue
        label_lower = label.lower()
        for tool_name, entry in catalog.items():
            if tool_name in granted or tool_name in cooldown:
                continue
            for keyword in entry.trigger_keywords:
                if keyword.lower() in label_lower or label_lower in keyword.lower():
                    candidates.append(WishCandidate(
                        tool_name=tool_name,
                        concept_id=concept.get("id", 0),
                        concept_label=label,
                        match_keyword=keyword,
                        activity_count=activity,
                    ))
                    break
    return candidates
