"""Web search via DuckDuckGo HTML API."""
from __future__ import annotations
from typing import Any
import httpx


async def _fetch_ddg(query: str) -> list[dict[str, str]]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
        )
        data = resp.json()
    results = []
    if data.get("Abstract"):
        results.append({"title": data.get("Heading", ""), "url": data.get("AbstractURL", ""), "snippet": data["Abstract"]})
    for topic in data.get("RelatedTopics", [])[:5]:
        if "Text" in topic:
            results.append({"title": topic.get("Text", "")[:80], "url": topic.get("FirstURL", ""), "snippet": topic.get("Text", "")})
    return results[:5]


async def web_search(query: str) -> list[dict[str, str]]:
    return await _fetch_ddg(query)
