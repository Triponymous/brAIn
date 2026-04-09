"""Tests for grant API endpoints."""
import tempfile
from pathlib import Path
import pytest
from httpx import AsyncClient, ASGITransport
from capabilities.grants import GrantStore
from server.grants import build_grants_router


@pytest.mark.asyncio
async def test_list_wishes_empty():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        router = build_grants_router(store, refresh_fn=lambda: None)
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/wishes")
        assert resp.status_code == 200
        assert resp.json()["wishes"] == []


@pytest.mark.asyncio
async def test_grant_wish():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        wish_id = store.record_wish("web_search", concept_id=12, concept_label="googlen")
        refreshed = []
        router = build_grants_router(store, refresh_fn=lambda: refreshed.append(True))
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/wishes/grant", json={"wish_id": wish_id, "tool_name": "web_search"})
        assert resp.status_code == 200
        assert store.is_granted("web_search")
        assert len(refreshed) == 1


@pytest.mark.asyncio
async def test_list_grants():
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "test.sqlite")
        store.grant("shell")
        router = build_grants_router(store, refresh_fn=lambda: None)
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/grants")
        assert resp.status_code == 200
        assert "shell" in resp.json()["grants"]
