"""Tests for capability tools — web search, shell, local files."""
import asyncio
import tempfile
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, patch
from capabilities.catalog import get_catalog, CatalogEntry
from capabilities.tools.web_search import web_search
from capabilities.tools.shell import safe_shell
from capabilities.tools.local_files import read_file, write_file, list_files


def test_catalog_has_three_entries():
    catalog = get_catalog()
    assert "web_search" in catalog
    assert "shell" in catalog
    assert "local_files" in catalog
    for name, entry in catalog.items():
        assert isinstance(entry, CatalogEntry)
        assert entry.trigger_keywords
        assert entry.tool_schema


@pytest.mark.asyncio
async def test_web_search_mock():
    with patch("capabilities.tools.web_search._fetch_ddg", new_callable=AsyncMock,
               return_value=[{"title": "Test", "url": "https://test.com", "snippet": "A test result."}]):
        result = await web_search(query="test query")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0]["title"] == "Test"


@pytest.mark.asyncio
async def test_shell_allowed_command():
    result = await safe_shell(command="date")
    assert "stdout" in result
    assert result["returncode"] == 0


@pytest.mark.asyncio
async def test_shell_blocked_command():
    result = await safe_shell(command="rm -rf /")
    assert "error" in result
    assert "blocked" in result["error"].lower() or "not" in result["error"].lower()


@pytest.mark.asyncio
async def test_shell_blocked_sudo():
    result = await safe_shell(command="sudo ls")
    assert "error" in result


def test_local_files_write_and_read():
    with tempfile.TemporaryDirectory() as tmp:
        result = write_file(path="test.md", content="# Hello", base_dir=tmp)
        assert result["status"] == "ok"
        content = read_file(path="test.md", base_dir=tmp)
        assert content["content"] == "# Hello"


def test_local_files_list():
    with tempfile.TemporaryDirectory() as tmp:
        write_file(path="a.md", content="A", base_dir=tmp)
        write_file(path="b.txt", content="B", base_dir=tmp)
        files = list_files(base_dir=tmp)
        assert len(files["files"]) == 2


def test_local_files_path_traversal_blocked():
    with tempfile.TemporaryDirectory() as tmp:
        result = read_file(path="../../../etc/passwd", base_dir=tmp)
        assert "error" in result
