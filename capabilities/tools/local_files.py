"""Local file read/write/list — sandboxed to a base directory."""
from __future__ import annotations
from pathlib import Path
from typing import Any

_DEFAULT_BASE = str(Path.home() / "Notes")


def _safe_path(path: str, base_dir: str) -> Path | None:
    base = Path(base_dir).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        return None
    return target


def read_file(path: str, base_dir: str = _DEFAULT_BASE) -> dict[str, Any]:
    target = _safe_path(path, base_dir)
    if target is None:
        return {"error": "Path traversal blocked"}
    if not target.exists():
        return {"error": f"File not found: {path}"}
    return {"content": target.read_text(encoding="utf-8"), "path": str(path)}


def write_file(path: str, content: str, base_dir: str = _DEFAULT_BASE) -> dict[str, Any]:
    target = _safe_path(path, base_dir)
    if target is None:
        return {"error": "Path traversal blocked"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"status": "ok", "path": str(path), "bytes": len(content.encode())}


def list_files(base_dir: str = _DEFAULT_BASE, pattern: str = "*") -> dict[str, Any]:
    base = Path(base_dir)
    if not base.exists():
        return {"files": [], "base_dir": str(base)}
    files = [str(f.relative_to(base)) for f in sorted(base.rglob(pattern)) if f.is_file()]
    return {"files": files[:100], "base_dir": str(base)}
