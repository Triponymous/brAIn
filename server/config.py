"""Config management — reads config.json, provides API to read/update at runtime."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter


_CONFIG_PATH = Path(__file__).parent.parent / "config.json"
_config: dict[str, Any] = {}


def load_config(path: Path | None = None) -> dict[str, Any]:
    global _config
    p = path or _CONFIG_PATH
    if p.exists():
        _config = json.loads(p.read_text())
    else:
        _config = {}
    return _config


def get_config() -> dict[str, Any]:
    if not _config:
        load_config()
    return _config


def get(section: str, key: str, default: Any = None) -> Any:
    cfg = get_config()
    return cfg.get(section, {}).get(key, default)


def set_and_save(section: str, key: str, value: Any) -> None:
    cfg = get_config()
    if section not in cfg:
        cfg[section] = {}
    cfg[section][key] = value
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def build_config_router() -> APIRouter:
    api = APIRouter()

    @api.get("/api/config")
    async def get_all_config() -> dict[str, Any]:
        return get_config()

    @api.post("/api/config")
    async def update_config(updates: dict[str, Any]) -> dict[str, str]:
        for section, values in updates.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    set_and_save(section, key, val)
        return {"status": "ok", "message": "Config updated. Restart daemon for some changes to take effect."}

    @api.get("/api/models")
    async def list_available_models() -> dict[str, Any]:
        """List installed Ollama models."""
        import httpx
        ollama_url = get("llm", "ollama_url", "http://localhost:11434")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{ollama_url}/api/tags")
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
        except Exception:
            models = []
        return {
            "installed": models,
            "current": get("llm", "local_model", "qwen2.5:7b-instruct"),
            "cloud_model": get("llm", "cloud_model"),
            "cloud_enabled": get("llm", "cloud_enabled", False),
        }

    @api.post("/api/models/switch")
    async def switch_model(body: dict[str, str]) -> dict[str, str]:
        model = body.get("model", "")
        if model:
            set_and_save("llm", "local_model", model)
            return {"status": "ok", "model": model, "message": f"Switched to {model}. Takes effect on next chat."}
        return {"status": "error", "message": "No model specified"}

    return api
