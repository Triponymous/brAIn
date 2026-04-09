"""Grant persistence — SQLite store for capability wishes and grants."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS capability_wishes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    trigger_concept_id INTEGER,
    trigger_concept_label TEXT,
    reason TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS capability_grants (
    tool_name TEXT PRIMARY KEY,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    concept_label TEXT,
    reason TEXT,
    params JSON
);
CREATE TABLE IF NOT EXISTS capability_denials (
    tool_name TEXT PRIMARY KEY,
    denied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cooldown_until TIMESTAMP
);
"""


class GrantStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def grant(self, tool_name: str, concept_label: str = "", reason: str = "", params: dict | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO capability_grants(tool_name, granted_at, concept_label, reason, params) VALUES (?, ?, ?, ?, ?)",
            (tool_name, datetime.now().isoformat(), concept_label, reason, json.dumps(params or {})),
        )
        self._conn.execute("DELETE FROM capability_denials WHERE tool_name = ?", (tool_name,))
        self._conn.commit()

    def deny(self, tool_name: str, cooldown_days: int = 7) -> None:
        until = (datetime.now() + timedelta(days=cooldown_days)).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO capability_denials(tool_name, denied_at, cooldown_until) VALUES (?, ?, ?)",
            (tool_name, datetime.now().isoformat(), until),
        )
        self._conn.commit()

    def is_granted(self, tool_name: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM capability_grants WHERE tool_name = ?", (tool_name,)).fetchone()
        return row is not None

    def is_in_cooldown(self, tool_name: str) -> bool:
        row = self._conn.execute("SELECT cooldown_until FROM capability_denials WHERE tool_name = ?", (tool_name,)).fetchone()
        if row is None:
            return False
        return datetime.fromisoformat(row["cooldown_until"]) > datetime.now()

    def list_grants(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM capability_grants").fetchall()
        return {r["tool_name"]: dict(r) for r in rows}

    def record_wish(self, tool_name: str, concept_id: int = 0, concept_label: str = "", reason: str = "") -> int:
        cur = self._conn.execute(
            "INSERT INTO capability_wishes(tool_name, trigger_concept_id, trigger_concept_label, reason) VALUES (?, ?, ?, ?)",
            (tool_name, concept_id, concept_label, reason),
        )
        self._conn.commit()
        return cur.lastrowid

    def list_wishes(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute("SELECT * FROM capability_wishes WHERE status = ?", (status,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM capability_wishes").fetchall()
        return [dict(r) for r in rows]

    def resolve_wish(self, wish_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE capability_wishes SET status = ?, resolved_at = ? WHERE id = ?",
            (status, datetime.now().isoformat(), wish_id),
        )
        self._conn.commit()
