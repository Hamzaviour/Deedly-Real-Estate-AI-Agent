"""Long-term buyer memory in SQLite. Short-term memory is the agent's in-session message list."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import runtime_dir

CATEGORIES = {"preference", "constraint", "shortlist", "note"}


class LongTermMemory:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or runtime_dir() / "memory.db"
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    user_id TEXT, key TEXT, value TEXT, category TEXT, updated_at TEXT,
                    PRIMARY KEY (user_id, key))"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def save(self, user_id: str, key: str, value: str, category: str = "preference") -> None:
        if category not in CATEGORIES:
            raise ValueError(f"Category must be one of {sorted(CATEGORIES)}")
        if not key.strip() or not value.strip():
            raise ValueError("Key and value must not be empty")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO memories VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (user_id, key) DO UPDATE SET
                value = excluded.value, category = excluded.category,
                updated_at = excluded.updated_at""",
                (user_id, key.strip(), value.strip(), category, now),
            )

    def forget(self, user_id: str, key: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM memories WHERE user_id = ? AND key = ?", (user_id, key)
            )
            return cur.rowcount > 0

    def load(self, user_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value, category, updated_at FROM memories "
                "WHERE user_id = ? ORDER BY category, key",
                (user_id,),
            ).fetchall()
        return [dict(zip(("key", "value", "category", "updated_at"), r)) for r in rows]
