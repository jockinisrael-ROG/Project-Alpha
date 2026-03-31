import sqlite3
from pathlib import Path
from typing import Dict, List


def init_memory_db(db_path: str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(id)"
        )
        conn.commit()


def save_message(db_path: str, role: str, content: str) -> None:
    text = content.strip()
    if role not in {"user", "assistant"} or not text:
        return

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (role, text),
        )
        conn.commit()


def load_recent_messages(db_path: str, limit: int) -> List[Dict[str, str]]:
    if limit <= 0:
        return []

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]
