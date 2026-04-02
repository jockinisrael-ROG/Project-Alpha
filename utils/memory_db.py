"""Message persistence layer using SQLite for conversation history."""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def init_memory_db(db_path: str) -> None:
    """Initialize the message database with schema if not exists.
    
    Args:
        db_path: Path to the SQLite database file
        
    Raises:
        RuntimeError: If database initialization fails
    """
    try:
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
        logger.info(f"Memory database initialized at {path}")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize memory database: {e}")
        raise RuntimeError(f"Database initialization failed: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error initializing memory database: {e}")
        raise RuntimeError(f"Unexpected error: {e}") from e


def save_message(db_path: str, role: str, content: str) -> None:
    """Save a message to the conversation history.
    
    Args:
        db_path: Path to the SQLite database file
        role: Message role ("user" or "assistant")
        content: Message text content
        
    Returns:
        None silently if role is invalid or content is empty
    """
    text = content.strip()
    if role not in {"user", "assistant"} or not text:
        return

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO messages (role, content) VALUES (?, ?)",
                (role, text),
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to save message: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving message: {e}")


def load_recent_messages(db_path: str, limit: int) -> List[Dict[str, str]]:
    """Load recent messages from conversation history.
    
    Args:
        db_path: Path to the SQLite database file
        limit: Maximum number of messages to load
        
    Returns:
        List of message dictionaries in chronological order
    """
    if limit <= 0:
        return []

    try:
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
    except sqlite3.Error as e:
        logger.error(f"Failed to load recent messages: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading recent messages: {e}")
        return []
