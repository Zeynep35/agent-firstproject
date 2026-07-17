import os
import sqlite3
from datetime import datetime
from logger_config import logger


MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "memory.sqlite")


def get_connection():
    return sqlite3.connect(MEMORY_DB_PATH)


def init_memory_db():
    try:
        with get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    kind TEXT DEFAULT 'note',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

        logger.info("Memory DB hazır: %s", MEMORY_DB_PATH)

    except Exception:
        logger.exception("Memory DB oluşturulurken hata oluştu.")


def add_memory(user_id: str, content: str, kind: str = "note"):
    content = (content or "").strip()

    if not content:
        return None

    created_at = datetime.utcnow().isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO user_memories (user_id, content, kind, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, content, kind, created_at)
        )
        conn.commit()

        memory_id = cursor.lastrowid

    return {
        "id": memory_id,
        "user_id": user_id,
        "content": content,
        "kind": kind,
        "created_at": created_at
    }


def list_memories(user_id: str, limit: int = 20):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, content, kind, created_at
            FROM user_memories
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()

    return [
        {
            "id": row[0],
            "user_id": row[1],
            "content": row[2],
            "kind": row[3],
            "created_at": row[4]
        }
        for row in rows
    ]


def delete_memory(user_id: str, memory_id: int):
    with get_connection() as conn:
        cursor = conn.execute(
            """
            DELETE FROM user_memories
            WHERE user_id = ? AND id = ?
            """,
            (user_id, memory_id)
        )
        conn.commit()

    return cursor.rowcount


def clear_memories(user_id: str):
    with get_connection() as conn:
        cursor = conn.execute(
            """
            DELETE FROM user_memories
            WHERE user_id = ?
            """,
            (user_id,)
        )
        conn.commit()

    return cursor.rowcount


def build_memory_context(user_id: str, limit: int = 10):
    memories = list_memories(user_id=user_id, limit=limit)

    if not memories:
        return "Kayıtlı kullanıcı hafızası yok."

    lines = []

    for memory in reversed(memories):
        lines.append(f"- {memory['content']}")

    return "\n".join(lines)