import json
import sqlite3
from chorus.infrastructure.config import DB_PATH

_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS user_profile "
    "(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, "
    "value_vector TEXT, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
)


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_value_vector(username: str) -> dict[str, float] | None:
    """Load Schwartz value vector for a user. Returns None if not set."""
    conn = _get_conn()
    try:
        conn.execute(_CREATE_TABLE)
        row = conn.execute(
            "SELECT value_vector FROM user_profile WHERE username = ?", (username,)
        ).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()


def save_value_vector(username: str, value_vector: dict[str, float]) -> None:
    conn = _get_conn()
    try:
        conn.execute(_CREATE_TABLE)
        conn.execute(
            "INSERT INTO user_profile (username, value_vector) VALUES (?, ?) "
            "ON CONFLICT(username) DO UPDATE SET value_vector = excluded.value_vector, "
            "updated_at = CURRENT_TIMESTAMP",
            (username, json.dumps(value_vector)),
        )
        conn.commit()
    finally:
        conn.close()
