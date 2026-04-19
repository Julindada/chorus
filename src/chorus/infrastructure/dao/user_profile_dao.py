import json
import sqlite3

DB_PATH = "data/chorus.db"

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
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    finally:
        conn.close()
    return None


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
