import json
import sqlite3
from chorus.utils.constants import SCHWARTZ_DIMS

DB_PATH = "chorus.db"


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_value_vector() -> dict[str, float] | None:
    """Load Schwartz value vector from SQLite. Returns None if not set."""
    conn = _get_conn()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_profile "
            "(id INTEGER PRIMARY KEY, value_vector TEXT, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        row = conn.execute(
            "SELECT value_vector FROM user_profile ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    finally:
        conn.close()
    return None


def save_value_vector(value_vector: dict[str, float]) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_profile "
            "(id INTEGER PRIMARY KEY, value_vector TEXT, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO user_profile (value_vector) VALUES (?)",
            (json.dumps(value_vector),),
        )
        conn.commit()
    finally:
        conn.close()
