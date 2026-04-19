import json
import sqlite3
from typing import get_args
from chorus.infrastructure.config import DB_PATH
from chorus.utils.constants import DecisionType, DECISION_TYPES

assert set(DECISION_TYPES) == set(get_args(DecisionType)), (
    f"DECISION_TYPES keys {set(DECISION_TYPES)} must match DecisionType {set(get_args(DecisionType))}"
)


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scene_templates (
            decision_type TEXT PRIMARY KEY,
            weights       TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _ensure_seeds(conn: sqlite3.Connection) -> None:
    for dtype, meta in DECISION_TYPES.items():
        conn.execute(
            "INSERT OR IGNORE INTO scene_templates (decision_type, weights) VALUES (?, ?)",
            (dtype, json.dumps(meta["weights"])),
        )
    conn.commit()


def find_scene_template(decision_type: str) -> dict[str, float] | None:
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_table(conn)
        _ensure_seeds(conn)
        row = conn.execute(
            "SELECT weights FROM scene_templates WHERE decision_type = ?",
            (decision_type,),
        ).fetchone()
    return json.loads(row[0]) if row else None


def save_scene_template(decision_type: str, weights: dict[str, float]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_table(conn)
        conn.execute(
            """
            INSERT INTO scene_templates (decision_type, weights)
            VALUES (?, ?)
            ON CONFLICT(decision_type) DO UPDATE SET weights = excluded.weights
            """,
            (decision_type, json.dumps(weights)),
        )
        conn.commit()
