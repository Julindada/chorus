import sqlite3
from chorus.infrastructure.config import DB_PATH


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def insert_decision_records(records: list[dict]) -> None:
    """Write per-agent alignment records to decision_history after each decision."""
    if not records:
        return
    conn = _get_conn()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS decision_history "
            "(id INTEGER PRIMARY KEY, decision_type TEXT, agent_name TEXT, "
            "initial_stance REAL, final_alignment REAL, timestamp DATETIME)"
        )
        conn.executemany(
            "INSERT INTO decision_history "
            "(decision_type, agent_name, initial_stance, final_alignment, timestamp) "
            "VALUES (:decision_type, :agent_name, :initial_stance, :final_alignment, :timestamp)",
            records,
        )
        conn.commit()
    finally:
        conn.close()
