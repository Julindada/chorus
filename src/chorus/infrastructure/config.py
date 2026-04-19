import os

DB_PATH: str = os.getenv("CHORUS_DB_PATH", "data/chorus.db")
LLM_MODEL: str = os.getenv("CHORUS_LLM_MODEL", "qwen3.6-plus")
