import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR: str = os.getenv("CHORUS_DATA_DIR", "data")
DB_PATH: str  = os.getenv("CHORUS_DB_PATH", f"{DATA_DIR}/chorus.db")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.6-plus")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
DASHSCOPE_BASE_URL: str = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
