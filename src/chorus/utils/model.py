from langchain_openai import ChatOpenAI
from chorus.infrastructure.config import LLM_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

_DEFAULT_TEMPERATURE = 0


def get_model(temperature: float = _DEFAULT_TEMPERATURE) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        temperature=temperature,
    )
