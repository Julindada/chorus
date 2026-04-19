from langchain_community.chat_models.tongyi import ChatTongyi
from chorus.infrastructure.config import LLM_MODEL

_DEFAULT_TEMPERATURE = 0.7


def get_model(temperature: float = _DEFAULT_TEMPERATURE) -> ChatTongyi:
    return ChatTongyi(model=LLM_MODEL, temperature=temperature)
