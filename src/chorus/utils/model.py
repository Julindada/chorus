from langchain_anthropic import ChatAnthropic

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_TEMPERATURE = 1.0


def get_model(temperature: float = _DEFAULT_TEMPERATURE) -> ChatAnthropic:
    return ChatAnthropic(model=_DEFAULT_MODEL, temperature=temperature)
