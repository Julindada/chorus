import json

from pydantic import BaseModel, Field

from chorus.state import DecisionState
from chorus.utils import AGENT_PROMPTS, get_model

_INSTRUCTION = """
根据用户的决策叙述和选项，从你的视角给出评估。

输入包含：
- user_narrative：用户的决策叙述
- decision_options：待决策的选项列表
- value_vector：用户的 Schwartz 10 维价值观权重（0.0–1.0）
- bias_flags：已检测到的认知偏误（仅供参考，不要对偏误本身发表评判）

输出要求：
- stance：你的倾向分（-1.0 = 强烈不建议，0.0 = 中立，+1.0 = 强烈建议）
- reasoning：立足于你的心理维度的评估依据，不超过 150 字
- confidence：你对此判断的信心分（0.0–1.0）"""


class _StanceResult(BaseModel):
    stance: float = Field(ge=-1.0, le=1.0)
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


def agent_node(state: DecisionState) -> dict:
    agent_name = state["agent_name"]
    try:
        result: _StanceResult = (
            get_model()
            .with_structured_output(_StanceResult)
            .invoke(_build_messages(agent_name, state))
        )
        return {"agent_stances": {agent_name: result.model_dump()}}
    except Exception:
        if agent_name in state["critical_agents"]:
            raise
        return {}


def _build_messages(agent_name: str, state: DecisionState) -> list[dict]:
    return [
        {"role": "system", "content": AGENT_PROMPTS[agent_name] + _INSTRUCTION},
        {"role": "user", "content": json.dumps({
            "user_narrative":   state["user_narrative"],
            "decision_options": state["decision_options"],
            "value_vector":     state["value_vector"],
            "bias_flags":       state["bias_flags"],
        }, ensure_ascii=False)},
    ]
