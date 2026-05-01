import json

from chorus.state import DecisionState, StanceResult
from chorus.utils import AGENT_PROMPTS, get_model

_INSTRUCTION = """
根据用户的决策叙述，从你的视角对每个候选选项独立评估。

输入包含：
- user_narrative：用户的决策叙述
- decision_options：待决策的选项列表
- value_vector：用户的 Schwartz 10 维价值观权重（0.0–1.0）
- bias_flags：已检测到的认知偏误（仅供参考，不要对偏误本身发表评判）

输出要求：
- option_scores：对每个候选选项的评分，key 为选项名称，value 为 -1.0（强烈不建议）到 +1.0（强烈推荐），必须包含所有选项
- reasoning：立足于你的心理维度的综合评估依据，不超过 60 字，尽量少使用心理学术语，用日常语言表达
- confidence：你对此判断的信心分（0.0–1.0）
- 请以 JSON 格式返回结果"""


async def agent_node(state: DecisionState) -> dict:
    agent_name = state["agent_name"]
    try:
        result: StanceResult = await (
            get_model()
            .with_structured_output(StanceResult)
            .ainvoke(_build_messages(agent_name, state))
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
