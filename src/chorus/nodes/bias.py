from typing import Literal
from pydantic import BaseModel
from chorus.state import DecisionState
from chorus.utils import AGENT_NAMES, get_model

AgentName = Literal[
    "Arbiter", "Empath", "Soothsayer", "Compass", "Narrator", "Conscience", "Guardian"
]

KNOWN_BIASES = [
    ("Sunk Cost Fallacy",    "已投入大量时间/金钱/精力，难以放弃，即使继续代价更高"),
    ("Recency Effect",       "近期发生的事件主导了整体判断，忽视了更长时间跨度的信息"),
    ("Bandwagon Effect",     "以他人的选择或意见作为自己决策的主要依据"),
    ("Black-and-White Thinking", "将复杂情况简化为非此即彼，忽视中间路径"),
    ("Confirmation Bias",    "叙述中正面/负面信息严重失衡，倾向于寻找支持已有立场的证据"),
    ("Loss Aversion",        "对可能的损失的恐惧远超对等值收益的期待，导致过度保守"),
    ("Status Quo Bias",      "倾向于维持现状，将改变本身视为风险而非机会"),
    ("Anchoring Effect",     "叙述被某个具体数字、时间点或初始条件过度锚定"),
    ("Overconfidence Bias",  "对自身能力或结果的乐观程度超出实际依据"),
    ("Emotional Reasoning",  "以当下情绪状态作为判断事实的依据（感觉不对 = 事情不对）"),
    ("Planning Fallacy",     "低估所需时间、资源或困难，计划过于乐观"),
    ("Catastrophizing",      "将负面结果想象到极端最坏情形，忽视更可能的中间结果"),
    ("Should Statements",    "用强烈的道德义务感（必须/应该）替代对自身真实意愿的探索"),
]

_BIAS_CATALOGUE = "\n".join(f"- {name}：{desc}" for name, desc in KNOWN_BIASES)
_AGENT_LIST = "、".join(AGENT_NAMES)

_SYSTEM_PROMPT = f"""你是一位认知偏误识别专家。你的唯一职责是识别用户决策叙述中存在的认知偏误，不做任何价值判断，不给出建议。

已知认知偏误参考列表（不限于此）：
{_BIAS_CATALOGUE}

对每个检测到的偏误，输出：
- bias：偏误名称
- target_agents：最需要警惕此偏误的心理维度，从以下选择（可多选）：{_AGENT_LIST}
- note：针对该叙述的具体说明，不超过 40 字，只描述现象不裁判对错

规则：
- 仅报告有明确证据的偏误，不要过度推断
- 同一偏误只报告一次
- 若未检测到任何偏误，返回空列表"""


class BiasFlag(BaseModel):
    bias: str
    target_agents: list[AgentName]
    note: str


class BiasDetectionResult(BaseModel):
    flags: list[BiasFlag]


def bias_detection_node(state: DecisionState) -> dict:
    narrative = state["user_narrative"]

    llm = get_model(temperature=0.0).with_structured_output(BiasDetectionResult)
    result: BiasDetectionResult = llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": narrative},
    ])

    return {"bias_flags": [f.model_dump() for f in result.flags]}
