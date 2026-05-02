from typing import Literal
from pydantic import BaseModel, Field, model_validator
from chorus.state import DecisionState
from chorus.utils import AGENT_NAMES, get_model

AgentName = Literal[
    "Arbiter", "Empath", "Soothsayer", "Compass", "Narrator", "Conscience", "Guardian"
]

KNOWN_BIASES = [
    ("沉没成本谬误", "已投入大量时间/金钱/精力，难以放弃，即使继续代价更高"),
    ("近因效应",     "近期发生的事件主导了整体判断，忽视了更长时间跨度的信息"),
    ("从众效应",     "以他人的选择或意见作为自己决策的主要依据"),
    ("非黑即白思维", "将复杂情况简化为非此即彼，忽视中间路径"),
    ("确认偏误",     "叙述中正面/负面信息严重失衡，倾向于寻找支持已有立场的证据"),
    ("损失厌恶",     "对可能的损失的恐惧远超对等值收益的期待，导致过度保守"),
    ("现状偏误",     "倾向于维持现状，将改变本身视为风险而非机会"),
    ("锚定效应",     "叙述被某个具体数字、时间点或初始条件过度锚定"),
    ("过度自信偏误", "对自身能力或结果的乐观程度超出实际依据"),
    ("情绪化推理",   "以当下情绪状态作为判断事实的依据（感觉不对 = 事情不对）"),
    ("计划谬误",     "低估所需时间、资源或困难，计划过于乐观"),
    ("灾难化思维",   "将负面结果想象到极端最坏情形，忽视更可能的中间结果"),
    ("应该陈述",     "用强烈的道德义务感（必须/应该）替代对自身真实意愿的探索"),
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
- 若未检测到任何偏误，flags 返回空列表

请以 JSON 格式返回结果，结构为：{{"flags": [...]}}"""


class BiasFlag(BaseModel):
    bias: str = Field(description="认知偏误名称")
    target_agents: list[AgentName] = Field(description="最需要警惕此偏误的心理维度 Agent 列表")
    note: str = Field(description="针对该叙述的具体说明，只描述现象不裁判对错，不超过 40 字")


class BiasDetectionResult(BaseModel):
    flags: list[BiasFlag] = Field(description="检测到的认知偏误列表，未检测到则为空列表")

    @model_validator(mode="before")
    @classmethod
    def _wrap_list(cls, v):
        if isinstance(v, list):
            return {"flags": v}
        return v


async def bias_detection_node(state: DecisionState) -> dict:
    narrative = state["user_narrative"]

    llm = get_model(temperature=0.0).with_structured_output(BiasDetectionResult)
    result: BiasDetectionResult = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": narrative},
    ])

    return {"bias_flags": [f.model_dump() for f in result.flags]}
