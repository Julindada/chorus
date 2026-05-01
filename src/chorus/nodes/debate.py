import asyncio
import json

from chorus.state import DecisionState, StanceResult
from chorus.utils import AGENT_PROMPTS, get_model

_DEBATE_INSTRUCTION = """
你正在参与一场关于用户决策的辩论。

输入包含：
- initial_stance：你在第一轮独立评估时的原始评分（各选项）
- my_stance：你当前对各选项的评分
- opponent_stance：对方对各选项的评分和论据
- ally_stances：与你观点相近的其他声音（可为空）
- debate_history：辩论历史摘要
- bias_note：用户叙述中检测到的潜在认知盲点（请主动审视，不要用于攻击对方）

要求：
- 认真回应对方的核心论点，不要回避
- 可以更新各选项评分，但需说明是什么论据改变了你的想法
- 若坚持原评分，需给出新的理由
- 输出与第一轮相同的结构：option_scores、reasoning、confidence
- reasoning 不超过 60 字，尽量少使用心理学术语，用日常语言表达
- 请以 JSON 格式返回结果"""

_COMPRESSION_PROMPT = (
    "将以下辩论记录压缩为核心分歧摘要，保留关键立场和论点，不超过 200 字。"
)


async def debate_node(state: DecisionState) -> dict:
    rep_a, rep_b, allies_a, allies_b, contested_option = _select_representatives(state)
    debate_context = await _get_debate_context(state["debate_history"])
    stances = state["agent_stances"]

    llm = get_model().with_structured_output(StanceResult)
    new_a, new_b = await asyncio.gather(
        llm.ainvoke(_build_messages(rep_a, stances[rep_b], allies_a, state, debate_context)),
        llm.ainvoke(_build_messages(rep_b, stances[rep_a], allies_b, state, debate_context)),
    )

    return {
        "agent_stances":  {rep_a: new_a.model_dump(), rep_b: new_b.model_dump()},
        "debate_round":   state["debate_round"] + 1,
        "debate_history": [{
            "round":  state["debate_round"],
            "type":   state["conflict_type"],
            "reason": _selection_reason(rep_a, rep_b, state["conflict_type"], contested_option),
            rep_a:    new_a.model_dump(),
            rep_b:    new_b.model_dump(),
        }],
    }


def _selection_reason(rep_a: str, rep_b: str, conflict_type: str, contested_option: str = "") -> str:
    return f"在「{contested_option}」上评分分歧最大：{rep_a}（评分最高）与 {rep_b}（评分最低）展开辩论"


def _agent_mean(stance: dict) -> float:
    scores = stance["option_scores"].values()
    return sum(scores) / len(scores) if scores else 0.0


def _option_std(stances: dict, option: str) -> float:
    import math
    values = [s["option_scores"].get(option, 0.0) for s in stances.values()]
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _select_representatives(
    state: DecisionState,
) -> tuple[str, str, dict, dict, str]:
    stances = state["agent_stances"]
    conflicting = state["conflicting_agents"]

    # find the most contested option across all agents (highest std dev)
    all_options: set[str] = set()
    for s in stances.values():
        all_options.update(s["option_scores"].keys())
    contested_option = max(all_options, key=lambda opt: _option_std(stances, opt))
    opt_scores = {n: stances[n]["option_scores"].get(contested_option, 0.0) for n in conflicting}

    if state["conflict_type"] == "binary":
        # dynamic median split on contested option scores — no fixed threshold
        sorted_agents = sorted(conflicting, key=lambda n: opt_scores[n])
        mid = len(sorted_agents) // 2
        lower_camp = sorted_agents[:mid]
        upper_camp = sorted_agents[mid:]
        rep_a    = upper_camp[-1]   # highest score on contested option
        rep_b    = lower_camp[0]    # lowest score on contested option
        allies_a = {n: stances[n] for n in upper_camp if n != rep_a}
        allies_b = {n: stances[n] for n in lower_camp if n != rep_b}
    else:  # outlier
        # most extreme agent on contested option vs closest to group mean
        all_scores = {n: stances[n]["option_scores"].get(contested_option, 0.0) for n in stances}
        group_mean = sum(all_scores.values()) / len(all_scores)
        rep_a    = max(all_scores, key=lambda n: abs(all_scores[n] - group_mean))
        rep_b    = min((n for n in stances if n != rep_a), key=lambda n: abs(all_scores[n] - group_mean))
        allies_a = {}
        allies_b = {}

    return rep_a, rep_b, allies_a, allies_b, contested_option


async def _get_debate_context(history: list[dict]) -> list[dict]:
    if len(history) <= 1:
        return history
    summary = await get_model().ainvoke([
        {"role": "system", "content": _COMPRESSION_PROMPT},
        {"role": "user",   "content": json.dumps(history[:-1], ensure_ascii=False)},
    ])
    return [{"round": "summary", "content": summary.content}, history[-1]]


def _build_messages(
    agent_name: str,
    opponent_stance: dict,
    ally_stances: dict,
    state: DecisionState,
    debate_context: list[dict],
) -> list[dict]:
    bias_note = (
        f"用户叙述中检测到以下潜在认知盲点，请在推理时主动审视：{state['bias_flags']}"
        if state["bias_flags"] else ""
    )
    return [
        {"role": "system", "content": AGENT_PROMPTS[agent_name] + _DEBATE_INSTRUCTION},
        {"role": "user", "content": json.dumps({
            "initial_stance":  state["initial_stances"].get(agent_name),
            "my_stance":       state["agent_stances"][agent_name],
            "opponent_stance": opponent_stance,
            "ally_stances":    ally_stances,
            "debate_history":  debate_context,
            "bias_note":       bias_note,
        }, ensure_ascii=False)},
    ]
