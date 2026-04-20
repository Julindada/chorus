import json

from chorus.state import DecisionState, StanceResult
from chorus.utils import AGENT_PROMPTS, STANCE_BOUNDARY, get_model

_DEBATE_INSTRUCTION = """
你正在参与一场关于用户决策的辩论。

输入包含：
- initial_stance：你在第一轮独立评估时的原始立场（Phase 1 锚点）
- my_stance：你当前的立场
- opponent_stance：对方的立场和论据
- ally_stances：与你同阵营的其他声音（可为空）
- debate_history：辩论历史摘要
- bias_note：用户叙述中检测到的潜在认知盲点（请主动审视，不要用于攻击对方）

要求：
- 认真回应对方的核心论点，不要回避
- 可以更新立场，但需说明是什么论据改变了你的想法
- 若坚持原立场，需给出新的理由
- 输出与 Phase 1 相同的结构：stance、reasoning、confidence"""

_COMPRESSION_PROMPT = (
    "将以下辩论记录压缩为核心分歧摘要，保留关键立场和论点，不超过 200 字。"
)


def debate_node(state: DecisionState) -> dict:
    rep_a, rep_b, allies_a, allies_b = _select_representatives(state)
    debate_context = _get_debate_context(state["debate_history"])
    stances = state["agent_stances"]

    llm = get_model().with_structured_output(StanceResult)
    new_a: StanceResult = llm.invoke(_build_messages(rep_a, stances[rep_b], allies_a, state, debate_context))
    new_b: StanceResult = llm.invoke(_build_messages(rep_b, stances[rep_a], allies_b, state, debate_context))

    return {
        "agent_stances":  {rep_a: new_a.model_dump(), rep_b: new_b.model_dump()},
        # increment is the hard guarantee against infinite debate loops
        "debate_round":   state["debate_round"] + 1,
        "debate_history": [{
            "round": state["debate_round"],
            "type":  state["conflict_type"],
            rep_a:   new_a.model_dump(),
            rep_b:   new_b.model_dump(),
        }],
    }


def _select_representatives(
    state: DecisionState,
) -> tuple[str, str, dict, dict]:
    stances = state["agent_stances"]
    conflicting = state["conflicting_agents"]
    scores = {n: stances[n]["stance"] for n in conflicting}

    if state["conflict_type"] == "binary":
        positive = [n for n in conflicting if scores[n] >  STANCE_BOUNDARY]
        negative = [n for n in conflicting if scores[n] < -STANCE_BOUNDARY]
        # most extreme agent on each side speaks for the camp
        rep_a = max(positive, key=lambda n: abs(scores[n]))
        rep_b = max(negative, key=lambda n: abs(scores[n]))
        allies_a = {n: stances[n] for n in positive if n != rep_a}
        allies_b = {n: stances[n] for n in negative if n != rep_b}
    else:  # outlier
        rep_a = conflicting[0]
        mean = sum(s["stance"] for s in stances.values()) / len(stances)
        # agent closest to mean speaks for the consensus
        rep_b = min(stances, key=lambda n: abs(stances[n]["stance"] - mean))
        # non-debating stances not passed to either side — prevents bandwagon pressure
        allies_a = {}
        allies_b = {}

    return rep_a, rep_b, allies_a, allies_b


def _get_debate_context(history: list[dict]) -> list[dict]:
    if len(history) <= 1:
        return history
    # compress older rounds to stay within context window; latest round kept verbatim
    summary = get_model().invoke([
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
