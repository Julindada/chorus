import json

from chorus.state import DecisionState
from chorus.utils import AGENT_VALUE_MAPPING, get_model

_DEFAULT_VALUE_SCORE  = 0.5
_DEFAULT_SCENE_WEIGHT = 1.0

_SYSTEM_PROMPT = (
    "你是决策综合分析器。根据各心理维度对每个候选选项的评分和辩论过程，"
    "生成客观、平衡的最终决策建议，不超过 300 字。"
    "若存在不可调和的分歧，如实呈现，不要强行给出单一结论。"
    "尽量少使用心理学术语，用日常语言表达。"
)


def consensus_node(state: DecisionState) -> dict:
    stances = state["agent_stances"]
    weights = _compute_weights(stances, state["value_vector"], state["scene_template"])
    option_scores = _compute_option_scores(stances, weights)
    top_option = max(option_scores, key=option_scores.get)

    result = get_model().invoke(_build_messages(state, weights, option_scores))

    return {
        "consensus": {
            "option_scores": option_scores,
            "top_option":    top_option,
            "weights":       weights,
        },
        "final_recommendation": result.content,
    }


def _compute_option_scores(stances: dict, weights: dict) -> dict[str, float]:
    options: set[str] = set()
    for s in stances.values():
        options.update(s["option_scores"].keys())

    return {
        opt: sum(
            stances[a]["option_scores"].get(opt, 0.0) * weights[a]
            for a in stances
        )
        for opt in options
    }


def _compute_weights(
    stances: dict,
    value_vector: dict[str, float],
    scene_template: dict[str, float],
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for agent in stances:
        dims = AGENT_VALUE_MAPPING.get(agent, [])
        value_score = (
            sum(value_vector.get(d, _DEFAULT_VALUE_SCORE) for d in dims) / len(dims)
            if dims else _DEFAULT_VALUE_SCORE
        )
        raw[agent] = scene_template.get(agent, _DEFAULT_SCENE_WEIGHT) * value_score

    total = sum(raw.values()) or 1.0
    return {agent: w / total for agent, w in raw.items()}


def _build_messages(
    state: DecisionState,
    weights: dict[str, float],
    option_scores: dict[str, float],
) -> list[dict]:
    ranked = sorted(option_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({
            "user_narrative":   state["user_narrative"],
            "decision_options": state["decision_options"],
            "agent_stances":    state["agent_stances"],
            "weights":          weights,
            "option_scores":    option_scores,
            "ranked_options":   ranked,
            "debate_history":   state["debate_history"],
            "antagonism_flags": state["antagonism_flags"],
            "bias_flags":       state["bias_flags"],
        }, ensure_ascii=False)},
    ]
