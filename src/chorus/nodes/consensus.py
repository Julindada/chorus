import json

from chorus.state import DecisionState
from chorus.utils import AGENT_VALUE_MAPPING, get_model

_DEFAULT_VALUE_SCORE  = 0.5  # neutral importance for unknown Schwartz dims
_DEFAULT_SCENE_WEIGHT = 1.0  # equal weight for agents absent from scene_template

_SYSTEM_PROMPT = (
    "你是决策综合分析器。根据各心理维度的评估结果和辩论过程，"
    "生成客观、平衡的最终决策报告。若存在不可调和的拮抗，如实呈现，不要强行给出单一结论。"
)


def consensus_node(state: DecisionState) -> dict:
    stances = state["agent_stances"]
    weights = _compute_weights(stances, state["value_vector"], state["scene_template"])
    weighted_score = sum(stances[a]["stance"] * weights[a] for a in stances)

    result = get_model().invoke(_build_messages(state, weights, weighted_score))

    return {
        "consensus": {"weighted_score": weighted_score, "weights": weights},
        "final_recommendation": result.content,
    }


def _compute_weights(
    stances: dict,
    value_vector: dict[str, float],
    scene_template: dict[str, float],
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for agent in stances:
        dims = AGENT_VALUE_MAPPING.get(agent, [])
        # average the user's importance scores for this agent's Schwartz dimensions
        value_score = (
            sum(value_vector.get(d, _DEFAULT_VALUE_SCORE) for d in dims) / len(dims)
            if dims else _DEFAULT_VALUE_SCORE
        )
        # scene_template sets base weight per decision type; value_score personalizes it
        raw[agent] = scene_template.get(agent, _DEFAULT_SCENE_WEIGHT) * value_score

    total = sum(raw.values()) or 1.0  # guard against all-zero weights
    return {agent: w / total for agent, w in raw.items()}


def _build_messages(
    state: DecisionState,
    weights: dict[str, float],
    weighted_score: float,
) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({
            "user_narrative":   state["user_narrative"],
            "agent_stances":    state["agent_stances"],
            "weights":          weights,
            "weighted_score":   weighted_score,
            "debate_history":   state["debate_history"],
            "antagonism_flags": state["antagonism_flags"],
            "bias_flags":       state["bias_flags"],
        }, ensure_ascii=False)},
    ]
