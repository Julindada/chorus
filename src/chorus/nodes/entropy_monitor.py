import math

from chorus.state import DecisionState
from chorus.utils import (
    AGENT_NAMES, AGENT_VALUE_MAPPING,
    SCHWARTZ_CONSERVATION_DIMS, SCHWARTZ_OPENNESS_DIMS, SCORE_RANGE,
)

# base entropy threshold to trigger debate (~15% of max possible weighted std dev)
_ENTROPY_BASE        = 0.15
# max threshold reduction for high-conservation users (security/conformity/tradition)
_CONSERVATION_ADJUST = 0.08
# max threshold increase for high-openness users (self_direction/stimulation)
_OPENNESS_ADJUST     = 0.05
# min change between rounds; below this the debate is considered stagnant
_OSCILLATION_EPS     = SCORE_RANGE * 0.025
# min weighted std dev across agent means; below this agents are considered converged
_CONVERGE_STD        = SCORE_RANGE * 0.05
# removing the outlier candidate must reduce std dev by at least this fraction
_OUTLIER_VARIANCE_RATIO = 0.5


async def entropy_monitor_node(state: DecisionState) -> dict:
    stances = dict(state["agent_stances"])

    missing = [a for a in AGENT_NAMES if a not in stances]
    if any(a in state["critical_agents"] for a in missing):
        return {
            "failed_agents":    missing,
            "antagonism_flags": ["SYSTEM_PARTIAL_FAILURE"],
        }
    options = state["decision_options"]
    for agent in missing:
        stances[agent] = {
            "option_scores": {opt: 0.0 for opt in options},
            "reasoning": "[missing, neutral fill]",
            "confidence": 0.0,
        }

    updates: dict = {}
    if state["debate_round"] == 0:
        updates["initial_stances"] = dict(stances)

    conflict_type, conflicting_agents = _classify_conflict(stances)
    if conflict_type == "multi_polar":
        updates["antagonism_flags"] = conflicting_agents

    return {
        **updates,
        "agent_stances":      stances,
        "entropy_score":      _compute_entropy(stances, state["value_vector"]),
        "last_entropy_score": state.get("entropy_score"),
        "conflict_type":      conflict_type,
        "conflicting_agents": conflicting_agents,
        "failed_agents":      missing,
        "stance_history":     [{name: s["option_scores"] for name, s in stances.items()}],
    }


def route_after_entropy(state: DecisionState) -> str:
    if state.get("conflict_type") in ("multi_polar", "converged"):
        return "consensus_node"

    if state["debate_round"] >= state["max_debate_rounds"]:
        return "consensus_node"

    if state["entropy_score"] < _compute_threshold(state["value_vector"]):
        return "consensus_node"

    last = state.get("last_entropy_score")
    if last is not None and state["debate_round"] > 0 and abs(state["entropy_score"] - last) < _OSCILLATION_EPS:
        return "consensus_node"

    if state["debate_round"] > 0:
        history = state.get("stance_history", [])
        if len(history) >= 3:
            current  = _flatten(history[-1])
            two_ago  = _flatten(history[-3])
            delta = sum(abs(current.get(k, 0) - two_ago.get(k, 0)) for k in current)
            if delta < _OSCILLATION_EPS * len(current):
                return "consensus_node"

    return "debate_node"


def _flatten(snapshot: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{agent}:{opt}": score
        for agent, scores in snapshot.items()
        for opt, score in scores.items()
    }


def _compute_entropy(stances: dict, value_vector: dict) -> float:
    agent_weights = {
        a: sum(value_vector.get(d, 0.5) for d in AGENT_VALUE_MAPPING.get(a, [])) / max(len(AGENT_VALUE_MAPPING.get(a, [])), 1)
        for a in stances
    }
    weight_sum = sum(agent_weights.values()) or 1.0

    options: set[str] = set()
    for s in stances.values():
        options.update(s["option_scores"].keys())
    if not options:
        return 0.0

    entropies = []
    for opt in options:
        w_mean = sum(agent_weights[a] * stances[a]["option_scores"].get(opt, 0.0) for a in stances) / weight_sum
        w_var  = sum(agent_weights[a] * (stances[a]["option_scores"].get(opt, 0.0) - w_mean) ** 2 for a in stances) / weight_sum
        entropies.append(math.sqrt(w_var))
    return sum(entropies) / len(entropies)


def _agent_mean(stance: dict) -> float:
    scores = stance["option_scores"].values()
    return sum(scores) / len(scores) if scores else 0.0


def _option_std(stances: dict, option: str) -> float:
    values = [s["option_scores"].get(option, 0.0) for s in stances.values()]
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _classify_conflict(stances: dict) -> tuple[str, list[str]]:
    options = {opt for s in stances.values() for opt in s["option_scores"]}

    # convergence: no option has meaningful per-option disagreement
    # max per-option std avoids false convergence when agents have compensating
    # preferences (e.g. A prefers X, B prefers Y — their means cancel out)
    max_opt_std = max((_option_std(stances, opt) for opt in options), default=0.0)
    if max_opt_std < _CONVERGE_STD:
        return "converged", []

    means = {name: _agent_mean(s) for name, s in stances.items()}
    values = list(means.values())
    global_mean = sum(values) / len(values)
    total_std = math.sqrt(sum((v - global_mean) ** 2 for v in values) / len(values))

    # outlier: removing the most extreme agent cuts std dev by more than half
    outlier_candidate = max(means, key=lambda n: abs(means[n] - global_mean))
    rest = [v for n, v in means.items() if n != outlier_candidate]
    rest_mean = sum(rest) / len(rest)
    rest_std = math.sqrt(sum((v - rest_mean) ** 2 for v in rest) / len(rest))
    if total_std > 0 and rest_std / total_std < _OUTLIER_VARIANCE_RATIO:
        return "outlier", [outlier_candidate]

    # binary: two camps via median split
    median = sorted(values)[len(values) // 2]
    above  = [n for n in means if means[n] > median]
    below  = [n for n in means if means[n] < median]
    ties   = [n for n in means if means[n] == median]
    if len(above) <= len(below):
        above += ties
    else:
        below += ties

    if len(above) >= 2 and len(below) >= 2:
        return "binary", above + below

    return "multi_polar", list(means.keys())


def _compute_threshold(value_vector: dict) -> float:
    mean_conservation = sum(value_vector.get(d, 0.5) for d in SCHWARTZ_CONSERVATION_DIMS) / len(SCHWARTZ_CONSERVATION_DIMS)
    mean_openness     = sum(value_vector.get(d, 0.5) for d in SCHWARTZ_OPENNESS_DIMS)     / len(SCHWARTZ_OPENNESS_DIMS)
    return (
        _ENTROPY_BASE
        + (1 - mean_conservation) * _CONSERVATION_ADJUST
        + mean_openness           * _OPENNESS_ADJUST
    )
