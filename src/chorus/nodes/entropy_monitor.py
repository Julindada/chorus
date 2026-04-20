import math

from chorus.state import DecisionState
from chorus.utils import AGENT_NAMES, STANCE_BOUNDARY

_ENTROPY_BASE          = 0.15
_CONSERVATION_ADJUST   = 0.08
_OPENNESS_ADJUST       = 0.05
_OSCILLATION_EPS       = 0.05
_CONSERVATION_DIMS     = ["conformity", "tradition", "security"]
_OPENNESS_DIMS         = ["self_direction", "stimulation"]


def entropy_monitor_node(state: DecisionState) -> dict:
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
        "entropy_score":      _compute_entropy(stances),
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
    if last is not None and abs(state["entropy_score"] - last) < _OSCILLATION_EPS:
        return "consensus_node"

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


def _compute_entropy(stances: dict) -> float:
    options: set[str] = set()
    for s in stances.values():
        options.update(s["option_scores"].keys())
    if not options:
        return 0.0

    entropies = []
    for opt in options:
        values = [s["option_scores"].get(opt, 0.0) for s in stances.values()]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        entropies.append(math.sqrt(variance))
    return sum(entropies) / len(entropies)


def _agent_mean(stance: dict) -> float:
    scores = stance["option_scores"].values()
    return sum(scores) / len(scores) if scores else 0.0


def _classify_conflict(stances: dict) -> tuple[str, list[str]]:
    means = {name: _agent_mean(s) for name, s in stances.items()}
    positive = [n for n, v in means.items() if v >  STANCE_BOUNDARY]
    negative = [n for n, v in means.items() if v < -STANCE_BOUNDARY]

    if len(positive) >= 2 and len(negative) >= 2:
        return "binary", positive + negative

    if len(positive) == 0 and len(negative) == 0:
        return "converged", []

    if len(positive) == 0 or len(negative) == 0:
        global_mean = sum(means.values()) / len(means)
        outlier = max(means, key=lambda n: abs(means[n] - global_mean))
        return "outlier", [outlier]

    return "multi_polar", list(means.keys())


def _compute_threshold(value_vector: dict) -> float:
    mean_conservation = sum(value_vector.get(d, 0.5) for d in _CONSERVATION_DIMS) / len(_CONSERVATION_DIMS)
    mean_openness     = sum(value_vector.get(d, 0.5) for d in _OPENNESS_DIMS)     / len(_OPENNESS_DIMS)
    return (
        _ENTROPY_BASE
        + (1 - mean_conservation) * _CONSERVATION_ADJUST
        + mean_openness           * _OPENNESS_ADJUST
    )
