import math

from chorus.state import DecisionState
from chorus.utils import AGENT_NAMES, STANCE_BOUNDARY

_ENTROPY_BASE          = 0.15
_CONSERVATION_ADJUST   = 0.08   # high conservation → tighter exit threshold
_OPENNESS_ADJUST       = 0.05   # high openness → looser exit threshold
_OSCILLATION_EPS       = 0.05
_CONSERVATION_DIMS     = ["conformity", "tradition", "security"]
_OPENNESS_DIMS         = ["self_direction", "stimulation"]


def entropy_monitor_node(state: DecisionState) -> dict:
    # copy to avoid mutating shared state when filling neutral values for missing agents
    stances = dict(state["agent_stances"])

    missing = [a for a in AGENT_NAMES if a not in stances]
    if any(a in state["critical_agents"] for a in missing):
        return {
            "failed_agents": missing,
            "antagonism_flags": ["SYSTEM_PARTIAL_FAILURE"],
        }
    for agent in missing:
        stances[agent] = {"stance": 0.0, "reasoning": "[missing, neutral fill]", "confidence": 0.0}

    updates: dict = {}
    # snapshot Phase 1 stances before any debate overwrites agent_stances via operator.or_
    if state["debate_round"] == 0:
        updates["initial_stances"] = dict(stances)

    conflict_type, conflicting_agents = _classify_conflict(stances)
    if conflict_type == "multi_polar":
        updates["antagonism_flags"] = conflicting_agents

    return {
        **updates,
        "agent_stances":      stances,
        "entropy_score":      _compute_entropy(stances),
        # carry current entropy forward so next round can detect no-progress
        "last_entropy_score": state.get("entropy_score"),
        "conflict_type":      conflict_type,
        "conflicting_agents": conflicting_agents,
        "failed_agents":      missing,
        # wrapped in list because operator.add reducer appends lists each round
        "stance_history":     [{name: s["stance"] for name, s in stances.items()}],
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
        # compare round N with round N-2 to detect period-2 oscillation
        current, two_ago = history[-1], history[-3]
        delta = sum(abs(current.get(k, 0) - two_ago.get(k, 0)) for k in current)
        if delta < _OSCILLATION_EPS * len(current):
            return "consensus_node"

    return "debate_node"


def _compute_entropy(stances: dict) -> float:
    # std dev of stances: 0 = full consensus, 1.0 = maximum polarization
    values = [s["stance"] for s in stances.values()]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _compute_threshold(value_vector: dict) -> float:
    # conservation dims lower threshold (prefer closure); openness dims raise it (tolerate ambiguity)
    mean_conservation = sum(value_vector.get(d, 0.5) for d in _CONSERVATION_DIMS) / len(_CONSERVATION_DIMS)
    mean_openness     = sum(value_vector.get(d, 0.5) for d in _OPENNESS_DIMS)     / len(_OPENNESS_DIMS)
    return (
        _ENTROPY_BASE
        + (1 - mean_conservation) * _CONSERVATION_ADJUST
        + mean_openness           * _OPENNESS_ADJUST
    )


def _classify_conflict(stances: dict) -> tuple[str, list[str]]:
    scores = {name: s["stance"] for name, s in stances.items()}
    positive = [n for n, v in scores.items() if v >  STANCE_BOUNDARY]
    negative = [n for n, v in scores.items() if v < -STANCE_BOUNDARY]

    if len(positive) >= 2 and len(negative) >= 2:
        return "binary", positive + negative

    # no agent holds a strong view — group has already converged near neutral
    if len(positive) == 0 and len(negative) == 0:
        return "converged", []

    if len(positive) == 0 or len(negative) == 0:
        mean = sum(scores.values()) / len(scores)
        outlier = max(scores, key=lambda n: abs(scores[n] - mean))
        return "outlier", [outlier]

    return "multi_polar", list(scores.keys())
