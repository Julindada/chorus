"""Unit tests for route_after_entropy routing logic."""
import pytest
from chorus.nodes.entropy_monitor import route_after_entropy

_DEFAULT_VALUE_VECTOR = {
    "self_direction": 0.5, "stimulation": 0.5, "hedonism": 0.5,
    "achievement": 0.5, "power": 0.5, "security": 0.5,
    "conformity": 0.5, "tradition": 0.5, "benevolence": 0.5, "universalism": 0.5,
}

# threshold ≈ 0.15 + (1-0.5)*0.08 + 0.5*0.05 = 0.215 with default value vector


def _state(**overrides) -> dict:
    base = {
        "conflict_type":      "binary",
        "debate_round":       0,
        "max_debate_rounds":  3,
        "entropy_score":      0.35,
        "last_entropy_score": None,
        "stance_history":     [],
        "value_vector":       _DEFAULT_VALUE_VECTOR,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# normal debate trigger
# ---------------------------------------------------------------------------

def test_routes_to_debate_when_conditions_met():
    assert route_after_entropy(_state()) == "debate_node"


# ---------------------------------------------------------------------------
# conflict_type exits
# ---------------------------------------------------------------------------

def test_converged_skips_debate():
    assert route_after_entropy(_state(conflict_type="converged")) == "consensus_node"

def test_multi_polar_skips_debate():
    assert route_after_entropy(_state(conflict_type="multi_polar")) == "consensus_node"


# ---------------------------------------------------------------------------
# round limit
# ---------------------------------------------------------------------------

def test_max_rounds_reached_skips_debate():
    assert route_after_entropy(_state(debate_round=3, max_debate_rounds=3)) == "consensus_node"


# ---------------------------------------------------------------------------
# entropy threshold
# ---------------------------------------------------------------------------

def test_low_entropy_skips_debate():
    # threshold ≈ 0.215; entropy 0.10 is well below it
    assert route_after_entropy(_state(entropy_score=0.10)) == "consensus_node"

def test_entropy_just_above_threshold_triggers_debate():
    assert route_after_entropy(_state(entropy_score=0.25)) == "debate_node"


# ---------------------------------------------------------------------------
# oscillation check only fires after round 0
# ---------------------------------------------------------------------------

def test_oscillation_ignored_on_round_0():
    """
    Regression test for the bug where last_entropy_score from a previous graph
    execution caused the oscillation check to fire on the very first round,
    preventing debate from ever starting.
    """
    state = _state(
        debate_round=0,
        entropy_score=0.340,
        last_entropy_score=0.294,  # carried over from previous run — diff = 0.046 < eps
    )
    assert route_after_entropy(state) == "debate_node"

def test_oscillation_fires_after_round_1():
    """After at least one debate round, stagnant entropy should exit."""
    state = _state(
        debate_round=1,
        entropy_score=0.340,
        last_entropy_score=0.294,  # diff = 0.046 < _OSCILLATION_EPS = 0.05
    )
    assert route_after_entropy(state) == "consensus_node"


# ---------------------------------------------------------------------------
# stance history oscillation only fires after round 0
# ---------------------------------------------------------------------------

def _snapshot(val: float) -> dict:
    return {"A": {"X": val}, "B": {"X": val}}

def test_stance_history_oscillation_ignored_on_round_0():
    history = [_snapshot(0.8), _snapshot(0.5), _snapshot(0.8)]  # periodic
    state = _state(debate_round=0, stance_history=history)
    assert route_after_entropy(state) == "debate_node"

def test_stance_history_oscillation_fires_after_round_1():
    history = [_snapshot(0.8), _snapshot(0.5), _snapshot(0.8)]  # periodic
    state = _state(debate_round=1, stance_history=history)
    assert route_after_entropy(state) == "consensus_node"
