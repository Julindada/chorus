"""Unit tests for entropy_monitor conflict classification."""
import pytest
from chorus.nodes.entropy_monitor import _classify_conflict


def _stance(scores: dict) -> dict:
    return {"option_scores": scores, "reasoning": "", "confidence": 0.8}


# ---------------------------------------------------------------------------
# converged: all agents agree on every option
# ---------------------------------------------------------------------------

def test_converged_when_all_agree():
    stances = {a: _stance({"X": 0.5, "Y": -0.3}) for a in "ABCDEFG"}
    conflict_type, agents = _classify_conflict(stances)
    assert conflict_type == "converged"
    assert agents == []


# ---------------------------------------------------------------------------
# compensating preferences must NOT be treated as converged
# This is the bug this fix addresses:
# agents have similar means but strongly disagree on which option is best
# ---------------------------------------------------------------------------

def test_compensating_preferences_not_converged():
    """
    Arbiter: loves X, dislikes Y  → mean ≈ 0.15
    Compass: loves Y, neutral X   → mean ≈ 0.33
    All means cluster near 0.2, but per-option std on Y is large.
    Old mean-based check would return converged; new check must not.
    """
    stances = {
        "Arbiter":    _stance({"X": 0.85, "PM": -0.3, "Y": -0.1}),
        "Empath":     _stance({"X": 0.80, "PM": -0.5, "Y":  0.5}),
        "Soothsayer": _stance({"X": 0.70, "PM": -0.4, "Y": -0.2}),
        "Compass":    _stance({"X": 0.50, "PM": -0.3, "Y":  0.8}),
        "Narrator":   _stance({"X": 0.30, "PM": -0.3, "Y":  0.8}),
        "Conscience": _stance({"X": 0.40, "PM": -0.3, "Y":  0.8}),
        "Guardian":   _stance({"X": 0.80, "PM":  0.2, "Y": -0.2}),
    }
    conflict_type, agents = _classify_conflict(stances)
    assert conflict_type != "converged", (
        "agents strongly disagree on Y (std≈0.45) — should not be converged"
    )
    assert len(agents) >= 2


# ---------------------------------------------------------------------------
# binary: two clear camps
# ---------------------------------------------------------------------------

def test_binary_two_clear_camps():
    stances = {
        "A": _stance({"X":  0.8}),
        "B": _stance({"X":  0.7}),
        "C": _stance({"X":  0.6}),
        "D": _stance({"X": -0.5}),
        "E": _stance({"X": -0.6}),
        "F": _stance({"X": -0.7}),
        "G": _stance({"X": -0.4}),
    }
    conflict_type, agents = _classify_conflict(stances)
    assert conflict_type == "binary"
    assert set(agents) == {"A", "B", "C", "D", "E", "F", "G"}


# ---------------------------------------------------------------------------
# outlier: one agent clearly separated
# ---------------------------------------------------------------------------

def test_outlier_one_extreme_agent():
    stances = {
        "A": _stance({"X":  0.1}),
        "B": _stance({"X":  0.2}),
        "C": _stance({"X":  0.15}),
        "D": _stance({"X":  0.1}),
        "E": _stance({"X":  0.2}),
        "F": _stance({"X":  0.1}),
        "G": _stance({"X": -0.9}),
    }
    conflict_type, agents = _classify_conflict(stances)
    assert conflict_type == "outlier"
    assert agents == ["G"]
