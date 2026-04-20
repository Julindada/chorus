"""Unit tests for debate representative selection logic."""
import pytest
from chorus.nodes.debate import _select_representatives, _option_std


def _make_state(conflict_type: str, conflicting: list[str], stances: dict) -> dict:
    return {
        "conflict_type":      conflict_type,
        "conflicting_agents": conflicting,
        "agent_stances":      stances,
    }


def _stance(scores: dict) -> dict:
    return {"option_scores": scores, "reasoning": "", "confidence": 0.8}


# ---------------------------------------------------------------------------
# binary: classic absolute split (positive vs negative)
# ---------------------------------------------------------------------------

def test_binary_classic_split():
    """Highest scorer becomes rep_a, lowest becomes rep_b."""
    stances = {
        "A": _stance({"X": 0.8, "Y": -0.2}),
        "B": _stance({"X": 0.5, "Y":  0.1}),
        "C": _stance({"X": -0.6, "Y":  0.3}),
        "D": _stance({"X": -0.4, "Y":  0.2}),
    }
    state = _make_state("binary", ["A", "B", "C", "D"], stances)
    rep_a, rep_b, allies_a, allies_b, contested = _select_representatives(state)

    assert contested == "X"        # X has higher std dev than Y
    assert rep_a == "A"            # highest on X
    assert rep_b == "C"            # lowest on X
    assert "B" in allies_a         # B is in upper camp but not rep
    assert "D" in allies_b         # D is in lower camp but not rep


# ---------------------------------------------------------------------------
# binary: all-positive scores — fixed threshold would have missed this
# ---------------------------------------------------------------------------

def test_binary_all_positive_no_fixed_threshold():
    """
    All agents score positive on X (0.1–0.9), but there's real spread.
    Fixed ±0.3 threshold would find no 'negative' camp and miss the conflict.
    Median split should still pick the highest vs lowest.
    """
    stances = {
        "A": _stance({"X": 0.9}),
        "B": _stance({"X": 0.7}),
        "C": _stance({"X": 0.2}),
        "D": _stance({"X": 0.1}),
    }
    state = _make_state("binary", ["A", "B", "C", "D"], stances)
    rep_a, rep_b, _, _, contested = _select_representatives(state)

    assert contested == "X"
    assert rep_a == "A"   # 0.9 — highest
    assert rep_b == "D"   # 0.1 — lowest


# ---------------------------------------------------------------------------
# binary: ally assignment
# ---------------------------------------------------------------------------

def test_binary_allies_correctly_assigned():
    stances = {
        "A": _stance({"X":  0.8}),
        "B": _stance({"X":  0.6}),
        "C": _stance({"X":  0.4}),
        "D": _stance({"X": -0.3}),
        "E": _stance({"X": -0.7}),
        "F": _stance({"X": -0.5}),
    }
    state = _make_state("binary", list(stances.keys()), stances)
    rep_a, rep_b, allies_a, allies_b, _ = _select_representatives(state)

    assert rep_a == "A"
    assert rep_b == "E"
    assert set(allies_a.keys()) == {"B", "C"}
    assert set(allies_b.keys()) == {"D", "F"}


# ---------------------------------------------------------------------------
# binary: contested option is correctly selected by std dev
# ---------------------------------------------------------------------------

def test_binary_contested_option_by_std_dev():
    """Option Y has higher std dev than X — Y should be contested."""
    stances = {
        "A": _stance({"X": 0.5, "Y":  0.9}),
        "B": _stance({"X": 0.4, "Y":  0.6}),
        "C": _stance({"X": 0.3, "Y": -0.5}),
        "D": _stance({"X": 0.2, "Y": -0.8}),
    }
    state = _make_state("binary", list(stances.keys()), stances)
    _, _, _, _, contested = _select_representatives(state)
    assert contested == "Y"


# ---------------------------------------------------------------------------
# outlier: one agent clearly separated from the rest
# ---------------------------------------------------------------------------

def test_outlier_most_extreme_vs_closest_to_mean():
    stances = {
        "A": _stance({"X":  0.1}),
        "B": _stance({"X":  0.2}),
        "C": _stance({"X":  0.15}),
        "D": _stance({"X":  0.1}),
        "E": _stance({"X":  0.2}),
        "F": _stance({"X":  0.1}),
        "G": _stance({"X": -0.9}),   # outlier
    }
    # conflicting_agents for outlier only contains the outlier itself,
    # but _select_representatives iterates all stances for outlier branch
    state = _make_state("outlier", ["G"], stances)
    rep_a, rep_b, allies_a, allies_b, contested = _select_representatives(state)

    assert rep_a == "G"            # furthest from group mean
    # rep_b should be one of the middle-scoring agents (closest to mean)
    group_mean = sum(s["option_scores"]["X"] for s in stances.values()) / len(stances)
    assert abs(stances[rep_b]["option_scores"]["X"] - group_mean) == pytest.approx(
        min(abs(stances[n]["option_scores"]["X"] - group_mean) for n in stances if n != "G"),
        abs=1e-9,
    )
    assert allies_a == {}
    assert allies_b == {}


# ---------------------------------------------------------------------------
# _option_std helper
# ---------------------------------------------------------------------------

def test_option_std_known_value():
    stances = {
        "A": _stance({"X": 1.0}),
        "B": _stance({"X": -1.0}),
    }
    std = _option_std(stances, "X")
    assert std == pytest.approx(1.0)


def test_option_std_zero_variance():
    stances = {
        "A": _stance({"X": 0.5}),
        "B": _stance({"X": 0.5}),
        "C": _stance({"X": 0.5}),
    }
    assert _option_std(stances, "X") == pytest.approx(0.0)
