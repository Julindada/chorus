from typing import Annotated, TypedDict
import operator


class AgentStance(TypedDict):
    stance: float       # -1.0 to 1.0
    reasoning: str
    confidence: float   # 0.0 to 1.0


class DecisionState(TypedDict):
    # Input
    username: str
    user_narrative: str
    decision_options: list[str]

    # User profile
    value_vector: dict[str, float]      # Schwartz 10-dim weights, 0.0–1.0
    scene_template: dict[str, float]    # per-agent scene weights

    # Bias metadata
    bias_flags: list[dict]

    # Injected per Send branch, not stored in shared state
    agent_name: str

    # Phase 1: parallel eval — operator.or_ merges Send branches into one dict
    agent_stances: Annotated[dict[str, AgentStance], operator.or_]
    initial_stances: dict[str, AgentStance]  # Phase 1 snapshot, set once, read-only in debate

    # Fan-in health check
    critical_agents: list[str]          # missing any → crash graph
    failed_agents: list[str]            # detected missing after fan-in

    # Entropy & debate control
    entropy_score: float
    last_entropy_score: float           # previous round, for oscillation detection
    conflict_type: str                  # "binary" | "multi_polar" | "outlier"
    conflicting_agents: list[str]
    debate_round: int
    max_debate_rounds: int

    # Stance snapshot per round — appended by entropy_monitor for oscillation detection
    stance_history: Annotated[list[dict[str, float]], operator.add]

    # Debate history — appended each round
    debate_history: Annotated[list[dict], operator.add]

    # Output
    consensus: dict
    antagonism_flags: list[str]
    final_recommendation: str
