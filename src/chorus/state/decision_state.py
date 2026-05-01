from typing import Annotated
from typing_extensions import TypedDict
import operator

from pydantic import BaseModel, Field


class AgentStance(TypedDict):
    option_scores: dict[str, float]  # option → score (-1.0 to 1.0)
    reasoning: str
    confidence: float                # 0.0 to 1.0


class StanceResult(BaseModel):
    option_scores: dict[str, float] = Field(description="每个候选选项的评分，key 为选项名称，value 为 -1.0（强烈不建议）到 +1.0（强烈推荐），必须包含所有选项")
    reasoning: str = Field(description="立足于你的心理维度的综合评估依据，不超过 60 字，用日常语言表达")
    confidence: float = Field(ge=0.0, le=1.0, description="对此判断的信心分，0.0 到 1.0")


class DecisionState(TypedDict):
    # Input
    username: str
    user_narrative: str
    decision_options: list[str]

    # User profile
    value_vector: dict[str, float]      # Schwartz 10-dim weights, 0.0–1.0
    decision_type: str                  # career | finance | relationship | ...
    scene_template: dict[str, float]    # per-agent scene weights

    # Bias metadata
    bias_flags: list[dict]

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
    final_report: str
