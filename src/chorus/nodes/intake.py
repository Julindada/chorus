from langgraph.types import interrupt

from chorus.state import DecisionState
from chorus.utils import AGENT_NAMES, SCHWARTZ_DIMS
from chorus.infrastructure.dao import load_value_vector, save_value_vector


def intake_node(state: DecisionState) -> dict:
    # ── 加载用户画像 ──────────────────────────────────────────────
    username = state["username"]
    value_vector = load_value_vector(username)
    if value_vector is None:
        user_input = interrupt({
            "action": "fill_value_vector",
            "message": "请为以下 10 个 Schwartz 价值观维度打分（0.0–1.0）：",
            "dims": SCHWARTZ_DIMS,
        })
        value_vector = {
            dim: max(0.0, min(1.0, float(user_input.get(dim, 0.5))))
            for dim in SCHWARTZ_DIMS
        }
        save_value_vector(username, value_vector)

    # ── 初始化控制字段 ────────────────────────────────────────────
    return {
        "value_vector":       value_vector,
        "critical_agents":    [AGENT_NAMES[0], AGENT_NAMES[1]],
        "failed_agents":      [],
        "debate_round":       0,
        "max_debate_rounds":  3,
        "last_entropy_score": None,
        "stance_history":     [],
        "debate_history":     [],
        "antagonism_flags":   [],
        "agent_stances":      {},
    }
