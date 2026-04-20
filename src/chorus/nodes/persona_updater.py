from datetime import datetime

from chorus.state import DecisionState
from chorus.infrastructure.dao.decision_history_dao import insert_decision_records


def persona_updater_node(state: DecisionState) -> dict:
    weighted_score = state["consensus"]["weighted_score"]
    decision_type = state["decision_type"]
    # single timestamp for all records so they're identifiable as one decision event
    timestamp = datetime.now().isoformat()

    records = [
        {
            "decision_type":   decision_type,
            "agent_name":      agent_name,
            # record Phase 1 stance, not the post-debate one, to track original position
            "initial_stance":  state["initial_stances"][agent_name]["stance"],
            # alignment ∈ [0,1]: 1 = fully aligned with final decision, 0 = opposite
            # divide by 2 to normalize |stance - weighted_score| from [0,2] to [0,1]
            "final_alignment": round(1.0 - abs(stance["stance"] - weighted_score) / 2.0, 4),
            "timestamp":       timestamp,
        }
        for agent_name, stance in state["agent_stances"].items()
    ]

    insert_decision_records(records)
    return {}
