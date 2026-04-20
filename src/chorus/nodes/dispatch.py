from langgraph.types import Send

from chorus.state import DecisionState
from chorus.utils import AGENT_NAMES


def dispatch_node(state: DecisionState) -> list[Send]:
    return [
        Send("agent_node", {**state, "agent_name": agent})
        for agent in AGENT_NAMES
    ]
