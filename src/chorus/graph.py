import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.pregel import RetryPolicy

from chorus.infrastructure.config import DB_PATH
from chorus.state import DecisionState
from chorus.nodes import (
    intake_node,
    decision_classifier_node,
    bias_detection_node,
    dispatch_node,
    agent_node,
    entropy_monitor_node,
    debate_node,
    consensus_node,
    persona_updater_node,
    route_after_entropy,
)

builder = StateGraph(DecisionState)

builder.add_node("intake_node",               intake_node)
builder.add_node("decision_classifier_node",  decision_classifier_node)
builder.add_node("bias_detection_node",       bias_detection_node)
builder.add_node("dispatch_node",             dispatch_node)
builder.add_node("agent_node",                agent_node,
                 retry=RetryPolicy(max_attempts=3, retry_on=Exception))
builder.add_node("entropy_monitor_node",      entropy_monitor_node)
builder.add_node("debate_node",               debate_node)
builder.add_node("consensus_node",            consensus_node)
builder.add_node("persona_updater_node",      persona_updater_node)

builder.add_edge(START,                        "intake_node")
builder.add_edge("intake_node",                "decision_classifier_node")
builder.add_edge("decision_classifier_node",   "bias_detection_node")
builder.add_edge("bias_detection_node",        "dispatch_node")
builder.add_edge("agent_node",                 "entropy_monitor_node")
builder.add_edge("debate_node",                "entropy_monitor_node")
builder.add_edge("consensus_node",             "persona_updater_node")
builder.add_edge("persona_updater_node",       END)

builder.add_conditional_edges("entropy_monitor_node", route_after_entropy)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
graph = builder.compile(checkpointer=SqliteSaver(conn))
