import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from chorus.state import AgentState
from chorus.nodes import call_model, tool_node, should_continue

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

conn = sqlite3.connect("chorus.db", check_same_thread=False)
graph = workflow.compile(checkpointer=SqliteSaver(conn))
