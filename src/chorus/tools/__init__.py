from langchain_core.tools import tool


@tool
def search_database(query: str) -> str:
    """Search for relevant information to support a decision. Provide a clear, specific query."""
    # TODO: wire up Tavily or other search backend
    return f"Search results for: {query}"
