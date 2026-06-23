"""
Search Agent — searches the web (via Tavily) for the
latest health news, outbreak alerts, and current information
that the static knowledge base does not contain.
"""

from tavily import TavilyClient


def search_agent(query: str, tavily_client: TavilyClient) -> str:
    """
    Searches the web for recent health-related information.
    Returns a clean, formatted summary of top results.
    """
    try:
        results = tavily_client.search(f"{query} health India")
        output = ""
        for r in results.get("results", [])[:3]:
            output += f"Title: {r.get('title', 'N/A')}\n"
            output += f"Info: {r.get('content', '')[:250]}\n\n"
        return output if output else "No recent news found on this topic."
    except Exception as e:
        return f"Web search unavailable: {str(e)}"
