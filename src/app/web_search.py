"""Web search via Tavily. Returns clean text results for the LLM."""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

API_KEY = os.getenv("TAVILY_API_KEY")

if not API_KEY:
    raise ValueError(
        "TAVILY_API_KEY not found. Make sure your .env file has the key."
    )

client = TavilyClient(api_key=API_KEY)


def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web with Tavily and return results as a single string.

    query:       what to search for
    max_results: how many top results to fetch (default 3)
    """
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
    )

    results = response.get("results", [])
    if not results:
        return ""

    # Format each result as: title, url, content snippet
    formatted = []
    for r in results:
        formatted.append(
            f"Title: {r.get('title', 'No title')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Content: {r.get('content', '')}"
        )

    return "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    # Quick test when run directly
    test_query = "Who won the FIFA World Cup 2022?"
    print(f"🔍 Searching: {test_query}\n")
    result = search_web(test_query)
    print(result)