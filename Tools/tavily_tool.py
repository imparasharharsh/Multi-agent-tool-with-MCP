"""
Tavily Search Tool for Travel Planning Agent.
Provides web search capabilities for destinations, attractions, hotels, dining, and local activities.
"""

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv()


class TavilySearchInput(BaseModel):
    query: str = Field(description="Search query for travel research (e.g. 'top attractions in Tokyo', 'budget hotels in Paris')")
    max_results: int = Field(default=5, description="Maximum number of search results to return")


@tool(args_schema=TavilySearchInput)
def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily for comprehensive and accurate travel research.
    Retrieves information on attractions, accommodations, weather, culture, and itinerary recommendations.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if api_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, max_results=max_results)
            results = response.get("results", [])

            if not results:
                return f"No relevant results found for '{query}'."

            formatted = [f"Search results for '{query}':\n"]
            for idx, item in enumerate(results, 1):
                title = item.get("title", "No Title")
                content = item.get("content", "")
                url = item.get("url", "")
                formatted.append(f"{idx}. {title}\n   {content}\n   Source: {url}\n")
            return "\n".join(formatted)
        except Exception as e:
            return f"Tavily API search error: {e}. Falling back to default search."

    # Fallback simulated response if TAVILY_API_KEY is not configured
    return (
        f"Search results for '{query}' (API Key not set; simulated overview):\n"
        f"1. Top Highlights for '{query}': Popular recommendations include city center tours, iconic historic landmarks, and cultural museums.\n"
        f"2. Local Dining & Food: Authentic local restaurants and street food hotspots with 4.5+ star ratings.\n"
        f"3. Practical Tips: Best times to visit, local transit card suggestions, and walkable neighborhoods.\n"
        f"[Tip: Configure TAVILY_API_KEY in .env for live web search results]"
    )


@tool
def search_destination_info(destination: str, topic: str = "attractions") -> str:
    """
    Look up specialized travel information for a specific destination by topic.

    Args:
        destination: City or country name (e.g., 'Rome', 'Kyoto', 'Barcelona').
        topic: Category to research ('attractions', 'food', 'neighborhoods', 'safety').
    """
    query = f"best {topic} in {destination} travel guide"
    return tavily_search.invoke({"query": query, "max_results": 3})
