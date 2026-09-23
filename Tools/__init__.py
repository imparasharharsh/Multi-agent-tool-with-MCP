"""
Tools package for Multi-Agent Trip Planner.
Exposes flight search, Tavily research, and destination tools.
"""

from Tools.flight_tool import search_flights, get_flight_details
from Tools.tavily_tool import tavily_search, search_destination_info

tools = [
    search_flights,
    get_flight_details,
    tavily_search,
    search_destination_info,
]

__all__ = [
    "search_flights",
    "get_flight_details",
    "tavily_search",
    "search_destination_info",
    "tools",
]
