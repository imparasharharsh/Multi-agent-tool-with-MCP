"""
Test script for Trip Planning Tools (Tavily search tools & AviationStack flight tool).
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from Tools.tavily_tool import tavily_search, search_destination_info
from Tools.flight_tool import search_flights, get_flight_details


def test_tavily_search():
    # 1. Test generic search (tavily_search)
    # query = "best hotels in Udaipur"
    # print(f"🔎 [1/2] Testing tavily_search with query: '{query}'")
    # print("=" * 60)
    # result_hotels = tavily_search.invoke({"query": query, "max_results": 5})
    # print(result_hotels)
    # print("=" * 60)

    # 2. Test structured destination search (search_destination_info)
    destination = "Udaipur"
    topic = "weather"
    print(f"\n🏰 Testing search_destination_info for destination: '{destination}', topic: '{topic}'")
    print("=" * 60)
    result_attractions = search_destination_info.invoke({"destination": destination, "topic": topic})
    print(result_attractions)
    print("=" * 60)
    print("\n✅ Tavily test execution completed successfully.")


def test_flight_tool():
    print("=" * 60)
    print("✈️ Testing Flight Search Tool (AviationStack)")
    print("=" * 60)

    # 1. Natural language query string test
    query_str = "Plan a Trip to US on 8th of Dec"
    print(f"\n[1/2] Natural language query: '{query_str}'")
    print("-" * 60)
    result_query = search_flights.invoke(query_str)
    print(result_query)

    # 2. Destination only test (origin defaulted from .env)
    dest = "LHR"
    print(f"\n[2/2] Query with destination only: '{dest}' (origin defaulted from .env)")
    print("-" * 60)
    result_dest = search_flights.invoke({"destination": dest, "departure_date": "2026-12-08"})
    print(result_dest)

    print("\n✅ Flight tool test execution completed successfully.")


if __name__ == "__main__":
    # Run flight tool test (or toggle test_tavily_search() as needed)
    test_flight_tool()
