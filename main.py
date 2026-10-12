"""
Multi-Agent Trip Planner Backend using LangGraph, Groq LLM, and MCP Tools.

Architecture:
- Intent Router: Analyzes user request to determine if the query is:
    1. 'flight_only': Routes ONLY to Flight Agent -> END
    2. 'hotel_only': Routes ONLY to Hotel Agent -> END
    3. 'full_trip': Routes through Flight Agent -> Hotel Agent -> Itinerary Agent -> END
- State: TripPlannerState tracks intent, flight data, hotel data, itinerary, and messages.
- Agent 1: Flight Agent (uses Tools.flight_tool with AviationStack API)
- Agent 2: Hotel Agent (uses Tools.tavily_tool for lodging and accommodations)
- Agent 3: Itinerary Agent (uses Tools.tavily_tool to compile day-by-day schedule with flights & hotels)
"""

import os
from typing import Annotated, Any, Dict, List, Optional
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# Import custom tools
from Tools.flight_tool import get_flight_details, search_flights
from Tools.tavily_tool import search_destination_info, tavily_search

# Load environment variables (.env)
load_dotenv()


# ==============================================================================
# 1. State Definition
# ==============================================================================

class TripPlannerState(TypedDict):
    """Shared state across the multi-agent trip planning graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_request: str
    intent: Optional[str]
    flight_data: Optional[str]
    hotel_data: Optional[str]
    itinerary_data: Optional[str]
    final_plan: Optional[str]


# ==============================================================================
# 2. LLM Initialization (Groq)
# ==============================================================================

def get_groq_llm(model: Optional[str] = None) -> ChatGroq:
    """Returns an instance of ChatGroq configured with API key from .env."""
    api_key = (
        os.getenv("Groq_API_Key")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("groq_api_key")
    )
    if not api_key:
        raise ValueError("Groq API key not found. Please set 'Groq_API_Key' in your .env file.")

    # Default to available Groq model (gpt-oss-20b is fast with generous rate limits)
    model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.2,
    )


def execute_agent_with_tools(
    system_prompt: str,
    user_prompt: str,
    tools: List[Any],
    model: Optional[ChatGroq] = None,
    max_steps: int = 3,
) -> str:
    """
    Executes an autonomous agent loop with bound tools using Groq.
    Handles tool calls, captures observations, and returns the agent's final reasoning.
    """
    llm = model or get_groq_llm()
    tool_map = {t.name: t for t in tools}
    tool_bound_llm = llm.bind_tools(tools)

    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    for step in range(max_steps):
        # On the last step, guide the model to wrap up without calling further tools
        if step == max_steps - 1:
            messages.append(
                SystemMessage(
                    content="You have gathered the necessary information. Please provide your comprehensive final response and recommendations now without making further tool calls."
                )
            )

        response = tool_bound_llm.invoke(messages)
        messages.append(response)

        # If the LLM returned text and no further tool calls, return it
        if not getattr(response, "tool_calls", None):
            if response.content:
                return str(response.content)
            continue

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            call_id = tool_call.get("id")

            if tool_name in tool_map:
                try:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_result = f"Error executing {tool_name}: {e}"
            else:
                tool_result = f"Tool '{tool_name}' not recognized."

            # Truncate large tool outputs to protect token budget
            tool_result_str = str(tool_result)
            if len(tool_result_str) > 1200:
                tool_result_str = tool_result_str[:1200] + "... [truncated]"

            messages.append(
                ToolMessage(
                    tool_call_id=call_id,
                    name=tool_name,
                    content=tool_result_str,
                )
            )

    # If loop completed with pending tool messages, invoke once more for final synthesis
    messages.append(
        SystemMessage(
            content="Please summarize and synthesize your final recommendations based on all collected findings above."
        )
    )
    final_resp = tool_bound_llm.invoke(messages)
    if final_resp.content:
        return str(final_resp.content)

    # Fail-safe synthesis: if tool-bound call generated another tool call instead of text,
    # synthesize directly from collected tool observations using a clean prompt
    tool_observations = [str(msg.content) for msg in messages if isinstance(msg, ToolMessage)]
    if tool_observations:
        synthesis_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Original Request: {user_prompt}\n\n"
                f"Collected Research Findings:\n" + "\n\n".join(tool_observations) + "\n\n"
                f"Please provide your complete, detailed recommendations and analysis now."
            ),
        ]
        synth_resp = llm.invoke(synthesis_messages)
        if synth_resp.content:
            return str(synth_resp.content)

    # Fallback to last available text in message history
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)

    return "Completed analysis."



# ==============================================================================
# 3. Intent Router Node
# ==============================================================================

def classify_intent(user_request: str) -> str:
    """Classifies user request into 'flight_only', 'hotel_only', or 'full_trip'."""
    req_lower = user_request.lower()

    # Keywords indicating a complete vacation/itinerary/trip plan
    has_plan = any(k in req_lower for k in [
        "itinerary", "itenary", "plan a trip", "trip plan", "plan my trip",
        "vacation", "holiday", "things to do", "sightseeing", "activities",
        "schedule", "day 1", "tour"
    ])
    has_flight = any(k in req_lower for k in ["flight", "flights", "airline", "airfare", "fly", "flying"])
    has_hotel = any(k in req_lower for k in ["hotel", "hotels", "stay", "resort", "resorts", "lodging", "hostel", "accommodation"])

    # Explicit single-domain queries
    if has_plan:
        return "full_trip"
    if has_hotel and not has_flight:
        return "hotel_only"
    if has_flight and not has_hotel:
        return "flight_only"

    # Default to full trip if both or general query
    return "full_trip"


def router_node(state: TripPlannerState) -> Dict[str, Any]:
    """Identifies the traveler's goal and routes accordingly."""
    user_req = state.get("user_request", "")
    intent = classify_intent(user_req)
    print(f"\n🧭 [Intent Router] Detected Intent: '{intent}' for request: '{user_req}'")
    return {"intent": intent}


def route_from_router(state: TripPlannerState) -> str:
    """Routes to the appropriate starting agent based on intent."""
    intent = state.get("intent", "full_trip")
    if intent == "hotel_only":
        return "hotel_agent"
    return "flight_agent"


def route_from_flight(state: TripPlannerState) -> str:
    """Routes after flight agent: terminates if flight_only, else proceeds to hotel."""
    intent = state.get("intent", "full_trip")
    if intent == "flight_only":
        return "__end__"
    return "hotel_agent"


def route_from_hotel(state: TripPlannerState) -> str:
    """Routes after hotel agent: terminates if hotel_only, else proceeds to itinerary."""
    intent = state.get("intent", "full_trip")
    if intent == "hotel_only":
        return "__end__"
    return "itinerary_agent"


# ==============================================================================
# 4. Agent 1: Flight Agent (AviationStack Tool)
# ==============================================================================

FLIGHT_AGENT_PROMPT = """You are an expert Flight Logistics Specialist Agent for a trip planning system.
Your job:
1. Identify the traveler's departure origin, destination, and requested dates from their trip request.
2. ALWAYS use the `search_flights` tool to find available flight options.
3. If specific flight numbers are found, you may use `get_flight_details` to verify baggage/gate information.
4. Provide a structured summary of the best flight options including:
   - Airline & Flight Number
   - Departure and Arrival Times & Terminals
   - Stops and Any Delays
   - Clear recommendation on the most convenient flight
"""

def flight_agent_node(state: TripPlannerState) -> Dict[str, Any]:
    """Agent 1: Researches flights using the AviationStack flight tool."""
    user_req = state.get("user_request", "")
    intent = state.get("intent", "full_trip")
    print("\n🛫 [Flight Agent] Researching flight options...")

    prompt = f"Analyze the traveler's trip request and find the best flight options:\n\"{user_req}\""
    flight_summary = execute_agent_with_tools(
        system_prompt=FLIGHT_AGENT_PROMPT,
        user_prompt=prompt,
        tools=[search_flights, get_flight_details],
    )

    updates: Dict[str, Any] = {
        "flight_data": flight_summary,
        "messages": [AIMessage(content=f"### Flight Options\n{flight_summary}")],
    }

    # If the user only asked for flights, this is the final output
    if intent == "flight_only":
        updates["final_plan"] = flight_summary

    return updates


# ==============================================================================
# 5. Agent 2: Hotel Agent (Tavily Search Tool)
# ==============================================================================

HOTEL_AGENT_PROMPT = """You are a Luxury & Boutique Hotel Concierge Specialist Agent.
Your job:
1. Extract the destination city from the trip request and flight details.
2. ALWAYS use the `tavily_search` or `search_destination_info` tools to find top-rated hotels and accommodations.
3. Recommend 3 diverse lodging options:
   - Luxury / Iconic Heritage stay
   - Mid-range / Modern Comfort stay
   - Budget-friendly / Value-for-money stay
4. For each hotel, mention its neighborhood, unique highlights, and proximity to major sights.
"""

def hotel_agent_node(state: TripPlannerState) -> Dict[str, Any]:
    """Agent 2: Researches hotel and lodging options using Tavily web search."""
    user_req = state.get("user_request", "")
    flight_data = state.get("flight_data", "")
    intent = state.get("intent", "full_trip")
    print("\n🏨 [Hotel Agent] Researching hotel options...")

    prompt = (
        f"Trip Request:\n\"{user_req}\"\n\n"
        f"Flight Context:\n{flight_data}\n\n"
        f"Find top hotel options with neighborhood details and ratings for this destination."
    )

    hotel_summary = execute_agent_with_tools(
        system_prompt=HOTEL_AGENT_PROMPT,
        user_prompt=prompt,
        tools=[tavily_search, search_destination_info],
    )

    updates: Dict[str, Any] = {
        "hotel_data": hotel_summary,
        "messages": [AIMessage(content=f"### Hotel Recommendations\n{hotel_summary}")],
    }

    # If the user only asked for hotels, this is the final output
    if intent == "hotel_only":
        updates["final_plan"] = hotel_summary

    return updates


# ==============================================================================
# 6. Agent 3: Itinerary Agent (Tavily Search Tool + Synthesis)
# ==============================================================================

ITINERARY_AGENT_PROMPT = """You are a Master Travel Planner and Itinerary Architect.
Your job:
1. Review the traveler's trip request, chosen flights, and selected hotel recommendations.
2. Use `tavily_search` or `search_destination_info` to look up top must-see attractions, local dining gems, and cultural activities.
3. Synthesize a complete, polished, day-by-day travel itinerary that harmoniously combines:
   - ✈️ Flight Logistics: Arrival, departure, and airport transit buffer times
   - 🏨 Hotel Base: Check-in, location benefits, and evening returns
   - 🗺️ Daily Schedule: Morning, Afternoon, and Evening activities with dining recommendations
   - 💡 Practical Tips: Local transport, ticketing advice, and packing tips
Format your output cleanly in Markdown with emojis and clear headings.
"""

def itinerary_agent_node(state: TripPlannerState) -> Dict[str, Any]:
    """Agent 3: Gathers destination highlights and synthesizes the final comprehensive itinerary."""
    user_req = state.get("user_request", "")
    flight_data = state.get("flight_data", "No flight details available.")
    hotel_data = state.get("hotel_data", "No hotel details available.")
    print("\n🗺️ [Itinerary Agent] Compiling master itinerary...")

    prompt = (
        f"Traveler's Original Request:\n\"{user_req}\"\n\n"
        f"Selected Flight Logistics:\n{flight_data}\n\n"
        f"Selected Hotel Options:\n{hotel_data}\n\n"
        f"Research top activities, sights, and dining, and build a complete day-by-day itinerary."
    )

    itinerary_summary = execute_agent_with_tools(
        system_prompt=ITINERARY_AGENT_PROMPT,
        user_prompt=prompt,
        tools=[tavily_search, search_destination_info],
    )

    return {
        "itinerary_data": itinerary_summary,
        "final_plan": itinerary_summary,
        "messages": [AIMessage(content=f"### Master Trip Itinerary\n{itinerary_summary}")],
    }


# ==============================================================================
# 7. LangGraph StateGraph Construction
# ==============================================================================

def build_trip_planner_graph() -> StateGraph:
    """Builds and compiles the multi-agent trip planner workflow graph."""
    builder = StateGraph(TripPlannerState)

    # Add agent nodes
    builder.add_node("router", router_node)
    builder.add_node("flight_agent", flight_agent_node)
    builder.add_node("hotel_agent", hotel_agent_node)
    builder.add_node("itinerary_agent", itinerary_agent_node)

    # Start with intent classification
    builder.add_edge(START, "router")

    # Conditional routing from router
    builder.add_conditional_edges(
        "router",
        route_from_router,
        {
            "flight_agent": "flight_agent",
            "hotel_agent": "hotel_agent",
        },
    )

    # Conditional routing from flight_agent
    builder.add_conditional_edges(
        "flight_agent",
        route_from_flight,
        {
            "hotel_agent": "hotel_agent",
            "__end__": END,
        },
    )

    # Conditional routing from hotel_agent
    builder.add_conditional_edges(
        "hotel_agent",
        route_from_hotel,
        {
            "itinerary_agent": "itinerary_agent",
            "__end__": END,
        },
    )

    # Itinerary agent terminates the full trip workflow
    builder.add_edge("itinerary_agent", END)

    return builder.compile()


# Compiled LangGraph workflow instance
trip_planner_app = build_trip_planner_graph()


# ==============================================================================
# 8. Helper Function for Backend & API Calls
# ==============================================================================

def plan_trip(user_request: str) -> Dict[str, Any]:
    """
    High-level entrypoint for trip planning.
    
    Args:
        user_request: The user's travel request string.
        
    Returns:
        The final state containing flight_data, hotel_data, and final_plan.
    """
    initial_state: TripPlannerState = {
        "messages": [HumanMessage(content=user_request)],
        "user_request": user_request,
        "intent": None,
        "flight_data": None,
        "hotel_data": None,
        "itinerary_data": None,
        "final_plan": None,
    }

    final_state = trip_planner_app.invoke(initial_state)
    return final_state


# ==============================================================================
# 9. Main CLI Runner
# ==============================================================================

if __name__ == "__main__":
    # Test sample: hotel only request
    sample_query = "Give me 3-day hotel stay in Bengaluru from 15th of Dec to 17th of Dec"
    print("=" * 70)
    print("🌟 Multi-Agent Trip Planner Backend Initialized")
    print(f"📌 Query: '{sample_query}'")
    print("=" * 70)

    result = plan_trip(sample_query)

    print("\n" + "=" * 70)
    print("🎉 FINAL RESULT:")
    print("=" * 70)
    print(result.get("final_plan", "No plan generated."))
