"""
FastAPI Backend Server for Multi-Agent Trip Planner.
Exposes REST API endpoints and serves the interactive frontend web application.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import trip planner engine and tools
from main import plan_trip
from Tools.flight_tool import search_flights, get_flight_details
from Tools.tavily_tool import tavily_search, search_destination_info

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="AI Multi-Agent Trip Planner API",
    description="Multi-agent travel coordination system powered by LangGraph, Groq, AviationStack, and Tavily.",
    version="1.0.0",
)

# Enable CORS for local development and client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==============================================================================
# Request & Response Models
# ==============================================================================

class TripPlanRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural language trip, hotel, or flight request.",
        examples=["Plan a 3-day trip to Udaipur from Delhi on 8th of Dec"],
    )


class FlightRequest(BaseModel):
    destination: str = Field(..., description="Arrival city, country, or IATA code.")
    origin: Optional[str] = Field(None, description="Departure origin (optional).")
    departure_date: Optional[str] = Field(None, description="Departure date in YYYY-MM-DD or natural language.")


class HotelRequest(BaseModel):
    destination: str = Field(..., description="Destination city or region.")
    query: Optional[str] = Field(None, description="Specific hotel query or preferences.")


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/", tags=["Frontend"])
def serve_index():
    """Serves the main interactive web interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AI Multi-Agent Trip Planner API is running. Access /docs for API documentation."}


@app.get("/api/health", tags=["Health"])
def health_check():
    """Returns system status, active agents, and environment configuration status."""
    return {
        "status": "healthy",
        "service": "AI Multi-Agent Trip Planner",
        "agents": ["Intent Router", "Flight Agent", "Hotel Agent", "Itinerary Agent"],
        "keys_configured": {
            "groq": bool(os.getenv("Groq_API_Key") or os.getenv("GROQ_API_KEY")),
            "tavily": bool(os.getenv("TAVILY_API_KEY")),
            "aviationstack": bool(os.getenv("Aviationstack_API_key") or os.getenv("AVIATIONSTACK_API_KEY")),
        },
    }


@app.post("/api/plan", tags=["Trip Planner"])
def create_trip_plan(request: TripPlanRequest) -> Dict[str, Any]:
    """
    Main Multi-Agent Planning Endpoint.
    Classifies intent, dynamically routes to required agents (Flight, Hotel, Itinerary),
    and returns structured output.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result_state = plan_trip(query)

        return {
            "status": "success",
            "query": query,
            "intent": result_state.get("intent", "full_trip"),
            "final_plan": result_state.get("final_plan") or "No plan could be generated.",
            "flight_data": result_state.get("flight_data"),
            "hotel_data": result_state.get("hotel_data"),
            "itinerary_data": result_state.get("itinerary_data"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing multi-agent planning: {str(e)}")


@app.post("/api/flights", tags=["Direct Tools"])
def search_flights_endpoint(request: FlightRequest):
    """Direct API endpoint to search flights via AviationStack."""
    try:
        flights_out = search_flights.invoke({
            "destination": request.destination,
            "origin": request.origin,
            "departure_date": request.departure_date,
        })
        return {"status": "success", "flights": flights_out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hotels", tags=["Direct Tools"])
def search_hotels_endpoint(request: HotelRequest):
    """Direct API endpoint to search hotels via Tavily research."""
    try:
        q = request.query or f"best hotels and boutique stays in {request.destination}"
        hotels_out = tavily_search.invoke({"query": q, "max_results": 3})
        return {"status": "success", "hotels": hotels_out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Development Server Runner
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting FastAPI Trip Planner server on http://localhost:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
