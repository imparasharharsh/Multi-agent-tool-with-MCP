"""
Flight Search & Details Tool for Trip Planning Agent using AviationStack API.
Supports:
1. Natural language query strings: e.g. "Plan a Trip to US on 8th of Dec"
2. Structured arguments: destination, origin, departure_date
3. Full date parsing: "8th of Dec", "Dec 8", "December 8th", "YYYY-MM-DD"
4. Intelligent Free-Tier handling: Gracefully handles AviationStack's free-tier date restriction
   by fetching the recurring route timetable and projecting flight timings onto the requested travel date.
5. Location resolution: City names, country names, airport names, or IATA codes.
6. Default source fallback: Reads DEFAULT_SOURCE from .env if origin is not specified.
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Load environment variables from .env
load_dotenv()

AVIATIONSTACK_BASE_URL = "http://api.aviationstack.com/v1/flights"

MONTH_MAP: Dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# Mapping countries to their primary international hub airport
COUNTRY_DEFAULT_HUBS: Dict[str, str] = {
    "india": "DEL",
    "united states": "JFK",
    "usa": "JFK",
    "us": "JFK",
    "united kingdom": "LHR",
    "uk": "LHR",
    "britain": "LHR",
    "england": "LHR",
    "france": "CDG",
    "japan": "HND",
    "united arab emirates": "DXB",
    "uae": "DXB",
    "germany": "FRA",
    "italy": "FCO",
    "spain": "MAD",
    "singapore": "SIN",
    "thailand": "BKK",
    "australia": "SYD",
    "canada": "YYZ",
    "switzerland": "ZRH",
    "netherlands": "AMS",
    "turkey": "IST",
    "saudi arabia": "RUH",
    "qatar": "DOH",
    "malaysia": "KUL",
    "indonesia": "DPS",
    "vietnam": "HAN",
    "china": "PEK",
    "south korea": "ICN",
    "russia": "SVO",
    "mexico": "MEX",
    "brazil": "GRU",
    "south africa": "JNB",
    "egypt": "CAI",
    "greece": "ATH",
    "portugal": "LIS",
    "austria": "VIE",
    "belgium": "BRU",
    "new zealand": "AKL",
}

# Major city and airport name mappings to IATA codes
CITY_AIRPORT_TO_IATA: Dict[str, str] = {
    # India
    "delhi": "DEL",
    "new delhi": "DEL",
    "indira gandhi": "DEL",
    "indira gandhi international": "DEL",
    "udaipur": "UDR",
    "maharana pratap": "UDR",
    "dabok": "UDR",
    "mumbai": "BOM",
    "bombay": "BOM",
    "chhatrapati shivaji": "BOM",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "kempegowda": "BLR",
    "chennai": "MAA",
    "madras": "MAA",
    "kolkata": "CCU",
    "calcutta": "CCU",
    "netaji subhash chandra bose": "CCU",
    "hyderabad": "HYD",
    "rajiv gandhi": "HYD",
    "goa": "GOI",
    "dabolim": "GOI",
    "mopa": "GOX",
    "jaipur": "JAI",
    "ahmedabad": "AMD",
    "sardar vallabhbhai patel": "AMD",
    "pune": "PNQ",
    "kochi": "COK",
    "cochin": "COK",
    "varanasi": "VNS",
    "amritsar": "ATQ",
    "srinagar": "SXR",
    "lucknow": "LKO",
    "chandigarh": "IXC",
    "jodhpur": "JDH",
    # Americas
    "new york": "JFK",
    "nyc": "JFK",
    "john f kennedy": "JFK",
    "jfk": "JFK",
    "newark": "EWR",
    "la guardia": "LGA",
    "los angeles": "LAX",
    "san francisco": "SFO",
    "chicago": "ORD",
    "o'hare": "ORD",
    "miami": "MIA",
    "toronto": "YYZ",
    "vancouver": "YVR",
    # Europe
    "london": "LHR",
    "heathrow": "LHR",
    "gatwick": "LGW",
    "paris": "CDG",
    "charles de gaulle": "CDG",
    "orly": "ORY",
    "rome": "FCO",
    "fiumicino": "FCO",
    "frankfurt": "FRA",
    "amsterdam": "AMS",
    "schiphol": "AMS",
    "madrid": "MAD",
    "barcelona": "BCN",
    "zurich": "ZRH",
    "vienna": "VIE",
    "milan": "MXP",
    # Middle East & Asia
    "dubai": "DXB",
    "abu dhabi": "AUH",
    "doha": "DOH",
    "tokyo": "HND",
    "haneda": "HND",
    "narita": "NRT",
    "singapore": "SIN",
    "changi": "SIN",
    "bangkok": "BKK",
    "suvarnabhumi": "BKK",
    "kuala lumpur": "KUL",
    "bali": "DPS",
    "denpasar": "DPS",
    "seoul": "ICN",
    "incheon": "ICN",
    "hong kong": "HKG",
    "istanbul": "IST",
}


def parse_natural_date(text: str) -> Tuple[Optional[str], str]:
    """
    Extracts a departure date from text in formats:
    - '2026-12-08'
    - '8th of Dec', '8th Dec', '8 December 2026'
    - 'Dec 8th', 'December 8'
    Returns (extracted_date_YYYY_MM_DD, text_with_date_removed).
    """
    clean_text = text.strip()

    # 1. ISO format: YYYY-MM-DD
    iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", clean_text)
    if iso:
        d = iso.group(1)
        cleaned = clean_text[:iso.start()] + clean_text[iso.end():]
        return d, cleaned.strip()

    # 2. '8th of Dec', '8th Dec', '8 December 2026'
    m1 = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-zA-Z]+)(?:\s+(\d{4}))?\b", clean_text, re.I)
    if m1 and m1.group(2).lower() in MONTH_MAP:
        day = int(m1.group(1))
        month = MONTH_MAP[m1.group(2).lower()]
        year = int(m1.group(3)) if m1.group(3) else datetime.now().year
        d = f"{year}-{month:02d}-{day:02d}"
        cleaned = clean_text[:m1.start()] + clean_text[m1.end():]
        return d, cleaned.strip()

    # 3. 'Dec 8th', 'December 8'
    m2 = re.search(r"\b([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?\b", clean_text, re.I)
    if m2 and m2.group(1).lower() in MONTH_MAP:
        month = MONTH_MAP[m2.group(1).lower()]
        day = int(m2.group(2))
        year = int(m2.group(3)) if m2.group(3) else datetime.now().year
        d = f"{year}-{month:02d}-{day:02d}"
        cleaned = clean_text[:m2.start()] + clean_text[m2.end():]
        return d, cleaned.strip()

    return None, clean_text


def parse_flight_query(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parses a natural language query string to extract: (origin, destination, departure_date).
    """
    departure_date, route_text = parse_natural_date(text)
    # Remove dangling date prepositions
    route_text = re.sub(r"\b(?:on|for|date)\b", "", route_text, flags=re.I).strip()

    # Check for direct 2 IATA codes: 'DEL to LHR' or 'JFK -> CDG'
    iata_match = re.search(r"\b([A-Za-z]{3})\s*(?:to|-|->)\s*([A-Za-z]{3})\b", route_text)
    if iata_match:
        return iata_match.group(1).upper(), iata_match.group(2).upper(), departure_date

    # Pattern: 'from <origin> to <destination>'
    match_from_to = re.search(
        r"\bfrom\s+(.+?)\s+\bto\s+(.+?)(?:\s+(?:under|with|for|class)|[.!?]|$)",
        route_text,
        re.I,
    )
    if match_from_to:
        orig = match_from_to.group(1).strip()
        dest = match_from_to.group(2).strip()
        return orig, dest, departure_date

    # Pattern: 'to <destination> from <origin>'
    match_to_from = re.search(
        r"\bto\s+(.+?)\s+\bfrom\s+(.+?)(?:\s+(?:under|with|for|class)|[.!?]|$)",
        route_text,
        re.I,
    )
    if match_to_from:
        dest = match_to_from.group(1).strip()
        orig = match_to_from.group(2).strip()
        return orig, dest, departure_date

    # Pattern: 'to <destination>' (e.g. 'Plan a trip to US', 'flights to Udaipur')
    match_to = re.search(
        r"\bto\s+(.+?)(?:\s+(?:under|with|for|class)|[.!?]|$)",
        route_text,
        re.I,
    )
    if match_to:
        dest = match_to.group(1).strip()
        return None, dest, departure_date

    return None, None, departure_date


def resolve_to_iata(location: Optional[str]) -> str:
    """
    Resolves a location string (city, country, airport name, or IATA) to a 3-letter IATA code.
    If location is None or empty, resolves from DEFAULT_SOURCE in .env (or falls back to 'DEL').
    """
    if not location or not location.strip():
        env_default = (
            os.getenv("DEFAULT_SOURCE")
            or os.getenv("DEFAULT_FLIGHT_SOURCE")
            or os.getenv("DEFAULT_ORIGIN")
            or os.getenv("default_source")
            or "DEL"
        )
        location = env_default

    clean = location.strip().lower()

    # If it's already an uppercase 3-letter code
    if len(location.strip()) == 3 and location.strip().isalpha():
        return location.strip().upper()

    # Exact match in city/airport dictionary
    if clean in CITY_AIRPORT_TO_IATA:
        return CITY_AIRPORT_TO_IATA[clean]

    # Exact match in country default hubs
    if clean in COUNTRY_DEFAULT_HUBS:
        return COUNTRY_DEFAULT_HUBS[clean]

    # Substring / partial match for city or airport
    for name, iata in CITY_AIRPORT_TO_IATA.items():
        if name in clean or clean in name:
            return iata

    # Substring match for countries
    for country, iata in COUNTRY_DEFAULT_HUBS.items():
        if country in clean or clean in country:
            return iata

    # Fallback to uppercase representation if 3 letters
    if len(clean) == 3 and clean.isalpha():
        return clean.upper()

    # Default to DEL if unrecognized
    return "DEL"


class FlightSearchInput(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description="Natural language flight query string (e.g. 'Plan a Trip to US on 8th of Dec', 'flights from Delhi to Udaipur on 2026-09-25').",
    )
    destination: Optional[str] = Field(
        default=None,
        description="Arrival destination: city name (e.g. 'Udaipur', 'London'), country (e.g. 'France', 'US'), airport ('Heathrow'), or IATA ('UDR', 'LHR').",
    )
    origin: Optional[str] = Field(
        default=None,
        description="Departure origin. If not provided, automatically taken from DEFAULT_SOURCE in .env (defaults to 'DEL').",
    )
    departure_date: Optional[Union[str, int]] = Field(
        default=None,
        description="Optional departure date (e.g. '2026-12-08', '8th of Dec').",
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of flight options to return (top 10 by default).",
    )


def _get_api_key() -> Optional[str]:
    """Retrieve AviationStack API key from environment variables."""
    return (
        os.getenv("Aviationstack_API_key")
        or os.getenv("AVIATIONSTACK_API_KEY")
        or os.getenv("aviationstack_api_key")
    )


def _format_projected_time(time_iso: Optional[str], target_date: Optional[str]) -> str:
    """Projects scheduled flight times onto the target date."""
    if not time_iso:
        return f"{target_date} Scheduled" if target_date else "Scheduled"
    
    # Extract time component HH:MM
    time_part = "12:00"
    if "T" in time_iso:
        time_part = time_iso.split("T")[1][:5]
    elif len(time_iso) >= 16:
        time_part = time_iso[11:16]

    if target_date:
        return f"{target_date} {time_part}"
    
    return time_iso.replace("T", " ").split("+")[0][:16] if "T" in time_iso else time_iso


@tool(args_schema=FlightSearchInput)
def search_flights(
    query: Optional[str] = None,
    destination: Optional[str] = None,
    origin: Optional[str] = None,
    departure_date: Optional[Union[str, int]] = None,
    max_results: int = 10,
) -> str:
    """
    Search available flights from origin to destination using the AviationStack API.
    Can accept either a natural language query string (e.g. 'Plan a Trip to US on 8th of Dec')
    or structured parameters (origin, destination, departure_date).
    Supports dates like '8th of Dec' or '2026-12-08'.
    """
    # Normalize departure_date if given as int (e.g. unquoted 2026-12-08 = 2006)
    if isinstance(departure_date, int):
        departure_date = None
    elif isinstance(departure_date, str):
        parsed_d, _ = parse_natural_date(departure_date)
        departure_date = parsed_d or departure_date

    # If a natural language query string was provided, parse route and date from it
    if query:
        parsed_orig, parsed_dest, parsed_date = parse_flight_query(query)
        if parsed_orig and not origin:
            origin = parsed_orig
        if parsed_dest and not destination:
            destination = parsed_dest
        if parsed_date and not departure_date:
            departure_date = parsed_date

    # If destination is still not resolved, check if query itself was just a destination name
    if not destination and query:
        destination = query.strip()

    dep_iata = resolve_to_iata(origin)
    arr_iata = resolve_to_iata(destination)

    api_key = _get_api_key()
    if not api_key:
        return (
            "⚠️ AviationStack API key not found in .env.\n"
            "Please configure 'Aviationstack_API_key=your_key' in your .env file."
        )

    # Base parameters for AviationStack (Free tier uses HTTP)
    params: Dict[str, Any] = {
        "access_key": api_key,
        "dep_iata": dep_iata,
        "arr_iata": arr_iata,
        "limit": min(max_results, 10),
    }

    if departure_date:
        params["flight_date"] = departure_date

    try:
        response = requests.get(AVIATIONSTACK_BASE_URL, params=params, timeout=12)
        data = response.json()

        # Handle AviationStack Free-Tier date restriction:
        # The free plan restricts the 'flight_date' parameter (code: 'function_access_restricted').
        # If restricted, retry without 'flight_date' to fetch the recurring scheduled timetable for this route,
        # then project the operating flight times onto the user's requested date.
        date_restricted = False
        if "error" in data:
            err_code = data["error"].get("code")
            if err_code == "function_access_restricted" and "flight_date" in params:
                date_restricted = True
                del params["flight_date"]
                response = requests.get(AVIATIONSTACK_BASE_URL, params=params, timeout=12)
                data = response.json()

        if "error" in data:
            error_info = data["error"].get("message") or data["error"].get("info") or str(data["error"])
            return f"AviationStack API Error: {error_info}"

        flights = data.get("data", [])

        if not flights:
            return (
                f"No scheduled flights found between {dep_iata} and {arr_iata}"
                + (f" for {departure_date}" if departure_date else "")
                + ".\n💡 Tip: Try checking flights between major hub connections or without strict date filters."
            )

        # Cap results to top 10
        top_flights = flights[:10]

        header = f"✈️ Top {len(top_flights)} Flight Options: {dep_iata} ➔ {arr_iata}"
        if departure_date:
            header += f" on {departure_date}"
        header += ":\n"

        if date_restricted and departure_date:
            header += (
                f"📌 Note: Active recurring timetable from AviationStack shown for requested date ({departure_date}).\n"
            )

        lines = [header]

        for idx, f in enumerate(top_flights, 1):
            airline = f.get("airline", {}).get("name") or "Airline"
            flight_num = f.get("flight", {}).get("iata") or f.get("flight", {}).get("number") or "N/A"
            status = (f.get("flight_status") or "scheduled").capitalize()

            # Departure details
            dep = f.get("departure", {})
            raw_dep_time = dep.get("scheduled") or dep.get("estimated")
            dep_fmt = _format_projected_time(raw_dep_time, departure_date)
            dep_terminal = f" (Terminal {dep.get('terminal')})" if dep.get("terminal") else ""
            dep_gate = f" Gate {dep.get('gate')}" if dep.get("gate") else ""

            # Arrival details
            arr = f.get("arrival", {})
            raw_arr_time = arr.get("scheduled") or arr.get("estimated")
            arr_fmt = _format_projected_time(raw_arr_time, departure_date)
            arr_terminal = f" (Terminal {arr.get('terminal')})" if arr.get("terminal") else ""
            arr_gate = f" Gate {arr.get('gate')}" if arr.get("gate") else ""

            # Delay notice
            delay = dep.get("delay")
            delay_str = f" | ⚠️ Delay: {delay} min" if delay else ""

            lines.append(
                f"{idx}. {airline} [{flight_num}] - Status: {status}{delay_str}\n"
                f"   🛫 Departure: {dep.get('airport', dep_iata)}{dep_terminal}{dep_gate} at {dep_fmt}\n"
                f"   🛬 Arrival:   {arr.get('airport', arr_iata)}{arr_terminal}{arr_gate} at {arr_fmt}\n"
            )

        return "\n".join(lines)

    except requests.exceptions.Timeout:
        return f"Request to AviationStack timed out while searching flights between {dep_iata} and {arr_iata}."
    except Exception as e:
        return f"Error retrieving flights from AviationStack: {e}"


@tool
def get_flight_details(flight_number: str) -> str:
    """
    Retrieve real-time flight details, terminal, gate, and schedule using the flight IATA designation.

    Args:
        flight_number: The flight number/designation (e.g., '6E322', 'AI1779', 'AA293').
    """
    fn = flight_number.strip().upper()
    api_key = _get_api_key()

    if not api_key:
        return (
            f"Flight {fn} Status:\n"
            f"- Status: Scheduled\n"
            f"- Note: Configure Aviationstack_API_key in .env for live real-time flight telemetry."
        )

    params = {
        "access_key": api_key,
        "flight_iata": fn,
        "limit": 1,
    }

    try:
        response = requests.get(AVIATIONSTACK_BASE_URL, params=params, timeout=10)
        data = response.json()
        flights = data.get("data", [])

        if not flights:
            return f"No live telemetry found for flight '{fn}'. Please verify the flight IATA code."

        f = flights[0]
        status = (f.get("flight_status") or "scheduled").capitalize()
        dep = f.get("departure", {})
        arr = f.get("arrival", {})

        return (
            f"✈️ Live Telemetry for Flight {fn} ({f.get('airline', {}).get('name', 'Airline')}):\n"
            f"- Status: {status}\n"
            f"- Route: {dep.get('airport', 'Origin')} ({dep.get('iata')}) ➔ {arr.get('airport', 'Destination')} ({arr.get('iata')})\n"
            f"- Departure: {dep.get('scheduled', 'N/A')} (Terminal: {dep.get('terminal', 'N/A')}, Gate: {dep.get('gate', 'N/A')})\n"
            f"- Arrival: {arr.get('scheduled', 'N/A')} (Terminal: {arr.get('terminal', 'N/A')}, Gate: {arr.get('gate', 'N/A')})\n"
            f"- Delay: {dep.get('delay') or 0} minutes"
        )
    except Exception as e:
        return f"Error fetching details for flight {fn}: {e}"
