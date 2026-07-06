"""
Gemini AI Agent module for the EquiCity platform.
Uses the NEW google-genai SDK with function calling.
"""

from google import genai
from google.genai import types
import json
import pandas as pd
from src.data_loader import load_complaints, filter_complaints, get_summary_stats
from src.fairness import compute_fairness_scores, get_disparity_report, get_recommendations


# --- Tool function definitions (what Gemini can call) ---

def tool_query_complaints(
    neighborhood: str = "",
    category: str = "",
    status: str = "",
    min_severity: int = 0,
) -> str:
    """
    Query and filter citizen complaints data.
    Use this to look up complaints by neighborhood, category, status, or severity.

    Args:
        neighborhood: Filter by neighborhood name (e.g., "Sector 1", "Sector 2")
        category: Filter by complaint category (e.g., "pothole", "streetlight", "garbage", "water_leak", "noise", "road_damage")
        status: Filter by status ("resolved", "pending", "in_progress")
        min_severity: Minimum severity level (1-5)
    """
    df = load_complaints()
    filtered = filter_complaints(
        df,
        neighborhood=neighborhood if neighborhood else None,
        category=category if category else None,
        status=status if status else None,
        min_severity=min_severity if min_severity else None,
    )

    if len(filtered) == 0:
        return json.dumps({"message": "No complaints found matching the criteria."})

    resolved = filtered[filtered["status"] == "resolved"]
    result = {
        "total_matching": len(filtered),
        "resolved": len(resolved),
        "pending": len(filtered[filtered["status"].isin(["pending", "in_progress"])]),
        "avg_resolution_days": round(resolved["resolution_days"].mean(), 1) if len(resolved) > 0 else None,
        "avg_severity": round(filtered["severity"].mean(), 1),
        "by_neighborhood": filtered.groupby("neighborhood").size().to_dict(),
        "by_category": filtered.groupby("category").size().to_dict(),
        "sample_complaints": filtered.head(5)[
            ["id", "category", "neighborhood", "severity", "resolution_days", "status", "citizen_feedback"]
        ].to_dict(orient="records"),
    }
    return json.dumps(result, default=str)


def tool_compute_fairness(category: str = "") -> str:
    """
    Compute fairness scores across all neighborhoods.
    Returns a fairness score (0-100) for each neighborhood, showing whether
    resource allocation and complaint resolution is equitable.

    Args:
        category: Optional - filter fairness analysis to a specific complaint category
    """
    df = load_complaints()
    result = compute_fairness_scores(df, category=category if category else None)
    return json.dumps(result, default=str)


def tool_get_disparity_report() -> str:
    """
    Generate a disparity report comparing complaint resolution across
    income bands (low, medium, high income neighborhoods).
    This reveals systemic bias in resource allocation.
    """
    df = load_complaints()
    result = get_disparity_report(df)
    return json.dumps(result, default=str)


def tool_get_recommendations(neighborhood: str = "") -> str:
    """
    Generate prioritized action recommendations to improve equity.
    Returns specific, data-backed actions the city should take.

    Args:
        neighborhood: Optional - focus recommendations on a specific neighborhood
    """
    df = load_complaints()
    result = get_recommendations(df, neighborhood=neighborhood if neighborhood else None)
    return json.dumps(result, default=str)


def tool_get_summary_stats() -> str:
    """
    Get high-level summary statistics about all complaints in the system.
    """
    df = load_complaints()
    result = get_summary_stats(df)
    return json.dumps(result, default=str)


# --- Map function names to actual functions ---
TOOL_FUNCTIONS = {
    "tool_query_complaints": tool_query_complaints,
    "tool_compute_fairness": tool_compute_fairness,
    "tool_get_disparity_report": tool_get_disparity_report,
    "tool_get_recommendations": tool_get_recommendations,
    "tool_get_summary_stats": tool_get_summary_stats,
}

# --- System prompt ---
SYSTEM_PROMPT = """You are EquiCity AI, an Equity Intelligence Analyst for a city government. Your job is to analyze citizen complaint data and identify fairness disparities in how different neighborhoods are treated.

CRITICAL RULES:
1. Always cite specific numbers (resolution days, severity scores, complaint counts).
2. When you find a disparity, state the exact ratio (e.g., "3.2x longer").
3. Be direct and actionable — don't just describe data, recommend what to do.
4. Frame findings in terms of equity and fairness, not just efficiency.
5. When neighborhoods are disadvantaged, name them and explain why.
6. Use the available tools to back every claim with data.

AVAILABLE DATA:
- Citizen complaints from 6 neighborhoods (Sectors 1-6) across 6 categories
- Each complaint has: category, neighborhood, income band, severity (1-5), resolution time, status
- Neighborhoods span low, medium, and high income bands

When asked about fairness or equity, always use the compute_fairness and disparity_report tools.
When asked what to do or for recommendations, use the get_recommendations tool.
When asked about specific complaints or neighborhoods, use the query_complaints tool.

Keep responses concise but data-rich. Use bullet points for clarity."""


def create_agent(api_key: str):
    """Create and return a configured Gemini client."""
    client = genai.Client(api_key=api_key)
    return client


def run_agent_query(client, chat_history: list, user_query: str) -> tuple:
    """
    Send a user query to the Gemini agent, handle function calls,
    and return the final response.

    Returns:
        (response_text, functions_called, data_cited, updated_history)
    """
    functions_called = []
    data_cited = {}

    # Add user message to history
    chat_history.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_query)]
    ))

    # Define tools
    tools = [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="tool_query_complaints",
            description="Query and filter citizen complaints data by neighborhood, category, status, or severity.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "neighborhood": types.Schema(type="STRING", description="Filter by neighborhood (e.g., 'Sector 1')"),
                    "category": types.Schema(type="STRING", description="Filter by category (e.g., 'pothole', 'streetlight', 'garbage', 'water_leak', 'noise', 'road_damage')"),
                    "status": types.Schema(type="STRING", description="Filter by status ('resolved', 'pending', 'in_progress')"),
                    "min_severity": types.Schema(type="INTEGER", description="Minimum severity level (1-5)"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="tool_compute_fairness",
            description="Compute fairness scores (0-100) across all neighborhoods showing whether complaint resolution is equitable.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "category": types.Schema(type="STRING", description="Optional category filter"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="tool_get_disparity_report",
            description="Generate a disparity report comparing complaint resolution across income bands (low, medium, high).",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="tool_get_recommendations",
            description="Generate prioritized action recommendations to improve equity.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "neighborhood": types.Schema(type="STRING", description="Optional neighborhood filter"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="tool_get_summary_stats",
            description="Get high-level summary statistics about all complaints.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
    ])]

    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        import time

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=chat_history,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=tools,
                    ),
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(30)
                    continue
                raise e

        # Check if there are function calls
        has_function_call = False
        function_response_parts = []

        for part in response.candidates[0].content.parts:
            if part.function_call:
                has_function_call = True
                fn_name = part.function_call.name
                fn_args = dict(part.function_call.args) if part.function_call.args else {}

                functions_called.append({"function": fn_name, "args": fn_args})

                # Execute the function
                if fn_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[fn_name](**fn_args)
                    data_cited[fn_name] = json.loads(result)
                else:
                    result = json.dumps({"error": f"Unknown function: {fn_name}"})

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": result}
                    )
                )

        if has_function_call:
            # Add model's function call to history
            chat_history.append(response.candidates[0].content)
            # Add function results to history
            chat_history.append(types.Content(
                role="user",
                parts=function_response_parts
            ))
        else:
            # No function calls — we have the final text response
            break

    # Extract final text response
    response_text = ""
    for part in response.candidates[0].content.parts:
        if part.text:
            response_text += part.text

    # Add assistant response to history
    chat_history.append(types.Content(
        role="model",
        parts=[types.Part.from_text(text=response_text)]
    ))

    if not response_text:
        response_text = "I analyzed the data but couldn't generate a response. Please try rephrasing your question."

    return response_text, functions_called, data_cited, chat_history
