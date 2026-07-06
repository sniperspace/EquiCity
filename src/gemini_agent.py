"""
Gemini/Groq AI Agent module for the EquiCity platform.
Uses Groq (free, fast) with Llama model + tool calling.
"""

from groq import Groq
import json
import pandas as pd
from src.data_loader import load_complaints, filter_complaints, get_summary_stats
from src.fairness import compute_fairness_scores, get_disparity_report, get_recommendations


# --- Tool function definitions ---

def tool_query_complaints(neighborhood="", category="", status="", min_severity=0):
    df = load_complaints()
    min_severity = int(min_severity) if min_severity else 0
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


def tool_compute_fairness(category=""):
    df = load_complaints()
    result = compute_fairness_scores(df, category=category if category else None)
    return json.dumps(result, default=str)


def tool_get_disparity_report():
    df = load_complaints()
    result = get_disparity_report(df)
    return json.dumps(result, default=str)


def tool_get_recommendations(neighborhood=""):
    df = load_complaints()
    result = get_recommendations(df, neighborhood=neighborhood if neighborhood else None)
    return json.dumps(result, default=str)


def tool_get_summary_stats():
    df = load_complaints()
    result = get_summary_stats(df)
    return json.dumps(result, default=str)


TOOL_FUNCTIONS = {
    "tool_query_complaints": tool_query_complaints,
    "tool_compute_fairness": tool_compute_fairness,
    "tool_get_disparity_report": tool_get_disparity_report,
    "tool_get_recommendations": tool_get_recommendations,
    "tool_get_summary_stats": tool_get_summary_stats,
}

# OpenAI-style tool definitions for Groq
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_query_complaints",
            "description": "Query and filter citizen complaints data by neighborhood, category, status, or severity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "neighborhood": {"type": "string", "description": "Filter by neighborhood (e.g., 'Sector 1')"},
                    "category": {"type": "string", "description": "Filter by category (pothole, streetlight, garbage, water_leak, noise, road_damage)"},
                    "status": {"type": "string", "description": "Filter by status (resolved, pending, in_progress)"},
                    "min_severity": {"type": "string", "description": "Minimum severity level (1-5)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_compute_fairness",
            "description": "Compute fairness scores (0-100) across all neighborhoods showing whether complaint resolution is equitable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Optional category filter"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_get_disparity_report",
            "description": "Generate a disparity report comparing complaint resolution across income bands (low, medium, high).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_get_recommendations",
            "description": "Generate prioritized action recommendations to improve equity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "neighborhood": {"type": "string", "description": "Optional neighborhood filter"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_get_summary_stats",
            "description": "Get high-level summary statistics about all complaints.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = """You are EquiCity AI, an Equity Intelligence Analyst for a city government. Your job is to analyze citizen complaint data and identify fairness disparities in how different neighborhoods are treated.

RULES:
1. Always cite specific numbers (resolution days, severity scores, complaint counts).
2. When you find a disparity, state the exact ratio (e.g., "3.2x longer").
3. Be direct and actionable — recommend what to do.
4. Frame findings in terms of equity and fairness.
5. Use the available tools to back every claim with data.
6. Keep responses concise with bullet points."""


def create_agent(api_key: str):
    """Create and return a Groq client."""
    client = Groq(api_key=api_key)
    return client


def run_agent_query(client, chat_history: list, user_query: str) -> tuple:
    """
    Send a user query, handle tool calls, return response.
    Returns: (response_text, functions_called, data_cited, updated_history)
    """
    functions_called = []
    data_cited = {}

    chat_history.append({"role": "user", "content": user_query})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history

    max_iterations = 5
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in msg.tool_calls
                ]
            })

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                functions_called.append({"function": fn_name, "args": fn_args})

                if fn_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[fn_name](**fn_args)
                    data_cited[fn_name] = json.loads(result)
                else:
                    result = json.dumps({"error": f"Unknown function: {fn_name}"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            break

    response_text = msg.content or "I analyzed the data but couldn't generate a response. Please try rephrasing."

    chat_history.append({"role": "assistant", "content": response_text})

    return response_text, functions_called, data_cited, chat_history
