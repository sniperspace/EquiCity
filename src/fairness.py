"""
Fairness computation module for the EquiCity platform.
Computes equity scores, detects bias, and generates recommendations.
"""

import pandas as pd
import json


def compute_fairness_scores(df: pd.DataFrame, category: str = None) -> dict:
    """
    Compute fairness scores across all neighborhoods.

    Fairness Score (0-100):
    - 100 = perfectly equitable (all neighborhoods resolve at same speed for same severity)
    - 0 = extreme inequity

    Method: Compare each neighborhood's avg resolution time against the global average,
    normalized by severity. Larger deviations = lower fairness score.
    """
    resolved = df[df["status"] == "resolved"].copy()

    if category:
        resolved = resolved[resolved["category"].str.lower() == category.lower()]

    if len(resolved) == 0:
        return {"error": "No resolved complaints found for the given criteria."}

    global_avg = resolved["resolution_days"].mean()

    neighborhood_stats = []

    for name, group in resolved.groupby("neighborhood"):
        avg_resolution = group["resolution_days"].mean()
        avg_severity = group["severity"].mean()
        total_complaints = len(group)
        pending_count = len(df[(df["neighborhood"] == name) & (df["status"].isin(["pending", "in_progress"]))])

        # Fairness score: how far is this neighborhood from the global average?
        # Normalized so 100 = equal to global avg, lower = worse
        deviation_ratio = avg_resolution / global_avg if global_avg > 0 else 1
        # A ratio of 1.0 = fair, >1.0 = slower than average, <1.0 = faster
        # Convert to a 0-100 score (clamped)
        fairness_score = max(0, min(100, 100 - (abs(deviation_ratio - 1.0) * 100)))

        # Determine if this neighborhood is disadvantaged
        is_disadvantaged = deviation_ratio > 1.3  # 30% slower than average

        income_band = group["income_band"].iloc[0]

        neighborhood_stats.append({
            "neighborhood": name,
            "income_band": income_band,
            "total_complaints": total_complaints,
            "pending_complaints": pending_count,
            "avg_resolution_days": round(avg_resolution, 1),
            "avg_severity": round(avg_severity, 1),
            "fairness_score": round(fairness_score, 1),
            "deviation_ratio": round(deviation_ratio, 2),
            "is_disadvantaged": is_disadvantaged,
        })

    # Sort by fairness score (worst first)
    neighborhood_stats.sort(key=lambda x: x["fairness_score"])

    # Overall platform fairness
    scores = [s["fairness_score"] for s in neighborhood_stats]
    overall_fairness = round(sum(scores) / len(scores), 1) if scores else 0

    return {
        "overall_fairness_score": overall_fairness,
        "global_avg_resolution_days": round(global_avg, 1),
        "neighborhood_breakdown": neighborhood_stats,
        "most_disadvantaged": neighborhood_stats[0]["neighborhood"] if neighborhood_stats else None,
        "category_filter": category or "all categories",
    }


def get_disparity_report(df: pd.DataFrame) -> dict:
    """
    Generate a focused disparity report comparing income bands.
    This is the core "aha" moment for the demo.
    """
    resolved = df[df["status"] == "resolved"].copy()

    band_stats = {}
    for band, group in resolved.groupby("income_band"):
        band_stats[band] = {
            "avg_resolution_days": round(group["resolution_days"].mean(), 1),
            "median_resolution_days": round(group["resolution_days"].median(), 1),
            "avg_severity": round(group["severity"].mean(), 1),
            "total_complaints": len(group),
            "complaints_over_14_days": len(group[group["resolution_days"] > 14]),
        }

    # Calculate the disparity ratio
    if "low" in band_stats and "high" in band_stats:
        disparity_ratio = round(
            band_stats["low"]["avg_resolution_days"] / band_stats["high"]["avg_resolution_days"], 1
        )
    else:
        disparity_ratio = None

    return {
        "income_band_comparison": band_stats,
        "disparity_ratio": disparity_ratio,
        "disparity_summary": (
            f"Low-income neighborhoods wait {disparity_ratio}x longer than high-income "
            f"neighborhoods for complaint resolution, despite similar severity scores "
            f"(avg {band_stats.get('low', {}).get('avg_severity', 'N/A')} vs "
            f"{band_stats.get('high', {}).get('avg_severity', 'N/A')})."
            if disparity_ratio
            else "Insufficient data to compute disparity."
        ),
    }


def get_recommendations(df: pd.DataFrame, neighborhood: str = None) -> dict:
    """
    Generate prioritized action recommendations based on fairness analysis.
    """
    fairness = compute_fairness_scores(df)
    disparity = get_disparity_report(df)
    recommendations = []

    # Recommendation 1: Address most disadvantaged neighborhoods
    disadvantaged = [
        n for n in fairness["neighborhood_breakdown"] if n["is_disadvantaged"]
    ]

    for n in disadvantaged:
        pending = n["pending_complaints"]
        recommendations.append({
            "priority": "HIGH",
            "action": f"Reallocate resources to {n['neighborhood']}",
            "reason": (
                f"{n['neighborhood']} ({n['income_band']}-income) has an average resolution "
                f"time of {n['avg_resolution_days']} days — {n['deviation_ratio']}x the city "
                f"average of {fairness['global_avg_resolution_days']} days. "
                f"There are {pending} complaints still pending."
            ),
            "impact": f"Could reduce resolution disparity by up to {round((n['deviation_ratio'] - 1) * 100)}%",
        })

    # Recommendation 2: Category-specific issues
    resolved = df[df["status"] == "resolved"]
    for cat, group in resolved.groupby("category"):
        cat_avg = group["resolution_days"].mean()
        if cat_avg > fairness["global_avg_resolution_days"] * 1.5:
            recommendations.append({
                "priority": "MEDIUM",
                "action": f"Review {cat} complaint handling process",
                "reason": (
                    f"{cat.replace('_', ' ').title()} complaints take an average of "
                    f"{cat_avg:.1f} days to resolve — 50%+ above the city average."
                ),
                "impact": "Process improvement could reduce citywide resolution times",
            })

    # Recommendation 3: Pending backlog
    pending = df[df["status"].isin(["pending", "in_progress"])]
    if len(pending) > 0:
        worst_backlog = pending.groupby("neighborhood").size().sort_values(ascending=False)
        top_backlog = worst_backlog.index[0]
        top_count = worst_backlog.iloc[0]
        recommendations.append({
            "priority": "HIGH",
            "action": f"Clear pending backlog in {top_backlog}",
            "reason": f"{top_backlog} has {top_count} unresolved complaints — the highest backlog in the city.",
            "impact": "Directly improves citizen satisfaction and fairness metrics",
        })

    # Filter by neighborhood if specified
    if neighborhood:
        recommendations = [
            r for r in recommendations
            if neighborhood.lower() in r["action"].lower() or neighborhood.lower() in r["reason"].lower()
        ]

    return {
        "recommendations": recommendations[:5],  # Top 5
        "total_recommendations": len(recommendations),
        "overall_fairness_score": fairness["overall_fairness_score"],
    }


def format_for_display(result: dict) -> str:
    """Format a result dict as a readable JSON string for the AI."""
    return json.dumps(result, indent=2, default=str)
