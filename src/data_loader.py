"""
Data loader module for the EquiCity platform.
Loads and preprocesses the synthetic complaints CSV.
"""

import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "complaints.csv")


def load_complaints(path: str = None) -> pd.DataFrame:
    """Load complaints CSV and parse dates."""
    if path is None:
        path = DATA_PATH

    df = pd.read_csv(path)

    # Parse dates
    df["date_reported"] = pd.to_datetime(df["date_reported"])
    df["date_resolved"] = pd.to_datetime(df["date_resolved"], errors="coerce")

    # Ensure resolution_days is numeric
    df["resolution_days"] = pd.to_numeric(df["resolution_days"], errors="coerce")

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return high-level summary statistics."""
    resolved = df[df["status"] == "resolved"]
    pending = df[df["status"].isin(["pending", "in_progress"])]

    return {
        "total_complaints": len(df),
        "resolved": len(resolved),
        "pending": len(pending),
        "resolution_rate": f"{len(resolved) / len(df) * 100:.1f}%",
        "avg_resolution_days": f"{resolved['resolution_days'].mean():.1f}",
        "categories": df["category"].nunique(),
        "neighborhoods": df["neighborhood"].nunique(),
        "date_range": f"{df['date_reported'].min().strftime('%b %d, %Y')} to {df['date_reported'].max().strftime('%b %d, %Y')}",
    }


def filter_complaints(
    df: pd.DataFrame,
    neighborhood: str = None,
    category: str = None,
    status: str = None,
    start_date: str = None,
    end_date: str = None,
    min_severity: int = None,
) -> pd.DataFrame:
    """Filter complaints DataFrame based on criteria."""
    filtered = df.copy()

    if neighborhood:
        filtered = filtered[filtered["neighborhood"].str.lower() == neighborhood.lower()]
    if category:
        filtered = filtered[filtered["category"].str.lower() == category.lower()]
    if status:
        filtered = filtered[filtered["status"].str.lower() == status.lower()]
    if start_date:
        filtered = filtered[filtered["date_reported"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["date_reported"] <= pd.to_datetime(end_date)]
    if min_severity:
        filtered = filtered[filtered["severity"] >= min_severity]

    return filtered
