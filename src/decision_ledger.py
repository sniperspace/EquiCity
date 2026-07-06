"""
Decision Ledger module for the EquiCity platform.
Logs every AI recommendation with full audit trail.
"""

from datetime import datetime


class DecisionLedger:
    """
    Maintains an auditable log of all AI decisions and recommendations.
    Each entry records: what was asked, what data was used, what was recommended,
    and which neighborhoods were affected.
    """

    def __init__(self):
        self.entries = []

    def log_decision(
        self,
        query: str,
        functions_called: list,
        data_points_cited: dict,
        ai_response: str,
        neighborhoods_affected: list = None,
    ):
        """Log a single AI decision to the ledger."""
        entry = {
            "id": len(self.entries) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "functions_called": functions_called,
            "data_points_cited": data_points_cited,
            "ai_response": ai_response[:500],  # Truncate for display
            "neighborhoods_affected": neighborhoods_affected or [],
        }
        self.entries.append(entry)
        return entry

    def get_entries(self, limit: int = None) -> list:
        """Return ledger entries, newest first."""
        entries = list(reversed(self.entries))
        if limit:
            entries = entries[:limit]
        return entries

    def get_entry_count(self) -> int:
        """Return total number of logged decisions."""
        return len(self.entries)

    def get_neighborhoods_summary(self) -> dict:
        """
        Summarize how many times each neighborhood has been
        referenced in AI decisions — shows attention distribution.
        """
        counts = {}
        for entry in self.entries:
            for n in entry.get("neighborhoods_affected", []):
                counts[n] = counts.get(n, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def clear(self):
        """Clear all ledger entries."""
        self.entries = []
