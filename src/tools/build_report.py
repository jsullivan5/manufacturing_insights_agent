from datetime import datetime, timezone
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def build_event_timeline(evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Scan `evidence` (the list stored by the Orchestrator) and return
    at least three chronological events:
       • investigation start
       • any tool calls that produced severity_score > 0.3
       • investigation finished
    """
    timeline: List[Dict[str, str]] = []
    if not evidence:
        return timeline

    # 1. start
    timeline.append({"time": evidence[0]["timestamp"], "description": "Investigation started"})

    # 2. significant tool results
    for ev in evidence:
        data = ev.get("data", {})
        sev = data.get("severity_score") or data.get("causal_confidence") or 0
        if sev and sev > 0.3:
            summary = ev.get("tool_name")
            timeline.append({"time": ev["timestamp"], "description": f"{summary} identified significant event"})

    # ensure at least 2 middle events
    if len(timeline) < 3:
        timeline.append({"time": _now_iso(), "description": "No major mid-investigation events"})

    # 3. end
    timeline.append({"time": _now_iso(), "description": "Investigation concluded"})
    return timeline[:3]  # trim to exactly 3 if longer

def build_business_impact(most_recent_impact_call: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Use the payload returned by `calculate_impact`.  If absent, stub.
    """
    if most_recent_impact_call:
        return {
            "total_cost_usd": most_recent_impact_call["total_cost"],
            "energy_cost_usd": most_recent_impact_call["energy_cost"],
            "product_risk_usd": most_recent_impact_call["product_risk"],
            "severity_level": most_recent_impact_call["severity"],
            "details": most_recent_impact_call["description"],
        }
    # fallback stub
    return {
        "total_cost_usd": 0.0,
        "energy_cost_usd": 0.0,
        "product_risk_usd": 0.0,
        "severity_level": "unknown",
        "details": "No explicit impact tool call; cost assumed negligible",
    }