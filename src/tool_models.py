from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from pydantic import model_validator
from datetime import datetime, timezone

class BaseToolArguments(BaseModel):
    """Base model for all tool arguments."""
    # schema_version for the OpenAI function schema itself will be injected by the orchestrator
    # from settings. This keeps the Pydantic models focused on argument structure.
    pass

class FindInterestingWindowArgs(BaseToolArguments):
    """
    Purpose: Identifies the most relevant time window for investigation based on variance or significant events in a primary tag's data.
    Returns: A dictionary containing the 'start_time' and 'end_time' of the most interesting window, along with the strategy used and a significance score.
    """
    primary_tag: str = Field(..., description="The primary manufacturing tag to analyze for finding an interesting window (e.g., 'FREEZER01.TEMP.INTERNAL_C').")
    window_hours: int = Field(default=2, description="The desired duration of the interesting window in hours.")
    # start_time and end_time for the search range are optional; if not provided, tools should use data_loader.get_data_time_range()
    start_time: Optional[str] = Field(None, description="Optional start ISO 8601 UTC timestamp for the overall search range. If None, uses the beginning of available data for the primary_tag.")
    end_time: Optional[str] = Field(None, description="Optional end ISO 8601 UTC timestamp for the overall search range. If None, uses the end of available data for the primary_tag.")

class DetectNumericAnomaliesArgs(BaseToolArguments):
    """
    Purpose: Detects anomalies (spikes, drops) in numeric time-series data using Z-score analysis within a specified time window.
    Returns: A dictionary containing a list of detected anomalies, each with timestamp, value, Z-score, and description.
    """
    tag: str = Field(..., description="The numeric manufacturing tag to analyze for anomalies (e.g., 'FREEZER01.TEMP.INTERNAL_C').")
    start_time: str = Field(..., description="Start ISO 8601 UTC timestamp for the analysis window. Should come from a previously identified 'interesting window' or be explicitly set.")
    end_time: str = Field(..., description="End ISO 8601 UTC timestamp for the analysis window. Should come from a previously identified 'interesting window' or be explicitly set.")
    threshold: Optional[float] = Field(default=None, description="Z-score threshold for anomaly detection. If None, an adaptive threshold is used (e.g., 2.5 for temperatures, 3.0 for others).")

class DetectBinaryFlipsArgs(BaseToolArguments):
    """
    Purpose: Detects state changes (e.g., 0 to 1 or 1 to 0) in binary time-series data within a specified time window, and also identifies if the state was continuously high from the window's start.
    Returns: A dictionary containing a list of state changes, total high duration, and potentially a 'continuous_high_event' if the tag was high from the start for a minimum duration.
    """
    tag: str = Field(..., description="The binary manufacturing tag to analyze for state changes (e.g., 'FREEZER01.DOOR.STATUS').")
    start_time: str = Field(..., description="Start ISO 8601 UTC timestamp for the analysis window.")
    end_time: str = Field(..., description="End ISO 8601 UTC timestamp for the analysis window.")
    min_continuous_high_minutes: Optional[int] = Field(default=5, description="Minimum duration (in minutes) for a state to be considered continuously high from the window start if no initial flip is detected within the window.")

class TestCausalityArgs(BaseToolArguments):
    """
    Purpose: Tests for a causal relationship between two manufacturing tags within a specified time window, considering time lags.
    Returns: A dictionary with the best correlation coefficient, the corresponding time lag in minutes, and a causal confidence score.
    """
    cause_tag: str = Field(..., description="The manufacturing tag suspected to be the cause (e.g., 'FREEZER01.DOOR.STATUS').")
    effect_tag: str = Field(..., description="The manufacturing tag suspected to be affected (e.g., 'FREEZER01.TEMP.INTERNAL_C').")
    start_time: str = Field(..., description="Start ISO 8601 UTC timestamp for the analysis window.")
    end_time: str = Field(..., description="End ISO 8601 UTC timestamp for the analysis window.")
    max_lag_minutes: int = Field(default=15, description="Maximum time lag in minutes to test for a causal relationship. Keep this reasonable (e.g., 5-30 minutes for typical freezer events).")

class CalculateImpactArgs(BaseToolArguments):
    """
    Purpose: Calculates the estimated business impact (e.g., cost) of an operational event.
    Returns: A dictionary detailing energy cost, product risk, total cost, severity, and a description of the impact.
    """
    event_type: str = Field(..., description="Type of event (e.g., 'door_open', 'compressor_failure', 'temperature_spike').")
    duration_minutes: float = Field(..., description="Duration of the event in minutes.")
    severity: Optional[float] = Field(default=1.0, description="Severity multiplier for the event (0.0 to 1.0+).")
    # price_per_kwh is typically taken from settings, but can be overridden if needed for specific scenarios.
    price_per_kwh: Optional[float] = Field(None, description="Optional energy price in $/kWh. If None, system default is used.")

class CreateEventSequenceArgs(BaseToolArguments):
    """
    Purpose: Creates a chronological sequence of significant events (anomalies, state changes) across multiple related tags within a time window.
    Returns: A dictionary containing a list of events, each with timestamp, tag, value, event type, description, and severity.
    """
    primary_tag: str = Field(..., description="The main manufacturing tag of interest around which the sequence is built.")
    related_tags: List[str] = Field(..., description="A list of related manufacturing tag names to include in the event sequence.")
    start_time: str = Field(..., description="Start ISO 8601 UTC timestamp for the analysis window.")
    end_time: str = Field(..., description="End ISO 8601 UTC timestamp for the analysis window.")

# --- Finish Investigation Model ---
class FinishInvestigationArgs(BaseToolArguments):
    """
    Purpose: Called by the LLM to conclude the investigation and provide a final summary.
    Returns: This function doesn't return to the LLM; its arguments are the final output of the investigation.
    """
    summary_version: str = Field("0.1", frozen=True, description="Version of the summary data structure provided by the LLM.")
    root_cause_statement: str = Field(..., description="A concise (1-3 sentences) explanation of the primary root cause identified.")
    event_timeline_summary: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="A chronological list of key events. Each event: {'time': ISO_UTC_str, 'description': str}.")
    business_impact_summary: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Summary of business impact. Expected keys: 'total_cost_usd', 'energy_cost_usd', 'product_risk_usd', 'severity_level'.")
    recommendations: List[str] = Field(..., min_length=1, description="A list of 1-3 actionable recommendations.")
    final_confidence_score: float = Field(..., ge=0.0, le=1.0, description="The overall confidence score (0.0-1.0) LLM has in its findings.")

    @model_validator(mode='after')
    def populate_default_summaries_if_empty(cls, values: Any) -> Any:
        if isinstance(values, dict): # Pydantic v2 gives a dict for model_dump typically
            timeline = values.get('event_timeline_summary')
            impact = values.get('business_impact_summary')

            if not timeline: # Empty list or None
                values['event_timeline_summary'] = [
                    {"time": datetime.now(timezone.utc).isoformat(), "description": "No specific timeline events provided by LLM."}
                ]
            
            if not impact: # Empty dict or None
                values['business_impact_summary'] = {
                    "total_cost_usd": 0.0,
                    "energy_cost_usd": 0.0,
                    "product_risk_usd": 0.0,
                    "severity_level": "unknown",
                    "details": "No specific business impact provided by LLM."
                }
        # If `values` is the model instance itself (can happen depending on how validator is called or Pydantic version nuances)
        elif hasattr(values, 'event_timeline_summary') and hasattr(values, 'business_impact_summary'):
            if not values.event_timeline_summary:
                values.event_timeline_summary = [
                    {"time": datetime.now(timezone.utc).isoformat(), "description": "No specific timeline events provided by LLM."}
                ]
            if not values.business_impact_summary:
                values.business_impact_summary = {
                    "total_cost_usd": 0.0,
                    "energy_cost_usd": 0.0,
                    "product_risk_usd": 0.0,
                    "severity_level": "unknown",
                    "details": "No specific business impact provided by LLM."
                }
        return values

class ParseTimeRangeArgs(BaseModel):
    """
    Arguments for the parse_time_range tool.
    The LLM should extract potential start and end times from the user's query.
    Both start_time and end_time are optional; if the LLM cannot determine a precise one,
    it can omit it or provide a sensible default. Output must be ISO-8601 UTC.
    If a query is provided, the tool might be able to infer time from it.
    """
    start_time: Optional[str] = Field(None, description="Start time in ISO-8601 UTC format (e.g., YYYY-MM-DDTHH:MM:SSZ). Optional.")
    end_time: Optional[str] = Field(None, description="End time in ISO-8601 UTC format (e.g., YYYY-MM-DDTHH:MM:SSZ). Optional.")
    query: Optional[str] = Field(None, description="The user query, which may contain time-related phrases for the tool to parse if start/end times are not directly provided by LLM.")

# --- Base Tool Result Model --- 
class BaseToolResult(BaseModel):
    """Base model for all tool results, includes schema versioning and status."""
    schema_version: str = Field("0.1", frozen=True, description="Version of this tool result data structure.")
    status: Literal["success", "error", "skipped_duplicate"] = Field("success", description="Status of the tool execution.")
    error_message: Optional[str] = Field(None, description="Error message if status is 'error'.")
    # Actual tool-specific data will be in a sub-dictionary or a specific model inheriting this.

# Example: How a specific tool result might look (though tools currently return dicts for V1)
# class FindInterestingWindowResult(BaseToolResult):
#     window: Optional[Dict[str, str]] = None # {"start_time": "...", "end_time": "..."}
#     strategy: Optional[str] = None
#     significance_score: Optional[float] = None 