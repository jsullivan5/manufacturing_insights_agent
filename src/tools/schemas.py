#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Tool Schemas

Defines standardized data structures for tool outputs to ensure
consistency across the MCP system and enable objective confidence scoring.
"""

from typing import List, Dict, Optional, TypedDict, Literal, Any, Union
from datetime import datetime

class AnomalyPoint(TypedDict):
    """Individual anomaly point detected in time series data."""
    timestamp: str  # ISO format
    value: float
    z_score: float
    deviation: float  # Absolute deviation from normal
    description: str


class AnomalyResult(TypedDict):
    """Result from anomaly detection tools."""
    tag: str
    value_type: Literal['numeric', 'binary']
    analysis_window: Dict[str, str]  # {start, end} as ISO strings
    anomalies: List[AnomalyPoint]
    threshold: float
    severity_score: float  # 0-1 score based on max z-score and duration


class StateChange(TypedDict):
    """Binary tag state change event."""
    timestamp: str  # ISO format
    from_state: int  # 0 or 1
    to_state: int  # 0 or 1
    duration_minutes: Optional[float]  # For completed state periods
    description: str


class BinaryChangeResult(TypedDict):
    """Result from binary state change detection."""
    tag: str
    analysis_window: Dict[str, str]  # {start, end} as ISO strings
    changes: List[StateChange]
    total_changes: int
    total_high_duration_minutes: float  # Total time in "high" state (1)
    severity_score: float  # 0-1 score based on unusual durations


class ChangePoint(TypedDict):
    """Significant change point in time series data."""
    timestamp: str  # ISO format
    value_before: float
    value_after: float
    magnitude: float
    direction: Literal['increase', 'decrease', 'fluctuation']
    duration_minutes: float


class ChangePointResult(TypedDict):
    """Result from change point detection."""
    tag: str
    analysis_window: Dict[str, str]  # {start, end} as ISO strings
    change_points: List[ChangePoint]
    significance_score: float  # 0-1 score based on magnitude


class CorrelationResult(TypedDict):
    """Result from correlation analysis."""
    primary_tag: str
    secondary_tag: str
    analysis_window: Dict[str, str]  # {start, end} as ISO strings
    pearson_correlation: float  # -1.0 to 1.0
    lag_minutes: float
    p_value: float
    data_points: int
    strength: Literal['very_weak', 'weak', 'moderate', 'strong', 'very_strong']
    significance: Literal['not_significant', 'marginally_significant', 'significant', 'highly_significant']
    direction: Literal['positive', 'negative']


class CausalityResult(TypedDict):
    """Result from causality testing."""
    cause_tag: str
    effect_tag: str
    analysis_window: Dict[str, str]  # {start, end} as ISO strings
    best_lag_minutes: float
    best_correlation: float
    direction: Literal['positive', 'negative']
    causal_strength: float  # 0-1 score based on lag and correlation
    causal_confidence: float  # 0-1 probability of causal relationship
    physical_plausibility: Literal['low', 'medium', 'high']


class EventPoint(TypedDict):
    """Individual event in a timeline."""
    timestamp: str  # ISO format
    tag: str
    value: float
    event_type: Literal['anomaly_start', 'anomaly_end', 'state_change', 'change_point']
    description: str
    severity: float  # 0-1 scale


class EventSequence(TypedDict):
    """Chronological sequence of events."""
    events: List[EventPoint]
    time_window: Dict[str, str]  # {start, end} as ISO strings
    primary_tag: str
    related_tags: List[str]


class BusinessImpact(TypedDict):
    """Business impact calculation."""
    event_type: str  # e.g., 'door_open', 'compressor_failure'
    duration_minutes: float
    energy_cost: float  # in dollars
    product_risk: float  # in dollars
    total_cost: float  # in dollars
    severity: Literal['low', 'medium', 'high', 'critical']
    description: str


class ToolResult(TypedDict):
    """Generic tool result with common fields."""
    tool_name: str
    timestamp: str  # When the tool was run
    params: Dict[str, Any]
    result_type: str  # Type of result contained in 'data'
    data: Dict[str, Any]  # The actual result data
    error: Optional[str] 