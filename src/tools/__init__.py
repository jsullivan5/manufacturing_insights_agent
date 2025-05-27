"""
Manufacturing Copilot (MCP) - Tools Package

Provides atomic data analysis tools for investigating manufacturing time series data.
Each tool performs a specific function and returns structured output that can be
combined by the root cause agent to build causal explanations.
"""

# Import atomic tools and schemas
from .atomic_tools import (
    detect_numeric_anomalies,
    detect_binary_flips,
    detect_change_points,
    test_causality,
    calculate_impact,
    create_event_sequence
)

# Import tag intelligence
from .tag_intel import (
    get_tag_intelligence,
    get_tag_metadata,
    get_value_type,
    get_related_tags,
    is_anomaly
)

# Import data loader
from .data_loader import (
    load_data,
    get_available_tags,
    get_data_time_range
)

# Import legacy tools (for backward compatibility)
from .anomaly_detection import detect_spike
from .correlation import find_correlated_tags, cross_corr
from .metrics import summarize_metric

__all__ = [
    # Atomic tools
    'detect_numeric_anomalies',
    'detect_binary_flips',
    'detect_change_points',
    'test_causality',
    'calculate_impact',
    'create_event_sequence',
    
    # Tag intelligence
    'get_tag_intelligence',
    'get_tag_metadata',
    'get_value_type',
    'get_related_tags',
    'is_anomaly',
    
    # Data loading
    'load_data',
    'get_available_tags',
    'get_data_time_range',
    
    # Legacy tools
    'detect_spike',
    'find_correlated_tags',
    'summarize_metric',
    'cross_corr'
] 