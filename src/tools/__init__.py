"""
Manufacturing Copilot (MCP) - Tool Functions Module

This module provides the core analytical tools for processing time-series data
and generating insights for the Manufacturing Copilot CLI.

Tools are designed to be modular and composable, enabling complex analysis
workflows from simple building blocks.
"""

from .data_loader import load_data
from .metrics import summarize_metric
from .anomaly_detection import detect_spike, detect_consecutive_anomalies, analyze_anomaly_patterns
from .correlation import correlate_tags, find_correlated_tags
from .visualization import generate_chart, generate_correlation_chart
from .quality import quality_summary

__all__ = [
    'load_data',
    'summarize_metric', 
    'detect_spike',
    'detect_consecutive_anomalies',
    'analyze_anomaly_patterns',
    'correlate_tags',
    'find_correlated_tags',
    'generate_chart',
    'generate_correlation_chart',
    'quality_summary'
] 