"""
Manufacturing Copilot (MCP) - Tool Functions Module

This module provides the core analytical tools for processing time-series data
and generating insights for the Manufacturing Copilot CLI.

Tools are designed to be modular and composable, enabling complex analysis
workflows from simple building blocks.
"""

from .data_loader import load_data
from .metrics import summarize_metric
from .anomaly_detection import detect_spike
from .correlation import correlate_tags
from .visualization import generate_chart
from .quality import quality_summary

__all__ = [
    'load_data',
    'summarize_metric', 
    'detect_spike',
    'correlate_tags',
    'generate_chart',
    'quality_summary'
] 