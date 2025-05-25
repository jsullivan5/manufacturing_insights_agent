#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Data Quality Analysis Tools
"""

import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def quality_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Compute percentage of Good, Questionable, and Bad quality values."""
    if df.empty or 'Quality' not in df.columns:
        return {"error": "No quality data available"}
    
    quality_counts = df['Quality'].value_counts()
    total = len(df)
    
    return {
        "good_pct": (quality_counts.get('Good', 0) / total) * 100,
        "questionable_pct": (quality_counts.get('Questionable', 0) / total) * 100,
        "bad_pct": (quality_counts.get('Bad', 0) / total) * 100,
        "total_points": total
    } 