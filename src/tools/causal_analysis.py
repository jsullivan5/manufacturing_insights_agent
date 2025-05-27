#!/usr/bin/env python3
"""
Manufacturing Copilot - Causal Analysis Tool

Advanced temporal causality detection for manufacturing forensics.
Identifies cause → effect relationships with precise timing and confidence scoring.

This tool transforms weak correlations into forensic evidence by:
1. Detecting significant change points in operational data
2. Calculating time-lagged correlations between events
3. Building confidence-scored causal timelines
4. Generating business impact assessments

Example Output:
    14:30:00 - Door left open (trigger)
    14:30-14:47 - Temperature rises 3.3°C (18-minute exposure)  
    14:35:00 - Compressor activates (5-minute response delay)
    
    Confidence: 96% - Strong causal relationship detected
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChangePoint:
    """Represents a significant change in a time series."""
    timestamp: datetime
    tag: str
    value_before: float
    value_after: float
    magnitude: float
    direction: str  # 'increase', 'decrease', 'spike', 'drop'
    confidence: float

@dataclass
class CausalEvent:
    """Represents a cause → effect relationship between two events."""
    cause_event: ChangePoint
    effect_event: ChangePoint
    time_lag: timedelta
    correlation_strength: float
    confidence: float
    business_impact: str

@dataclass
class CausalTimeline:
    """Complete forensic timeline of causal relationships."""
    primary_anomaly: ChangePoint
    causal_chain: List[CausalEvent]
    overall_confidence: float
    root_cause: str
    business_impact: str
    recommendations: List[str]

def detect_change_points(df: pd.DataFrame, tag: str, 
                        sensitivity: float = 2.0,
                        min_duration: int = 5) -> List[ChangePoint]:
    """
    Detect significant change points in time series data.
    
    Uses rolling statistics and z-score analysis to identify
    sudden changes that could indicate operational events.
    
    Args:
        df: DataFrame with timestamp index and tag columns
        tag: Column name to analyze for change points
        sensitivity: Z-score threshold for change detection (lower = more sensitive)
        min_duration: Minimum minutes for sustained change
        
    Returns:
        List of detected change points with metadata
    """
    if tag not in df.columns:
        logger.warning(f"Tag {tag} not found in data")
        return []
    
    series = df[tag].dropna()
    if len(series) < 20:  # Need minimum data for analysis
        return []
    
    change_points = []
    
    # Calculate rolling statistics for baseline
    window = min(60, len(series) // 4)  # 1 hour or 25% of data
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()
    
    # Calculate z-scores for change detection
    z_scores = np.abs((series - rolling_mean) / rolling_std)
    
    # Find significant changes
    significant_changes = z_scores > sensitivity
    
    # Group consecutive changes and find start/end points
    change_groups = []
    in_change = False
    start_idx = None
    
    for idx, is_change in significant_changes.items():
        if is_change and not in_change:
            # Start of change period
            start_idx = idx
            in_change = True
        elif not is_change and in_change:
            # End of change period
            if start_idx is not None:
                duration = (idx - start_idx).total_seconds() / 60  # minutes
                if duration >= min_duration:
                    change_groups.append((start_idx, idx))
            in_change = False
            start_idx = None
    
    # Convert change groups to ChangePoint objects
    for start_time, end_time in change_groups:
        # Get values before and after change
        before_window = series.loc[:start_time].tail(10)
        after_window = series.loc[start_time:end_time]
        
        if len(before_window) > 0 and len(after_window) > 0:
            value_before = before_window.mean()
            value_after = after_window.mean()
            magnitude = abs(value_after - value_before)
            
            # Determine direction
            if value_after > value_before * 1.1:
                direction = 'increase'
            elif value_after < value_before * 0.9:
                direction = 'decrease'
            elif magnitude > value_before * 0.2:
                direction = 'spike' if value_after > value_before else 'drop'
            else:
                continue  # Skip minor changes
            
            # Calculate confidence based on magnitude and consistency
            relative_change = magnitude / (abs(value_before) + 0.001)  # Avoid division by zero
            consistency = max(0.0, 1.0 - (after_window.std() / (magnitude + 0.001)))  # Ensure non-negative
            confidence = min(0.95, max(0.0, relative_change * consistency))  # Ensure 0-0.95 range
            
            change_point = ChangePoint(
                timestamp=start_time,
                tag=tag,
                value_before=value_before,
                value_after=value_after,
                magnitude=magnitude,
                direction=direction,
                confidence=confidence
            )
            change_points.append(change_point)
    
    return sorted(change_points, key=lambda x: x.timestamp)

def calculate_time_lagged_correlations(df: pd.DataFrame, 
                                     cause_tag: str, 
                                     effect_tag: str,
                                     max_lag_minutes: int = 30) -> Dict[int, float]:
    """
    Calculate correlations between two tags at different time lags.
    
    This helps identify if changes in one tag consistently lead to
    changes in another tag after a specific delay.
    
    Args:
        df: DataFrame with timestamp index
        cause_tag: Tag that might be causing changes
        effect_tag: Tag that might be responding to changes
        max_lag_minutes: Maximum lag to test (in minutes)
        
    Returns:
        Dictionary mapping lag_minutes → correlation_coefficient
    """
    if cause_tag not in df.columns or effect_tag not in df.columns:
        return {}
    
    cause_series = df[cause_tag].dropna()
    effect_series = df[effect_tag].dropna()
    
    # Align series to common time range
    common_start = max(cause_series.index.min(), effect_series.index.min())
    common_end = min(cause_series.index.max(), effect_series.index.max())
    
    cause_aligned = cause_series.loc[common_start:common_end]
    effect_aligned = effect_series.loc[common_start:common_end]
    
    correlations = {}
    
    # Test different lag periods
    for lag_minutes in range(0, max_lag_minutes + 1, 1):
        lag_timedelta = timedelta(minutes=lag_minutes)
        
        # Shift effect series backward by lag amount
        effect_lagged = effect_aligned.shift(freq=f'-{lag_minutes}min')
        
        # Find overlapping time range
        overlap_start = max(cause_aligned.index.min(), effect_lagged.index.min())
        overlap_end = min(cause_aligned.index.max(), effect_lagged.index.max())
        
        if overlap_start < overlap_end:
            cause_overlap = cause_aligned.loc[overlap_start:overlap_end]
            effect_overlap = effect_lagged.loc[overlap_start:overlap_end]
            
            # Resample to ensure same frequency
            common_freq = '1min'
            cause_resampled = cause_overlap.resample(common_freq).mean()
            effect_resampled = effect_overlap.resample(common_freq).mean()
            
            # Calculate correlation
            if len(cause_resampled) > 10 and len(effect_resampled) > 10:
                correlation = cause_resampled.corr(effect_resampled)
                if not np.isnan(correlation):
                    correlations[lag_minutes] = correlation
    
    return correlations

def detect_causal_events(df: pd.DataFrame, 
                        primary_tag: str,
                        candidate_tags: List[str],
                        time_window_hours: int = 2) -> List[CausalEvent]:
    """
    Detect causal relationships between primary tag anomalies and candidate events.
    
    This is the main forensic function that identifies cause → effect chains
    with precise timing and confidence scoring.
    
    Args:
        df: DataFrame with manufacturing time series data
        primary_tag: Tag showing the anomaly we want to explain
        candidate_tags: Tags that might be causing the anomaly
        time_window_hours: How far back to look for causes
        
    Returns:
        List of causal events sorted by confidence
    """
    logger.info(f"🕵️ Investigating causal relationships for {primary_tag}")
    
    # Step 1: Find change points in primary tag (the effects)
    # Use more sensitive detection for temperature changes
    primary_changes = detect_change_points(df, primary_tag, sensitivity=1.5)
    
    if not primary_changes:
        logger.info("No significant changes detected in primary tag")
        return []
    
    causal_events = []
    
    # Step 2: For each primary change, look for potential causes
    for effect_change in primary_changes:
        effect_time = effect_change.timestamp
        search_start = effect_time - timedelta(hours=time_window_hours)
        
        logger.info(f"Investigating effect at {effect_time}")
        
        # Step 3: Find change points in candidate tags before the effect
        for candidate_tag in candidate_tags:
            if candidate_tag == primary_tag:
                continue
                
            # Get data window for analysis
            window_data = df.loc[search_start:effect_time]
            
            if len(window_data) < 10:
                continue
            
            # Find change points in candidate tag
            # Use high sensitivity for door/status changes
            sensitivity = 1.0 if 'DOOR' in candidate_tag.upper() or 'STATUS' in candidate_tag.upper() else 1.2
            candidate_changes = detect_change_points(window_data, candidate_tag, sensitivity=sensitivity)
            
            # Step 4: Calculate time-lagged correlations
            lag_correlations = calculate_time_lagged_correlations(
                df, candidate_tag, primary_tag, max_lag_minutes=120
            )
            
            # Step 5: Evaluate each candidate change as potential cause
            for cause_change in candidate_changes:
                time_lag = effect_change.timestamp - cause_change.timestamp
                lag_minutes = int(time_lag.total_seconds() / 60)
                
                # Skip if cause happens after effect
                if lag_minutes < 0:
                    continue
                
                # Get correlation at this time lag
                correlation = lag_correlations.get(lag_minutes, 0.0)
                
                # Calculate confidence score
                confidence = calculate_causal_confidence(
                    cause_change, effect_change, time_lag, correlation
                )
                
                # Generate business impact assessment
                business_impact = assess_business_impact(
                    cause_change, effect_change, time_lag
                )
                
                # Lower threshold for plausible relationships to catch more events
                if confidence > 0.2:  # Reduced from 0.3 to catch weaker but valid relationships
                    causal_event = CausalEvent(
                        cause_event=cause_change,
                        effect_event=effect_change,
                        time_lag=time_lag,
                        correlation_strength=correlation,
                        confidence=confidence,
                        business_impact=business_impact
                    )
                    causal_events.append(causal_event)
    
    # Sort by confidence (highest first)
    return sorted(causal_events, key=lambda x: x.confidence, reverse=True)

def calculate_causal_confidence(cause: ChangePoint, 
                              effect: ChangePoint,
                              time_lag: timedelta,
                              correlation: float) -> float:
    """
    Calculate confidence score for a causal relationship.
    
    Combines multiple factors:
    - Temporal proximity (closer = higher confidence)
    - Correlation strength
    - Magnitude of changes
    - Physical plausibility
    """
    # Temporal proximity factor (0-1, higher for shorter lags)
    lag_minutes = time_lag.total_seconds() / 60
    if lag_minutes <= 5:
        temporal_factor = 1.0
    elif lag_minutes <= 30:
        temporal_factor = 0.8
    elif lag_minutes <= 120:
        temporal_factor = 0.5
    else:
        temporal_factor = 0.2
    
    # Correlation factor (0-1)
    correlation_factor = abs(correlation)
    
    # Magnitude factor (higher for larger changes)
    cause_magnitude = cause.confidence
    effect_magnitude = effect.confidence
    magnitude_factor = (cause_magnitude + effect_magnitude) / 2
    
    # Physical plausibility (domain-specific rules)
    plausibility_factor = assess_physical_plausibility(cause, effect)
    
    # Combine factors with weights
    confidence = (
        temporal_factor * 0.3 +
        correlation_factor * 0.4 +
        magnitude_factor * 0.2 +
        plausibility_factor * 0.1
    )
    
    return min(0.99, confidence)  # Cap at 99%

def assess_physical_plausibility(cause: ChangePoint, effect: ChangePoint) -> float:
    """
    Assess physical plausibility of cause → effect relationship.
    
    Uses manufacturing domain knowledge to evaluate if the
    proposed causal relationship makes physical sense.
    """
    cause_tag = cause.tag.upper()
    effect_tag = effect.tag.upper()
    
    # Door → Temperature relationships
    if 'DOOR' in cause_tag and 'TEMP' in effect_tag:
        if cause.direction in ['increase', 'spike'] and effect.direction in ['increase', 'spike']:
            return 0.9  # Door opening causes temperature rise
    
    # Temperature → Compressor relationships  
    if 'TEMP' in cause_tag and 'COMPRESSOR' in effect_tag:
        if cause.direction in ['increase', 'spike'] and effect.direction in ['increase', 'spike']:
            return 0.8  # Temperature rise triggers compressor
    
    # Power → Temperature relationships
    if 'POWER' in cause_tag and 'TEMP' in effect_tag:
        if cause.direction in ['decrease', 'drop'] and effect.direction in ['increase', 'spike']:
            return 0.7  # Power loss causes temperature rise
    
    # Default plausibility for unknown relationships
    return 0.5

def assess_business_impact(cause: ChangePoint, 
                         effect: ChangePoint,
                         time_lag: timedelta) -> str:
    """
    Generate business impact assessment for causal relationship.
    
    Translates technical causality into business language
    with cost estimates and operational implications.
    """
    cause_tag = cause.tag.upper()
    effect_tag = effect.tag.upper()
    lag_minutes = int(time_lag.total_seconds() / 60)
    
    # Door-related impacts
    if 'DOOR' in cause_tag and 'TEMP' in effect_tag:
        temp_rise = abs(effect.value_after - effect.value_before)
        return f"Door left open for {lag_minutes} minutes caused {temp_rise:.1f}°C temperature rise. " \
               f"Estimated energy waste: ${lag_minutes * 0.5:.0f}. Product quality risk."
    
    # Compressor-related impacts
    if 'COMPRESSOR' in cause_tag:
        if cause.direction in ['decrease', 'drop']:
            return f"Compressor failure for {lag_minutes} minutes. " \
                   f"Estimated product loss risk: ${lag_minutes * 10:.0f}. Immediate maintenance required."
    
    # Power-related impacts
    if 'POWER' in cause_tag:
        power_change = abs(effect.value_after - effect.value_before)
        return f"Power fluctuation caused {power_change:.1f}kW change for {lag_minutes} minutes. " \
               f"Energy cost impact: ${power_change * lag_minutes * 0.12:.0f}."
    
    # Generic impact
    return f"Operational event lasting {lag_minutes} minutes with potential efficiency impact."

def build_event_timeline(causal_events: List[CausalEvent],
                        primary_tag: str) -> CausalTimeline:
    """
    Build a forensic timeline from detected causal events.
    
    Creates a comprehensive narrative of cause → effect relationships
    with overall confidence assessment and business recommendations.
    """
    if not causal_events:
        return CausalTimeline(
            primary_anomaly=None,
            causal_chain=[],
            overall_confidence=0.0,
            root_cause="No causal relationships detected",
            business_impact="Unable to determine impact",
            recommendations=["Investigate data quality", "Check sensor calibration"]
        )
    
    # Sort events chronologically
    sorted_events = sorted(causal_events, key=lambda x: x.cause_event.timestamp)
    
    # Find the primary anomaly (highest confidence effect)
    primary_anomaly = max(causal_events, key=lambda x: x.confidence).effect_event
    
    # Calculate overall confidence (weighted average)
    total_weight = sum(event.confidence for event in causal_events)
    overall_confidence = sum(event.confidence ** 2 for event in causal_events) / total_weight
    
    # Identify root cause
    root_cause_event = sorted_events[0] if sorted_events else None
    if root_cause_event:
        root_cause = f"{root_cause_event.cause_event.tag} {root_cause_event.cause_event.direction} " \
                    f"at {root_cause_event.cause_event.timestamp.strftime('%H:%M:%S')}"
    else:
        root_cause = "Unknown operational event"
    
    # Aggregate business impact
    business_impacts = [event.business_impact for event in causal_events]
    business_impact = " ".join(business_impacts)
    
    # Generate recommendations
    recommendations = generate_recommendations(causal_events, overall_confidence)
    
    return CausalTimeline(
        primary_anomaly=primary_anomaly,
        causal_chain=sorted_events,
        overall_confidence=overall_confidence,
        root_cause=root_cause,
        business_impact=business_impact,
        recommendations=recommendations
    )

def generate_recommendations(causal_events: List[CausalEvent], 
                           confidence: float) -> List[str]:
    """Generate actionable recommendations based on causal analysis."""
    recommendations = []
    
    # High confidence recommendations
    if confidence > 0.8:
        recommendations.append("High confidence causal relationship identified - implement immediate corrective action")
    
    # Door-related recommendations
    door_events = [e for e in causal_events if 'DOOR' in e.cause_event.tag.upper()]
    if door_events:
        recommendations.extend([
            "Implement door usage monitoring and alerts",
            "Train staff on proper door closure procedures",
            "Consider automatic door closure system"
        ])
    
    # Compressor-related recommendations
    compressor_events = [e for e in causal_events if 'COMPRESSOR' in e.cause_event.tag.upper()]
    if compressor_events:
        recommendations.extend([
            "Schedule immediate compressor inspection",
            "Implement predictive maintenance program",
            "Monitor compressor performance trends"
        ])
    
    # Power-related recommendations
    power_events = [e for e in causal_events if 'POWER' in e.cause_event.tag.upper()]
    if power_events:
        recommendations.extend([
            "Investigate electrical system stability",
            "Consider backup power systems",
            "Monitor power quality metrics"
        ])
    
    # Generic recommendations
    if not recommendations:
        recommendations = [
            "Continue monitoring for pattern confirmation",
            "Collect additional data for analysis",
            "Review operational procedures"
        ]
    
    return recommendations[:5]  # Limit to top 5 recommendations

# Main analysis function for integration with MCP
def analyze_causality(df: pd.DataFrame, 
                     primary_tag: str,
                     candidate_tags: Optional[List[str]] = None,
                     time_window_hours: int = 2) -> Dict[str, Any]:
    """
    Main function for causal analysis integration with Manufacturing Copilot.
    
    Args:
        df: Manufacturing time series data
        primary_tag: Tag showing anomaly to investigate
        candidate_tags: Potential cause tags (auto-detected if None)
        time_window_hours: Investigation time window
        
    Returns:
        Dictionary with causal analysis results for LLM interpretation
    """
    logger.info(f"🔍 Starting causal investigation for {primary_tag}")
    
    # Auto-detect candidate tags if not provided
    if candidate_tags is None:
        candidate_tags = [col for col in df.columns if col != primary_tag]
    
    # Detect causal events
    causal_events = detect_causal_events(df, primary_tag, candidate_tags, time_window_hours)
    
    # Build forensic timeline
    timeline = build_event_timeline(causal_events, primary_tag)
    
    # Format results for LLM consumption
    results = {
        'analysis_type': 'causal_investigation',
        'primary_tag': primary_tag,
        'investigation_confidence': timeline.overall_confidence,
        'root_cause': timeline.root_cause,
        'business_impact': timeline.business_impact,
        'recommendations': timeline.recommendations,
        'causal_events': [],
        'forensic_timeline': []
    }
    
    # Add detailed event information
    for event in timeline.causal_chain:
        event_info = {
            'cause_tag': event.cause_event.tag,
            'cause_time': event.cause_event.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'cause_type': event.cause_event.direction,
            'effect_tag': event.effect_event.tag,
            'effect_time': event.effect_event.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'time_lag_minutes': int(event.time_lag.total_seconds() / 60),
            'confidence': event.confidence,
            'correlation': event.correlation_strength,
            'business_impact': event.business_impact
        }
        results['causal_events'].append(event_info)
        
        # Add to forensic timeline
        timeline_entry = f"{event.cause_event.timestamp.strftime('%H:%M:%S')} - " \
                        f"{event.cause_event.tag} {event.cause_event.direction} " \
                        f"(confidence: {event.confidence:.0%})"
        results['forensic_timeline'].append(timeline_entry)
    
    logger.info(f"✅ Causal investigation complete. Confidence: {timeline.overall_confidence:.0%}")
    
    return results 