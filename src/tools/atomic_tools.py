#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Atomic Tools

Provides standardized tools for manufacturing data analysis with structured outputs.
These atomic tools are used by the RootCause agent to investigate anomalies
and build causal chains.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union

import pandas as pd
import numpy as np

# Import schema definitions
from .schemas import (
    AnomalyPoint, AnomalyResult,
    StateChange, BinaryChangeResult,
    ChangePoint, ChangePointResult,
    CorrelationResult, CausalityResult,
    BusinessImpact
)

# Import existing tools to wrap
from .anomaly_detection import detect_spike
from .correlation import cross_corr, correlate_tags
from .data_loader import load_data, get_data_time_range

# Import tag intelligence

from .tag_intel import get_tag_metadata, get_value_type, is_anomaly

# ------------------------------------------------------------------
# Tag metadata helper (alert/high state & thresholds)
from src.glossary import TagGlossary

_glossary_singleton = TagGlossary()

from datetime import timezone

def _ensure_utc_ts(ts: Union[str, datetime, pd.Timestamp, None]):
    if ts is None:
        return None
    ts = pd.to_datetime(ts)

    # >>> If we somehow got two timezone chunks, strip one <<<
    # e.g.  '2025-05-20T08:05:00+00:00+00:00'  -->  '2025-05-20T08:05:00+00:00'
    if isinstance(ts, str) and ts.count('+00:00') > 1:
        ts = ts.replace('+00:00+00:00', '+00:00')

    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts

def _alert_state(tag_name: str) -> int | None:
    """
    Return 1 or 0 if the glossary defines which binary value means
    "alert / on / high" for this tag, otherwise None.
    """
    info = _glossary_singleton.get_tag_info(tag_name)
    if not info:
        return None
    val = str(info.alert_state).lower() if getattr(info, "alert_state", None) is not None else ""
    if val in {"1", "high", "open", "true", "on"}:
        return 1
    if val in {"0", "low", "closed", "false", "off"}:
        return 0
    return None
# ------------------------------------------------------------------

# Configure logging
logger = logging.getLogger(__name__)

# Utility function to parse string dates
def parse_time_reference(time_ref: Union[str, datetime, None]) -> Optional[datetime]:
    """
    Parse string time references into datetime objects.
    
    Handles natural language like 'yesterday', 'last week', etc.
    
    Args:
        time_ref: String time reference or datetime object
        
    Returns:
        Datetime object or None if parsing fails
    """
    if time_ref is None:
        return None
        
    if isinstance(time_ref, datetime):
        return time_ref
        
    # Handle common string patterns
    now = datetime.now()
    time_ref = str(time_ref).lower().strip()
    
    try:
        if time_ref == 'now':
            return now
            
        elif 'yesterday' in time_ref:
            result = now - timedelta(days=1)
            # Handle time of day references
            if 'morning' in time_ref:
                return result.replace(hour=8, minute=0, second=0)
            elif 'afternoon' in time_ref:
                return result.replace(hour=14, minute=0, second=0)
            elif 'evening' in time_ref:
                return result.replace(hour=18, minute=0, second=0)
            return result.replace(hour=0, minute=0, second=0)
            
        elif 'today' in time_ref:
            # Handle time of day references
            if 'morning' in time_ref:
                return now.replace(hour=8, minute=0, second=0)
            elif 'afternoon' in time_ref:
                return now.replace(hour=14, minute=0, second=0)
            elif 'evening' in time_ref:
                return now.replace(hour=18, minute=0, second=0)
            return now.replace(hour=0, minute=0, second=0)
            
        elif 'last week' in time_ref:
            return now - timedelta(days=7)
            
        elif 'last month' in time_ref:
            return now - timedelta(days=30)
            
        # Try to parse standard date formats
        return pd.to_datetime(time_ref).to_pydatetime()
        
    except Exception as e:
        logger.warning(f"Failed to parse time reference '{time_ref}': {e}")
        return None

# Default energy costs (can be overridden via config)
ENERGY_COST_KWH = 0.12  # $/kWh
PRODUCT_LOSS_TEMP_THRESHOLD = -15.0  # °C
PRODUCT_LOSS_PER_DEGREE_HOUR = 5.0  # $/°C-hour above threshold


def detect_numeric_anomalies(
    tag: str, 
    start_time: Optional[Union[datetime, str]] = None,
    end_time: Optional[Union[datetime, str]] = None,
    threshold: Optional[float] = None,
    baseline_window_hours: int = 24, 
    default_expansion_minutes_total: int = 120, # e.g., 2 hours total window if start==end
    **kwargs
) -> Dict[str, Any]:
    """
    Detect anomalies in numeric tag data using statistical methods.
    Uses a baseline period outside the analysis window for more robust Z-score calculation.
    If start_time == end_time, expands the window by default_expansion_minutes_total.
    """
    value_type = get_value_type(tag)
    if value_type != 'numeric':
        logger.warning(f"Tag {tag} is not numeric, using detect_binary_flips instead")
        if value_type == 'binary':
            return detect_binary_flips(tag, start_time, end_time)
        return {'error': f"Tag {tag} is not a recognized type"}

    parsed_start_time = parse_time_reference(start_time)
    parsed_end_time = parse_time_reference(end_time)

    if parsed_start_time and parsed_end_time and parsed_start_time == parsed_end_time:
        logger.info(f"detect_numeric_anomalies: start_time equals end_time ({parsed_start_time}). Expanding window by {default_expansion_minutes_total} minutes.")
        half_expansion = timedelta(minutes=default_expansion_minutes_total / 2)
        parsed_start_time = parsed_start_time - half_expansion
        parsed_end_time = parsed_end_time + half_expansion
        logger.info(f"New expanded window: {parsed_start_time} to {parsed_end_time}")

    if not parsed_start_time or not parsed_end_time:
        logger.error(f"Invalid or missing start/end time for anomaly detection on {tag} after potential expansion.")
        data_range = get_data_time_range(tag=tag)
        parsed_start_time = parsed_start_time or data_range.get('start')
        parsed_end_time = parsed_end_time or data_range.get('end')
        if not parsed_start_time or not parsed_end_time:
             return {'error': "Could not determine a valid time range for analysis."}
    
    # Load data for the analysis window
    try:
        analysis_df = load_data(tag, parsed_start_time, parsed_end_time)
    except Exception as e:
        logger.error(f"Error loading analysis data for {tag}: {e}")
        return {'error': f"Analysis data loading failed: {e}"}

    if analysis_df.empty:
        logger.info(f"No data in analysis window for {tag} ({parsed_start_time} to {parsed_end_time})")
        # Return structure consistent with successful empty result
        return {
            'tag': tag, 'value_type': 'numeric',
            'analysis_window': {'start': parsed_start_time.isoformat(), 'end': parsed_end_time.isoformat()},
            'anomalies': [], 'threshold': threshold or 3.0, 'severity_score': 0.0,
            'message': 'No data in the specified analysis window.'
        }

    # Determine baseline period: X hours before the analysis start_time
    baseline_end_time = parsed_start_time - timedelta(seconds=1) # Ensure no overlap
    baseline_start_time = baseline_end_time - timedelta(hours=baseline_window_hours)
    
    baseline_mean = None
    baseline_std = None
    baseline_info = "No baseline data available or insufficient data."

    try:
        # Attempt to load baseline data from a period *before* the analysis window
        baseline_df = load_data(tag, baseline_start_time, baseline_end_time)
        if not baseline_df.empty and len(baseline_df) > 5: # Need a few points for meaningful baseline
            baseline_values = baseline_df['Value'].dropna()
            if not baseline_values.empty:
                baseline_mean = baseline_values.mean()
                baseline_std = baseline_values.std(ddof=0) # Population standard deviation
                if baseline_std == 0: # Avoid division by zero if baseline is flat
                    baseline_std = 1e-6 # A very small number
                baseline_info = f"Baseline from {baseline_start_time.isoformat()} to {baseline_end_time.isoformat()}: Mean={baseline_mean:.2f}, Std={baseline_std:.2f}"
                logger.info(f"Calculated baseline for {tag}: Mean={baseline_mean:.2f}, Std={baseline_std:.2f} from {len(baseline_values)} points.")
            else:
                logger.warning(f"Baseline data for {tag} had no numeric values after dropna.")
        else:
            logger.warning(f"Insufficient baseline data for {tag} ({len(baseline_df)} points). Will attempt rolling stats if possible.")
    except Exception as e:
        logger.warning(f"Error loading baseline data for {tag}: {e}. Will attempt rolling stats if possible.")

    # If external baseline couldn't be established, fallback to rolling mean from detect_spike (original behavior)
    # Or, if baseline_std is effectively zero, rolling might be better.
    if baseline_mean is None or baseline_std is None or baseline_std < 1e-5:
        logger.info(f"Using rolling statistics for {tag} as external baseline was not robust.")
        # Call original detect_spike which uses rolling window from analysis_df
        detected_anomalies_raw = detect_spike(analysis_df, threshold=(threshold or (2.5 if 'TEMP' in tag else 3.0)))
        baseline_info = "Used rolling statistics from analysis window for baseline."
    else:
        # Calculate Z-scores using the external baseline
        analysis_df_copy = analysis_df.copy()
        analysis_df_copy['z_score'] = np.abs((analysis_df_copy['Value'] - baseline_mean) / baseline_std)
        analysis_df_copy['z_score'] = analysis_df_copy['z_score'].fillna(0)
        
        current_threshold = threshold or (2.5 if 'TEMP' in tag else 3.0)
        anomalous_points_df = analysis_df_copy[analysis_df_copy['z_score'] > current_threshold]
        
        detected_anomalies_raw = []
        for _, row in anomalous_points_df.iterrows():
            desc = f"Value {row['Value']:.2f} is {row['z_score']:.1f} std devs from baseline mean {baseline_mean:.2f}"
            detected_anomalies_raw.append((row['Timestamp'], row['Value'], row['z_score'], desc))
        logger.info(f"Detected {len(detected_anomalies_raw)} anomalies for {tag} using external baseline.")

    # Common processing for anomalies
    anomaly_points = []
    max_z_score = 0.0
    analysis_mean = analysis_df['Value'].mean() # For deviation calculation

    for ts, val, z, desc in detected_anomalies_raw:
        anomaly_points.append({
            'timestamp': ts.isoformat() + 'Z', # Ensure UTC Z notation
            'value': float(val),
            'z_score': float(z),
            'deviation': float(abs(val - analysis_mean)) if pd.notna(analysis_mean) and pd.notna(val) else 0.0,
            'description': desc
        })
        max_z_score = max(max_z_score, abs(z))
    
    severity_score = 0.0
    if anomaly_points:
        z_score_component = min(1.0, max_z_score / 10.0)
        count_component = min(1.0, len(anomaly_points) / 5.0)
        severity_score = 0.7 * z_score_component + 0.3 * count_component
    
    return {
        'tag': tag,
        'value_type': 'numeric',
        'analysis_window': {
            'start': parsed_start_time.isoformat() + 'Z',
            'end': parsed_end_time.isoformat() + 'Z'
        },
        'baseline_info': baseline_info,
        'anomalies': anomaly_points,
        'threshold_used': threshold or (2.5 if 'TEMP' in tag else 3.0),
        'severity_score': severity_score
    }


def detect_binary_flips(
    tag: str,
    start_time: Optional[Union[datetime, str]] = None,
    end_time: Optional[Union[datetime, str]] = None,
    min_continuous_high_minutes: int = 5, # New parameter
    **kwargs
) -> Dict[str, Any]:
    """
    Detect state changes in binary tag data and periods of continuous high state.
    Uses diff() to identify changes in state on raw data.
    
    Args:
        tag: The binary tag to analyze.
        start_time: Optional start time for analysis window.
        end_time: Optional end time for analysis window.
        min_continuous_high_minutes: Min duration for a state to be considered continuously high from window start.
        
    Returns:
        Dictionary with state changes and continuous high state information.
    """
    value_type = get_value_type(tag)
    if value_type != 'binary':
        logger.warning(f"Tag {tag} is not binary for detect_binary_flips. Value type: {value_type}")
        return {'error': f"Tag {tag} is not a binary type for flip detection."}

    parsed_start_time = parse_time_reference(start_time)
    parsed_end_time = parse_time_reference(end_time)

    if not parsed_start_time or not parsed_end_time:
        data_range = get_data_time_range(tag=tag)
        parsed_start_time = parsed_start_time or data_range.get('start')
        parsed_end_time = parsed_end_time or data_range.get('end')
        if not parsed_start_time or not parsed_end_time:
             return {'error': "Could not determine a valid time range for binary flip detection."}

    try:
        df = load_data(tag, parsed_start_time, parsed_end_time)
    except Exception as e:
        return {'error': f"Data loading failed for {tag}: {e}"}

    analysis_window_out = {
        'start': parsed_start_time.isoformat() + "Z",
        'end': parsed_end_time.isoformat() + "Z"
    }

    if df.empty or len(df) < 2: # Need at least 2 points to detect a flip with diff
        logger.info(f"Insufficient data (len {len(df)}) for binary flip detection on {tag} in window.")
        return {
            'tag': tag, 'analysis_window': analysis_window_out,
            'changes': [], 'total_changes': 0,
            'total_high_duration_minutes': 0.0, 'severity_score': 0.0,
            'continuous_high_event': None,
            'message': 'Insufficient data for flip detection.'
        }

    # Ensure 'Value' is numeric (0 or 1)
    def robust_to_binary(val):
        if isinstance(val, str):
            if val.lower() in ["1", "true", "on", "yes"]: return 1
            if val.lower() in ["0", "false", "off", "no"]: return 0
        try:
            return 1 if float(val) > 0 else 0
        except (ValueError, TypeError):
            return 0 # Default to 0 if parsing fails

    df['state'] = df['Value'].apply(robust_to_binary)
    df = df.sort_values('Timestamp') # Ensure correct order for diff
    # Resolve which numeric value represents the "alert/high" state.
    alert_on_value = _alert_state(tag)
    if alert_on_value is None:
        alert_on_value = 1  # fallback to previous assumption
    
    # Detect changes using diff
    df['prev_state'] = df['state'].diff() # This will be NaN for first, 1 for 0->1, -1 for 1->0, 0 for no change

    changes = []
    total_high_duration = 0.0
    
    # For total_high_duration and continuous_high_event, we still need to iterate effectively
    # This part is tricky with pure diff(), as diff only tells you *where* changes occur.
    # We need to reconstruct durations.
    
    # Iterating through points where a change occurred or at the start/end of the window
    change_points_df = df[df['prev_state'] != 0].copy() # Points where state actually changed
    
    # Add first and last points of the original df to correctly calculate durations at boundaries
    # Ensure they are not duplicated if they are already change points
    boundary_indices = [df.index[0], df.index[-1]]
    for b_idx in boundary_indices:
        if b_idx not in change_points_df.index:
            # change_points_df = pd.concat([change_points_df, df.loc[[b_idx]]]) # This might re-introduce prev_state = 0
             # Add boundary points by creating temporary rows that look like change points for iteration
             # We are interested in the state *at* these points and their timestamps.
             # The 'prev_state' for these added boundaries doesn't strictly matter for the duration logic below
             # as long as they are processed in order.
             # A simpler way: iterate the original df to calculate durations.
             pass # Let's use a simpler iteration for durations.

    # Recalculate durations by iterating through the original dataframe
    # This is more robust for calculating time spent in each state.
    last_change_ts = df['Timestamp'].iloc[0]
    last_state = df['state'].iloc[0]
    
    for i in range(1, len(df)):
        current_ts = df['Timestamp'].iloc[i]
        current_state = df['state'].iloc[i]
        
        if current_state != last_state:
            duration_seconds = (current_ts - last_change_ts).total_seconds()
            duration_minutes = duration_seconds / 60.0
            
            if last_state == alert_on_value: # The state that just *ended* was 'high'
                total_high_duration += duration_minutes
            
            desc_key = "opened" if "DOOR" in tag.upper() else ("started" if "COMPRESSOR" in tag.upper() else "went_high")
            if current_state == (1 - alert_on_value): # Transition to low (meaning previous state was high and it just ended)
                 desc_key = "closed" if "DOOR" in tag.upper() else ("stopped" if "COMPRESSOR" in tag.upper() else "went_low")
            elif current_state == alert_on_value: # Transition to high
                 pass # desc_key already set for "went_high" like events

            changes.append({
                'timestamp': current_ts.isoformat() + 'Z', # Timestamp of the change
                'from_state': last_state,
                'to_state': current_state,
                'duration_of_previous_state_minutes': round(duration_minutes, 2),
                'description': f"{tag} {desc_key}. Was {last_state} for {duration_minutes:.1f} min until this change."
            })
            last_change_ts = current_ts
            last_state = current_state

    # Account for the duration of the very last state in the window
    if len(df) > 0:
        final_duration_seconds = (df['Timestamp'].iloc[-1] - last_change_ts).total_seconds()
        final_duration_minutes = final_duration_seconds / 60.0
        if last_state == alert_on_value:
            total_high_duration += final_duration_minutes

    # Continuous high event detection (from the start of the window)
    continuous_high_event = None
    first_point_timestamp = df['Timestamp'].iloc[0]
    initial_state_at_window_start = df['state'].iloc[0]

    if initial_state_at_window_start == alert_on_value:
        first_change_to_low = None
        # Check if any actual flips to low occurred
        for chg in changes:
            if chg['to_state'] == (1 - alert_on_value):
                # Parse the timestamp string from the change event into a datetime object
                chg_ts_str = chg.get('timestamp')
                if chg_ts_str:
                    first_change_to_low_ts = datetime.fromisoformat(chg_ts_str.replace('Z', '+00:00'))
                    if first_change_to_low is None or first_change_to_low_ts < first_change_to_low:
                        first_change_to_low = first_change_to_low_ts
                # No need to break here if we want the earliest, just iterate all and take min
        
        continuous_duration_minutes = 0
        if first_change_to_low: # It was high and then went low within the window
            continuous_duration_minutes = (first_change_to_low - first_point_timestamp).total_seconds() / 60.0
        else: # It remained high for the entire window (or there were no flips at all and it started high)
            continuous_duration_minutes = (df['Timestamp'].iloc[-1] - first_point_timestamp).total_seconds() / 60.0
            
        if continuous_duration_minutes >= min_continuous_high_minutes:
            continuous_high_event = {
                'timestamp': first_point_timestamp.isoformat().replace("+00:00", "Z"), # Changed from 'start_time' to 'timestamp' for consistency
                'event': 'state_high_continuous_from_window_start',
                'duration_minutes': round(continuous_duration_minutes, 1),
                'description': f"{tag} was continuously high for {continuous_duration_minutes:.1f} minutes from window start."
            }
    
    # If no flips were detected, but it started high and qualified for continuous_high_event
    # it means the state was consistently high for the whole period.
    if not changes and initial_state_at_window_start == alert_on_value and continuous_high_event:
        logger.info(f"No explicit flips detected for {tag}, but was continuously high. Event: {continuous_high_event}")
        # The continuous_high_event already captures this.

    severity_score = min(1.0, (len(changes) * 0.1) + (total_high_duration / 60.0) * 0.2) 
    if continuous_high_event: severity_score = max(severity_score, 0.5)

    # User-suggested block: Ensure continuous_high_event is set if no changes and started high
    if not changes and initial_state_at_window_start == alert_on_value:
        # Calculate duration from the first point to the last point in the DataFrame
        duration_if_continuously_high = (df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds() / 60.0
        if duration_if_continuously_high >= min_continuous_high_minutes:
            # If continuous_high_event was somehow not set by the previous logic, or to ensure it has this specific structure
            continuous_high_event = {
                'timestamp': df['Timestamp'].iloc[0].isoformat().replace("+00:00", "Z"), # Time of the first data point
                'event': 'state_high_continuous_from_window_start_fallback', # Indicate it's from this specific block
                'duration_minutes': round(duration_if_continuously_high, 1),
                'description': f"{tag} was continuously high for {duration_if_continuously_high:.1f} minutes from window start (verified by fallback)."
            }
            logger.info(f"Fallback: Set continuous_high_event for {tag} as no flips detected and started high. Event: {continuous_high_event}")

    # Fallback: no flips, started high, stayed high ≥ min_continuous_high_minutes
        if not changes and df['state'].iat[0] == alert_on_value:
            duration = (df['Timestamp'].iat[-1] - df['Timestamp'].iat[0]).total_seconds() / 60
            if duration >= min_continuous_high_minutes:
                continuous_high_event = {
                    "timestamp": df['Timestamp'].iat[0].isoformat() + "Z",
                    "event": "state_high_continuous_from_window_start",
                    "duration_minutes": round(duration, 1),
                    "description": f"{tag} was high for {duration:.1f} min from window start"
                }
    
    logger.debug(
        f"[dbg-binary] {tag}: flips={len(changes)}, che={bool(continuous_high_event)}"
    )
    return {
        'tag': tag, 'analysis_window': analysis_window_out,
        'changes': changes, 'total_changes': len(changes),
        'total_high_duration_minutes': round(total_high_duration, 2),
        'continuous_high_event': continuous_high_event,
        'severity_score': round(severity_score, 2)
    }


def detect_change_points(
    tag: str,
    start_time: Optional[Union[datetime, str]] = None,
    end_time: Optional[Union[datetime, str]] = None,
    sensitivity: float = 2.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Detect significant change points in time series data.
    
    Args:
        tag: The tag to analyze
        start_time: Optional start time for analysis window
        end_time: Optional end time for analysis window
        sensitivity: Lower values detect more changes (default: 2.0)
        
    Returns:
        ChangePointResult with significant changes
    """
    # Parse string time references
    start_time = parse_time_reference(start_time)
    end_time = parse_time_reference(end_time)
    
    # Get time range if not specified
    if start_time is None or end_time is None:
        data_range = get_data_time_range()
        if start_time is None:
            start_time = data_range['start']
        if end_time is None:
            end_time = data_range['end']
    
    # Load data
    try:
        df = load_data(tag, start_time, end_time)
    except Exception as e:
        logger.error(f"Error loading data for {tag}: {e}")
        return {'error': f"Data loading failed: {e}"}
    
    if df.empty or len(df) < 20:
        return {
            'tag': tag,
            'analysis_window': {
                'start': start_time.isoformat() if start_time else None,
                'end': end_time.isoformat() if end_time else None
            },
            'change_points': [],
            'significance_score': 0.0
        }
    
    # Implementation for change point detection
    series = df['Value'].dropna()
    
    # Calculate rolling statistics for baseline
    window = min(60, len(series) // 4)  # 1 hour or 25% of data
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()
    
    # Calculate z-scores for change detection
    z_scores = np.abs((series - rolling_mean) / rolling_std)
    
    # Find significant changes
    significant_changes = z_scores > sensitivity
    
    change_points = []
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
                # Convert timestamps to datetime if they're not already
                try:
                    if not isinstance(idx, datetime):
                        idx_dt = pd.to_datetime(idx)
                    else:
                        idx_dt = idx
                        
                    if not isinstance(start_idx, datetime):
                        start_idx_dt = pd.to_datetime(start_idx)
                    else:
                        start_idx_dt = start_idx
                        
                    duration = (idx_dt - start_idx_dt).total_seconds() / 60  # minutes
                except Exception as e:
                    logger.warning(f"Error calculating duration: {e}")
                    duration = 5.0  # Default fallback
                if duration >= 5:  # Minimum 5 minutes
                    # Get values before and after change
                    before_window = series.loc[:start_idx].tail(10)
                    after_window = series.loc[start_idx:idx]
                    
                    if len(before_window) > 0 and len(after_window) > 0:
                        value_before = before_window.mean()
                        value_after = after_window.mean()
                        magnitude = abs(value_after - value_before)
                        
                        # Determine direction
                        if value_after > value_before * 1.05:
                            direction = 'increase'
                        elif value_after < value_before * 0.95:
                            direction = 'decrease'
                        else:
                            direction = 'fluctuation'
                        
                        change_points.append({
                            'timestamp': start_idx.isoformat(),
                            'value_before': float(value_before),
                            'value_after': float(value_after),
                            'magnitude': float(magnitude),
                            'direction': direction,
                            'duration_minutes': float(duration)
                        })
            
            in_change = False
            start_idx = None
    
    # Sort by magnitude and return top 5
    change_points.sort(key=lambda x: x['magnitude'], reverse=True)
    change_points = change_points[:5]
    
    # Calculate significance score
    significance_score = 0.0
    if change_points:
        # Significance based on magnitude of largest change
        tag_metadata = get_tag_metadata(tag)
        if tag_metadata and tag_metadata.get('baseline_stats'):
            stats = tag_metadata['baseline_stats']
            range_width = stats['max_normal'] - stats['min_normal']
            if range_width > 0:
                normalized_magnitude = change_points[0]['magnitude'] / range_width
                significance_score = min(0.9, normalized_magnitude)
            else:
                significance_score = 0.5  # Default if range is zero
        else:
            # Fallback if no baseline stats
            significance_score = min(0.7, change_points[0]['magnitude'] / 10.0)
    
    return {
        'tag': tag,
        'analysis_window': {
            'start': start_time.isoformat() if start_time else None,
            'end': end_time.isoformat() if end_time else None
        },
        'change_points': change_points,
        'significance_score': significance_score
    }


def test_causality(
    cause_tag: str,
    effect_tag: str,
    window_start: Optional[Union[datetime, str]] = None,
    window_end: Optional[Union[datetime, str]] = None,
    max_lag_minutes: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Test causal relationship between two tags.
    
    Args:
        cause_tag: Potential cause tag
        effect_tag: Potential effect tag
        window_start: Optional start time for analysis window
        window_end: Optional end time for analysis window
        max_lag_minutes: Maximum lag to test in minutes
        
    Returns:
        CausalityResult with correlation and lag information
    """
    # Parse string time references
    window_start = parse_time_reference(window_start)
    window_end = parse_time_reference(window_end)
    
    # Get time range if not specified
    if window_start is None or window_end is None:
        data_range = get_data_time_range()
        if window_start is None:
            window_start = data_range['start']
        if window_end is None:
            window_end = data_range['end']
    
    # Load data for both tags
    try:
        cause_df = load_data(cause_tag, window_start, window_end)
        effect_df = load_data(effect_tag, window_start, window_end)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return {'error': f"Data loading failed: {e}"}
    
    if cause_df.empty or effect_df.empty:
        return {
            'cause_tag': cause_tag,
            'effect_tag': effect_tag,
            'analysis_window': {
                'start': window_start.isoformat() if window_start else None,
                'end': window_end.isoformat() if window_end else None
            },
            'best_lag_minutes': 0.0,
            'best_correlation': 0.0,
            'direction': 'positive',
            'causal_strength': 0.0,
            'causal_confidence': 0.0,
            'physical_plausibility': 'low'
        }
    
    # Check if cause_tag is binary and apply smoothing if so
    cause_tag_value_type = get_value_type(cause_tag)
    if cause_tag_value_type == 'binary':
        logger.info(f"Cause tag {cause_tag} is binary. Applying 3-minute rolling mean for causality test.")
        # Ensure Value is numeric (0 or 1) for rolling mean
        def robust_to_binary_for_rolling(val):
            if isinstance(val, str):
                if val.lower() in ["1", "true", "on", "yes"]: return 1.0
                if val.lower() in ["0", "false", "off", "no"]: return 0.0
            try:
                return 1.0 if float(val) > 0 else 0.0
            except (ValueError, TypeError):
                return 0.0 # Default to 0.0 if parsing fails
        
        cause_df['Value'] = cause_df['Value'].apply(robust_to_binary_for_rolling)
        
        # Set Timestamp as index for rolling operation
        cause_df_indexed = cause_df.set_index('Timestamp')
        # Apply rolling mean. Ensure the window string is valid, e.g., '3T' for 3 minutes.
        # The default data is 1-minute, so a window of 3 points is '3min' or '3T'.
        cause_series_smoothed = cause_df_indexed['Value'].rolling(window='3min', min_periods=1).mean()
        # The cross_corr function expects DataFrame inputs, so we need to reconstruct one for the smoothed cause
        cause_df_smoothed = pd.DataFrame({'Timestamp': cause_series_smoothed.index, 'Value': cause_series_smoothed.values})
        # The cross_corr function itself will handle resampling to a common index and then correlating.
        # We pass the modified cause_df (smoothed) instead of the original cause_df.
        result = cross_corr(
            cause_tag, # Pass original tag name for metadata inside cross_corr if needed
            effect_tag,
            window_start, # Pass the original window_start datetime
            window_end,   # Pass the original window_end datetime
            max_lag_minutes,
            cause_data=cause_df_smoothed, # Provide the pre-loaded and smoothed data
            effect_data=effect_df       # Provide the pre-loaded effect data
        )
    else:
        result = cross_corr(
            cause_tag,
            effect_tag,
            window_start,
            window_end,
            max_lag_minutes,
            cause_data=cause_df,  # Provide pre-loaded original data
            effect_data=effect_df # Provide pre-loaded original data
        )
    
    # Extract results
    correlation = result.get('best_correlation', 0.0)
    lag = result.get('best_lag', 0)
    
    # Convert lag to minutes (cross_corr returns in data points)
    # Assuming 1-minute data sampling
    lag_minutes = lag
    
    # Determine direction
    direction = 'positive' if correlation >= 0 else 'negative'
    
    # Calculate causal strength (0-1)
    abs_corr = abs(correlation)
    if abs_corr >= 0.8:
        corr_component = 1.0
    elif abs_corr >= 0.6:
        corr_component = 0.8
    elif abs_corr >= 0.4:
        corr_component = 0.5
    elif abs_corr >= 0.2:
        corr_component = 0.2
    else:
        corr_component = 0.0
    
    # Lag component (shorter lag = stronger causality)
    if abs(lag_minutes) <= 3:
        lag_component = 1.0
    elif abs(lag_minutes) <= 5:
        lag_component = 0.8
    elif abs(lag_minutes) <= 10:
        lag_component = 0.5
    else:
        lag_component = 0.2
    
    # Combine components
    causal_strength = 0.6 * corr_component + 0.4 * lag_component
    
    # Assess physical plausibility
    # Check if the relationship makes sense based on tag categories
    cause_type = get_tag_metadata(cause_tag)
    effect_type = get_tag_metadata(effect_tag)
    
    plausibility = 'low'
    if cause_type and effect_type:
        cause_cat = cause_type.get('category', '')
        effect_cat = effect_type.get('category', '')
        
        # Door → Temperature is plausible
        if 'DOOR' in cause_tag and 'TEMP' in effect_tag:
            plausibility = 'high'
        # Temperature → Compressor is plausible
        elif 'TEMP' in cause_tag and 'COMPRESSOR' in effect_tag:
            plausibility = 'high'
        # Same category relationships are somewhat plausible
        elif cause_cat == effect_cat:
            plausibility = 'medium'
    
    # Calculate final causal confidence
    causal_confidence = causal_strength
    if plausibility == 'high':
        causal_confidence *= 1.0
    elif plausibility == 'medium':
        causal_confidence *= 0.8
    else:
        causal_confidence *= 0.6
    
    return {
        'cause_tag': cause_tag,
        'effect_tag': effect_tag,
        'analysis_window': {
            'start': window_start.isoformat() if window_start else None,
            'end': window_end.isoformat() if window_end else None
        },
        'best_lag_minutes': float(lag_minutes),
        'best_correlation': float(correlation),
        'direction': direction,
        'causal_strength': float(causal_strength),
        'causal_confidence': float(causal_confidence),
        'physical_plausibility': plausibility
    }


def calculate_impact(
    event_type: str,
    duration_minutes: float,
    severity: float = 1.0,
    price_per_kwh: float | None = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate business impact of an operational event.
    
    Args:
        event_type: Type of event (e.g., 'door_open', 'compressor_failure')
        duration_minutes: Duration of the event in minutes
        severity: Severity multiplier (default: 1.0)
        price_per_kwh: Energy cost in $ per kWh (defaults to ENERGY_COST_KWH if omitted)
        
    Returns:
        BusinessImpact with cost and severity assessment
    """
    # ------------------------------------------------------------------
    # Ensure we have a valid energy price; fall back to the global default
    # if the caller didn't provide one.
    if price_per_kwh is None:
        price_per_kwh = ENERGY_COST_KWH
    # ------------------------------------------------------------------
    energy_cost = 0.0
    product_risk = 0.0
    description = ""
    
    # Calculate impact based on event type
    if event_type == 'door_open':
        # Energy waste from door open events
        base_cost_per_minute = 2.50 * price_per_kwh / 0.12  # Scale by energy price
        energy_cost = duration_minutes * base_cost_per_minute * severity
        
        # Additional costs for longer events
        if duration_minutes > 15:
            product_risk = (duration_minutes - 15) * 1.0 * severity  # Product quality risk
        
        description = f"Door left open for {duration_minutes:.1f} minutes causing energy waste and potential product temperature exposure."
    
    elif event_type == 'compressor_failure':
        # Compressor failure costs
        energy_cost = duration_minutes * 3.0 * price_per_kwh / 0.12 * severity
        product_risk = duration_minutes * 2.0 * severity
        
        description = f"Compressor failure for {duration_minutes:.1f} minutes resulting in energy inefficiency and temperature control issues."
    
    elif event_type == 'temperature_spike':
        # Temperature spike costs
        energy_cost = duration_minutes * 1.5 * price_per_kwh / 0.12 * severity
        
        # Product risk increases with duration
        if duration_minutes > 30:
            product_risk = duration_minutes * 3.0 * severity
        
        description = f"Temperature spike lasting {duration_minutes:.1f} minutes causing increased energy usage and potential product damage."
    
    else:
        # Generic operational event
        energy_cost = duration_minutes * 1.0 * price_per_kwh / 0.12 * severity
        description = f"Operational event lasting {duration_minutes:.1f} minutes with efficiency impact."
    
    # Calculate total cost
    total_cost = energy_cost + product_risk
    
    # Determine severity level
    if total_cost > 100:
        severity_level = 'critical'
    elif total_cost > 50:
        severity_level = 'high'
    elif total_cost > 20:
        severity_level = 'medium'
    else:
        severity_level = 'low'
    
    return {
        'event_type': event_type,
        'duration_minutes': float(duration_minutes),
        'energy_cost': float(energy_cost),
        'product_risk': float(product_risk),
        'total_cost': float(total_cost),
        'severity': severity_level,
        'description': description
    }


def create_event_sequence(
    primary_tag: str,
    related_tags: List[str],
    start_time: Optional[Union[datetime, str]] = None,
    end_time: Optional[Union[datetime, str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a chronological sequence of events across multiple tags.
    
    Args:
        primary_tag: Main tag of interest
        related_tags: List of related tags to include
        start_time: Optional start time for analysis window
        end_time: Optional end time for analysis window
        
    Returns:
        EventSequence with chronological events
    """
    # Parse string time references
    start_time = parse_time_reference(start_time)
    end_time = parse_time_reference(end_time)
    
    # Get time range if not specified
    if start_time is None or end_time is None:
        data_range = get_data_time_range()
        if start_time is None:
            start_time = data_range['start']
        if end_time is None:
            end_time = data_range['end']
    
    # Combine primary tag with related tags
    all_tags = [primary_tag] + related_tags
    
    # Collect events from all tags
    events = []
    
    for tag in all_tags:
        value_type = get_value_type(tag)
        
        if value_type == 'binary':
            # Get binary state changes
            result = detect_binary_flips(tag, start_time, end_time)
            
            for change in result.get('changes', []):
                if change.get('to_state') == 1:  # Only include transitions to high state
                    events.append({
                        'timestamp': change.get('timestamp'),
                        'tag': tag,
                        'value': 1.0,
                        'event_type': 'state_change',
                        'description': change.get('description', ''),
                        'severity': 0.7 if 'DOOR' in tag or 'COMPRESSOR' in tag else 0.5
                    })
        
        elif value_type == 'numeric':
            # Get anomalies and change points
            anomaly_result = detect_numeric_anomalies(tag, start_time, end_time)
            
            for anomaly in anomaly_result.get('anomalies', []):
                events.append({
                    'timestamp': anomaly.get('timestamp'),
                    'tag': tag,
                    'value': anomaly.get('value', 0.0),
                    'event_type': 'anomaly_start',
                    'description': anomaly.get('description', ''),
                    'severity': min(1.0, anomaly.get('z_score', 0.0) / 5.0)
                })
    
    # Sort events chronologically
    events.sort(key=lambda x: x.get('timestamp', ''))
    
    return {
        'events': events,
        'time_window': {
            'start': start_time.isoformat() if start_time else None,
            'end': end_time.isoformat() if end_time else None
        },
        'primary_tag': primary_tag,
        'related_tags': related_tags
    }


def find_interesting_window(
    primary_tag: str,
    start_time: Optional[Union[datetime, str]] = None,
    end_time: Optional[Union[datetime, str]] = None,
    window_hours: int = 2,
    default_expansion_minutes_total: int = 120,
    **kwargs
) -> Dict[str, Any]:
    """
    Find the most interesting sub-window of `window_hours` duration within a given scan range.
    If scan start_time == end_time, it's expanded by `default_expansion_minutes_total` first.
    If the (expanded) scan range is already <= window_hours, it returns this range.
    """
    logger.debug("=" * 60)
    logger.debug("=" * 60)
    logger.debug("=" * 60)
    logger.debug(f"find_interesting_window received call with primary_tag: {primary_tag}, start_time: {start_time}, end_time: {end_time}")
    logger.debug("=" * 60)
    logger.debug("=" * 60)
    logger.debug("=" * 60)
    parsed_scan_start_time = parse_time_reference(start_time)
    parsed_scan_end_time = parse_time_reference(end_time)
    logger.debug("=" * 60)
    logger.debug("=" * 60)
    logger.debug("=" * 60)
    logger.debug(f"find_interesting_window received call with parsed: {primary_tag}, parsed_scan_start_time: {parsed_scan_start_time}, parsed_scan_end_time: {parsed_scan_end_time}")
    logger.debug("=" * 60)
    logger.debug("=" * 60)
    logger.debug("=" * 60)

    # Ensure scan-range endpoints are timezone‑aware UTC before any further logic
    parsed_scan_start_time = _ensure_utc_ts(parsed_scan_start_time)
    parsed_scan_end_time   = _ensure_utc_ts(parsed_scan_end_time)

    # P0.3 Fix: Expand if called with a zero-duration scan window
    if parsed_scan_start_time and parsed_scan_end_time and parsed_scan_start_time == parsed_scan_end_time:
        logger.info(f"find_interesting_window: received zero-duration scan range ({parsed_scan_start_time}). Expanding by {default_expansion_minutes_total} mins.")
        half_expansion = timedelta(minutes=default_expansion_minutes_total / 2)
        parsed_scan_start_time = parsed_scan_start_time - half_expansion
        parsed_scan_end_time = parsed_scan_start_time + timedelta(minutes=default_expansion_minutes_total) # Corrected end time for expansion
        logger.info(f"New expanded scan range for find_interesting_window: {parsed_scan_start_time} to {parsed_scan_end_time}")
    
    # If no explicit scan range after potential expansion, use full data range for the tag
    if parsed_scan_start_time is None or parsed_scan_end_time is None:
        data_range = get_data_time_range(tag=primary_tag)
        if not data_range or not data_range.get('start') or not data_range.get('end'):
            logger.error(f"Could not get valid data range for tag {primary_tag}")
            fallback_start = datetime.now(timezone.utc) - timedelta(hours=window_hours)
            fallback_end = datetime.now(timezone.utc)
            return {
                'primary_tag': primary_tag,
                'full_range': {'start_time': None, 'end_time': None},
                'window': {'start_time': fallback_start.isoformat().replace("+00:00", "Z"), 'end_time': fallback_end.isoformat().replace("+00:00", "Z")},
                'strategy': 'error_no_data_range',
                'significance_score': 0.0 }
        parsed_scan_start_time = parsed_scan_start_time or data_range['start']
        parsed_scan_end_time = parsed_scan_end_time or data_range['end']

    # P0.3: If the (potentially expanded) scan range is already narrow enough, just use it.
    if parsed_scan_start_time and parsed_scan_end_time:
        scan_duration_hours = (parsed_scan_end_time - parsed_scan_start_time).total_seconds() / 3600
        if scan_duration_hours <= window_hours and scan_duration_hours > 0: # Ensure positive duration
            logger.info(f"Scan window for {primary_tag} ({scan_duration_hours:.2f}hrs) is within/equal to desired window_hours ({window_hours}hrs). Using it directly.")
            start_iso = parsed_scan_start_time.isoformat().replace("+00:00", "Z")
            end_iso = parsed_scan_end_time.isoformat().replace("+00:00", "Z")
            return {
                'primary_tag': primary_tag,
                'full_range': {'start_time': start_iso, 'end_time': end_iso},
                'window': {'start_time': start_iso, 'end_time': end_iso},
                'strategy': 'provided_window_used_directly',
                'significance_score': 0.75 
            }
    
    # ... (rest of the logic for loading df, resampling, finding variance/deviation, and returning the sub-window) ...
    # Ensure all df loading and processing uses parsed_scan_start_time and parsed_scan_end_time.
    # Ensure all returned timestamps are ISO and Z-suffixed.
    try:
        df = load_data(primary_tag, parsed_scan_start_time, parsed_scan_end_time)
        
    except Exception as e:
        logger.error(f"Error loading data for {primary_tag} in find_interesting_window: {e}")
        start_iso = parsed_scan_start_time.isoformat().replace("+00:00", "Z") if parsed_scan_start_time else None
        end_iso = (parsed_scan_start_time + timedelta(hours=window_hours)).isoformat().replace("+00:00", "Z") if parsed_scan_start_time else None
        if parsed_scan_end_time and (not parsed_scan_start_time or (parsed_scan_start_time + timedelta(hours=window_hours)) > parsed_scan_end_time):
             end_iso = parsed_scan_end_time.isoformat().replace("+00:00", "Z")
        return { 'primary_tag': primary_tag, 'full_range': {'start_time': parsed_scan_start_time.isoformat().replace("+00:00", "Z") if parsed_scan_start_time else None, 'end_time': parsed_scan_end_time.isoformat().replace("+00:00", "Z") if parsed_scan_end_time else None}, 'window': {'start_time': start_iso, 'end_time': end_iso}, 'strategy': 'error_data_load', 'significance_score': 0.0}

    if df.empty:
        start_iso = parsed_scan_start_time.isoformat().replace("+00:00", "Z") if parsed_scan_start_time else None
        end_iso = (parsed_scan_start_time + timedelta(hours=window_hours)).isoformat().replace("+00:00", "Z") if parsed_scan_start_time else None
        if parsed_scan_end_time and (not parsed_scan_start_time or (parsed_scan_start_time + timedelta(hours=window_hours)) > parsed_scan_end_time):
            end_iso = parsed_scan_end_time.isoformat().replace("+00:00", "Z")
        return { 'primary_tag': primary_tag, 'full_range': {'start_time': parsed_scan_start_time.isoformat().replace("+00:00", "Z") if parsed_scan_start_time else None, 'end_time': parsed_scan_end_time.isoformat().replace("+00:00", "Z") if parsed_scan_end_time else None}, 'window': {'start_time': start_iso, 'end_time': end_iso}, 'strategy': 'error_empty_data', 'significance_score': 0.0}
    
    tag_type = get_value_type(primary_tag)
    effective_scan_start = pd.to_datetime(df['Timestamp'].min()).tz_convert('UTC') if pd.to_datetime(df['Timestamp'].min()).tzinfo else pd.to_datetime(df['Timestamp'].min()).tz_localize('UTC')
    effective_scan_end = pd.to_datetime(df['Timestamp'].max()).tz_convert('UTC') if pd.to_datetime(df['Timestamp'].max()).tzinfo else pd.to_datetime(df['Timestamp'].max()).tz_localize('UTC')

    result_window_start = effective_scan_start
    result_window_end = min(effective_scan_end, effective_scan_start + timedelta(hours=window_hours))
    strategy = 'default_initial_window_or_too_short'
    significance_score = 0.1

    if tag_type == 'binary':
        binary_flips_result = detect_binary_flips(primary_tag, effective_scan_start, effective_scan_end)
        changes = binary_flips_result.get('data', {}).get('changes', []) if isinstance(binary_flips_result.get("data"), dict) else binary_flips_result.get('changes', [])
        if changes:
            # Ensure timestamps from changes are timezone-aware UTC datetimes for comparison
            change_times = sorted([
                datetime.fromisoformat(c['timestamp'].replace('Z', '+00:00')) 
                for c in changes if c.get('timestamp')
            ])
            # Ensure all elements in change_times are timezone-aware
            change_times = [
                ct.tz_localize('UTC') if ct.tzinfo is None else ct.astimezone(timezone.utc) 
                for ct in change_times
            ]

            if change_times:
                max_events_in_sub_window = 0; best_sub_window_start = None
                for t_start_loop_var in change_times: # Renamed t_start to avoid conflict
                    # Ensure t_start_loop_var is UTC aware for arithmetic
                    # (already done by list comprehension above, but double check context)
                    t_end_candidate = t_start_loop_var + timedelta(hours=window_hours)
                    events_count = sum(1 for ct in change_times if t_start_loop_var <= ct < t_end_candidate)
                    if events_count > max_events_in_sub_window:
                        max_events_in_sub_window = events_count
                        best_sub_window_start = t_start_loop_var
                if best_sub_window_start:
                    result_window_start = best_sub_window_start
                    result_window_end = min(effective_scan_end, best_sub_window_start + timedelta(hours=window_hours))
                    strategy = 'binary_max_density'
                    significance_score = min(0.9, max_events_in_sub_window * 0.2)
    else: 
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df_resampled = df.set_index('Timestamp')['Value'].resample('5min').mean().dropna().reset_index()
        if len(df_resampled) >= 3:
            rolling_window_size = int(window_hours * 60 / 5)
            if rolling_window_size < 1: rolling_window_size = 1
            rolling_var = df_resampled['Value'].rolling(window=rolling_window_size, center=True, min_periods=1).var()
            if not rolling_var.isnull().all():
                max_var_idx = rolling_var.idxmax()
                if pd.notna(max_var_idx) and max_var_idx < len(df_resampled):
                    center_time_resampled = pd.to_datetime(df_resampled.iloc[max_var_idx]['Timestamp'])
                    center_time_resampled = center_time_resampled.tz_convert('UTC') if center_time_resampled.tzinfo else center_time_resampled.tz_localize('UTC')
                    half_target_window_td = timedelta(hours=window_hours/2)
                    calc_start = center_time_resampled - half_target_window_td
                    calc_end = center_time_resampled + half_target_window_td
                    result_window_start = max(effective_scan_start, calc_start)
                    result_window_end = min(effective_scan_end, calc_end)
                    if result_window_start >= result_window_end : # Safety for very narrow effective scan ranges
                        result_window_start = effective_scan_start
                        result_window_end = min(effective_scan_end, effective_scan_start + timedelta(hours=window_hours))
                    elif (result_window_end - result_window_start).total_seconds() / 3600 < window_hours * 0.9: # If clamped window is too small
                        # Try to center it better if possible, or expand if against one boundary
                        if calc_start < effective_scan_start: # Clamped by start
                            result_window_end = min(effective_scan_end, result_window_start + timedelta(hours=window_hours))
                        elif calc_end > effective_scan_end: # Clamped by end
                            result_window_start = max(effective_scan_start, result_window_end - timedelta(hours=window_hours))

                    strategy = 'numeric_max_variance'
                    median_var = rolling_var.median()
                    if median_var > 0 and pd.notna(rolling_var.iloc[max_var_idx]): 
                        significance_score = float(min(0.9, (rolling_var.iloc[max_var_idx] / median_var) / 5.0))
                    else: significance_score = 0.3

    start_iso_out = result_window_start.isoformat().replace("+00:00", "Z")
    end_iso_out = result_window_end.isoformat().replace("+00:00", "Z")

    return {
        'primary_tag': primary_tag,
        'full_range': {
            'start_time': parsed_scan_start_time.isoformat().replace("+00:00", "Z") if parsed_scan_start_time else None,
            'end_time': parsed_scan_end_time.isoformat().replace("+00:00", "Z") if parsed_scan_end_time else None
        },
        'window': {'start_time': start_iso_out, 'end_time': end_iso_out},
        'strategy': strategy,
        'significance_score': round(significance_score, 3)
    }


def _safe_year(dt: datetime, now: datetime) -> datetime:
    """Ensures the year of dt is not in the past relative to now, correcting to now.year if it is."""
    # Ensure both are timezone-aware for comparison if one is, or make them naive for comparison.
    # Assuming now is timezone-aware (e.g., UTC)
    # If dt is naive, localize it or compare years directly.
    dt_year = dt.year
    now_year = now.year
    if dt_year < now_year:
        logger.warning(f"Correcting past year {dt_year} in date {dt} to current year {now_year}.")
        return dt.replace(year=now_year)
    return dt

def parse_time_range(start_time: Optional[str] = None, end_time: Optional[str] = None, query: Optional[str] = None, **kwargs) -> Dict[str, Optional[str]]:
    """
    Parses or infers a time range. 
    If start_time and end_time are provided by the LLM (as ISO 8601 UTC strings), they are validated and used.
    If not, and a query is provided, uses dateparser to find time references in the query.
    The base_date_for_relative_queries (effectively 'now') is taken from the orchestrator's today_iso.
    Returns a dictionary containing 'start_time' and 'end_time' as ISO 8601 UTC strings.
    """
    logger.info(f"parse_time_range received - LLM start: {start_time}, LLM end: {end_time}, Query: {query}")

    # Get 'now' from orchestrator if available (passed via kwargs), else use current time.
    # This is crucial for dateparser to correctly interpret relative dates like "yesterday".
    base_date_str = kwargs.get('today_iso_from_orchestrator')
    now_utc = pd.to_datetime(base_date_str).tz_convert(timezone.utc).to_pydatetime() if base_date_str else datetime.now(timezone.utc)

    processed_start_dt: Optional[datetime] = None
    processed_end_dt: Optional[datetime] = None

    if start_time: # LLM provided a start_time
        try:
            dt_start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            if dt_start.tzinfo is None: dt_start = dt_start.replace(tzinfo=timezone.utc)
            processed_start_dt = _safe_year(dt_start, now_utc)
        except ValueError:
            logger.warning(f"Could not parse LLM-provided start_time '{start_time}' as ISO. Will try query.")
            start_time = None # Invalidate to trigger query parsing

    if end_time: # LLM provided an end_time
        try:
            dt_end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            if dt_end.tzinfo is None: dt_end = dt_end.replace(tzinfo=timezone.utc)
            processed_end_dt = _safe_year(dt_end, now_utc)
        except ValueError:
            logger.warning(f"Could not parse LLM-provided end_time '{end_time}' as ISO. Will try query if start_time also failed.")
            end_time = None # Invalidate
    
    # If LLM didn't provide valid start/end, or only one, try parsing the query
    if not processed_start_dt and query:
        logger.info(f"Attempting to parse time from query: '{query}' using dateparser relative to {now_utc.isoformat()})")
        try:
            import dateparser.search
            # Settings for dateparser
            # PREFER_DATES_FROM: 'past' helps with "yesterday", "last week"
            # TIMEZONE: 'UTC' ensures parsed datetimes are in UTC if not specified
            # RETURN_AS_TIMEZONE_AWARE: Ensures datetime objects have tzinfo
            # NORMALIZE: True can help with some varied inputs
            dp_settings = {
                'PREFER_DATES_FROM': 'past',
                'TIMEZONE': 'UTC',
                'RETURN_AS_TIMEZONE_AWARE': True,
                'RELATIVE_BASE': now_utc # Critically important for relative dates
            }
            
            found_dates = dateparser.search.search_dates(query, languages=['en'], settings=dp_settings)
            
            if found_dates:
                logger.info(f"Dateparser found: {found_dates}")
                # Logic to handle found_dates: prioritize, set default windows
                # For simplicity, let's take the first one found as a reference point.
                # A more complex implementation could look for pairs or specific keywords.
                
                ref_text, ref_dt = found_dates[0]
                ref_dt = _safe_year(ref_dt, now_utc) # Ensure year is correct

                # Default windowing logic based on what dateparser found
                if "afternoon" in ref_text.lower():
                    processed_start_dt = ref_dt.replace(hour=12, minute=0, second=0, microsecond=0)
                    processed_end_dt = ref_dt.replace(hour=17, minute=0, second=0, microsecond=0)
                elif "morning" in ref_text.lower():
                    processed_start_dt = ref_dt.replace(hour=8, minute=0, second=0, microsecond=0)
                    processed_end_dt = ref_dt.replace(hour=12, minute=0, second=0, microsecond=0)
                elif "evening" in ref_text.lower() or "night" in ref_text.lower():
                    processed_start_dt = ref_dt.replace(hour=18, minute=0, second=0, microsecond=0)
                    processed_end_dt = ref_dt.replace(hour=23, minute=59, second=59, microsecond=0)
                elif "around" in ref_text.lower() or (ref_dt.hour != 0 or ref_dt.minute != 0 or ref_dt.second != 0):
                    # A specific time was mentioned, create a +/- 30min window or similar
                    processed_start_dt = ref_dt - timedelta(minutes=30)
                    processed_end_dt = ref_dt + timedelta(minutes=30)
                else: # Only a date, no specific time of day implies full day or common business hours
                    processed_start_dt = ref_dt.replace(hour=9, minute=0, second=0, microsecond=0)
                    processed_end_dt = ref_dt.replace(hour=17, minute=0, second=0, microsecond=0)
                
                # If only one of start/end was provided by LLM, and query parsing yielded a window,
                # prefer the query-parsed window for now. This logic can be refined.
                if start_time and not end_time and processed_start_dt and processed_end_dt: # LLM gave start, query gave window
                    logger.info("LLM gave start_time, query parsing yielded a window. Using query-parsed window for consistency.")
                elif end_time and not start_time and processed_start_dt and processed_end_dt: # LLM gave end_time, query gave window
                     logger.info("LLM gave end_time, query parsing yielded a window. Using query-parsed window for consistency.")

            else:
                logger.info("Dateparser found no specific dates in query. Falling back to default.")
                # Fallback if dateparser finds nothing and LLM didn't provide times
                pivot_time = now_utc.replace(hour=14, minute=30, second=0, microsecond=0) - timedelta(days=1)
                processed_start_dt = pivot_time - timedelta(minutes=30)
                processed_end_dt = pivot_time + timedelta(minutes=30)
                logger.info(f"Defaulted to yesterday around 14:30 UTC: {processed_start_dt.isoformat()} to {processed_end_dt.isoformat()}")

        except ImportError:
            logger.error("Dateparser library is not installed. Please install it: pip install dateparser")
            # Fallback to very basic yesterday logic if dateparser is missing
            if query and "yesterday" in query.lower():
                pivot_time = now_utc.replace(hour=14, minute=0, second=0, microsecond=0) - timedelta(days=1)
                processed_start_dt = pivot_time
                processed_end_dt   = pivot_time + timedelta(hours=1)
        except Exception as e:
            logger.exception(f"Error during dateparser processing: {e}")
            # Fallback on any other dateparser error
            if query and "yesterday" in query.lower():
                pivot_time = now_utc.replace(hour=14, minute=0, second=0, microsecond=0) - timedelta(days=1)
                processed_start_dt = pivot_time
                processed_end_dt   = pivot_time + timedelta(hours=1)

    # If only one of start/end is determined, create a default window around it
    if processed_start_dt and not processed_end_dt:
        processed_end_dt = processed_start_dt + timedelta(hours=1)
        logger.info(f"Only start_time determined. Setting end_time to 1 hour after: {processed_end_dt.isoformat()}")
    elif not processed_start_dt and processed_end_dt:
        processed_start_dt = processed_end_dt - timedelta(hours=1)
        logger.info(f"Only end_time determined. Setting start_time to 1 hour before: {processed_start_dt.isoformat()}")
    
    # Final check for start < end
    if processed_start_dt and processed_end_dt and processed_start_dt >= processed_end_dt:
        logger.warning(f"Corrected: start_time {processed_start_dt.isoformat()} was not before end_time {processed_end_dt.isoformat()}. Adjusting end_time.")
        processed_end_dt = processed_start_dt + timedelta(hours=1) # Ensure end is after start

    final_start_iso = processed_start_dt.isoformat().replace("+00:00", "Z") if processed_start_dt else None
    final_end_iso = processed_end_dt.isoformat().replace("+00:00", "Z") if processed_end_dt else None

    # Ensure this tool doesn't cause an error if it still can't determine times.
    # Consuming tools MUST validate the window they receive from this.
    if not final_start_iso and not final_end_iso:
        logger.warning("parse_time_range could not determine any time window. Returning None for both.")

    logger.info(f"parse_time_range returning: start_time='{final_start_iso}', end_time='{final_end_iso}'")
    return {"start_time": final_start_iso, "end_time": final_end_iso}


def main():
    """Demo function to test atomic tools."""
    print("🧰 Manufacturing Copilot - Atomic Tools Demo")
    print("=" * 60)
    
    # Demo data time range
    from datetime import datetime
    start_time = datetime(2025, 5, 25, 14, 0, 0)
    end_time = datetime(2025, 5, 25, 16, 0, 0)
    
    print(f"Demo time range: {start_time} to {end_time}")
    
    # Test interesting window finder
    print("\n🔍 Testing find_interesting_window...")
    interesting_window = find_interesting_window("FREEZER01.TEMP.INTERNAL_C", start_time, end_time)
    print(f"Found interesting window: {interesting_window.get('window', {}).get('start_time')} to {interesting_window.get('window', {}).get('end_time')}")
    print(f"Strategy: {interesting_window.get('strategy')}")
    print(f"Significance: {interesting_window.get('significance_score'):.2f}")
    
    # Test numeric anomaly detection
    print("\n📈 Testing detect_numeric_anomalies on temperature tag...")
    temp_anomalies = detect_numeric_anomalies("FREEZER01.TEMP.INTERNAL_C", start_time, end_time)
    print(f"Found {len(temp_anomalies.get('anomalies', []))} temperature anomalies")
    print(f"Severity score: {temp_anomalies.get('severity_score', 0.0):.2f}")
    
    # Test binary state change detection
    print("\n🔄 Testing detect_binary_flips on door tag...")
    door_changes = detect_binary_flips("FREEZER01.DOOR.STATUS", start_time, end_time)
    print(f"Found {door_changes.get('total_changes', 0)} door state changes")
    print(f"Total door open duration: {door_changes.get('total_high_duration_minutes', 0.0):.1f} minutes")
    
    # Test causality analysis
    print("\n🔍 Testing test_causality between door and temperature...")
    causality = test_causality("FREEZER01.DOOR.STATUS", "FREEZER01.TEMP.INTERNAL_C", start_time, end_time)
    print(f"Best lag: {causality.get('best_lag_minutes', 0.0):.1f} minutes")
    print(f"Correlation: {causality.get('best_correlation', 0.0):.3f}")
    print(f"Causal confidence: {causality.get('causal_confidence', 0.0):.2f}")
    
    # Test business impact calculation
    print("\n💰 Testing calculate_impact for door open event...")
    impact = calculate_impact("door_open", 15.0)
    print(f"Total cost: ${impact.get('total_cost', 0.0):.2f}")
    print(f"Impact severity: {impact.get('severity', 'unknown')}")
    
    print("\n" + "=" * 60)
    print("✅ Atomic tools demo complete!")


if __name__ == "__main__":
    main() 