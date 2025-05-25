#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Anomaly Detection Tools

Provides spike detection and anomaly identification for manufacturing data
using statistical methods like z-score analysis and rolling window detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def detect_spike(df: pd.DataFrame, threshold: float = 3.0, window_size: int = 10) -> List[Tuple[datetime, float, float, str]]:
    """
    Detect abnormal changes in time series data using z-score analysis.
    
    Uses rolling statistics to identify data points that deviate significantly
    from the local mean, indicating potential equipment malfunctions, sensor
    errors, or operational anomalies.
    
    Args:
        df: DataFrame with time-series data (must have 'Timestamp' and 'Value' columns)
        threshold: Z-score threshold for anomaly detection (default: 3.0)
        window_size: Rolling window size for calculating local statistics (default: 10)
        
    Returns:
        List of tuples containing (timestamp, value, z_score, reason) for detected anomalies
        
    Raises:
        ValueError: If required columns are missing or data is insufficient
    """
    logger.debug(f"Starting spike detection with threshold={threshold}, window_size={window_size}")
    
    # Validate input data
    if df.empty:
        logger.warning("Empty DataFrame provided to detect_spike")
        return []
    
    required_columns = ['Timestamp', 'Value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < window_size:
        logger.warning(f"Insufficient data points ({len(df)}) for window size ({window_size})")
        return []
    
    # Ensure data is sorted by timestamp
    df_sorted = df.sort_values('Timestamp').copy()
    
    # Calculate rolling statistics
    df_sorted['rolling_mean'] = df_sorted['Value'].rolling(window=window_size, center=True).mean()
    df_sorted['rolling_std'] = df_sorted['Value'].rolling(window=window_size, center=True).std()
    
    # Calculate z-scores (how many standard deviations from the rolling mean)
    df_sorted['z_score'] = np.abs((df_sorted['Value'] - df_sorted['rolling_mean']) / df_sorted['rolling_std'])
    
    # Handle division by zero (when std is 0)
    df_sorted['z_score'] = df_sorted['z_score'].fillna(0)
    
    # Identify anomalies
    anomalies = df_sorted[df_sorted['z_score'] > threshold].copy()
    
    if anomalies.empty:
        logger.info("No anomalies detected")
        return []
    
    # Generate anomaly descriptions
    anomaly_list = []
    for _, row in anomalies.iterrows():
        timestamp = row['Timestamp']
        value = row['Value']
        z_score = row['z_score']
        rolling_mean = row['rolling_mean']
        
        # Determine anomaly type
        if value > rolling_mean:
            if z_score > 5.0:
                reason = f"Extreme high spike ({z_score:.1f}σ above local mean)"
            else:
                reason = f"High spike ({z_score:.1f}σ above local mean)"
        else:
            if z_score > 5.0:
                reason = f"Extreme low spike ({z_score:.1f}σ below local mean)"
            else:
                reason = f"Low spike ({z_score:.1f}σ below local mean)"
        
        anomaly_list.append((timestamp, value, z_score, reason))
    
    logger.info(f"Detected {len(anomaly_list)} anomalies with threshold {threshold}")
    return anomaly_list


def detect_consecutive_anomalies(df: pd.DataFrame, threshold: float = 3.0, 
                                min_duration_minutes: int = 5) -> List[Tuple[datetime, datetime, str]]:
    """
    Detect periods of consecutive anomalies that might indicate sustained issues.
    
    Args:
        df: DataFrame with time-series data
        threshold: Z-score threshold for anomaly detection
        min_duration_minutes: Minimum duration for a sustained anomaly period
        
    Returns:
        List of tuples containing (start_time, end_time, description) for sustained anomalies
    """
    anomalies = detect_spike(df, threshold)
    
    if not anomalies:
        return []
    
    # Group consecutive anomalies
    periods = []
    current_start = None
    current_end = None
    
    for i, (timestamp, value, z_score, reason) in enumerate(anomalies):
        if current_start is None:
            current_start = timestamp
            current_end = timestamp
        else:
            # Check if this anomaly is close to the previous one (within 2 minutes)
            time_gap = (timestamp - current_end).total_seconds() / 60
            if time_gap <= 2:
                current_end = timestamp
            else:
                # End current period and start new one
                duration = (current_end - current_start).total_seconds() / 60
                if duration >= min_duration_minutes:
                    periods.append((current_start, current_end, 
                                  f"Sustained anomaly period ({duration:.1f} minutes)"))
                current_start = timestamp
                current_end = timestamp
    
    # Handle the last period
    if current_start is not None:
        duration = (current_end - current_start).total_seconds() / 60
        if duration >= min_duration_minutes:
            periods.append((current_start, current_end, 
                          f"Sustained anomaly period ({duration:.1f} minutes)"))
    
    logger.info(f"Detected {len(periods)} sustained anomaly periods")
    return periods


def analyze_anomaly_patterns(df: pd.DataFrame, threshold: float = 3.0) -> dict:
    """
    Analyze patterns in detected anomalies to provide insights.
    
    Args:
        df: DataFrame with time-series data
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        Dictionary with anomaly analysis results
    """
    anomalies = detect_spike(df, threshold)
    
    if not anomalies:
        return {
            'total_anomalies': 0,
            'anomaly_rate': 0.0,
            'severity_distribution': {},
            'time_distribution': {}
        }
    
    # Calculate anomaly rate
    total_points = len(df)
    anomaly_rate = len(anomalies) / total_points * 100
    
    # Analyze severity distribution
    severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'extreme': 0}
    for _, _, z_score, _ in anomalies:
        if z_score < 3.5:
            severity_counts['low'] += 1
        elif z_score < 5.0:
            severity_counts['medium'] += 1
        elif z_score < 7.0:
            severity_counts['high'] += 1
        else:
            severity_counts['extreme'] += 1
    
    # Analyze time distribution (by hour of day)
    time_counts = {}
    for timestamp, _, _, _ in anomalies:
        hour = timestamp.hour
        time_counts[hour] = time_counts.get(hour, 0) + 1
    
    return {
        'total_anomalies': len(anomalies),
        'anomaly_rate': anomaly_rate,
        'severity_distribution': severity_counts,
        'time_distribution': time_counts,
        'most_active_hour': max(time_counts.items(), key=lambda x: x[1])[0] if time_counts else None
    }


def main():
    """
    Demo and test the anomaly detection functions with synthetic data.
    
    This is a demo function for testing - not part of the core MCP pipeline.
    """
    print("🔍 Manufacturing Copilot - Anomaly Detection Demo")
    print("=" * 55)
    
    # Test with real data if available
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        from src.tools.data_loader import load_data
        
        print("📊 Testing with real freezer data...")
        
        # Load temperature data (should contain injected anomalies)
        temp_data = load_data("FREEZER01.TEMP.INTERNAL_C")
        print(f"   Loaded {len(temp_data)} temperature data points")
        
        # Test basic spike detection
        print("\n🎯 Basic Spike Detection (threshold=3.0):")
        anomalies = detect_spike(temp_data, threshold=3.0)
        
        if anomalies:
            print(f"   Found {len(anomalies)} anomalies:")
            for timestamp, value, z_score, reason in anomalies[:5]:  # Show first 5
                print(f"   • {timestamp}: {value:.2f}°C - {reason}")
            if len(anomalies) > 5:
                print(f"   ... and {len(anomalies) - 5} more")
        else:
            print("   No anomalies detected")
        
        # Test consecutive anomaly detection
        print("\n⏱️  Consecutive Anomaly Detection:")
        periods = detect_consecutive_anomalies(temp_data, threshold=3.0, min_duration_minutes=5)
        
        if periods:
            print(f"   Found {len(periods)} sustained anomaly periods:")
            for start, end, description in periods:
                duration = (end - start).total_seconds() / 60
                print(f"   • {start} to {end}: {description}")
        else:
            print("   No sustained anomaly periods detected")
        
        # Test pattern analysis
        print("\n📈 Anomaly Pattern Analysis:")
        patterns = analyze_anomaly_patterns(temp_data, threshold=3.0)
        print(f"   Total anomalies: {patterns['total_anomalies']}")
        print(f"   Anomaly rate: {patterns['anomaly_rate']:.2f}%")
        print(f"   Severity distribution: {patterns['severity_distribution']}")
        if patterns['most_active_hour'] is not None:
            print(f"   Most active hour: {patterns['most_active_hour']}:00")
        
        # Test with power data (should show different patterns)
        print("\n⚡ Testing with power consumption data...")
        power_data = load_data("FREEZER01.COMPRESSOR.POWER_KW")
        power_anomalies = detect_spike(power_data, threshold=2.5)  # Lower threshold for power
        
        if power_anomalies:
            print(f"   Found {len(power_anomalies)} power anomalies:")
            for timestamp, value, z_score, reason in power_anomalies[:3]:
                print(f"   • {timestamp}: {value:.2f}kW - {reason}")
        else:
            print("   No power anomalies detected")
            
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")
        print("📝 Testing with synthetic data instead...")
        
        # Create synthetic test data with known anomalies
        timestamps = pd.date_range('2025-01-01', periods=100, freq='1min')
        values = np.random.normal(20, 2, 100)  # Normal data around 20°C
        
        # Inject known anomalies
        values[30] = 35  # High spike
        values[31] = 34  # Consecutive high
        values[60] = 5   # Low spike
        
        test_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Value': values
        })
        
        print("🧪 Testing with synthetic data (100 points, 3 injected anomalies)...")
        
        anomalies = detect_spike(test_df, threshold=2.0)
        print(f"   Detected {len(anomalies)} anomalies:")
        for timestamp, value, z_score, reason in anomalies:
            print(f"   • {timestamp}: {value:.2f} - {reason}")
    
    print("\n" + "=" * 55)
    print("✅ Anomaly detection testing complete!")
    print("\n🔧 Usage in code:")
    print("   from src.tools.anomaly_detection import detect_spike")
    print("   anomalies = detect_spike(df, threshold=3.0)")


if __name__ == "__main__":
    main() 