#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Correlation Analysis Tools

Provides correlation analysis between manufacturing metrics to identify
relationships and potential root causes during anomaly periods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def correlate_tags(primary_df: pd.DataFrame, candidate_dfs: List[pd.DataFrame], 
                  window: Tuple[datetime, datetime]) -> List[Dict]:
    """
    Calculate correlation between a primary tag and candidate tags during a specific time window.
    
    Analyzes multiple types of correlations to identify relationships that might
    indicate causal connections during anomaly periods.
    
    Args:
        primary_df: DataFrame with primary metric data (Timestamp, Value columns)
        candidate_dfs: List of DataFrames with candidate metric data
        window: Tuple of (start_time, end_time) defining the analysis window
        
    Returns:
        List of dictionaries containing correlation results, sorted by relevance
        
    Raises:
        ValueError: If input data is invalid or insufficient
    """
    logger.debug(f"Starting correlation analysis for window {window[0]} to {window[1]}")
    
    # Validate inputs
    if primary_df.empty:
        logger.warning("Primary DataFrame is empty")
        return []
    
    if not candidate_dfs:
        logger.warning("No candidate DataFrames provided")
        return []
    
    start_time, end_time = window
    
    # Filter primary data to window
    primary_windowed = primary_df[
        (primary_df['Timestamp'] >= start_time) & 
        (primary_df['Timestamp'] <= end_time)
    ].copy()
    
    if primary_windowed.empty:
        logger.warning("No primary data in specified window")
        return []
    
    correlations = []
    
    for i, candidate_df in enumerate(candidate_dfs):
        if candidate_df.empty:
            continue
            
        # Get tag name from DataFrame if available, otherwise use index
        tag_name = candidate_df.get('TagName', [f'Tag_{i}'])[0] if 'TagName' in candidate_df.columns else f'Tag_{i}'
        
        try:
            correlation_result = _analyze_tag_correlation(
                primary_windowed, candidate_df, window, tag_name
            )
            if correlation_result:
                correlations.append(correlation_result)
                
        except Exception as e:
            logger.warning(f"Error analyzing correlation for {tag_name}: {e}")
            continue
    
    # Sort by overall correlation strength (absolute value)
    correlations.sort(key=lambda x: abs(x.get('pearson_correlation', 0)), reverse=True)
    
    logger.info(f"Completed correlation analysis: {len(correlations)} valid correlations found")
    return correlations


def _analyze_tag_correlation(primary_df: pd.DataFrame, candidate_df: pd.DataFrame, 
                           window: Tuple[datetime, datetime], tag_name: str) -> Optional[Dict]:
    """
    Analyze correlation between primary and candidate tag within a time window.
    
    Args:
        primary_df: Primary metric DataFrame (already filtered to window)
        candidate_df: Candidate metric DataFrame
        window: Time window tuple
        tag_name: Name of the candidate tag
        
    Returns:
        Dictionary with correlation analysis results or None if analysis fails
    """
    start_time, end_time = window
    
    # Filter candidate data to window
    candidate_windowed = candidate_df[
        (candidate_df['Timestamp'] >= start_time) & 
        (candidate_df['Timestamp'] <= end_time)
    ].copy()
    
    if candidate_windowed.empty:
        return None
    
    # Align timestamps by merging on nearest timestamp
    merged_df = pd.merge_asof(
        primary_df.sort_values('Timestamp'),
        candidate_windowed.sort_values('Timestamp'),
        on='Timestamp',
        suffixes=('_primary', '_candidate'),
        tolerance=pd.Timedelta('2min')  # Allow 2-minute tolerance for alignment
    )
    
    # Remove rows where either value is missing
    merged_df = merged_df.dropna(subset=['Value_primary', 'Value_candidate'])
    
    if len(merged_df) < 3:  # Need at least 3 points for meaningful correlation
        return None
    
    primary_values = merged_df['Value_primary'].values
    candidate_values = merged_df['Value_candidate'].values
    
    # Calculate Pearson correlation
    pearson_corr = np.corrcoef(primary_values, candidate_values)[0, 1]
    
    # Calculate change correlation (correlation of differences)
    primary_changes = np.diff(primary_values)
    candidate_changes = np.diff(candidate_values)
    
    change_corr = 0.0
    if len(primary_changes) > 1:
        change_corr = np.corrcoef(primary_changes, candidate_changes)[0, 1]
    
    # Calculate time-lagged correlation (candidate leading primary)
    lag_corr = _calculate_lagged_correlation(primary_values, candidate_values)
    
    # Calculate statistical significance
    n_points = len(merged_df)
    significance = _calculate_correlation_significance(pearson_corr, n_points)
    
    # Determine correlation strength and interpretation
    strength = _interpret_correlation_strength(pearson_corr)
    
    # Calculate value ranges for context
    primary_range = (primary_values.min(), primary_values.max())
    candidate_range = (candidate_values.min(), candidate_values.max())
    
    return {
        'tag_name': tag_name,
        'pearson_correlation': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
        'change_correlation': float(change_corr) if not np.isnan(change_corr) else 0.0,
        'lagged_correlation': lag_corr,
        'correlation_strength': strength,
        'statistical_significance': significance,
        'data_points': n_points,
        'primary_range': primary_range,
        'candidate_range': candidate_range,
        'time_window': window
    }


def _calculate_lagged_correlation(primary_values: np.ndarray, candidate_values: np.ndarray, 
                                max_lag: int = 5) -> Dict[str, float]:
    """
    Calculate correlation with time lags to detect leading/lagging relationships.
    
    Args:
        primary_values: Primary metric values
        candidate_values: Candidate metric values
        max_lag: Maximum lag to test (in data points)
        
    Returns:
        Dictionary with best lag correlation information
    """
    if len(primary_values) <= max_lag or len(candidate_values) <= max_lag:
        return {'best_lag': 0, 'best_correlation': 0.0}
    
    best_lag = 0
    best_corr = 0.0
    
    # Test positive lags (candidate leading primary)
    for lag in range(1, min(max_lag + 1, len(candidate_values) - 1)):
        if len(primary_values[lag:]) > 1 and len(candidate_values[:-lag]) > 1:
            corr = np.corrcoef(primary_values[lag:], candidate_values[:-lag])[0, 1]
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
    
    # Test negative lags (primary leading candidate)
    for lag in range(1, min(max_lag + 1, len(primary_values) - 1)):
        if len(primary_values[:-lag]) > 1 and len(candidate_values[lag:]) > 1:
            corr = np.corrcoef(primary_values[:-lag], candidate_values[lag:])[0, 1]
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = -lag
    
    return {
        'best_lag': best_lag,
        'best_correlation': float(best_corr) if not np.isnan(best_corr) else 0.0
    }


def _calculate_correlation_significance(correlation: float, n_points: int) -> str:
    """
    Calculate statistical significance of correlation coefficient.
    
    Args:
        correlation: Pearson correlation coefficient
        n_points: Number of data points
        
    Returns:
        String describing significance level
    """
    if np.isnan(correlation) or n_points < 3:
        return 'insufficient_data'
    
    # Calculate t-statistic for correlation
    t_stat = abs(correlation) * np.sqrt((n_points - 2) / (1 - correlation**2))
    
    # Rough significance thresholds (for quick assessment)
    if t_stat > 3.0:  # Approximately p < 0.01
        return 'highly_significant'
    elif t_stat > 2.0:  # Approximately p < 0.05
        return 'significant'
    elif t_stat > 1.0:
        return 'marginally_significant'
    else:
        return 'not_significant'


def _interpret_correlation_strength(correlation: float) -> str:
    """
    Interpret the strength of a correlation coefficient.
    
    Args:
        correlation: Pearson correlation coefficient
        
    Returns:
        String describing correlation strength
    """
    if np.isnan(correlation):
        return 'undefined'
    
    abs_corr = abs(correlation)
    
    if abs_corr >= 0.8:
        return 'very_strong'
    elif abs_corr >= 0.6:
        return 'strong'
    elif abs_corr >= 0.4:
        return 'moderate'
    elif abs_corr >= 0.2:
        return 'weak'
    else:
        return 'very_weak'


def find_correlated_tags(primary_tag: str, start_time: datetime, end_time: datetime, 
                        correlation_threshold: float = 0.3) -> List[Dict]:
    """
    Find tags that are correlated with a primary tag during a specific time window.
    
    Convenience function that loads data for all available tags and finds correlations.
    
    Args:
        primary_tag: Name of the primary tag to analyze
        start_time: Start of analysis window
        end_time: End of analysis window
        correlation_threshold: Minimum correlation strength to include in results
        
    Returns:
        List of correlation results above the threshold
    """
    try:
        from .data_loader import load_data, get_available_tags
        
        # Load primary tag data
        primary_df = load_data(primary_tag, start_time, end_time)
        
        if primary_df.empty:
            logger.warning(f"No data found for primary tag {primary_tag}")
            return []
        
        # Get all available tags except the primary one
        all_tags = get_available_tags()
        candidate_tags = [tag for tag in all_tags if tag != primary_tag]
        
        # Load candidate tag data
        candidate_dfs = []
        for tag in candidate_tags:
            try:
                tag_df = load_data(tag, start_time, end_time)
                if not tag_df.empty:
                    candidate_dfs.append(tag_df)
            except Exception as e:
                logger.debug(f"Could not load data for tag {tag}: {e}")
                continue
        
        # Perform correlation analysis
        correlations = correlate_tags(primary_df, candidate_dfs, (start_time, end_time))
        
        # Filter by threshold
        significant_correlations = [
            corr for corr in correlations 
            if abs(corr.get('pearson_correlation', 0)) >= correlation_threshold
        ]
        
        logger.info(f"Found {len(significant_correlations)} correlations above threshold {correlation_threshold}")
        return significant_correlations
        
    except Exception as e:
        logger.error(f"Error in find_correlated_tags: {e}")
        return []


def main():
    """
    Demo and test the correlation analysis functions.
    
    This is a demo function for testing - not part of the core MCP pipeline.
    """
    print("🔗 Manufacturing Copilot - Correlation Analysis Demo")
    print("=" * 58)
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        from src.tools.data_loader import load_data
        from datetime import datetime, timedelta
        
        print("📊 Testing correlation analysis with freezer data...")
        
        # Define a time window that should contain anomalies
        end_time = datetime(2025, 5, 23, 23, 59)
        start_time = end_time - timedelta(hours=24)
        
        print(f"   Analysis window: {start_time} to {end_time}")
        
        # Load primary tag (temperature)
        primary_df = load_data("FREEZER01.TEMP.INTERNAL_C", start_time, end_time)
        print(f"   Primary tag data points: {len(primary_df)}")
        
        # Load candidate tags
        candidate_tags = [
            "FREEZER01.COMPRESSOR.POWER_KW",
            "FREEZER01.COMPRESSOR.STATUS", 
            "FREEZER01.DOOR.STATUS",
            "FREEZER01.TEMP.AMBIENT_C"
        ]
        
        candidate_dfs = []
        for tag in candidate_tags:
            tag_df = load_data(tag, start_time, end_time)
            candidate_dfs.append(tag_df)
            print(f"   {tag}: {len(tag_df)} data points")
        
        # Perform correlation analysis
        print("\n🔍 Correlation Analysis Results:")
        correlations = correlate_tags(primary_df, candidate_dfs, (start_time, end_time))
        
        if correlations:
            for i, corr in enumerate(correlations, 1):
                print(f"\n   {i}. {corr['tag_name']}")
                print(f"      Pearson correlation: {corr['pearson_correlation']:.3f} ({corr['correlation_strength']})")
                print(f"      Change correlation: {corr['change_correlation']:.3f}")
                print(f"      Lagged correlation: {corr['lagged_correlation']['best_correlation']:.3f} (lag: {corr['lagged_correlation']['best_lag']})")
                print(f"      Significance: {corr['statistical_significance']}")
                print(f"      Data points: {corr['data_points']}")
        else:
            print("   No correlations found")
        
        # Test the convenience function
        print("\n🎯 Testing find_correlated_tags convenience function:")
        significant_corrs = find_correlated_tags(
            "FREEZER01.TEMP.INTERNAL_C", 
            start_time, 
            end_time, 
            correlation_threshold=0.2
        )
        
        print(f"   Found {len(significant_corrs)} significant correlations (threshold: 0.2)")
        for corr in significant_corrs:
            direction = "positive" if corr['pearson_correlation'] > 0 else "negative"
            print(f"   • {corr['tag_name']}: {corr['pearson_correlation']:.3f} ({direction})")
            
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")
        print("📝 Testing with synthetic data...")
        
        # Create synthetic correlated data
        timestamps = pd.date_range('2025-01-01', periods=100, freq='1min')
        
        # Primary signal (temperature)
        primary_values = 20 + 2 * np.sin(np.arange(100) * 0.1) + np.random.normal(0, 0.5, 100)
        primary_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Value': primary_values,
            'TagName': 'TEMP'
        })
        
        # Candidate 1: Positively correlated (power consumption)
        power_values = 5 + 0.5 * primary_values + np.random.normal(0, 0.3, 100)
        power_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Value': power_values,
            'TagName': 'POWER'
        })
        
        # Candidate 2: Negatively correlated (efficiency)
        efficiency_values = 100 - 2 * primary_values + np.random.normal(0, 1, 100)
        efficiency_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Value': efficiency_values,
            'TagName': 'EFFICIENCY'
        })
        
        # Candidate 3: Uncorrelated (random)
        random_values = np.random.normal(50, 5, 100)
        random_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Value': random_values,
            'TagName': 'RANDOM'
        })
        
        candidate_dfs = [power_df, efficiency_df, random_df]
        window = (timestamps[0], timestamps[-1])
        
        print("🧪 Testing with synthetic data (100 points, known correlations)...")
        correlations = correlate_tags(primary_df, candidate_dfs, window)
        
        print("   Expected: POWER (+), EFFICIENCY (-), RANDOM (~0)")
        for corr in correlations:
            print(f"   {corr['tag_name']}: {corr['pearson_correlation']:.3f} ({corr['correlation_strength']})")
    
    print("\n" + "=" * 58)
    print("✅ Correlation analysis testing complete!")
    print("\n🔧 Usage in code:")
    print("   from src.tools.correlation import correlate_tags, find_correlated_tags")
    print("   correlations = correlate_tags(primary_df, candidate_dfs, window)")


if __name__ == "__main__":
    main() 