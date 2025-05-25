#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Visualization Tools

Provides chart generation and visualization capabilities for manufacturing
time-series data with support for anomaly highlighting and trend analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

def generate_chart(df: pd.DataFrame, tag: str, highlights: Optional[List[Tuple[datetime, datetime]]] = None,
                  output_dir: str = "charts", figsize: Tuple[int, int] = (12, 6)) -> str:
    """
    Generate PNG chart of time series data with optional anomaly highlighting.
    
    Creates professional-quality time series visualizations with trend lines,
    statistical overlays, and highlighted anomaly periods for manufacturing
    data analysis and reporting.
    
    Args:
        df: DataFrame with time-series data (must have 'Timestamp' and 'Value' columns)
        tag: Tag name for chart title and filename
        highlights: Optional list of (start_time, end_time) tuples for anomaly highlighting
        output_dir: Directory to save chart files (default: "charts")
        figsize: Figure size as (width, height) in inches
        
    Returns:
        String path to the generated PNG file
        
    Raises:
        ValueError: If required columns are missing or data is insufficient
    """
    logger.debug(f"Generating chart for tag '{tag}' with {len(df)} data points")
    
    # Validate input data
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    required_columns = ['Timestamp', 'Value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < 2:
        raise ValueError("Need at least 2 data points to generate a chart")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort data by timestamp
    df_sorted = df.sort_values('Timestamp').copy()
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_sorted['Timestamp']):
        df_sorted['Timestamp'] = pd.to_datetime(df_sorted['Timestamp'])
    
    # Set up the plot with professional styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main time series
    ax.plot(df_sorted['Timestamp'], df_sorted['Value'], 
           color='#2E86AB', linewidth=1.5, alpha=0.8, label='Data')
    
    # Add trend line if we have enough data
    if len(df_sorted) > 10:
        _add_trend_line(ax, df_sorted)
    
    # Add statistical bands (mean ± std)
    _add_statistical_bands(ax, df_sorted)
    
    # Highlight anomaly periods if provided
    if highlights:
        _add_anomaly_highlights(ax, highlights, df_sorted)
    
    # Add data quality indicators if available
    if 'Quality' in df_sorted.columns:
        _add_quality_indicators(ax, df_sorted)
    
    # Format the chart
    _format_chart(ax, tag, df_sorted)
    
    # Generate filename and save
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag_name = tag.replace('/', '_').replace('\\', '_').replace(':', '_')
    filename = f"{safe_tag_name}_{timestamp_str}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Chart saved to {filepath}")
    return filepath


def _add_trend_line(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Add a trend line to the chart."""
    try:
        # Convert timestamps to numeric for trend calculation
        timestamps_numeric = mdates.date2num(df['Timestamp'])
        values = df['Value'].values
        
        # Calculate linear trend
        z = np.polyfit(timestamps_numeric, values, 1)
        p = np.poly1d(z)
        
        # Plot trend line
        ax.plot(df['Timestamp'], p(timestamps_numeric), 
               color='#F18F01', linestyle='--', linewidth=2, alpha=0.7, label='Trend')
        
    except Exception as e:
        logger.debug(f"Could not add trend line: {e}")


def _add_statistical_bands(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Add statistical bands (mean ± standard deviation) to the chart."""
    try:
        mean_value = df['Value'].mean()
        std_value = df['Value'].std()
        
        # Add horizontal lines for mean and std bands
        ax.axhline(y=mean_value, color='#A4243B', linestyle='-', alpha=0.5, linewidth=1, label='Mean')
        ax.axhline(y=mean_value + std_value, color='#A4243B', linestyle=':', alpha=0.3, linewidth=1)
        ax.axhline(y=mean_value - std_value, color='#A4243B', linestyle=':', alpha=0.3, linewidth=1)
        
        # Fill area between ±1 std
        ax.fill_between(df['Timestamp'], 
                       mean_value - std_value, 
                       mean_value + std_value,
                       alpha=0.1, color='#A4243B', label='±1σ')
        
    except Exception as e:
        logger.debug(f"Could not add statistical bands: {e}")


def _add_anomaly_highlights(ax: plt.Axes, highlights: List[Tuple[datetime, datetime]], 
                          df: pd.DataFrame) -> None:
    """Add highlighted regions for anomaly periods."""
    try:
        y_min, y_max = ax.get_ylim()
        
        for i, (start_time, end_time) in enumerate(highlights):
            # Add shaded region for anomaly period
            ax.axvspan(start_time, end_time, 
                      alpha=0.2, color='red', 
                      label='Anomaly Period' if i == 0 else "")
            
            # Add vertical lines at start and end
            ax.axvline(x=start_time, color='red', linestyle='--', alpha=0.6, linewidth=1)
            ax.axvline(x=end_time, color='red', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add annotation for the anomaly period
            duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes
            mid_time = start_time + (end_time - start_time) / 2
            
            # Find the y-position for annotation (avoid overlapping with data)
            y_pos = y_max * 0.9 - (i * 0.1 * (y_max - y_min))
            
            ax.annotate(f'Anomaly\n({duration:.0f}min)', 
                       xy=(mid_time, y_pos),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                       fontsize=8, ha='center')
        
    except Exception as e:
        logger.debug(f"Could not add anomaly highlights: {e}")


def _add_quality_indicators(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Add indicators for data quality issues."""
    try:
        # Mark questionable and bad quality data points
        questionable_data = df[df['Quality'] == 'Questionable']
        bad_data = df[df['Quality'] == 'Bad']
        
        if not questionable_data.empty:
            ax.scatter(questionable_data['Timestamp'], questionable_data['Value'],
                      color='orange', marker='o', s=20, alpha=0.7, 
                      label='Questionable Quality', zorder=5)
        
        if not bad_data.empty:
            ax.scatter(bad_data['Timestamp'], bad_data['Value'],
                      color='red', marker='x', s=30, alpha=0.8,
                      label='Bad Quality', zorder=5)
        
    except Exception as e:
        logger.debug(f"Could not add quality indicators: {e}")


def _format_chart(ax: plt.Axes, tag: str, df: pd.DataFrame) -> None:
    """Apply professional formatting to the chart."""
    # Set title and labels
    ax.set_title(f'Manufacturing Data: {tag}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=12)
    
    # Get unit from DataFrame if available
    unit = df.get('Units', [''])[0] if 'Units' in df.columns else ''
    y_label = f'Value ({unit})' if unit else 'Value'
    ax.set_ylabel(y_label, fontsize=12)
    
    # Format x-axis (time)
    time_range = df['Timestamp'].max() - df['Timestamp'].min()
    
    if time_range <= timedelta(hours=6):
        # For short time ranges, show hours and minutes
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    elif time_range <= timedelta(days=2):
        # For medium time ranges, show day and hour
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    else:
        # For long time ranges, show dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend if there are multiple elements
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Set background color
    ax.set_facecolor('#FAFAFA')
    
    # Add timestamp annotation
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax.text(0.99, 0.01, f'Generated: {timestamp_str}', 
           transform=ax.transAxes, fontsize=8, alpha=0.7,
           ha='right', va='bottom')


def generate_correlation_chart(primary_df: pd.DataFrame, candidate_df: pd.DataFrame,
                             primary_tag: str, candidate_tag: str,
                             correlation_value: float, output_dir: str = "charts") -> str:
    """
    Generate a dual-axis correlation chart showing two metrics over time.
    
    Args:
        primary_df: DataFrame with primary metric data
        candidate_df: DataFrame with candidate metric data  
        primary_tag: Name of primary tag
        candidate_tag: Name of candidate tag
        correlation_value: Correlation coefficient to display
        output_dir: Directory to save chart files
        
    Returns:
        String path to the generated PNG file
    """
    logger.debug(f"Generating correlation chart for {primary_tag} vs {candidate_tag}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot primary metric on left axis
    color1 = '#2E86AB'
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel(primary_tag, color=color1, fontsize=12)
    line1 = ax1.plot(primary_df['Timestamp'], primary_df['Value'], 
                     color=color1, linewidth=2, label=primary_tag)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create second y-axis for candidate metric
    ax2 = ax1.twinx()
    color2 = '#F18F01'
    ax2.set_ylabel(candidate_tag, color=color2, fontsize=12)
    line2 = ax2.plot(candidate_df['Timestamp'], candidate_df['Value'], 
                     color=color2, linewidth=2, label=candidate_tag)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add correlation information to title
    direction = "positive" if correlation_value > 0 else "negative"
    strength = _interpret_correlation_strength(abs(correlation_value))
    title = f'Correlation Analysis: {primary_tag} vs {candidate_tag}\n'
    title += f'Correlation: {correlation_value:.3f} ({strength} {direction})'
    
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Format time axis
    _format_time_axis(ax1, primary_df)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    # Generate filename and save
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_primary = primary_tag.replace('/', '_').replace('\\', '_').replace(':', '_')
    safe_candidate = candidate_tag.replace('/', '_').replace('\\', '_').replace(':', '_')
    filename = f"correlation_{safe_primary}_vs_{safe_candidate}_{timestamp_str}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Correlation chart saved to {filepath}")
    return filepath


def _interpret_correlation_strength(abs_correlation: float) -> str:
    """Interpret correlation strength for chart titles."""
    if abs_correlation >= 0.8:
        return 'very strong'
    elif abs_correlation >= 0.6:
        return 'strong'
    elif abs_correlation >= 0.4:
        return 'moderate'
    elif abs_correlation >= 0.2:
        return 'weak'
    else:
        return 'very weak'


def _format_time_axis(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Format time axis based on data range."""
    time_range = df['Timestamp'].max() - df['Timestamp'].min()
    
    if time_range <= timedelta(hours=6):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    elif time_range <= timedelta(days=2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def main():
    """
    Demo and test the visualization functions.
    
    This is a demo function for testing - not part of the core MCP pipeline.
    """
    print("📊 Manufacturing Copilot - Visualization Demo")
    print("=" * 52)
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        from src.tools.data_loader import load_data
        from src.tools.anomaly_detection import detect_spike
        from datetime import datetime, timedelta
        
        print("📈 Testing chart generation with freezer data...")
        
        # Load temperature data
        temp_data = load_data("FREEZER01.TEMP.INTERNAL_C")
        print(f"   Loaded {len(temp_data)} temperature data points")
        
        # Generate basic chart
        print("\n🎨 Generating basic time series chart...")
        chart_path = generate_chart(temp_data, "FREEZER01.TEMP.INTERNAL_C")
        print(f"   Chart saved to: {chart_path}")
        
        # Detect anomalies and generate chart with highlights
        print("\n🔍 Detecting anomalies for highlighting...")
        anomalies = detect_spike(temp_data, threshold=2.0)
        
        if anomalies:
            print(f"   Found {len(anomalies)} anomalies")
            
            # Convert anomalies to highlight periods (group consecutive anomalies)
            highlights = []
            if anomalies:
                # Simple approach: create 10-minute windows around each anomaly
                for timestamp, value, z_score, reason in anomalies[:3]:  # Limit to first 3
                    start_time = timestamp - timedelta(minutes=5)
                    end_time = timestamp + timedelta(minutes=5)
                    highlights.append((start_time, end_time))
            
            print("\n🎨 Generating chart with anomaly highlights...")
            highlighted_chart = generate_chart(
                temp_data, 
                "FREEZER01.TEMP.INTERNAL_C", 
                highlights=highlights
            )
            print(f"   Highlighted chart saved to: {highlighted_chart}")
        else:
            print("   No anomalies found for highlighting")
        
        # Test correlation chart
        print("\n🔗 Testing correlation chart generation...")
        power_data = load_data("FREEZER01.COMPRESSOR.POWER_KW")
        
        if not power_data.empty:
            correlation_chart = generate_correlation_chart(
                temp_data.head(1000),  # Limit data for faster processing
                power_data.head(1000),
                "FREEZER01.TEMP.INTERNAL_C",
                "FREEZER01.COMPRESSOR.POWER_KW",
                -0.65  # Example correlation value
            )
            print(f"   Correlation chart saved to: {correlation_chart}")
        
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")
        print("📝 Testing with synthetic data...")
        
        # Create synthetic test data
        timestamps = pd.date_range('2025-01-01', periods=100, freq='1min')
        values = 20 + 2 * np.sin(np.arange(100) * 0.1) + np.random.normal(0, 0.5, 100)
        
        # Add some anomalies
        values[30] = 35  # High spike
        values[60] = 5   # Low spike
        
        test_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Value': values,
            'Units': '°C',
            'Quality': ['Good'] * 100
        })
        
        # Mark some data as questionable
        test_df.loc[30, 'Quality'] = 'Questionable'
        test_df.loc[60, 'Quality'] = 'Bad'
        
        print("🧪 Testing with synthetic data...")
        
        # Create highlights for known anomalies
        highlights = [
            (timestamps[28], timestamps[32]),  # Around first anomaly
            (timestamps[58], timestamps[62])   # Around second anomaly
        ]
        
        synthetic_chart = generate_chart(
            test_df, 
            "TEST.TEMPERATURE", 
            highlights=highlights
        )
        print(f"   Synthetic chart saved to: {synthetic_chart}")
    
    print("\n" + "=" * 52)
    print("✅ Visualization testing complete!")
    print("\n🔧 Usage in code:")
    print("   from src.tools.visualization import generate_chart")
    print("   chart_path = generate_chart(df, tag_name, highlights=anomaly_periods)")


if __name__ == "__main__":
    main() 