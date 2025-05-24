#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Data Verification Script

Analyzes and visualizes the generated freezer system mock data to verify:
- Realistic operational patterns
- Proper anomaly injection
- Data quality and completeness

This script provides insights into the generated dataset characteristics
to ensure it's suitable for MCP insight demonstrations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def load_and_analyze_data(csv_file: str = "data/freezer_system_mock_data.csv") -> pd.DataFrame:
    """
    Load and perform basic analysis of the generated freezer data.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        DataFrame with timestamp as index
    """
    print("Loading freezer system data...")
    df = pd.read_csv(csv_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    print(f"Dataset overview:")
    print(f"- Total records: {len(df):,}")
    print(f"- Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"- Tags: {', '.join(df['TagName'].unique())}")
    print(f"- Quality distribution: {df['Quality'].value_counts().to_dict()}")
    
    return df

def pivot_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot data from long format to wide format for easier analysis.
    
    Args:
        df: Long format DataFrame
        
    Returns:
        Wide format DataFrame with timestamp as index
    """
    # Pivot to wide format for easier analysis
    wide_df = df.pivot(index='Timestamp', columns='TagName', values='Value')
    return wide_df

def detect_anomaly_periods(df_wide: pd.DataFrame) -> dict:
    """
    Automatically detect anomaly periods based on data patterns.
    
    Args:
        df_wide: Wide format DataFrame
        
    Returns:
        Dictionary of detected anomalies
    """
    anomalies = {}
    
    # Detect prolonged door open periods (>10 minutes for simplicity)
    door_open = df_wide['FREEZER01.DOOR.STATUS'] == 1.0
    
    # Simple approach: find continuous periods where door is open
    door_diff = door_open.astype(int).diff().fillna(0)
    open_starts = door_diff[door_diff == 1].index
    close_starts = door_diff[door_diff == -1].index
    
    for start in open_starts:
        # Find the next close event
        next_closes = close_starts[close_starts > start]
        if len(next_closes) > 0:
            end = next_closes[0]
        else:
            # Door stays open until end of data
            end = door_open.index[-1]
        
        duration_minutes = (end - start).total_seconds() / 60
        if duration_minutes > 10:  # Only flag significant door open events
            anomalies[f"prolonged_door_open_{start.strftime('%m%d_%H%M')}"] = {
                'type': 'prolonged_door_open',
                'start': start,
                'end': end,
                'duration_minutes': duration_minutes
            }
    
    # Detect compressor failure periods
    # Look for periods where temperature > -16°C and compressor is off for > 30 minutes
    high_temp = df_wide['FREEZER01.TEMP.INTERNAL_C'] > -16.0
    compressor_off = df_wide['FREEZER01.COMPRESSOR.STATUS'] == 0.0
    failure_condition = high_temp & compressor_off
    
    # Find start and end of failure periods
    failure_diff = failure_condition.astype(int).diff().fillna(0)
    failure_starts = failure_diff[failure_diff == 1].index
    failure_ends = failure_diff[failure_diff == -1].index
    
    for start in failure_starts:
        # Find the next end event
        next_ends = failure_ends[failure_ends > start]
        if len(next_ends) > 0:
            end = next_ends[0]
        else:
            end = failure_condition.index[-1]
        
        duration_minutes = (end - start).total_seconds() / 60
        if duration_minutes > 30:  # Only flag significant failures
            peak_temp = df_wide.loc[start:end, 'FREEZER01.TEMP.INTERNAL_C'].max()
            anomalies[f"compressor_failure_{start.strftime('%m%d_%H%M')}"] = {
                'type': 'compressor_failure',
                'start': start,
                'end': end,
                'duration_minutes': duration_minutes,
                'peak_temp': peak_temp
            }
    
    return anomalies

def create_overview_plot(df_wide: pd.DataFrame, anomalies: dict):
    """
    Create comprehensive overview plot showing all key metrics and anomalies.
    
    Args:
        df_wide: Wide format DataFrame
        anomalies: Dictionary of detected anomalies
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle('Freezer System Mock Data - 7-Day Overview', fontsize=16, fontweight='bold')
    
    # Temperature plot
    axes[0].plot(df_wide.index, df_wide['FREEZER01.TEMP.INTERNAL_C'], 
                label='Internal Temp', color='blue', alpha=0.7, linewidth=1)
    axes[0].plot(df_wide.index, df_wide['FREEZER01.TEMP.AMBIENT_C'], 
                label='Ambient Temp', color='orange', alpha=0.5, linewidth=1)
    axes[0].axhline(y=-18, color='red', linestyle='--', alpha=0.7, label='Setpoint')
    axes[0].axhline(y=-16, color='red', linestyle=':', alpha=0.7, label='High Alarm')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Door status and compressor status
    axes[1].fill_between(df_wide.index, 0, df_wide['FREEZER01.DOOR.STATUS'], 
                        alpha=0.6, color='red', label='Door Open')
    axes[1].fill_between(df_wide.index, 0, df_wide['FREEZER01.COMPRESSOR.STATUS'], 
                        alpha=0.4, color='green', label='Compressor On')
    axes[1].set_ylabel('Status (0=Off, 1=On)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Power consumption
    axes[2].plot(df_wide.index, df_wide['FREEZER01.COMPRESSOR.POWER_KW'], 
                color='purple', alpha=0.7, linewidth=1)
    axes[2].set_ylabel('Power (kW)')
    axes[2].grid(True, alpha=0.3)
    
    # Day/Night background
    for ax in axes[:3]:
        for day in pd.date_range(df_wide.index.min().date(), df_wide.index.max().date(), freq='D'):
            # Night periods (8 PM to 8 AM)
            night_start = pd.Timestamp(day) + pd.Timedelta(hours=20)
            night_end = pd.Timestamp(day + pd.Timedelta(days=1)) + pd.Timedelta(hours=8)
            ax.axvspan(night_start, night_end, alpha=0.1, color='gray', label='Night Shift' if day == df_wide.index.min().date() else "")
    
    # Highlight anomalies
    colors = ['red', 'orange', 'purple', 'brown']
    for i, (name, anomaly) in enumerate(anomalies.items()):
        color = colors[i % len(colors)]
        for ax in axes[:3]:
            ax.axvspan(anomaly['start'], anomaly['end'], alpha=0.3, color=color, 
                      label=f"Anomaly: {anomaly['type']}" if ax == axes[0] else "")
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/freezer_data_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_operational_patterns(df_wide: pd.DataFrame):
    """
    Analyze and report on operational patterns in the data.
    
    Args:
        df_wide: Wide format DataFrame
    """
    print("\n=== Operational Pattern Analysis ===")
    
    # Temperature statistics
    temp_stats = df_wide['FREEZER01.TEMP.INTERNAL_C'].describe()
    print(f"Internal Temperature Statistics:")
    print(f"  Mean: {temp_stats['mean']:.2f}°C")
    print(f"  Std Dev: {temp_stats['std']:.2f}°C")
    print(f"  Range: {temp_stats['min']:.2f}°C to {temp_stats['max']:.2f}°C")
    
    # Door activity analysis
    df_wide['hour'] = df_wide.index.hour
    door_by_hour = df_wide.groupby('hour')['FREEZER01.DOOR.STATUS'].mean()
    day_hours = door_by_hour[8:20].mean()
    night_hours = door_by_hour[list(range(20, 24)) + list(range(0, 8))].mean()
    
    print(f"\nDoor Activity Patterns:")
    print(f"  Day shift (8 AM - 8 PM) average: {day_hours:.3f}")
    print(f"  Night shift (8 PM - 8 AM) average: {night_hours:.3f}")
    print(f"  Day/Night ratio: {day_hours/night_hours:.1f}x more active during day")
    
    # Compressor cycling
    compressor_on_time = (df_wide['FREEZER01.COMPRESSOR.STATUS'].sum() / len(df_wide)) * 100
    avg_power = df_wide['FREEZER01.COMPRESSOR.POWER_KW'].mean()
    
    print(f"\nCompressor Performance:")
    print(f"  On-time percentage: {compressor_on_time:.1f}%")
    print(f"  Average power consumption: {avg_power:.2f} kW")
    
    # Quality analysis
    df_quality = pd.read_csv("data/freezer_system_mock_data.csv")
    quality_stats = df_quality['Quality'].value_counts()
    questionable_pct = (quality_stats.get('Questionable', 0) / len(df_quality)) * 100
    
    print(f"\nData Quality:")
    print(f"  Good quality data: {quality_stats.get('Good', 0):,} points")
    print(f"  Questionable data: {quality_stats.get('Questionable', 0):,} points ({questionable_pct:.2f}%)")

def main():
    """
    Main verification function to analyze and visualize freezer data.
    """
    print("Manufacturing Copilot - Freezer Data Verification")
    print("=" * 50)
    
    # Load and analyze data
    df = load_and_analyze_data()
    df_wide = pivot_for_analysis(df)
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    anomalies = detect_anomaly_periods(df_wide)
    
    print(f"Found {len(anomalies)} anomaly periods:")
    for name, anomaly in anomalies.items():
        duration = (anomaly['end'] - anomaly['start']).total_seconds() / 60
        print(f"  - {anomaly['type']}: {anomaly['start']} ({duration:.0f} minutes)")
    
    # Analyze patterns
    analyze_operational_patterns(df_wide)
    
    # Create visualization
    print("\nGenerating overview plot...")
    create_overview_plot(df_wide, anomalies)
    
    print("\nVerification complete! Check 'data/freezer_data_overview.png' for visualization.")

if __name__ == "__main__":
    main() 