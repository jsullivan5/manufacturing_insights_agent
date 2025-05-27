#!/usr/bin/env python3
"""
Debug script for causal analysis tool.
Investigates why the door opening → temperature spike isn't being detected.
"""

import pandas as pd
import matplotlib.pyplot as plt
from src.tools.causal_analysis import detect_change_points, calculate_time_lagged_correlations

def debug_causal_analysis():
    """Debug causal analysis step by step."""
    
    print('🔍 Debugging Causal Analysis Tool')
    print('=' * 50)
    
    # Load and pivot the data
    df_raw = pd.read_csv('data/freezer_system_mock_data.csv', parse_dates=['Timestamp'])
    df = df_raw.pivot_table(index='Timestamp', columns='TagName', values='Value', aggfunc='first')
    
    print(f'Data shape: {df.shape}')
    print(f'Date range: {df.index.min()} to {df.index.max()}')
    
    # Check for door opening events
    door_data = df['FREEZER01.DOOR.STATUS']
    door_opens = door_data[door_data > 0]
    print(f'\n🚪 Door Opening Analysis:')
    print(f'  Total door open timestamps: {len(door_opens)}')
    if len(door_opens) > 0:
        print(f'  First door open: {door_opens.index[0]}')
        print(f'  Last door open: {door_opens.index[-1]}')
        
        # Find continuous door open periods
        door_changes = door_data.diff().fillna(0)
        door_open_starts = door_changes[door_changes > 0].index
        door_close_starts = door_changes[door_changes < 0].index
        
        print(f'  Door opening events: {len(door_open_starts)}')
        print(f'  Door closing events: {len(door_close_starts)}')
        
        # Show longest door open period
        if len(door_open_starts) > 0 and len(door_close_starts) > 0:
            durations = []
            for open_time in door_open_starts:
                close_times = door_close_starts[door_close_starts > open_time]
                if len(close_times) > 0:
                    duration = (close_times[0] - open_time).total_seconds() / 60
                    durations.append((open_time, close_times[0], duration))
            
            if durations:
                longest = max(durations, key=lambda x: x[2])
                print(f'  Longest door open: {longest[2]:.1f} minutes from {longest[0]} to {longest[1]}')
    
    # Check temperature variations
    temp_data = df['FREEZER01.TEMP.INTERNAL_C']
    print(f'\n🌡️ Temperature Analysis:')
    print(f'  Temperature range: {temp_data.min():.1f}°C to {temp_data.max():.1f}°C')
    print(f'  Temperature std: {temp_data.std():.2f}°C')
    print(f'  Temperature mean: {temp_data.mean():.1f}°C')
    
    # Test change point detection with different sensitivities
    print(f'\n🔍 Change Point Detection Tests:')
    for sensitivity in [1.0, 1.5, 2.0, 2.5, 3.0]:
        temp_changes = detect_change_points(df, 'FREEZER01.TEMP.INTERNAL_C', sensitivity=sensitivity)
        door_changes = detect_change_points(df, 'FREEZER01.DOOR.STATUS', sensitivity=sensitivity)
        
        print(f'  Sensitivity {sensitivity}: {len(temp_changes)} temp changes, {len(door_changes)} door changes')
        
        if len(temp_changes) > 0:
            print(f'    First temp change: {temp_changes[0].timestamp} ({temp_changes[0].direction})')
        if len(door_changes) > 0:
            print(f'    First door change: {door_changes[0].timestamp} ({door_changes[0].direction})')
    
    # Test time-lagged correlations
    print(f'\n📊 Time-Lagged Correlation Analysis:')
    correlations = calculate_time_lagged_correlations(
        df, 'FREEZER01.DOOR.STATUS', 'FREEZER01.TEMP.INTERNAL_C', max_lag_minutes=30
    )
    
    if correlations:
        best_lag = max(correlations.items(), key=lambda x: abs(x[1]))
        print(f'  Best correlation: {best_lag[1]:.3f} at {best_lag[0]} minute lag')
        
        # Show top 5 correlations
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f'  Top correlations:')
        for lag, corr in sorted_corrs:
            print(f'    {lag} min lag: {corr:.3f}')
    else:
        print('  No correlations calculated')
    
    # Create a visualization of the data around the longest door opening
    if 'longest' in locals():
        print(f'\n📈 Creating visualization around longest door opening...')
        
        # Get data around the event
        start_time = longest[0] - pd.Timedelta(hours=1)
        end_time = longest[1] + pd.Timedelta(hours=1)
        
        event_data = df.loc[start_time:end_time]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot temperature
        ax1.plot(event_data.index, event_data['FREEZER01.TEMP.INTERNAL_C'], 'b-', linewidth=2)
        ax1.axvspan(longest[0], longest[1], alpha=0.3, color='red', label=f'Door Open ({longest[2]:.1f} min)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature vs Door Opening Event')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot door status
        ax2.plot(event_data.index, event_data['FREEZER01.DOOR.STATUS'], 'r-', linewidth=2)
        ax2.set_ylabel('Door Status')
        ax2.set_xlabel('Time')
        ax2.set_title('Door Status')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('charts/causality_debug.png', dpi=150, bbox_inches='tight')
        print(f'  Saved visualization to charts/causality_debug.png')

if __name__ == '__main__':
    debug_causal_analysis() 