#!/usr/bin/env python3
"""
Debug door change point detection specifically.
"""

import pandas as pd
from src.tools.causal_analysis import detect_change_points

def debug_door_detection():
    """Debug why door change points aren't being detected."""
    
    print('🔍 Debugging Door Change Point Detection')
    print('=' * 50)
    
    # Load and pivot the data
    df_raw = pd.read_csv('data/freezer_system_mock_data.csv', parse_dates=['Timestamp'])
    df = df_raw.pivot_table(index='Timestamp', columns='TagName', values='Value', aggfunc='first')
    
    # Focus on the door opening period
    focus_start = pd.Timestamp('2025-05-23 13:00:00')
    focus_end = pd.Timestamp('2025-05-23 16:00:00')
    df_focused = df.loc[focus_start:focus_end]
    
    # Analyze door status changes
    door_data = df_focused['FREEZER01.DOOR.STATUS']
    print(f'Door data shape: {door_data.shape}')
    print(f'Door values: {door_data.unique()}')
    print(f'Door open count: {(door_data > 0).sum()}')
    
    # Show the actual door opening pattern
    door_changes = door_data.diff().fillna(0)
    door_opens = door_changes[door_changes > 0]
    door_closes = door_changes[door_changes < 0]
    
    print(f'\n🚪 Door Opening Pattern:')
    print(f'  Opening events: {len(door_opens)}')
    print(f'  Closing events: {len(door_closes)}')
    
    if len(door_opens) > 0:
        print(f'  First open: {door_opens.index[0]}')
        print(f'  Last open: {door_opens.index[-1]}')
        
        # Find the longest continuous opening
        current_open = None
        longest_duration = 0
        longest_period = None
        
        for timestamp, value in door_data.items():
            if value > 0 and current_open is None:
                current_open = timestamp
            elif value == 0 and current_open is not None:
                duration = (timestamp - current_open).total_seconds() / 60
                if duration > longest_duration:
                    longest_duration = duration
                    longest_period = (current_open, timestamp)
                current_open = None
        
        if longest_period:
            print(f'  Longest opening: {longest_duration:.1f} minutes from {longest_period[0]} to {longest_period[1]}')
    
    # Test change point detection with various sensitivities
    print(f'\n🔍 Change Point Detection Results:')
    for sensitivity in [0.5, 1.0, 1.5, 2.0]:
        door_changes = detect_change_points(df_focused, 'FREEZER01.DOOR.STATUS', sensitivity=sensitivity, min_duration=1)
        temp_changes = detect_change_points(df_focused, 'FREEZER01.TEMP.INTERNAL_C', sensitivity=sensitivity, min_duration=1)
        
        print(f'  Sensitivity {sensitivity}:')
        print(f'    Door changes: {len(door_changes)}')
        print(f'    Temp changes: {len(temp_changes)}')
        
        if door_changes:
            for i, change in enumerate(door_changes[:3]):  # Show first 3
                print(f'      Door {i+1}: {change.timestamp} - {change.direction} (conf: {change.confidence:.2f})')
        
        if temp_changes:
            for i, change in enumerate(temp_changes[:3]):  # Show first 3
                print(f'      Temp {i+1}: {change.timestamp} - {change.direction} (conf: {change.confidence:.2f})')
    
    # Show raw data around the longest opening
    if 'longest_period' in locals() and longest_period:
        print(f'\n📊 Raw Data Around Longest Opening:')
        start_time = longest_period[0] - pd.Timedelta(minutes=30)
        end_time = longest_period[1] + pd.Timedelta(minutes=30)
        
        sample_data = df.loc[start_time:end_time]
        
        print(f'  Time range: {start_time} to {end_time}')
        print(f'  Door status during event:')
        door_sample = sample_data['FREEZER01.DOOR.STATUS']
        for timestamp, value in door_sample.items():
            if timestamp >= longest_period[0] and timestamp <= longest_period[1]:
                print(f'    {timestamp}: {value} ← DOOR OPEN')
            elif abs((timestamp - longest_period[0]).total_seconds()) < 600:  # Within 10 minutes
                print(f'    {timestamp}: {value}')
        
        print(f'\n  Temperature during event:')
        temp_sample = sample_data['FREEZER01.TEMP.INTERNAL_C']
        temp_before = temp_sample.loc[:longest_period[0]].tail(5).mean()
        temp_during = temp_sample.loc[longest_period[0]:longest_period[1]].mean()
        temp_after = temp_sample.loc[longest_period[1]:].head(5).mean()
        
        print(f'    Before: {temp_before:.1f}°C')
        print(f'    During: {temp_during:.1f}°C')
        print(f'    After: {temp_after:.1f}°C')
        print(f'    Temperature rise: {temp_during - temp_before:.1f}°C')

if __name__ == '__main__':
    debug_door_detection() 