#!/usr/bin/env python3
"""
Focused test for the specific door opening event on May 23rd at 14:30.
"""

import pandas as pd
from src.tools.causal_analysis import analyze_causality

def test_door_event():
    """Test causal analysis specifically around the door opening event."""
    
    print('🚪 Testing Door Opening Event Detection')
    print('=' * 50)
    
    # Load and pivot the data
    df_raw = pd.read_csv('data/freezer_system_mock_data.csv', parse_dates=['Timestamp'])
    df = df_raw.pivot_table(index='Timestamp', columns='TagName', values='Value', aggfunc='first')
    
    # Focus on the time period around the door opening (May 23rd, 14:30)
    focus_start = pd.Timestamp('2025-05-23 13:00:00')
    focus_end = pd.Timestamp('2025-05-23 16:00:00')
    
    df_focused = df.loc[focus_start:focus_end]
    
    print(f'Focused data shape: {df_focused.shape}')
    print(f'Time range: {df_focused.index.min()} to {df_focused.index.max()}')
    
    # Check door status in this period
    door_data = df_focused['FREEZER01.DOOR.STATUS']
    door_opens = door_data[door_data > 0]
    print(f'\n🚪 Door events in focus period:')
    print(f'  Door open timestamps: {len(door_opens)}')
    if len(door_opens) > 0:
        print(f'  First: {door_opens.index[0]}')
        print(f'  Last: {door_opens.index[-1]}')
    
    # Check temperature changes in this period
    temp_data = df_focused['FREEZER01.TEMP.INTERNAL_C']
    print(f'\n🌡️ Temperature in focus period:')
    print(f'  Range: {temp_data.min():.1f}°C to {temp_data.max():.1f}°C')
    print(f'  Std: {temp_data.std():.2f}°C')
    
    # Run causal analysis on this focused period
    results = analyze_causality(
        df=df_focused,
        primary_tag='FREEZER01.TEMP.INTERNAL_C',
        candidate_tags=['FREEZER01.DOOR.STATUS'],
        time_window_hours=1  # Shorter window for focused analysis
    )
    
    print('\n🎯 FOCUSED CAUSAL ANALYSIS RESULTS')
    print('=' * 50)
    print(f'Investigation Confidence: {results["investigation_confidence"]:.1%}')
    print(f'Root Cause: {results["root_cause"]}')
    
    if results['causal_events']:
        print(f'\n✅ Found {len(results["causal_events"])} causal events:')
        for i, event in enumerate(results['causal_events'], 1):
            print(f'\n  Event {i}:')
            print(f'    Cause: {event["cause_tag"]} {event["cause_type"]} at {event["cause_time"]}')
            print(f'    Effect: {event["effect_tag"]} at {event["effect_time"]}')
            print(f'    Time Lag: {event["time_lag_minutes"]} minutes')
            print(f'    Confidence: {event["confidence"]:.1%}')
            print(f'    Impact: {event["business_impact"]}')
    else:
        print('\n❌ No door-related causal events detected in focused period')
        
        # Let's also try the full dataset but with door-only analysis
        print('\n🔍 Trying full dataset with door-only analysis...')
        
        full_results = analyze_causality(
            df=df,
            primary_tag='FREEZER01.TEMP.INTERNAL_C',
            candidate_tags=['FREEZER01.DOOR.STATUS'],
            time_window_hours=1
        )
        
        if full_results['causal_events']:
            print(f'✅ Found {len(full_results["causal_events"])} door events in full dataset:')
            for i, event in enumerate(full_results['causal_events'][:3], 1):  # Show top 3
                print(f'  {i}. {event["cause_time"]} → {event["effect_time"]} (confidence: {event["confidence"]:.1%})')
        else:
            print('❌ Still no door events detected')

if __name__ == '__main__':
    test_door_event() 