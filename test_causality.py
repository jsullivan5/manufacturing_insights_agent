#!/usr/bin/env python3
"""
Test script for causal analysis tool.
Validates that we can detect the 18-minute door opening → temperature spike relationship.
"""

import pandas as pd
from src.tools.causal_analysis import analyze_causality

def test_causal_analysis():
    """Test causal analysis with freezer anomaly data."""
    
    print('🔍 Testing Causal Analysis Tool')
    print('=' * 50)
    
    # Load the freezer data (PI System format)
    df_raw = pd.read_csv('data/freezer_system_mock_data.csv', parse_dates=['Timestamp'])
    
    print(f'Raw data shape: {df_raw.shape}')
    print(f'Date range: {df_raw["Timestamp"].min()} to {df_raw["Timestamp"].max()}')
    print(f'Available tags: {df_raw["TagName"].unique()}')
    
    # Convert from PI System format (long) to wide format for analysis
    df = df_raw.pivot_table(
        index='Timestamp', 
        columns='TagName', 
        values='Value', 
        aggfunc='first'
    )
    
    print(f'\nPivoted data shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    
    # Test causal analysis on temperature anomalies
    results = analyze_causality(
        df=df,
        primary_tag='FREEZER01.TEMP.INTERNAL_C',
        candidate_tags=['FREEZER01.DOOR.STATUS', 'FREEZER01.COMPRESSOR.STATUS', 'FREEZER01.COMPRESSOR.POWER_KW'],
        time_window_hours=2
    )
    
    print('\n🎯 CAUSAL INVESTIGATION RESULTS')
    print('=' * 50)
    print(f'Investigation Confidence: {results["investigation_confidence"]:.1%}')
    print(f'Root Cause: {results["root_cause"]}')
    print(f'Business Impact: {results["business_impact"]}')
    
    print('\n📋 Forensic Timeline:')
    for entry in results['forensic_timeline']:
        print(f'  {entry}')
    
    print('\n🔧 Recommendations:')
    for i, rec in enumerate(results['recommendations'], 1):
        print(f'  {i}. {rec}')
    
    print(f'\n✅ Found {len(results["causal_events"])} causal events')
    
    # Show detailed event information
    if results['causal_events']:
        print('\n🕵️ Detailed Causal Events:')
        for i, event in enumerate(results['causal_events'], 1):
            print(f'\n  Event {i}:')
            print(f'    Cause: {event["cause_tag"]} {event["cause_type"]} at {event["cause_time"]}')
            print(f'    Effect: {event["effect_tag"]} at {event["effect_time"]}')
            print(f'    Time Lag: {event["time_lag_minutes"]} minutes')
            print(f'    Confidence: {event["confidence"]:.1%}')
            print(f'    Correlation: {event["correlation"]:.3f}')
            print(f'    Impact: {event["business_impact"]}')
    else:
        print('\n⚠️  No causal events detected. Let\'s check the data...')
        
        # Debug: Show some sample data
        print('\n📊 Sample Temperature Data:')
        temp_data = df['FREEZER01.TEMP.INTERNAL_C'].dropna()
        print(f'  Temperature range: {temp_data.min():.1f}°C to {temp_data.max():.1f}°C')
        print(f'  Temperature std: {temp_data.std():.2f}°C')
        
        print('\n🚪 Sample Door Data:')
        door_data = df['FREEZER01.DOOR.STATUS'].dropna()
        print(f'  Door status range: {door_data.min()} to {door_data.max()}')
        print(f'  Door open events: {(door_data > 0).sum()} timestamps')

if __name__ == '__main__':
    test_causal_analysis() 