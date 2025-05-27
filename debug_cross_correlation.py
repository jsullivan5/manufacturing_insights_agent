#!/usr/bin/env python3
"""Debug cross-correlation to see why door-temperature correlation is 0.000."""

import pandas as pd
from src.tools.data_loader import load_data, get_data_time_range
from src.ai_detective import cross_corr
from datetime import timedelta

# Get the data range and focus on the door event period
data_range = get_data_time_range()
end_time = data_range['end']

# Focus on the door event period (14:30-15:30)
door_event_start = pd.to_datetime("2025-05-25 14:00")
door_event_end = pd.to_datetime("2025-05-25 16:00")

print(f"Testing cross-correlation around door event: {door_event_start} to {door_event_end}")

# Load door and temperature data
door_df = load_data('FREEZER01.DOOR.STATUS', door_event_start, door_event_end)
temp_df = load_data('FREEZER01.TEMP.INTERNAL_C', door_event_start, door_event_end)

print(f"Door data points: {len(door_df)}")
print(f"Temperature data points: {len(temp_df)}")

# Check the actual values
print(f"\nDoor status values:")
print(f"  Min: {door_df['Value'].min()}, Max: {door_df['Value'].max()}")
print(f"  Mean: {door_df['Value'].mean():.3f}")
print(f"  Unique values: {door_df['Value'].unique()}")

print(f"\nTemperature values:")
print(f"  Min: {temp_df['Value'].min():.2f}, Max: {temp_df['Value'].max():.2f}")
print(f"  Mean: {temp_df['Value'].mean():.2f}")

# Show door status during the event
door_event_period = door_df[(door_df['Timestamp'] >= pd.to_datetime("2025-05-25 14:30")) & 
                           (door_df['Timestamp'] <= pd.to_datetime("2025-05-25 14:50"))]
print(f"\nDoor status during event (14:30-14:50):")
for _, row in door_event_period[::5].iterrows():  # Every 5 minutes
    print(f"  {row['Timestamp']}: {row['Value']}")

# Test cross-correlation
print(f"\nTesting cross-correlation:")
result = cross_corr('FREEZER01.DOOR.STATUS', 'FREEZER01.TEMP.INTERNAL_C', 
                   door_event_start, door_event_end, max_lag=10)
print(f"Result: {result}")

# Test with different window sizes
print(f"\nTesting different window sizes:")
for hours in [1, 2, 4]:
    window_start = pd.to_datetime("2025-05-25 14:00")
    window_end = window_start + timedelta(hours=hours)
    result = cross_corr('FREEZER01.DOOR.STATUS', 'FREEZER01.TEMP.INTERNAL_C', 
                       window_start, window_end, max_lag=10)
    print(f"  {hours} hour window: r={result['correlation']:.3f}, lag={result['lag']} min")

# Manual correlation check
print(f"\nManual correlation check:")
door_values = door_df.set_index('Timestamp')['Value']
temp_values = temp_df.set_index('Timestamp')['Value']

# Resample to common frequency
door_resampled = door_values.resample('1min').mean().fillna(method='ffill')
temp_resampled = temp_values.resample('1min').mean().fillna(method='ffill')

# Find common time range
common_start = max(door_resampled.index.min(), temp_resampled.index.min())
common_end = min(door_resampled.index.max(), temp_resampled.index.max())

door_aligned = door_resampled.loc[common_start:common_end]
temp_aligned = temp_resampled.loc[common_start:common_end]

print(f"Common time range: {common_start} to {common_end}")
print(f"Aligned data points: {len(door_aligned)}")

if len(door_aligned) > 0:
    correlation = door_aligned.corr(temp_aligned)
    print(f"Direct correlation: {correlation:.3f}")
    
    # Test with lag
    for lag in range(0, 11):
        if lag == 0:
            shifted_door = door_aligned
        else:
            shifted_door = door_aligned.shift(lag)
        
        valid_mask = ~(shifted_door.isna() | temp_aligned.isna())
        if valid_mask.sum() > 10:
            corr = shifted_door[valid_mask].corr(temp_aligned[valid_mask])
            if not pd.isna(corr):
                print(f"  Lag {lag} min: r={corr:.3f}") 