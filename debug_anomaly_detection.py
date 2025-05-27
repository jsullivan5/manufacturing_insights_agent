#!/usr/bin/env python3
"""Debug anomaly detection to see why temperature spikes aren't being found."""

import pandas as pd
from src.tools.data_loader import load_data, get_data_time_range
from src.ai_detective import dynamic_sigma
from src.tools.anomaly_detection import detect_spike
from datetime import timedelta

# Get the last 24 hours of temperature data
data_range = get_data_time_range()
end_time = data_range['end']
start_time = end_time - timedelta(hours=24)

print(f"Analyzing temperature data from {start_time} to {end_time}")

df = load_data('FREEZER01.TEMP.INTERNAL_C', start_time, end_time)
print(f"Loaded {len(df)} data points")
print(f"Temperature range: {df['Value'].min():.2f} to {df['Value'].max():.2f}")
print(f"Temperature std: {df['Value'].std():.2f}")
print(f"Temperature mean: {df['Value'].mean():.2f}")

# Test dynamic threshold
threshold = dynamic_sigma(df['Value'])
print(f"\nDynamic threshold: {threshold:.2f}")

# Test different thresholds
print("\nTesting different thresholds:")
for t in [1.0, 1.5, 2.0, 2.5, 3.0]:
    anomalies = detect_spike(df, threshold=t)
    print(f"Threshold {t}: {len(anomalies)} anomalies")
    if len(anomalies) > 0:
        print(f"  First anomaly: {anomalies[0]}")

# Look for the specific door event time
door_event_time = "2025-05-25 14:30"
print(f"\nLooking for data around door event time: {door_event_time}")

# Filter data around the door event
event_start = pd.to_datetime(door_event_time)
event_end = event_start + timedelta(minutes=30)

event_data = df[(df['Timestamp'] >= event_start) & (df['Timestamp'] <= event_end)]
if not event_data.empty:
    print(f"Temperature during door event:")
    print(f"  Before: {df[df['Timestamp'] < event_start]['Value'].tail(5).mean():.2f}")
    print(f"  During: {event_data['Value'].mean():.2f}")
    print(f"  Peak: {event_data['Value'].max():.2f}")
    
    # Show the actual temperature progression
    print(f"\nTemperature progression around door event:")
    extended_window = df[(df['Timestamp'] >= event_start - timedelta(minutes=10)) & 
                        (df['Timestamp'] <= event_start + timedelta(minutes=30))]
    for _, row in extended_window[::5].iterrows():  # Every 5 minutes
        print(f"  {row['Timestamp']}: {row['Value']:.2f}°C")
else:
    print("No data found during door event time")

# Check if we're looking at the right time window
print(f"\nChecking if 14:30 is in our analysis window:")
print(f"Analysis window: {start_time} to {end_time}")
print(f"Door event: {door_event_time}")
print(f"Is door event in window? {start_time <= pd.to_datetime(door_event_time) <= end_time}")

# Try analyzing just the door event period
print(f"\nAnalyzing just the door event period:")
door_start = pd.to_datetime("2025-05-25 14:00")
door_end = pd.to_datetime("2025-05-25 15:30")
door_df = load_data('FREEZER01.TEMP.INTERNAL_C', door_start, door_end)
print(f"Door period data points: {len(door_df)}")
if not door_df.empty:
    door_anomalies = detect_spike(door_df, threshold=2.0)
    print(f"Anomalies in door period: {len(door_anomalies)}")
    for anomaly in door_anomalies:
        print(f"  {anomaly}") 