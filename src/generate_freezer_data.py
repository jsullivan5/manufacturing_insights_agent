#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Freezer System Mock Data Generator

Generates realistic time series data simulating a freezer system monitored by 
AVEVA PI System. Produces data in long format CSV compatible with PI System 
data structures for demonstration of anomaly detection and causal analysis.

The generated dataset enables downstream analysis to identify patterns like:
- Door left open → temperature spike → compressor response
- Compressor failure → temperature rise despite cooling demand
- Sensor malfunctions → data quality issues

Author: Manufacturing Copilot Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Quality(Enum):
    """PI System data quality indicators"""
    GOOD = "Good"
    BAD = "Bad" 
    QUESTIONABLE = "Questionable"

@dataclass
class DataPoint:
    """Represents a single PI System data point"""
    timestamp: datetime
    tag_name: str
    value: float
    units: str
    quality: Quality = Quality.GOOD

class FreezorDataGenerator:
    """
    Generates realistic freezer system time series data for MCP demonstrations.
    
    Simulates normal operational patterns, shift-based variations, and injectable
    anomalies to demonstrate manufacturing insight capabilities.
    """
    
    def __init__(self, start_date: Optional[str] = None, duration_days: int = 7):
        """
        Initialize the freezer data generator.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (optional, defaults to 7 days ago)
            duration_days: Number of days to generate data for
        """
        if start_date is None:
            # Default to ending at today's midnight, going back duration_days
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            self.end_datetime = today
            self.start_datetime = today - timedelta(days=duration_days)
        else:
            self.start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            self.end_datetime = self.start_datetime + timedelta(days=duration_days)
            
        self.sampling_interval = timedelta(minutes=1)
        
        # Freezer operational parameters
        self.temp_setpoint = -18.0  # Target temperature (°C)
        self.temp_deadband = 2.0    # Hysteresis band (°C)
        self.compressor_power_base = 8.5    # Base power when running (kW)
        self.compressor_power_idle = 0.5    # Standby power (kW)
        
        # Environmental parameters
        self.ambient_temp_base = 22.0       # Base ambient temperature (°C)
        self.thermal_mass_factor = 0.98     # Temperature change resistance
        
        # Operational patterns
        self.day_shift_start = 8    # 8 AM
        self.day_shift_end = 20     # 8 PM
        
        self.data_points: List[DataPoint] = []
        
    def _is_day_shift(self, timestamp: datetime) -> bool:
        """Determine if timestamp falls in day shift (8 AM - 8 PM)"""
        hour = timestamp.hour
        return self.day_shift_start <= hour < self.day_shift_end
    
    def _generate_ambient_temperature(self, timestamp: datetime) -> float:
        """
        Generate realistic ambient temperature with daily cycles.
        
        Simulates warmer temperatures during day, cooler at night,
        with some random variation to represent weather changes.
        """
        # Daily temperature cycle (peak around 2 PM, minimum around 6 AM)
        hour_angle = (timestamp.hour + timestamp.minute/60.0 - 6) * np.pi / 12
        daily_variation = 3.0 * np.sin(hour_angle)
        
        # Weekly variation (slight warming trend)
        days_elapsed = (timestamp - self.start_datetime).days
        weekly_trend = 0.5 * days_elapsed / 7.0
        
        # Random noise
        noise = np.random.normal(0, 0.8)
        
        return self.ambient_temp_base + daily_variation + weekly_trend + noise
    
    def _generate_door_events(self, timestamp: datetime) -> bool:
        """
        Generate realistic door opening patterns based on shift operations.
        
        Day shift: More frequent door openings (staff activity)
        Night shift: Minimal door activity (occasional checks)
        """
        if self._is_day_shift(timestamp):
            # Day shift: Higher probability of door events
            # Peak activity around meal times (12 PM, 6 PM)
            hour = timestamp.hour
            if hour in [12, 18]:  # Meal times
                base_prob = 0.008  # ~0.8% chance per minute
            elif 10 <= hour <= 16:  # Active work hours
                base_prob = 0.004  # ~0.4% chance per minute
            else:
                base_prob = 0.002  # ~0.2% chance per minute
        else:
            base_prob = 0.0     # no chance of door events during night shift
            
        return np.random.random() < base_prob
    
    def _calculate_heat_transfer(self, internal_temp: float, ambient_temp: float, 
                               door_open: bool, compressor_running: bool) -> float:
        """
        Calculate temperature change based on heat transfer physics.
        
        Factors:
        - Heat leak through insulation
        - Increased heat transfer when door is open
        - Cooling effect when compressor is running
        """
        # Base heat leak (always present)
        temp_diff = ambient_temp - internal_temp
        base_leak = temp_diff * 0.002  # Insulation factor
        
        # Door open multiplier (significant heat transfer)
        if door_open:
            door_effect = temp_diff * 0.05  # Much higher heat transfer
        else:
            door_effect = 0.0
            
        # Compressor cooling effect
        if compressor_running:
            cooling_effect = -0.8  # Strong cooling
        else:
            cooling_effect = 0.0
            
        # Add some randomness for realistic sensor noise
        noise = np.random.normal(0, 0.05)
        
        total_change = base_leak + door_effect + cooling_effect + noise
        return total_change * self.thermal_mass_factor
    
    def _should_compressor_run(self, internal_temp: float, current_state: bool) -> bool:
        """
        Determine compressor state based on temperature and hysteresis control.
        
        Implements realistic thermostat behavior with deadband to prevent
        rapid cycling (compressor short-cycling protection).
        """
        if current_state:  # Currently running
            # Turn off when temperature drops below setpoint
            return internal_temp > self.temp_setpoint
        else:  # Currently off
            # Turn on when temperature rises above setpoint + deadband
            return internal_temp > (self.temp_setpoint + self.temp_deadband)
    
    def _calculate_compressor_power(self, running: bool, internal_temp: float) -> float:
        """
        Calculate realistic compressor power consumption.
        
        Power varies based on:
        - Running state (on/off)
        - Temperature differential (higher differential = higher power)
        - Some operational noise
        """
        if not running:
            return self.compressor_power_idle
        
        # Power increases with temperature differential
        temp_factor = max(0, internal_temp - self.temp_setpoint) * 0.3
        base_power = self.compressor_power_base + temp_factor
        
        # Add realistic power fluctuations (motor load variations)
        noise = np.random.normal(0, 0.2)
        
        return max(0.5, base_power + noise)  # Minimum power consumption
    
    def generate_normal_operation(self) -> None:
        """
        Generate baseline time series data representing normal freezer operation.
        
        Creates realistic patterns of:
        - Temperature cycling around setpoint
        - Door opening events based on shift patterns  
        - Compressor cycling with appropriate delays
        - Ambient temperature variations
        """
        logger.info("Generating normal operation data...")
        
        # Initialize state variables
        current_time = self.start_datetime
        internal_temp = self.temp_setpoint + 0.5  # Start slightly warm
        compressor_running = False
        door_open = False
        door_open_duration = 0  # Track how long door has been open
        
        while current_time < self.end_datetime:
            # Generate ambient temperature
            ambient_temp = self._generate_ambient_temperature(current_time)
            
            # Handle door events
            if door_open:
                door_open_duration += 1
                # Door typically stays open for 1-5 minutes during normal operations
                if door_open_duration > np.random.randint(1, 6):
                    door_open = False
                    door_open_duration = 0
            else:
                door_open = self._generate_door_events(current_time)
                if door_open:
                    door_open_duration = 1
            
            # Calculate temperature change
            temp_change = self._calculate_heat_transfer(
                internal_temp, ambient_temp, door_open, compressor_running
            )
            internal_temp += temp_change
            
            # Update compressor state
            compressor_running = self._should_compressor_run(internal_temp, compressor_running)
            
            # Calculate compressor power
            compressor_power = self._calculate_compressor_power(compressor_running, internal_temp)
            
            # Create data points for all tags
            data_points = [
                DataPoint(current_time, "FREEZER01.TEMP.INTERNAL_C", 
                         round(internal_temp, 2), "°C"),
                DataPoint(current_time, "FREEZER01.TEMP.AMBIENT_C", 
                         round(ambient_temp, 2), "°C"),
                DataPoint(current_time, "FREEZER01.DOOR.STATUS", 
                         1.0 if door_open else 0.0, "Boolean"),
                DataPoint(current_time, "FREEZER01.COMPRESSOR.POWER_KW", 
                         round(compressor_power, 2), "kW"),
                DataPoint(current_time, "FREEZER01.COMPRESSOR.STATUS", 
                         1.0 if compressor_running else 0.0, "Boolean"),
            ]
            
            self.data_points.extend(data_points)
            current_time += self.sampling_interval
    
    def inject_anomaly_prolonged_door_open(self, start_time: str, duration_minutes: int = 15) -> None:
        """
        Inject anomaly: Door left open for extended period.
        
        Simulates scenario where door is accidentally left open, causing:
        - Rapid temperature rise
        - Extended compressor operation
        - Higher power consumption
        
        Args:
            start_time: When anomaly starts (YYYY-MM-DD HH:MM)
            duration_minutes: How long door stays open
        """
        logger.info(f"Injecting prolonged door open anomaly at {start_time} for {duration_minutes} minutes")
        
        anomaly_start = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        anomaly_end = anomaly_start + timedelta(minutes=duration_minutes)
        
        # Find and modify affected data points
        for point in self.data_points:
            if anomaly_start <= point.timestamp <= anomaly_end:
                if point.tag_name == "FREEZER01.DOOR.STATUS":
                    point.value = 1.0  # Force door open
                elif point.tag_name == "FREEZER01.TEMP.INTERNAL_C":
                    # Accelerate temperature rise
                    minutes_elapsed = (point.timestamp - anomaly_start).total_seconds() / 60
                    temp_rise = min(8.0, minutes_elapsed * 0.3)  # Cap at 8°C rise
                    point.value += temp_rise
                elif point.tag_name == "FREEZER01.COMPRESSOR.POWER_KW":
                    # Increase power consumption due to higher load
                    point.value *= 1.3
                    
    def inject_anomaly_compressor_failure(self, start_time: str, duration_minutes: int = 60) -> None:
        """
        Inject anomaly: Compressor hard‑off event – unit stays completely off, causing a rapid temperature rise that is easy to spot in demos.
        
        Simulates mechanical failure where compressor doesn't respond to
        temperature control signals, resulting in continuous temperature rise.
        
        Args:
            start_time: When anomaly starts (YYYY-MM-DD HH:MM)
            duration_minutes: How long compressor remains failed
        """
        logger.info(f"Injecting compressor failure anomaly at {start_time} for {duration_minutes} minutes")
        
        anomaly_start = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        anomaly_end = anomaly_start + timedelta(minutes=duration_minutes)
        
        for point in self.data_points:
            if anomaly_start <= point.timestamp <= anomaly_end:
                if point.tag_name == "FREEZER01.COMPRESSOR.STATUS":
                    point.value = 0.0  # Force compressor off
                elif point.tag_name == "FREEZER01.COMPRESSOR.POWER_KW":
                    point.value = 0.0  # Compressor draws no power during hard‑off
                elif point.tag_name == "FREEZER01.TEMP.INTERNAL_C":
                    # Accelerate temperature rise so the effect is obvious in a 1‑hour slice
                    minutes_elapsed = (point.timestamp - anomaly_start).total_seconds() / 60
                    temp_rise = min(15.0, minutes_elapsed * 0.30)  # Up to +15 °C at 50 min
                    point.value += temp_rise
                elif point.tag_name == "FREEZER01.COMPRESSOR.STATUS":
                    # Make sure status is firmly off for the entire interval
                    point.value = 0.0
    
    def inject_anomaly_sensor_flatline(self, start_time: str, duration_hours: int = 6) -> None:
        """
        Inject anomaly: Door sensor reports constant "closed" despite high temperature.
        
        Simulates sensor malfunction where door status is stuck, creating
        inconsistent data pattern (high temp but door shows closed).
        
        Args:
            start_time: When anomaly starts (YYYY-MM-DD HH:MM)
            duration_hours: How long sensor remains stuck
        """
        logger.info(f"Injecting sensor flatline anomaly at {start_time} for {duration_hours} hours")
        
        anomaly_start = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        anomaly_end = anomaly_start + timedelta(hours=duration_hours)
        
        for point in self.data_points:
            if anomaly_start <= point.timestamp <= anomaly_end:
                if point.tag_name == "FREEZER01.DOOR.STATUS":
                    point.value = 0.0  # Stuck at "closed"
                    point.quality = Quality.QUESTIONABLE  # Mark as suspect data
                elif point.tag_name == "FREEZER01.TEMP.INTERNAL_C":
                    # Temperature rises as if door is actually open
                    minutes_elapsed = (point.timestamp - anomaly_start).total_seconds() / 60
                    temp_rise = min(6.0, minutes_elapsed * 0.1)
                    point.value += temp_rise
    
    def inject_anomaly_power_fluctuation(self, start_time: str, duration_minutes: int = 30) -> None:
        """
        Inject anomaly: Electrical power fluctuations affecting compressor.
        
        Simulates power quality issues causing erratic compressor behavior
        and inconsistent cooling performance.
        
        Args:
            start_time: When anomaly starts (YYYY-MM-DD HH:MM)  
            duration_minutes: Duration of power issues
        """
        logger.info(f"Injecting power fluctuation anomaly at {start_time} for {duration_minutes} minutes")
        
        anomaly_start = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        anomaly_end = anomaly_start + timedelta(minutes=duration_minutes)
        
        for point in self.data_points:
            if anomaly_start <= point.timestamp <= anomaly_end:
                if point.tag_name == "FREEZER01.COMPRESSOR.POWER_KW":
                    # Add significant power fluctuations
                    fluctuation = np.random.normal(0, 2.0)  # Large variations
                    point.value = max(0.5, point.value + fluctuation)
                elif point.tag_name == "FREEZER01.TEMP.INTERNAL_C":
                    # Less effective cooling due to power issues
                    minutes_elapsed = (point.timestamp - anomaly_start).total_seconds() / 60
                    temp_drift = minutes_elapsed * 0.05  # Slight temperature drift
                    point.value += temp_drift
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert generated data points to pandas DataFrame in PI System long format.
        
        Returns:
            DataFrame with columns: Timestamp, TagName, Value, Units, Quality
        """
        data = []
        for point in self.data_points:
            data.append({
                'Timestamp': point.timestamp.isoformat(),
                'TagName': point.tag_name,
                'Value': point.value,
                'Units': point.units,
                'Quality': point.quality.value
            })
        
        df = pd.DataFrame(data)
        return df.sort_values(['Timestamp', 'TagName']).reset_index(drop=True)
    
    def export_to_csv(self, filename: str = "freezer_system_data.csv") -> None:
        """
        Export generated data to CSV file in PI System compatible format.
        
        Args:
            filename: Output CSV filename
        """
        df = self.to_dataframe()
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(df)} data points to {filename}")
        
        # Log summary statistics
        unique_tags = df['TagName'].nunique()
        date_range = f"{df['Timestamp'].min()} to {df['Timestamp'].max()}"
        logger.info(f"Dataset summary: {unique_tags} tags, {date_range}")


def main():
    """
    Main function to generate freezer system mock data with realistic anomalies.
    
    Creates a comprehensive dataset demonstrating various failure modes and
    operational patterns that enable MCP insight demonstrations.
    """
    logger.info("Starting Manufacturing Copilot freezer data generation...")
    
    # Initialize generator for one week of data ending at today's midnight
    generator = FreezorDataGenerator(duration_days=7)
    
    # Generate baseline normal operation
    generator.generate_normal_operation()
    
    # Calculate relative anomaly times based on the generated date range
    # Inject anomalies at strategic times relative to the end date
    end_date = generator.end_datetime
    
    # Inject modular anomalies at strategic times
    # Anomaly 1: Door left open during day shift (YESTERDAY afternoon - easy to find!)
    anomaly1_time = (end_date - timedelta(days=1)).replace(hour=14, minute=30)
    generator.inject_anomaly_prolonged_door_open(
        anomaly1_time.strftime("%Y-%m-%d %H:%M"), 
        duration_minutes=19  # 19 minutes for the $47 cost calculation
    )
    
    # Anomaly 2: Compressor failure during night (4 days ago, early morning)
    anomaly2_time = (end_date - timedelta(days=4)).replace(hour=2, minute=15)
    generator.inject_anomaly_compressor_failure(
        anomaly2_time.strftime("%Y-%m-%d %H:%M"), 
        duration_minutes=55
    )
    
    # Anomaly 3: Sensor malfunction (3 days ago, afternoon)
    anomaly3_time = (end_date - timedelta(days=3)).replace(hour=16, minute=0)
    generator.inject_anomaly_sensor_flatline(
        anomaly3_time.strftime("%Y-%m-%d %H:%M"), 
        duration_hours=4
    )
    
    # Anomaly 4: Power fluctuations (2 days ago, late morning)
    anomaly4_time = (end_date - timedelta(days=2)).replace(hour=11, minute=45)
    generator.inject_anomaly_power_fluctuation(
        anomaly4_time.strftime("%Y-%m-%d %H:%M"), 
        duration_minutes=25
    )
    
    # Export to CSV for MCP analysis
    generator.export_to_csv("data/freezer_system_mock_data.csv")
    
    # Generate summary for verification
    df = generator.to_dataframe()
    print(f"\n=== Dataset Summary ===")
    print(f"Total data points: {len(df):,}")
    print(f"Unique tags: {df['TagName'].nunique()}")
    print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"Quality distribution:")
    print(df['Quality'].value_counts())
    print(f"\nTag breakdown:")
    print(df['TagName'].value_counts())
    
    # Show anomaly timing for reference
    print(f"\n=== Injected Anomalies ===")
    print(f"1. Prolonged door open: {anomaly1_time.strftime('%Y-%m-%d %H:%M')} (19 min)")
    print(f"2. Compressor failure: {anomaly2_time.strftime('%Y-%m-%d %H:%M')} (55 min)")
    print(f"3. Sensor flatline: {anomaly3_time.strftime('%Y-%m-%d %H:%M')} (4 hours)")
    print(f"4. Power fluctuations: {anomaly4_time.strftime('%Y-%m-%d %H:%M')} (25 min)")
    
    logger.info("Freezer data generation completed successfully!")


if __name__ == "__main__":
    main() 