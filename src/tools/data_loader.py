#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Data Loading Tools

Provides functions for loading and filtering time-series data from PI System
CSV exports. Handles tag validation, time range filtering, and data quality
checks for manufacturing data analysis.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

def load_data(
    tag: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    data_file: str = "data/freezer_system_mock_data.csv"
) -> pd.DataFrame:
    """
    Load time-series data from CSV file filtered by tag name and time range.
    
    Loads PI System long format data and filters for a specific tag within
    the specified time window. Provides helpful error messages if tag is
    not found or data is missing.
    
    Args:
        tag: PI System tag name (e.g., "FREEZER01.TEMP.INTERNAL_C")
        start: Start datetime for filtering (optional, defaults to data start)
        end: End datetime for filtering (optional, defaults to data end)
        data_file: Path to CSV file containing time-series data
        
    Returns:
        DataFrame with columns: Timestamp, TagName, Value, Units, Quality
        Filtered for the specified tag and time range, sorted by timestamp
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If tag is not found in the dataset
        RuntimeError: If no data exists in the specified time range
    """
    logger.debug(f"Loading data for tag '{tag}' from {data_file}")
    
    try:
        # Load the entire dataset
        df = pd.read_csv(data_file)
        
        # Convert timestamp column to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Get list of available tags for error reporting
        available_tags = df['TagName'].unique().tolist()
        
        # Validate tag exists
        if tag not in available_tags:
            similar_tags = [t for t in available_tags if tag.lower() in t.lower()]
            error_msg = f"Tag '{tag}' not found in dataset."
            if similar_tags:
                error_msg += f" Did you mean one of: {similar_tags[:3]}"
            else:
                error_msg += f" Available tags: {available_tags[:5]}{'...' if len(available_tags) > 5 else ''}"
            raise ValueError(error_msg)
        
        # Filter by tag
        tag_data = df[df['TagName'] == tag].copy()
        
        if tag_data.empty:
            raise RuntimeError(f"No data found for tag '{tag}'")
        
        # Apply time range filtering if specified
        if start is not None:
            tag_data = tag_data[tag_data['Timestamp'] >= start]
            
        if end is not None:
            tag_data = tag_data[tag_data['Timestamp'] <= end]
            
        # Check if we have data after time filtering
        if tag_data.empty:
            data_start = df[df['TagName'] == tag]['Timestamp'].min()
            data_end = df[df['TagName'] == tag]['Timestamp'].max()
            raise RuntimeError(
                f"No data found for tag '{tag}' in specified time range. "
                f"Available data: {data_start} to {data_end}"
            )
        
        # Sort by timestamp and reset index
        tag_data = tag_data.sort_values('Timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(tag_data)} data points for tag '{tag}'")
        
        return tag_data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_file}")
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"Data file is empty: {data_file}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise RuntimeError(f"Failed to load data: {e}")


def get_available_tags(data_file: str = "data/freezer_system_mock_data.csv") -> List[str]:
    """
    Get list of all available tags in the dataset.
    
    Args:
        data_file: Path to CSV file containing time-series data
        
    Returns:
        List of unique tag names available in the dataset
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    try:
        df = pd.read_csv(data_file)
        return sorted(df['TagName'].unique().tolist())
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_file}")


def get_data_time_range(
    tag: Optional[str] = None,
    data_file: str = "data/freezer_system_mock_data.csv"
) -> Dict[str, datetime]:
    """
    Get the time range of available data for a tag or entire dataset.
    
    Args:
        tag: Specific tag to check (optional, defaults to entire dataset)
        data_file: Path to CSV file containing time-series data
        
    Returns:
        Dictionary with 'start' and 'end' datetime keys
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If specified tag is not found
    """
    try:
        df = pd.read_csv(data_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        if tag is not None:
            if tag not in df['TagName'].unique():
                raise ValueError(f"Tag '{tag}' not found in dataset")
            df = df[df['TagName'] == tag]
        
        return {
            'start': df['Timestamp'].min(),
            'end': df['Timestamp'].max()
        }
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_file}")


def preview_data(
    tag: str,
    num_points: int = 10,
    data_file: str = "data/freezer_system_mock_data.csv"
) -> pd.DataFrame:
    """
    Get a preview of data for a specific tag.
    
    Args:
        tag: PI System tag name
        num_points: Number of recent data points to return
        data_file: Path to CSV file containing time-series data
        
    Returns:
        DataFrame with the most recent data points for the tag
    """
    tag_data = load_data(tag, data_file=data_file)
    return tag_data.tail(num_points) 