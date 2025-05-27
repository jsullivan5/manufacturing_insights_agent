#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Tag Intelligence

Extends the tag glossary with automatic type detection and statistical baselines.
Provides semantic metadata needed for intelligent tool selection and causality analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Literal, TypedDict, Optional, Union, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Path to cache file
CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "tag_intelligence.json")

class BaselineStats(TypedDict):
    """Statistical baseline for numeric tags."""
    mean: float
    std: float
    q01: float
    q99: float
    min_normal: float
    max_normal: float
    min: float
    max: float


class TagIntelligence(TypedDict):
    """Enhanced tag metadata with value type and statistics."""
    tag: str
    description: str
    value_type: Literal['numeric', 'binary']
    unit: str
    category: str
    baseline_stats: Optional[BaselineStats]


def compute_tag_intelligence(force_refresh: bool = False) -> Dict[str, TagIntelligence]:
    """
    Compute and cache extended tag metadata with statistical baselines.
    
    Args:
        force_refresh: If True, recompute even if cache exists
        
    Returns:
        Dictionary of tag intelligence records by tag name
    """
    # Check for cached data first
    if not force_refresh and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                
            # Convert string dates back to datetime if needed
            return cache_data
        except Exception as e:
            logger.warning(f"Failed to load tag intelligence cache: {e}, recomputing...")
    
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Import here to avoid circular imports
    from src.glossary import TagGlossary
    from src.tools.data_loader import get_available_tags, load_data, get_data_time_range
    
    # Get tag glossary and data time range
    glossary = TagGlossary()
    all_tags_info = glossary.list_all_tags()
    data_range = get_data_time_range()
    
    tag_intelligence = {}
    
    for tag_info in all_tags_info:
        tag_name = tag_info['tag']
        logger.info(f"Computing intelligence for tag: {tag_name}")
        
        # Determine value type from unit or by inspecting data
        value_type = 'binary' if tag_info['unit'].lower() == 'boolean' else 'numeric'
        
        # Create base intelligence record
        intel_record = TagIntelligence(
            tag=tag_name,
            description=tag_info['description'],
            value_type=value_type,
            unit=tag_info['unit'],
            category=tag_info['category'],
            baseline_stats=None
        )
        
        # For numeric tags, compute baseline statistics
        if value_type == 'numeric':
            try:
                # Load one week of data
                df = load_data(tag_name, data_range['start'], data_range['end'])
                
                if not df.empty and len(df) >= 10:  # Need at least 10 points
                    values = df['Value'].dropna()
                    
                    # Compute statistics
                    q01 = values.quantile(0.01)
                    q99 = values.quantile(0.99)
                    mean = values.mean()
                    std = values.std()
                    
                    # Add 10% padding to normal range
                    range_width = q99 - q01
                    padding = range_width * 0.1
                    
                    baseline_stats = BaselineStats(
                        mean=float(mean),
                        std=float(std),
                        q01=float(q01),
                        q99=float(q99),
                        min_normal=float(q01 - padding),
                        max_normal=float(q99 + padding),
                        min=float(values.min()),
                        max=float(values.max())
                    )
                    
                    intel_record['baseline_stats'] = baseline_stats
                    
                    logger.info(f"Computed baseline stats for {tag_name}: "
                                f"mean={mean:.2f}, range=[{q01:.2f}, {q99:.2f}]")
                else:
                    logger.warning(f"Insufficient data for {tag_name}, skipping baseline computation")
                    
            except Exception as e:
                logger.error(f"Error computing baseline stats for {tag_name}: {e}")
        
        # Add to collection
        tag_intelligence[tag_name] = intel_record
    
    # Cache the results
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(tag_intelligence, f, indent=2)
        logger.info(f"Cached tag intelligence to {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Failed to cache tag intelligence: {e}")
    
    return tag_intelligence


def get_tag_intelligence(refresh: bool = False) -> Dict[str, TagIntelligence]:
    """
    Get tag intelligence data, computing it if necessary.
    
    Args:
        refresh: If True, force recomputation of intelligence data
        
    Returns:
        Dictionary of tag intelligence records by tag name
    """
    return compute_tag_intelligence(force_refresh=refresh)


def get_tag_metadata(tag: str) -> Optional[TagIntelligence]:
    """
    Get intelligence record for a specific tag.
    
    Args:
        tag: Tag name to look up
        
    Returns:
        TagIntelligence record if found, None otherwise
    """
    intelligence = get_tag_intelligence()
    return intelligence.get(tag)


def get_value_type(tag: str) -> str:
    """
    Get the value type (numeric/binary) for a tag.
    
    Args:
        tag: Tag name to check
        
    Returns:
        'numeric' or 'binary'
    """
    intel = get_tag_metadata(tag)
    if intel:
        return intel['value_type']
    return 'unknown'


def get_related_tags(tag: str, n: int = 5) -> List[str]:
    """
    Get semantically related tags based on category.
    
    Args:
        tag: Primary tag
        n: Number of related tags to return
        
    Returns:
        List of related tag names
    """
    intel = get_tag_metadata(tag)
    if not intel:
        return []
    
    # Get category
    category = intel['category']
    
    # Find other tags in same category
    all_intel = get_tag_intelligence()
    
    related = [
        t for t, info in all_intel.items()
        if info['category'] == category and t != tag
    ]
    
    return related[:n]


def is_anomaly(tag: str, value: float) -> bool:
    """
    Check if a value is anomalous for a given tag based on baseline stats.
    
    Args:
        tag: Tag name
        value: Value to check
        
    Returns:
        True if value is outside normal range, False otherwise
    """
    intel = get_tag_metadata(tag)
    if not intel or intel['value_type'] != 'numeric' or not intel['baseline_stats']:
        return False
    
    baseline = intel['baseline_stats']
    return value < baseline['min_normal'] or value > baseline['max_normal']


def main():
    """Demo function to test tag intelligence computation."""
    print("🧠 Manufacturing Copilot - Tag Intelligence Demo")
    print("=" * 60)
    
    # Force recomputation for demo
    print("Computing tag intelligence with statistical baselines...")
    intelligence = compute_tag_intelligence(force_refresh=True)
    
    print(f"\nProcessed {len(intelligence)} tags:")
    
    # Show stats for each type
    numeric_tags = [tag for tag, info in intelligence.items() 
                    if info['value_type'] == 'numeric']
    binary_tags = [tag for tag, info in intelligence.items() 
                   if info['value_type'] == 'binary']
    
    print(f"• {len(numeric_tags)} numeric tags")
    print(f"• {len(binary_tags)} binary tags")
    
    # Print detailed info for one numeric tag
    if numeric_tags:
        example = intelligence[numeric_tags[0]]
        print(f"\nExample numeric tag: {example['tag']}")
        print(f"Description: {example['description']}")
        print(f"Unit: {example['unit']}")
        print(f"Category: {example['category']}")
        if example['baseline_stats']:
            stats = example['baseline_stats']
            print("Baseline statistics:")
            print(f"• Mean: {stats['mean']:.2f} {example['unit']}")
            print(f"• Standard deviation: {stats['std']:.2f}")
            print(f"• Normal range: [{stats['min_normal']:.2f}, {stats['max_normal']:.2f}] {example['unit']}")
    
    # Print info for one binary tag
    if binary_tags:
        example = intelligence[binary_tags[0]]
        print(f"\nExample binary tag: {example['tag']}")
        print(f"Description: {example['description']}")
        print(f"Unit: {example['unit']}")
        print(f"Category: {example['category']}")
    
    print("\n" + "=" * 60)
    print("✅ Tag intelligence computation complete!")


if __name__ == "__main__":
    main() 