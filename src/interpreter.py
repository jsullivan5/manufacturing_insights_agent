#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Natural Language Query Interpreter

Interprets natural language queries about manufacturing operations and translates
them into structured data requests. Uses semantic search to find relevant tags
and dateparser to extract time ranges from natural language.

Enhanced with intelligent tool routing for anomaly detection, correlation analysis,
and visualization based on query intent analysis.

This module enables queries like:
- "Show me freezer temperatures last night" → Basic analysis + chart
- "What anomalies happened with the compressor yesterday?" → Anomaly detection + chart
- "What caused the temperature spike?" → Anomaly detection + correlation analysis
- "Are temperature and power related?" → Correlation analysis + dual chart
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import re

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dateparser
from pydantic import BaseModel, Field, field_validator
import pandas as pd

from src.glossary import TagGlossary
from src.tools import (
    load_data, summarize_metric, 
    detect_spike, correlate_tags, find_correlated_tags,
    generate_chart, generate_correlation_chart
)
from src.tools.data_loader import get_data_time_range, get_available_tags

# Configure logging
logger = logging.getLogger(__name__)

# Global glossary instance to avoid collection conflicts
_glossary_instance = None

def get_glossary() -> TagGlossary:
    """Get or create a singleton TagGlossary instance."""
    global _glossary_instance
    if _glossary_instance is None:
        _glossary_instance = TagGlossary()
    return _glossary_instance

class QueryParams(BaseModel):
    """
    Structured parameters extracted from natural language queries.
    
    Represents the essential components needed to execute a manufacturing
    data analysis query: which tag to analyze and what time range to examine.
    """
    tag: str = Field(..., description="PI System tag name to analyze")
    start: datetime = Field(..., description="Start time for data analysis")
    end: datetime = Field(..., description="End time for data analysis")
    
    @field_validator('start', 'end')
    @classmethod
    def validate_datetime(cls, v):
        """Ensure datetime objects are timezone-naive for consistency."""
        if v.tzinfo is not None:
            return v.replace(tzinfo=None)
        return v
    
    @field_validator('end')
    @classmethod
    def end_after_start(cls, v, info):
        """Ensure end time is after start time."""
        if 'start' in info.data and v <= info.data['start']:
            raise ValueError("End time must be after start time")
        return v


class QueryIntent(BaseModel):
    """
    Analysis of query intent to determine which tools to apply.
    
    Enables intelligent routing of queries to appropriate analytical tools
    based on natural language patterns and keywords.
    """
    needs_anomaly_detection: bool = Field(default=False, description="Query asks about anomalies, spikes, or unusual patterns")
    needs_correlation: bool = Field(default=False, description="Query asks about relationships or root causes")
    needs_visualization: bool = Field(default=False, description="Query asks to show, display, or visualize data")
    correlation_threshold: float = Field(default=0.3, description="Minimum correlation strength to report")
    anomaly_threshold: float = Field(default=3.0, description="Z-score threshold for anomaly detection")


def _analyze_query_intent(query: str) -> QueryIntent:
    """
    Analyze natural language query to determine which tools should be applied.
    
    Uses keyword matching and pattern recognition to understand user intent
    and route the query to appropriate analytical functions.
    
    Args:
        query: Natural language query string
        
    Returns:
        QueryIntent object with flags for required tools
    """
    query_lower = query.lower()
    
    # Anomaly detection keywords
    anomaly_keywords = [
        'anomal', 'spike', 'unusual', 'abnormal', 'strange', 'weird',
        'problem', 'issue', 'fault', 'error', 'wrong', 'unexpected',
        'outlier', 'deviation', 'irregular', 'malfunction'
    ]
    
    # Correlation/causation keywords  
    correlation_keywords = [
        'cause', 'caused', 'why', 'reason', 'because', 'due to',
        'relationship', 'related', 'correlat', 'connect', 'link',
        'affect', 'impact', 'influence', 'trigger', 'lead to',
        'follow', 'after', 'before', 'during'
    ]
    
    # Visualization keywords
    visualization_keywords = [
        'show', 'display', 'plot', 'chart', 'graph', 'visualiz',
        'see', 'look', 'view', 'picture', 'image'
    ]
    
    # Check for anomaly detection needs
    needs_anomaly = any(keyword in query_lower for keyword in anomaly_keywords)
    
    # Check for correlation analysis needs
    needs_correlation = any(keyword in query_lower for keyword in correlation_keywords)
    
    # Check for visualization needs (default to True for "show me" type queries)
    needs_visualization = (
        any(keyword in query_lower for keyword in visualization_keywords) or
        query_lower.startswith(('show', 'display', 'what happened'))
    )
    
    # Adjust thresholds based on query specificity
    anomaly_threshold = 3.0
    correlation_threshold = 0.3
    
    if any(word in query_lower for word in ['extreme', 'major', 'significant']):
        anomaly_threshold = 2.0  # More sensitive for explicit requests
        correlation_threshold = 0.5  # Higher threshold for significance
    elif any(word in query_lower for word in ['minor', 'small', 'slight']):
        anomaly_threshold = 4.0  # Less sensitive
        correlation_threshold = 0.2  # Lower threshold
    
    logger.debug(f"Query intent analysis: anomaly={needs_anomaly}, correlation={needs_correlation}, "
                f"visualization={needs_visualization}")
    
    return QueryIntent(
        needs_anomaly_detection=needs_anomaly,
        needs_correlation=needs_correlation,
        needs_visualization=needs_visualization,
        anomaly_threshold=anomaly_threshold,
        correlation_threshold=correlation_threshold
    )


def _extract_time_references(query: str) -> List[Tuple[str, str]]:
    """
    Extract potential time references from natural language query.
    
    Args:
        query: Natural language query string
        
    Returns:
        List of tuples containing (original_text, normalized_text) for time references
    """
    # Common time reference patterns
    time_patterns = [
        r'\b(last night|yesterday|today|tonight)\b',
        r'\b(last|past|previous)\s+(hour|hours|day|days|week|weeks)\b',
        r'\b(this|last)\s+(morning|afternoon|evening|night)\b',
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s+\d{1,2}\b',
        r'\b\d{1,2}[:/]\d{1,2}(?:[:/]\d{1,2})?\s*(am|pm)?\b',
        r'\b\d{1,2}\s*(am|pm)\b',
        r'\b(from|since|until|to|between)\s+[\w\s:/-]+\b'
    ]
    
    time_refs = []
    for pattern in time_patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            time_refs.append((match.group(), match.group()))
    
    return time_refs


def _parse_time_range(query: str) -> Tuple[datetime, datetime]:
    """
    Parse time range from natural language query using dateparser and heuristics.
    
    Args:
        query: Natural language query containing time references
        
    Returns:
        Tuple of (start_datetime, end_datetime)
        
    Raises:
        ValueError: If no valid time range can be extracted
    """
    # Get the data time range to provide context
    try:
        data_range = get_data_time_range()
        data_start = data_range['start']
        data_end = data_range['end']
    except Exception:
        # Fallback if data range can't be determined
        data_end = datetime.now()
        data_start = data_end - timedelta(days=7)
    
    # Extract time references from query
    time_refs = _extract_time_references(query)
    
    # Try to parse with dateparser
    parsed_dates = []
    if time_refs:
        for original, normalized in time_refs:
            parsed = dateparser.parse(normalized, settings={
                'RELATIVE_BASE': data_end,  # Use data end as reference point
                'PREFER_DATES_FROM': 'past',
                'RETURN_AS_TIMEZONE_AWARE': False
            })
            if parsed:
                parsed_dates.append(parsed)
    
    # If no specific dates found, try parsing the entire query
    if not parsed_dates:
        parsed = dateparser.parse(query, settings={
            'RELATIVE_BASE': data_end,
            'PREFER_DATES_FROM': 'past',
            'RETURN_AS_TIMEZONE_AWARE': False
        })
        if parsed:
            parsed_dates.append(parsed)
    
    # Determine time range based on parsed dates and query context
    if len(parsed_dates) >= 2:
        # Multiple dates found - use as start and end
        start_time = min(parsed_dates)
        end_time = max(parsed_dates)
    elif len(parsed_dates) == 1:
        # Single date found - determine range based on query context
        reference_time = parsed_dates[0]
        
        if any(term in query.lower() for term in ['last night', 'yesterday']):
            # Yesterday: 00:00 to 23:59 of the previous day
            start_time = reference_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = reference_time.replace(hour=23, minute=59, second=59, microsecond=0)
        elif 'night' in query.lower():
            # Night time: 8 PM to 6 AM
            if reference_time.hour < 12:  # If morning, assume previous night
                end_time = reference_time.replace(hour=6, minute=0, second=0, microsecond=0)
                start_time = (reference_time - timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)
            else:  # If afternoon/evening, assume that night
                start_time = reference_time.replace(hour=20, minute=0, second=0, microsecond=0)
                end_time = (reference_time + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
        elif any(term in query.lower() for term in ['morning', 'afternoon', 'evening']):
            # Specific time of day
            if 'morning' in query.lower():
                start_time = reference_time.replace(hour=6, minute=0, second=0, microsecond=0)
                end_time = reference_time.replace(hour=12, minute=0, second=0, microsecond=0)
            elif 'afternoon' in query.lower():
                start_time = reference_time.replace(hour=12, minute=0, second=0, microsecond=0)
                end_time = reference_time.replace(hour=18, minute=0, second=0, microsecond=0)
            elif 'evening' in query.lower():
                start_time = reference_time.replace(hour=18, minute=0, second=0, microsecond=0)
                end_time = reference_time.replace(hour=23, minute=59, second=59, microsecond=0)
        else:
            # Default to full day
            start_time = reference_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = reference_time.replace(hour=23, minute=59, second=59, microsecond=0)
    else:
        # No specific dates found - use default based on query context
        if any(term in query.lower() for term in ['last', 'yesterday', 'previous']):
            # Default to last 24 hours
            end_time = data_end
            start_time = end_time - timedelta(hours=24)
        elif any(term in query.lower() for term in ['week', 'weekly']):
            # Default to last week
            end_time = data_end
            start_time = end_time - timedelta(days=7)
        else:
            # Default to last 24 hours
            end_time = data_end
            start_time = end_time - timedelta(hours=24)
    
    # Ensure times are within data range
    start_time = max(start_time, data_start)
    end_time = min(end_time, data_end)
    
    # Ensure we have a valid time range
    if start_time >= end_time:
        # Fallback to last 24 hours of available data
        end_time = data_end
        start_time = max(data_start, end_time - timedelta(hours=24))
    
    logger.debug(f"Parsed time range: {start_time} to {end_time}")
    return start_time, end_time


def parse_query(query: str) -> QueryParams:
    """
    Parse natural language query into structured parameters.
    
    Uses semantic search to find the most relevant PI tag and dateparser
    to extract time ranges from natural language time references.
    
    Args:
        query: Natural language query about manufacturing operations
        
    Returns:
        QueryParams object with tag, start, and end times
        
    Raises:
        ValueError: If no relevant tags found or time range cannot be parsed
    """
    logger.info(f"Parsing query: '{query}'")
    
    # Step 1: Find relevant tag using semantic search
    tag_results = get_glossary().search_tags(query, top_k=1)
    
    if not tag_results:
        raise ValueError(f"No relevant tags found for query: '{query}'")
    
    primary_tag = tag_results[0]['tag']
    similarity_score = tag_results[0]['similarity_score']
    
    logger.debug(f"Selected tag: {primary_tag} (similarity: {similarity_score:.3f})")
    
    # Step 2: Parse time range from query
    try:
        start_time, end_time = _parse_time_range(query)
    except Exception as e:
        logger.warning(f"Time parsing failed: {e}, using default 24-hour window")
        # Fallback to default time range
        try:
            data_range = get_data_time_range(primary_tag)
            end_time = data_range['end']
            start_time = end_time - timedelta(hours=24)
        except Exception:
            # Ultimate fallback
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
    
    # Step 3: Create and validate QueryParams
    try:
        params = QueryParams(
            tag=primary_tag,
            start=start_time,
            end=end_time
        )
        
        logger.info(f"Parsed query successfully: tag={params.tag}, "
                   f"time_range={params.start} to {params.end}")
        
        return params
        
    except Exception as e:
        raise ValueError(f"Failed to create valid query parameters: {e}")


def interpret_query(query: str) -> str:
    """
    Interpret natural language query and return comprehensive analysis results.
    
    Enhanced version that intelligently routes queries to appropriate analytical
    tools based on intent analysis. Provides anomaly detection, correlation
    analysis, and visualization as needed.
    
    Args:
        query: Natural language question about manufacturing operations
        
    Returns:
        Markdown-formatted string with comprehensive analysis results
        
    Raises:
        ValueError: If query cannot be interpreted or data cannot be loaded
    """
    try:
        # Step 1: Parse query into structured parameters
        params = parse_query(query)
        
        # Step 2: Analyze query intent to determine which tools to apply
        intent = _analyze_query_intent(query)
        
        # Step 3: Load data for the specified tag and time range
        df = load_data(
            tag=params.tag,
            start=params.start,
            end=params.end
        )
        
        if df.empty:
            return f"❌ No data found for tag '{params.tag}' in the specified time range."
        
        # Step 4: Generate basic summary statistics
        stats = summarize_metric(df)
        
        if 'error' in stats:
            return f"❌ Error analyzing data: {stats['error']}"
        
        # Step 5: Get tag metadata for units
        tag_results = get_glossary().search_tags(params.tag, top_k=1)
        unit = tag_results[0]['unit'] if tag_results else "units"
        
        # Step 6: Build comprehensive analysis results
        result_sections = []
        
        # Basic summary section
        start_str = params.start.strftime('%b %d %I:%M%p').replace(' 0', ' ')
        end_str = params.end.strftime('%b %d %I:%M%p').replace(' 0', ' ')
        
        # Determine trend
        change_pct = stats['change_pct']
        if abs(change_pct) < 1.0:
            trend = "Stable"
        elif change_pct > 5.0:
            trend = "Rising"
        elif change_pct < -5.0:
            trend = "Falling"
        elif change_pct > 0:
            trend = "Slight increase"
        else:
            trend = "Slight decrease"
        
        basic_summary = f"""✅ **Analysis Summary for {params.tag}**
→ **Time Range**: {start_str} – {end_str}
→ **Statistics**: Mean: {stats['mean']:.1f}{unit} | Min: {stats['min']:.1f}{unit} | Max: {stats['max']:.1f}{unit} | Trend: {trend}
→ **Data Points**: {stats['count']:,} | Change: {stats['change']:+.1f}{unit} ({change_pct:+.1f}%)"""
        
        # Add data quality info if relevant
        if 'Quality' in df.columns:
            quality_counts = df['Quality'].value_counts()
            if len(quality_counts) > 1 or 'Good' not in quality_counts:
                good_pct = (quality_counts.get('Good', 0) / len(df)) * 100
                basic_summary += f"\n→ **Data Quality**: {good_pct:.1f}% Good"
                if quality_counts.get('Questionable', 0) > 0:
                    questionable_pct = (quality_counts.get('Questionable', 0) / len(df)) * 100
                    basic_summary += f", {questionable_pct:.1f}% Questionable"
        
        result_sections.append(basic_summary)
        
        # Step 7: Apply anomaly detection if requested
        anomaly_highlights = []
        if intent.needs_anomaly_detection:
            try:
                anomalies = detect_spike(df, threshold=intent.anomaly_threshold)
                
                if anomalies:
                    anomaly_section = f"\n\n🔍 **Anomaly Detection Results** (threshold: {intent.anomaly_threshold}σ)"
                    anomaly_section += f"\n→ **Found {len(anomalies)} anomalies:**"
                    
                    for i, (timestamp, value, z_score, reason) in enumerate(anomalies[:5]):  # Show top 5
                        time_str = timestamp.strftime('%b %d %I:%M%p').replace(' 0', ' ')
                        anomaly_section += f"\n   • **{time_str}**: {value:.2f}{unit} - {reason}"
                        
                        # Create highlight periods for visualization (±5 minutes around anomaly)
                        start_highlight = timestamp - timedelta(minutes=5)
                        end_highlight = timestamp + timedelta(minutes=5)
                        anomaly_highlights.append((start_highlight, end_highlight))
                    
                    if len(anomalies) > 5:
                        anomaly_section += f"\n   • ... and {len(anomalies) - 5} more anomalies"
                    
                    result_sections.append(anomaly_section)
                else:
                    result_sections.append(f"\n\n✅ **No anomalies detected** (threshold: {intent.anomaly_threshold}σ)")
                    
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")
                result_sections.append(f"\n\n⚠️ **Anomaly detection failed**: {e}")
        
        # Step 8: Apply correlation analysis if requested
        if intent.needs_correlation:
            try:
                correlations = find_correlated_tags(
                    params.tag, 
                    params.start, 
                    params.end, 
                    correlation_threshold=intent.correlation_threshold
                )
                
                if correlations:
                    corr_section = f"\n\n🔗 **Correlation Analysis** (threshold: {intent.correlation_threshold:.1f})"
                    corr_section += f"\n→ **Found {len(correlations)} significant correlations:**"
                    
                    for corr in correlations[:3]:  # Show top 3 correlations
                        direction = "positively" if corr['pearson_correlation'] > 0 else "negatively"
                        strength = corr['correlation_strength'].replace('_', ' ')
                        corr_section += f"\n   • **{corr['tag_name']}**: {corr['pearson_correlation']:.3f} ({strength} {direction} correlated)"
                        
                        # Add lagged correlation info if significant
                        lag_info = corr['lagged_correlation']
                        if abs(lag_info['best_correlation']) > 0.3 and lag_info['best_lag'] != 0:
                            lag_direction = "leads" if lag_info['best_lag'] > 0 else "follows"
                            corr_section += f" - {corr['tag_name']} {lag_direction} by {abs(lag_info['best_lag'])} points"
                    
                    if len(correlations) > 3:
                        corr_section += f"\n   • ... and {len(correlations) - 3} more correlations"
                    
                    result_sections.append(corr_section)
                else:
                    result_sections.append(f"\n\n📊 **No significant correlations found** (threshold: {intent.correlation_threshold:.1f})")
                    
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
                result_sections.append(f"\n\n⚠️ **Correlation analysis failed**: {e}")
        
        # Step 9: Generate visualization if requested
        chart_info = ""
        if intent.needs_visualization:
            try:
                chart_path = generate_chart(
                    df, 
                    params.tag, 
                    highlights=anomaly_highlights if anomaly_highlights else None
                )
                
                chart_filename = os.path.basename(chart_path)
                chart_info = f"\n\n📊 **Visualization Generated**\n→ Chart saved to: `{chart_filename}`"
                
                if anomaly_highlights:
                    chart_info += f"\n→ Anomaly periods highlighted in red"
                
                result_sections.append(chart_info)
                
            except Exception as e:
                logger.warning(f"Chart generation failed: {e}")
                result_sections.append(f"\n\n⚠️ **Chart generation failed**: {e}")
        
        # Step 10: Add actionable insights and recommendations
        insights_section = "\n\n💡 **Insights & Recommendations**"
        
        # Trend-based insights
        if abs(change_pct) > 10:
            direction = "increase" if change_pct > 0 else "decrease"
            insights_section += f"\n→ **Significant {direction}** of {abs(change_pct):.1f}% detected - investigate operational changes"
        
        # Anomaly-based insights
        if intent.needs_anomaly_detection and anomaly_highlights:
            insights_section += f"\n→ **{len(anomaly_highlights)} anomaly periods** identified - check for equipment issues or process changes"
        
        # Correlation-based insights
        if intent.needs_correlation and 'correlations' in locals() and correlations:
            top_corr = correlations[0]
            if abs(top_corr['pearson_correlation']) > 0.6:
                insights_section += f"\n→ **Strong correlation** with {top_corr['tag_name']} suggests potential causal relationship"
        
        # General recommendations
        if not intent.needs_anomaly_detection and not intent.needs_correlation:
            insights_section += "\n→ Consider running anomaly detection or correlation analysis for deeper insights"
        
        result_sections.append(insights_section)
        
        # Combine all sections
        final_result = "".join(result_sections)
        
        logger.info(f"Successfully interpreted query with {len(result_sections)} analysis sections")
        return final_result
        
    except Exception as e:
        logger.error(f"Error interpreting query '{query}': {e}")
        return f"❌ Error interpreting query: {e}"


def main():
    """
    Demo function showing interpreter capabilities with example queries.
    """
    print("Manufacturing Copilot - Query Interpreter Demo")
    print("=" * 50)
    
    # Example queries to demonstrate capabilities
    demo_queries = [
        "Show me freezer temperatures last night",
        "What happened with the compressor yesterday?",
        "Door activity patterns from yesterday morning",
        "Power consumption in the last 6 hours",
        "Temperature readings on Monday"
    ]
    
    print("Demo: Natural Language Query Interpretation")
    print("-" * 45)
    
    for query in demo_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 30)
        
        try:
            result = interpret_query(query)
            print(result)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("=" * 50)
    print("Demo completed!")


if __name__ == "__main__":
    main() 