#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - LLM-Powered Interpreter

Uses GPT-4 to provide expert-level manufacturing insights, root cause analysis,
and operational recommendations. This is the intelligent reasoning layer that
transforms raw data into actionable manufacturing intelligence.

This module demonstrates the power of LLM-driven analysis for:
- Natural language query understanding
- Multi-step analysis planning
- Expert-level insight generation
- Manufacturing domain expertise
"""

import logging
import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
from pydantic import BaseModel, Field
import pandas as pd

from src.glossary import TagGlossary
from src.tools import (
    load_data, summarize_metric, 
    detect_spike, find_correlated_tags,
    generate_chart
)
from src.tools.data_loader import get_data_time_range, get_available_tags

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Global glossary instance
_glossary_instance = None

def get_glossary() -> TagGlossary:
    """Get or create a singleton TagGlossary instance."""
    global _glossary_instance
    if _glossary_instance is None:
        _glossary_instance = TagGlossary()
    return _glossary_instance


class AnalysisPlan(BaseModel):
    """
    LLM-generated analysis plan for manufacturing data investigation.
    
    Represents the intelligent analysis strategy determined by GPT-4
    based on the user's query and available data sources.
    """
    primary_tag: str = Field(..., description="Most relevant PI tag for the query")
    start_time: datetime = Field(..., description="Analysis start time")
    end_time: datetime = Field(..., description="Analysis end time")
    analysis_steps: List[str] = Field(..., description="Ordered list of analysis tools to execute")
    reasoning: str = Field(..., description="LLM explanation of why this approach was chosen")
    sensitivity_level: str = Field(default="normal", description="Anomaly detection sensitivity: low, normal, high")


def parse_query_with_llm(query: str) -> AnalysisPlan:
    """
    Use GPT-4 to parse natural language query and create intelligent analysis plan.
    
    This replaces deterministic keyword matching with true language understanding
    and manufacturing domain expertise.
    
    Args:
        query: Natural language query about manufacturing operations
        
    Returns:
        AnalysisPlan with intelligent analysis strategy
        
    Raises:
        ValueError: If LLM cannot parse the query or create a valid plan
    """
    logger.info(f"Using LLM to parse query: '{query}'")
    
    # Get available tags and data range for context
    try:
        available_tags = get_available_tags()
        data_range = get_data_time_range()
        tag_descriptions = _get_tag_descriptions()
    except Exception as e:
        logger.warning(f"Could not get full context: {e}")
        available_tags = ["FREEZER01.TEMP.INTERNAL_C", "FREEZER01.COMPRESSOR.POWER_KW", 
                         "FREEZER01.DOOR.STATUS", "FREEZER01.TEMP.AMBIENT_C", "FREEZER01.COMPRESSOR.STATUS"]
        data_range = {"start": datetime.now() - timedelta(days=7), "end": datetime.now()}
        tag_descriptions = {}
    
    # Create intelligent prompt for GPT-4
    system_prompt = """You are a senior manufacturing engineer and data analyst specializing in industrial refrigeration systems. 
    You have deep expertise in:
    - PI System data analysis and interpretation
    - Freezer/refrigeration system troubleshooting
    - Root cause analysis for temperature and power anomalies
    - Predictive maintenance and operational optimization
    
    Your job is to parse user queries and create intelligent analysis plans."""
    
    user_prompt = f"""
    QUERY: "{query}"
    
    AVAILABLE DATA:
    - Tags: {available_tags}
    - Tag Descriptions: {tag_descriptions}
    - Data Range: {data_range['start']} to {data_range['end']}
    - Available Analysis Tools: detect_anomalies, correlate_tags, generate_chart, basic_statistics
    
    Create an analysis plan as JSON with:
    {{
        "primary_tag": "most relevant tag name",
        "start_time": "YYYY-MM-DD HH:MM:SS",
        "end_time": "YYYY-MM-DD HH:MM:SS", 
        "analysis_steps": ["ordered", "list", "of", "tools"],
        "reasoning": "expert explanation of analysis approach",
        "sensitivity_level": "low/normal/high for anomaly detection"
    }}
    
    GUIDELINES:
    - Choose the most relevant tag based on the query context
    - Parse time references intelligently (yesterday, last week, etc.)
    - Select appropriate analysis tools based on what the user is asking
    - If asking about problems/issues/anomalies: include detect_anomalies
    - If asking about causes/relationships: include correlate_tags  
    - If asking to show/display/visualize: include generate_chart
    - Always include basic_statistics as foundation
    - Provide expert reasoning for your choices
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent parsing
            max_tokens=1000
        )
        
        # Parse the JSON response
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response (handle potential markdown formatting)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        elif "{" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_text = response_text[json_start:json_end]
        else:
            raise ValueError("No JSON found in LLM response")
        
        plan_data = json.loads(json_text)
        
        # Convert string dates to datetime objects
        plan_data['start_time'] = datetime.fromisoformat(plan_data['start_time'].replace('Z', '+00:00')).replace(tzinfo=None)
        plan_data['end_time'] = datetime.fromisoformat(plan_data['end_time'].replace('Z', '+00:00')).replace(tzinfo=None)
        
        # Create and validate the plan
        plan = AnalysisPlan(**plan_data)
        
        logger.info(f"LLM created analysis plan: {plan.primary_tag}, {len(plan.analysis_steps)} steps")
        return plan
        
    except Exception as e:
        logger.error(f"LLM query parsing failed: {e}")
        raise ValueError(f"Could not parse query with LLM: {e}")


def execute_analysis_plan(plan: AnalysisPlan) -> Dict[str, Any]:
    """
    Execute the LLM-generated analysis plan by running tools in sequence.
    
    Builds comprehensive data context that will be fed to the LLM
    for expert-level insight generation.
    
    Args:
        plan: LLM-generated analysis plan
        
    Returns:
        Dictionary with all analysis results and context
    """
    logger.info(f"Executing analysis plan with {len(plan.analysis_steps)} steps")
    
    context = {
        'query_plan': plan,
        'primary_tag': plan.primary_tag,
        'time_range': (plan.start_time, plan.end_time),
        'analysis_results': {}
    }
    
    try:
        # Step 1: Always load data and get basic statistics
        logger.info(f"Loading data for {plan.primary_tag}")
        df = load_data(plan.primary_tag, plan.start_time, plan.end_time)
        
        if df.empty:
            context['error'] = f"No data found for {plan.primary_tag} in specified time range"
            return context
        
        context['data_points'] = len(df)
        context['analysis_results']['basic_statistics'] = summarize_metric(df)
        
        # Step 2: Execute planned analysis steps
        for step in plan.analysis_steps:
            logger.info(f"Executing analysis step: {step}")
            
            if step == "detect_anomalies":
                # Set threshold based on sensitivity level
                threshold_map = {"low": 4.0, "normal": 3.0, "high": 2.0}
                threshold = threshold_map.get(plan.sensitivity_level, 3.0)
                
                anomalies = detect_spike(df, threshold=threshold)
                context['analysis_results']['anomalies'] = {
                    'count': len(anomalies),
                    'threshold': threshold,
                    'details': [
                        {
                            'timestamp': str(ts),
                            'value': float(val),
                            'z_score': float(z),
                            'description': reason
                        }
                        for ts, val, z, reason in anomalies[:10]  # Limit to top 10
                    ]
                }
                
            elif step == "correlate_tags":
                correlations = find_correlated_tags(
                    plan.primary_tag, 
                    plan.start_time, 
                    plan.end_time,
                    correlation_threshold=0.2  # Lower threshold for more insights
                )
                context['analysis_results']['correlations'] = {
                    'count': len(correlations),
                    'details': [
                        {
                            'tag': corr['tag_name'],
                            'correlation': float(corr['pearson_correlation']),
                            'strength': corr['correlation_strength'],
                            'significance': corr['statistical_significance'],
                            'data_points': corr['data_points']
                        }
                        for corr in correlations[:5]  # Top 5 correlations
                    ]
                }
                
            elif step == "generate_chart":
                # Generate chart with anomaly highlights if available
                highlights = None
                if 'anomalies' in context['analysis_results']:
                    anomaly_details = context['analysis_results']['anomalies']['details']
                    highlights = [
                        (
                            datetime.fromisoformat(a['timestamp']) - timedelta(minutes=5),
                            datetime.fromisoformat(a['timestamp']) + timedelta(minutes=5)
                        )
                        for a in anomaly_details[:5]  # Highlight top 5 anomalies
                    ]
                
                chart_path = generate_chart(df, plan.primary_tag, highlights=highlights)
                context['analysis_results']['visualization'] = {
                    'chart_path': chart_path,
                    'chart_filename': os.path.basename(chart_path),
                    'anomaly_highlights': len(highlights) if highlights else 0
                }
        
        logger.info(f"Analysis plan execution completed successfully")
        return context
        
    except Exception as e:
        logger.error(f"Error executing analysis plan: {e}")
        context['error'] = str(e)
        return context


def generate_expert_insights(query: str, context: Dict[str, Any]) -> str:
    """
    Use GPT-4 to generate expert-level manufacturing insights and recommendations.
    
    This is where the magic happens - the LLM analyzes all the data and provides
    insights like a senior process engineer would.
    
    Args:
        query: Original user query
        context: Complete analysis results and data context
        
    Returns:
        Expert-level insights and recommendations in natural language
    """
    logger.info("Generating expert insights with LLM")
    
    if 'error' in context:
        return f"❌ **Analysis Error**: {context['error']}"
    
    # Create comprehensive context for the LLM
    system_prompt = """You are a senior manufacturing engineer with 20+ years of experience in industrial refrigeration systems, 
    PI System data analysis, and predictive maintenance. You specialize in:

    - Root cause analysis for temperature and power anomalies
    - Compressor and refrigeration system troubleshooting  
    - Predictive maintenance and operational optimization
    - Manufacturing process improvement and efficiency

    Your job is to analyze data results and provide expert insights that would help plant operators and maintenance teams.
    Write like you're briefing the plant manager - be clear, actionable, and authoritative."""
    
    # Build detailed context prompt
    plan = context['query_plan']
    results = context['analysis_results']
    
    context_prompt = f"""
    USER QUERY: "{query}"
    
    ANALYSIS PERFORMED:
    - Primary Tag: {plan.primary_tag}
    - Time Range: {plan.start_time} to {plan.end_time}
    - Data Points: {context['data_points']}
    - Analysis Steps: {', '.join(plan.analysis_steps)}
    
    RESULTS:
    """
    
    # Add basic statistics
    if 'basic_statistics' in results:
        stats = results['basic_statistics']
        context_prompt += f"""
    BASIC STATISTICS:
    - Mean: {stats.get('mean', 'N/A')}
    - Min: {stats.get('min', 'N/A')} | Max: {stats.get('max', 'N/A')}
    - Change: {stats.get('change', 'N/A')} ({stats.get('change_pct', 'N/A')}%)
    - Standard Deviation: {stats.get('std', 'N/A')}
    """
    
    # Add anomaly information
    if 'anomalies' in results:
        anomalies = results['anomalies']
        context_prompt += f"""
    ANOMALY DETECTION:
    - Found: {anomalies['count']} anomalies (threshold: {anomalies['threshold']}σ)
    - Details: {json.dumps(anomalies['details'], indent=2)}
    """
    
    # Add correlation information
    if 'correlations' in results:
        correlations = results['correlations']
        context_prompt += f"""
    CORRELATION ANALYSIS:
    - Found: {correlations['count']} significant correlations
    - Details: {json.dumps(correlations['details'], indent=2)}
    """
    
    # Add visualization info
    if 'visualization' in results:
        viz = results['visualization']
        context_prompt += f"""
    VISUALIZATION:
    - Chart generated: {viz['chart_filename']}
    - Anomaly highlights: {viz['anomaly_highlights']}
    """
    
    context_prompt += """
    
    PROVIDE EXPERT ANALYSIS:
    1. **What Happened?** - Summarize the key findings in manufacturing terms
    2. **Root Cause Analysis** - What likely caused any issues or patterns?
    3. **Operational Impact** - How does this affect production/efficiency?
    4. **Recommended Actions** - Specific next steps for operators/maintenance
    5. **Preventive Measures** - How to avoid similar issues in the future
    
    Format your response with clear sections and actionable recommendations.
    Use manufacturing terminology and be specific about equipment and processes.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_prompt}
            ],
            temperature=0.3,  # Some creativity for insights, but stay factual
            max_tokens=2000
        )
        
        expert_insights = response.choices[0].message.content.strip()
        
        # Add chart reference if available
        if 'visualization' in results:
            chart_info = f"\n\n📊 **Visualization**: Chart saved as `{results['visualization']['chart_filename']}`"
            if results['visualization']['anomaly_highlights'] > 0:
                chart_info += f" with {results['visualization']['anomaly_highlights']} anomaly periods highlighted"
            expert_insights += chart_info
        
        logger.info("Expert insights generated successfully")
        return expert_insights
        
    except Exception as e:
        logger.error(f"Error generating expert insights: {e}")
        return f"❌ **Error generating insights**: {e}"


def llm_interpret_query(query: str) -> str:
    """
    Main LLM-powered query interpretation function.
    
    This is the complete pipeline that replaces the deterministic interpreter:
    1. Parse query with LLM intelligence
    2. Execute multi-step analysis plan
    3. Generate expert-level insights
    
    Args:
        query: Natural language query about manufacturing operations
        
    Returns:
        Expert-level analysis and recommendations
    """
    try:
        logger.info(f"🧠 LLM-powered analysis starting for: '{query}'")
        
        # Step 1: LLM parses query and creates analysis plan
        plan = parse_query_with_llm(query)
        logger.info(f"📋 Analysis plan: {plan.reasoning}")
        
        # Step 2: Execute the analysis plan
        context = execute_analysis_plan(plan)
        
        # Step 3: Generate expert insights
        insights = generate_expert_insights(query, context)
        
        logger.info("✅ LLM-powered analysis completed successfully")
        return insights
        
    except Exception as e:
        logger.error(f"LLM interpretation failed: {e}")
        return f"❌ **LLM Analysis Failed**: {e}\n\nPlease check your OpenAI API key and try again."


def _get_tag_descriptions() -> Dict[str, str]:
    """Get tag descriptions for LLM context."""
    try:
        glossary = get_glossary()
        tags = glossary.list_all_tags()
        return {tag['tag']: tag['description'] for tag in tags}
    except Exception:
        return {
            "FREEZER01.TEMP.INTERNAL_C": "Internal freezer temperature in Celsius",
            "FREEZER01.TEMP.AMBIENT_C": "Ambient room temperature in Celsius", 
            "FREEZER01.COMPRESSOR.POWER_KW": "Compressor power consumption in kilowatts",
            "FREEZER01.COMPRESSOR.STATUS": "Compressor on/off status",
            "FREEZER01.DOOR.STATUS": "Freezer door open/closed status"
        }


def main():
    """
    Demo function showing LLM-powered interpreter capabilities.
    """
    print("🧠 Manufacturing Copilot - LLM-Powered Analysis Demo")
    print("=" * 60)
    
    # Demo queries that showcase LLM intelligence
    demo_queries = [
        "What caused the temperature problems in the freezer yesterday?",
        "Show me any unusual patterns in the compressor power consumption",
        "Why did the freezer temperature spike last week? What should we check?",
        "Are there any correlations between door activity and temperature changes?",
        "Give me a complete analysis of the freezer system performance"
    ]
    
    print("🎯 Demo: LLM-Powered Manufacturing Intelligence")
    print("-" * 50)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Query {i}: '{query}'")
        print("=" * 60)
        
        try:
            result = llm_interpret_query(query)
            print(result)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)
    
    print("✅ LLM-powered demo completed!")


if __name__ == "__main__":
    main() 