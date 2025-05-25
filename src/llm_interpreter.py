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
from typing import List, Dict, Any, Optional, Tuple

# Suppress tokenizer warnings and model loading verbosity
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

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

# Configure logging - suppress verbose HTTP and telemetry logs
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format for cleaner output
)
logger = logging.getLogger(__name__)

# Suppress verbose third-party logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

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
        RuntimeError: If cannot access required data sources
    """
    # Get available tags and data range for context - fail if not available
    try:
        available_tags = get_available_tags()
        data_range = get_data_time_range()
        tag_descriptions = _get_tag_descriptions()
    except Exception as e:
        logger.error(f"Failed to get data context: {e}")
        raise RuntimeError(f"Cannot access manufacturing data sources: {e}. Please check data connections and try again.")
    
    # Validate we have actual data to work with
    if not available_tags:
        raise RuntimeError("No manufacturing tags available. Please check data source connections.")
    
    if not data_range or 'start' not in data_range or 'end' not in data_range:
        raise RuntimeError("Cannot determine data time range. Please check data source connections.")
    
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
    
    You MUST respond with a valid JSON object in this exact format:
    {{
        "primary_tag": "most relevant tag name from available tags OR 'NO_RELEVANT_TAG'",
        "start_time": "YYYY-MM-DD HH:MM:SS",
        "end_time": "YYYY-MM-DD HH:MM:SS", 
        "analysis_steps": ["basic_statistics", "generate_chart"],
        "reasoning": "expert explanation of analysis approach",
        "sensitivity_level": "normal"
    }}
    
    GUIDELINES:
    - If the query asks for a tag that clearly doesn't exist in the available tags, set primary_tag to "NO_RELEVANT_TAG"
    - Only choose from the available tags if there's a reasonable match
    - If no tag is clearly relevant, use "NO_RELEVANT_TAG" and explain in reasoning
    - Parse time references intelligently (yesterday, last week, etc.)
    - If no time reference is given, use the last 24 hours
    - Select appropriate analysis tools based on what the user is asking
    - If asking about problems/issues/anomalies: include detect_anomalies
    - If asking about causes/relationships: include correlate_tags  
    - If asking to show/display/visualize: include generate_chart
    - Always include basic_statistics as foundation
    - Provide expert reasoning for your choices
    
    CRITICAL: You must ALWAYS return valid JSON. If no relevant tag exists, use "NO_RELEVANT_TAG".
    """
    
    try:
        print("Consulting GPT-4 manufacturing expert...")
        
        # Use streaming for more interactive experience
        stream = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent parsing
            max_tokens=1000,
            stream=True
        )
        
        # Collect the streamed response
        print("🧠 GPT-4 Reasoning: ", end="", flush=True)
        response_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                response_text += content
        
        print()  # New line after streaming
        
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
        
        # Convert string dates to datetime objects with robust error handling
        try:
            start_time_str = plan_data['start_time']
            end_time_str = plan_data['end_time']
            
            # Handle N/A or invalid dates by falling back to reasonable defaults
            if start_time_str in ['N/A', 'null', None] or end_time_str in ['N/A', 'null', None]:
                logger.warning("LLM returned invalid dates, using default time range")
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=24)
            else:
                start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00')).replace(tzinfo=None)
                end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00')).replace(tzinfo=None)
            
            plan_data['start_time'] = start_time
            plan_data['end_time'] = end_time
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse LLM dates: {e}, using default time range")
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            plan_data['start_time'] = start_time
            plan_data['end_time'] = end_time
        
        # Create and validate the plan
        plan = AnalysisPlan(**plan_data)
        
        print("✅ Analysis plan created")
        return plan
        
    except Exception as e:
        logger.error(f"LLM query parsing failed: {e}")
        raise ValueError(f"Could not parse query with LLM: {e}")


def execute_analysis_plan(plan: AnalysisPlan, demo_mode: bool = False) -> Dict[str, Any]:
    """
    Execute the LLM-generated analysis plan by running tools in sequence.
    
    Builds comprehensive data context that will be fed to the LLM
    for expert-level insight generation.
    
    Args:
        plan: LLM-generated analysis plan
        demo_mode: If True, pause at key points for demo narration
        
    Returns:
        Dictionary with all analysis results and context
    """
    context = {
        'query_plan': plan,
        'primary_tag': plan.primary_tag,
        'time_range': (plan.start_time, plan.end_time),
        'analysis_results': {}
    }
    
    try:
        # Step 1: Always load data and get basic statistics
        print(f"Loading data for {plan.primary_tag}...")
        df = load_data(plan.primary_tag, plan.start_time, plan.end_time)
        
        if df.empty:
            context['error'] = f"No data found for {plan.primary_tag} in specified time range"
            return context
        
        print(f"✅ Loaded {len(df):,} data points")
        context['data_points'] = len(df)
        context['analysis_results']['basic_statistics'] = summarize_metric(df)
        
        # Step 2: Execute planned analysis steps
        for i, step in enumerate(plan.analysis_steps, 1):
            if demo_mode and i > 1:  # Don't pause before the first step
                demo_pause(f"Moving to analysis step {i}: {step.replace('_', ' ').title()}")
            
            print(f"Step {i}/{len(plan.analysis_steps)}: {step.replace('_', ' ').title()}...")
            
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
                print(f"   Found {len(anomalies)} anomalies (threshold: {threshold}σ)")
                
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
                print(f"   Found {len(correlations)} significant correlations")
                
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
                print(f"   Generated chart: {os.path.basename(chart_path)}")
        
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
        print("Generating manufacturing insights...")
        
        # Use streaming for more interactive experience
        stream = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_prompt}
            ],
            temperature=0.3,  # Some creativity for insights, but stay factual
            max_tokens=2000,
            stream=True
        )
        
        # Collect the streamed response
        print("🧠 Expert Analysis: ", end="", flush=True)
        expert_insights = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                expert_insights += content
        
        print()  # New line after streaming
        
        # Add chart reference if available
        if 'visualization' in results:
            chart_info = f"\n\n📊 **Visualization**: Chart saved as `{results['visualization']['chart_filename']}`"
            if results['visualization']['anomaly_highlights'] > 0:
                chart_info += f" with {results['visualization']['anomaly_highlights']} anomaly periods highlighted"
            expert_insights += chart_info
            print(chart_info)  # Also print the chart info immediately
        
        return expert_insights
        
    except Exception as e:
        logger.error(f"Error generating expert insights: {e}")
        return f"❌ **Error generating insights**: {e}"


def validate_analysis_plan(plan: AnalysisPlan) -> Tuple[bool, str]:
    """
    Validate the LLM-generated analysis plan for correctness.
    
    Ensures the plan references valid tags and has a sensible time range
    before proceeding with expensive data loading and analysis operations.
    
    Args:
        plan: LLM-generated analysis plan to validate
        
    Returns:
        Tuple of (can_continue, error_reason)
        - can_continue: True if plan is valid, False if invalid
        - error_reason: Empty string if valid, descriptive error if invalid
    """
    # Check if the primary tag exists in available tags
    if plan.primary_tag == "NO_RELEVANT_TAG":
        try:
            available_tags = get_available_tags()
            return False, f"No relevant tag found for your query. Available tags are: {', '.join(available_tags)}. Please try asking about one of these manufacturing metrics."
        except Exception as e:
            logger.error(f"Cannot get available tags for validation: {e}")
            return False, f"No relevant tag found for your query and cannot access tag list: {e}. Please check data connections."
    
    try:
        available_tags = get_available_tags()
        if plan.primary_tag not in available_tags:
            return False, f"Unknown tag '{plan.primary_tag}'. Available tags: {', '.join(available_tags)}"
    except Exception as e:
        logger.error(f"Cannot validate tag against available tags: {e}")
        return False, f"Cannot validate tag '{plan.primary_tag}': {e}. Please check data connections."
    
    # Check if time range is valid
    if plan.start_time >= plan.end_time:
        return False, f"Invalid time range: start time ({plan.start_time}) must be before end time ({plan.end_time})"
    
    # Check if time range is reasonable (not too far in the past or future)
    now = datetime.now()
    max_past = now - timedelta(days=365)  # 1 year ago
    max_future = now + timedelta(days=1)  # 1 day in future
    
    if plan.end_time < max_past:
        return False, f"Time range too far in the past: {plan.end_time} (data may not be available)"
    
    if plan.start_time > max_future:
        return False, f"Time range in the future: {plan.start_time} (no data available yet)"
    
    # Check if analysis steps are valid
    valid_steps = ["basic_statistics", "detect_anomalies", "correlate_tags", "generate_chart"]
    invalid_steps = [step for step in plan.analysis_steps if step not in valid_steps]
    if invalid_steps:
        return False, f"Invalid analysis steps: {', '.join(invalid_steps)}. Valid steps: {', '.join(valid_steps)}"
    
    return True, ""


def demo_pause(message: str) -> None:
    """
    Pause execution in demo mode for video narration.
    
    Args:
        message: Message to display before pausing
    """
    print(f"\n🎬 {message}")
    print("Press Enter to continue...")
    input()


def llm_interpret_query(query: str, demo_mode: bool = False) -> str:
    """
    Main LLM-powered query interpretation function.
    
    This is the complete pipeline that replaces the deterministic interpreter:
    1. Parse query with LLM intelligence
    2. Execute multi-step analysis plan
    3. Generate expert-level insights
    
    Args:
        query: Natural language query about manufacturing operations
        demo_mode: If True, pause at key points for demo narration
        
    Returns:
        Expert-level analysis and recommendations
    """
    try:
        print("\n🔍 QUERY PARSING")
        print("=" * 60)
        print(f"Processing: '{query}'")
        
        if demo_mode:
            demo_pause("About to analyze your query with GPT-4 manufacturing expert...")
        
        # Step 1: LLM parses query and creates analysis plan
        plan = parse_query_with_llm(query)
        
        print("\n📋 ANALYSIS PLAN")
        print("=" * 60)
        print(f"Primary Tag: {plan.primary_tag}")
        print(f"Time Range: {plan.start_time.strftime('%Y-%m-%d %H:%M')} → {plan.end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Analysis Steps: {', '.join(plan.analysis_steps)}")
        print(f"Reasoning: {plan.reasoning}")
        
        # Step 2: Validate the analysis plan
        can_continue, error_reason = validate_analysis_plan(plan)
        if not can_continue:
            print(f"\n❌ VALIDATION FAILED: {error_reason}")
            return f"❌ **Analysis Plan Validation Failed**: {error_reason}"
        
        print("\n✅ Plan validated successfully")
        
        if demo_mode:
            demo_pause("GPT-4 has created an intelligent analysis plan. Now executing the data analysis steps...")
        
        # Step 3: Execute the analysis plan
        print("\n📈 EXECUTION")
        print("=" * 60)
        context = execute_analysis_plan(plan, demo_mode=demo_mode)
        
        if demo_mode:
            demo_pause("Data analysis complete! Now generating expert manufacturing insights with GPT-4...")
        
        # Step 4: Generate expert insights with streaming
        print("\n🤖 GENERATING INSIGHTS")
        print("=" * 60)
        insights = generate_expert_insights(query, context)
        
        print("\n✅ ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Return the insights without duplicating the streamed content
        return "Analysis completed successfully. See insights above."
        
    except Exception as e:
        logger.error(f"❌ LLM Analysis Failed: {e}")
        return f"❌ **LLM Analysis Failed**: {e}\n\nPlease check your OpenAI API key and try again."


def _get_tag_descriptions() -> Dict[str, str]:
    """
    Get tag descriptions for LLM context.
    
    Raises:
        RuntimeError: If cannot access tag glossary
    """
    try:
        glossary = get_glossary()
        tags = glossary.list_all_tags()
        return {tag['tag']: tag['description'] for tag in tags}
    except Exception as e:
        logger.error(f"Failed to get tag descriptions: {e}")
        raise RuntimeError(f"Cannot access tag glossary: {e}. Please check glossary data source.")


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