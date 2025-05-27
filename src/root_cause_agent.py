#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Root Cause Agent

LLM-powered agent that investigates manufacturing anomalies through iterative 
analysis of time series data. Uses a structured reasoning chain and atomic tools
to build causal explanations with quantifiable confidence.

This demonstrates how LLMs can:
- Choose appropriate tools based on metadata
- Build evidence through sequential analysis
- Form and test causal hypotheses
- Quantify confidence deterministically
- Explain findings in business terms
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple

import openai

# Import the glossary for semantic tag search
from src.glossary import TagGlossary, search_tags
from src.tools.tag_intel import get_tag_intelligence, get_tag_metadata, get_related_tags
from src.tools.atomic_tools import (
    detect_numeric_anomalies,
    detect_binary_flips,
    detect_change_points,
    test_causality,
    calculate_impact,
    create_event_sequence,
    find_interesting_window
)
from src.confidence.scorer import get_confidence_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tool registration for LLM
TOOL_REGISTRY = {
    "detect_numeric_anomalies": detect_numeric_anomalies,
    "detect_binary_flips": detect_binary_flips,
    "detect_change_points": detect_change_points,
    "test_causality": test_causality,
    "calculate_impact": calculate_impact,
    "create_event_sequence": create_event_sequence,
    "find_interesting_window": find_interesting_window
}

class RootCauseAgent:
    """
    LLM-driven agent that investigates anomalies in manufacturing data.
    
    Uses sequential reasoning, tool selection, and evidence-based confidence
    scoring to build comprehensive causal explanations.
    """
    
    def __init__(self, max_steps: int = 6, confidence_threshold: float = 0.9):
        """
        Initialize the root cause agent.
        
        Args:
            max_steps: Maximum number of investigation steps
            confidence_threshold: Confidence threshold for early stopping
        """
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.evidence = []
        self.current_step = 0
        self.current_confidence = 0.0
        self.glossary = TagGlossary()  # Initialize the glossary for tag search
        
        # Store the optimal time window for investigation
        self.window_start = None
        self.window_end = None
        self.window_identified = False
        
        # Ensure OpenAI API key is set
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize client
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def investigate(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Investigate a query about manufacturing data.
        
        Args:
            query: Natural language query about manufacturing anomalies
            verbose: Whether to print detailed logs
            
        Returns:
            Investigation result with confidence, root cause, and timeline
        """
        # Reset state for new investigation
        self.evidence = []
        self.current_step = 0
        self.current_confidence = 0.0
        
        # Begin investigation
        if verbose:
            print("🕵️ Root Cause Agent - Starting Investigation")
            print("=" * 60)
            print(f"Query: '{query}'")
            print("=" * 60)
        
        # Find relevant tags through semantic search
        relevant_tags = self._find_relevant_tags(query)
        if verbose and relevant_tags:
            print("\n🔍 RELEVANT TAGS IDENTIFIED")
            for i, tag_info in enumerate(relevant_tags, 1):
                print(f"  {i}. {tag_info['tag']} - {tag_info['description']}")
                print(f"     Similarity: {tag_info['similarity_score']:.2f}")
        
        # Continue investigation until confidence threshold or max steps reached
        while self.current_step < self.max_steps:
            self.current_step += 1
            
            if verbose:
                print(f"\n🔍 STEP {self.current_step}/{self.max_steps}")
                print(f"Current confidence: {self.current_confidence:.1%}")
            
            # For the first step, explicitly use find_interesting_window
            if self.current_step == 1:
                # Find the most relevant numeric tag
                primary_tag = None
                for tag_info in relevant_tags:
                    tag_name = tag_info['tag']
                    tag_type = self._get_tag_type(tag_name)
                    if tag_type == 'numeric':
                        primary_tag = tag_name
                        break
                
                # If no numeric tag is found, use the first available tag
                if primary_tag is None and relevant_tags:
                    primary_tag = relevant_tags[0]['tag']
                # Fallback to a known temperature tag if nothing is found
                if primary_tag is None:
                    primary_tag = "FREEZER01.TEMP.INTERNAL_C"
                
                next_step = {
                    "tool": "find_interesting_window",
                    "params": {"primary_tag": primary_tag, "window_hours": 2},
                    "reasoning": f"First identifying the most interesting time window for {primary_tag} to focus our investigation and avoid time range issues."
                }
            else:
                # For subsequent steps, use LLM to plan
                next_step = self._plan_step(query, relevant_tags)
            
            # Execute the step
            if verbose:
                print(f"Tool selected: {next_step['tool']}")
                print(f"Reasoning: {next_step['reasoning']}")
                print(f"Parameters: {next_step['params']}")
            
            # Execute the tool
            result = self._execute_tool(next_step['tool'], next_step['params'])
            
            # Add to evidence
            self.evidence.append({
                'step': self.current_step,
                'tool': next_step['tool'],
                'params': next_step['params'],
                'reasoning': next_step['reasoning'],
                'result': result
            })
            
            # Calculate confidence score
            confidence_report = get_confidence_report(self.evidence)
            self.current_confidence = confidence_report['confidence']
            
            if verbose:
                print(f"Step completed - New confidence: {self.current_confidence:.1%}")
                print(f"Explanation: {confidence_report['explanation']}")
            
            # Check if we've reached the confidence threshold
            if self.current_confidence >= self.confidence_threshold:
                if verbose:
                    print(f"\n✅ Confidence threshold reached! ({self.current_confidence:.1%})")
                break
        
        # Generate final report
        report = self._generate_final_report(query)
        
        if verbose:
            print("\n📋 INVESTIGATION COMPLETE")
            print("=" * 60)
            print(f"Final confidence: {self.current_confidence:.1%}")
            print(f"Root cause: {report['root_cause']}")
            
            # Safely access business impact data
            if 'business_impact' in report and isinstance(report['business_impact'], dict):
                impact = report['business_impact']
                if 'total_cost' in impact:
                    print(f"Business impact: ${impact['total_cost']:.2f}")
            
            if 'timeline' in report:
                print("\nTimeline:")
                for event in report['timeline']:
                    if 'time' in event and 'description' in event:
                        print(f"  • {event['time']}: {event['description']}")
        
        return report
    
    def _find_relevant_tags(self, query: str) -> List[Dict[str, Any]]:
        """
        Find relevant tags using semantic search.
        
        Args:
            query: Natural language query
            
        Returns:
            List of relevant tags with similarity scores
        """
        # Use glossary semantic search to find relevant tags
        try:
            # First, search for tags semantically related to the query
            relevant_tags = self.glossary.search_tags(query, top_k=5)
            
            # If we have a primary tag, find related tags as well
            if relevant_tags:
                primary_tag = relevant_tags[0]['tag']
                related_tags = get_related_tags(primary_tag, n=3)
                
                # Add related tags to results if they're not already included
                existing_tags = {tag['tag'] for tag in relevant_tags}
                
                for related_tag in related_tags:
                    if related_tag not in existing_tags:
                        tag_info = self.glossary.get_tag_info(related_tag)
                        if tag_info:
                            relevant_tags.append({
                                'tag': tag_info.tag,
                                'description': tag_info.description,
                                'unit': tag_info.unit,
                                'similarity_score': 0.6  # Default similarity for related tags
                            })
            
            return relevant_tags
            
        except Exception as e:
            logger.error(f"Error finding relevant tags: {e}")
            return []
    
    def _plan_step(self, query: str, relevant_tags: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to plan the next investigation step.
        
        Args:
            query: The original investigation query
            relevant_tags: List of relevant tags with similarity scores
            
        Returns:
            Dictionary with tool selection and parameters
        """
        # Build context from previous evidence
        evidence_context = self._build_evidence_context()
        
        # Format relevant tags for LLM
        tags_context = self._format_relevant_tags(relevant_tags)
        
        # Format tool descriptions for LLM
        tools_description = self._format_tool_descriptions()
        
        # Build the prompt
        system_prompt = f"""You are an AI manufacturing detective that investigates anomalies in time series data.
You analyze manufacturing systems step-by-step, using a variety of tools to find causal relationships.

Your goal is to build evidence for the root cause of the issue, focusing on finding causal relationships
between different manufacturing tags (sensors).

{tags_context}

{tools_description}

{f"IMPORTANT: The optimal time window for investigation has been identified as: {self.window_start} to {self.window_end}. Focus on this time period." if self.window_identified else ""}

When choosing tools, remember:
1. Start by detecting anomalies in the most relevant tag for the query
2. For binary tags (doors, switches), use detect_binary_flips
3. For numeric tags (temperature, power), use detect_numeric_anomalies
4. For correlations, use test_causality between potentially related tags
5. Choose related tags based on what would physically make sense in a freezer system
6. Focus on building sequential evidence: cause → effect → impact
7. Test causal relationships AFTER you've found anomalies in both candidate tags
8. Always use the identified time window in your parameters

Only choose one tool to call. You'll get another chance to choose a different tool in the next step.
"""
        
        user_prompt = f"""
INVESTIGATION QUERY: "{query}"

CURRENT STEP: {self.current_step}
CURRENT CONFIDENCE: {self.current_confidence:.1%}

{evidence_context}

{f"IMPORTANT: The optimal time window for investigation has been identified as: {self.window_start} to {self.window_end}. Focus on this time period." if self.window_identified else ""}

Based on the current evidence, determine the next tool to call and its parameters.
Respond with a valid JSON object containing:
1. "tool": The name of the tool to call
2. "params": Parameters for the tool as a JSON object
3. "reasoning": Brief explanation of why you're choosing this tool

IMPORTANT: Choose the most logical next step in the investigation, focusing on building 
causal relationships between different sensors.
"""
        
        # Call the LLM to plan the next step
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            # Extract JSON response
            plan_text = response.choices[0].message.content
            
            # Parse JSON
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', plan_text)
            if json_match:
                plan_json = json.loads(json_match.group(1))
            else:
                # Try to find JSON without markdown formatting
                json_start = plan_text.find('{')
                json_end = plan_text.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    plan_json = json.loads(plan_text[json_start:json_end])
                else:
                    raise ValueError("Failed to extract JSON from LLM response")
            
            return plan_json
            
        except Exception as e:
            logger.error(f"Error planning next step: {e}")
            # Fallback to a simple plan using the most relevant tag
            if relevant_tags:
                tag = relevant_tags[0]['tag']
                tag_type = self._get_tag_type(tag)
                
                if tag_type == 'numeric':
                    return {
                        "tool": "detect_numeric_anomalies",
                        "params": {"tag": tag},
                        "reasoning": f"Falling back to checking {tag} anomalies due to LLM error."
                    }
                else:
                    return {
                        "tool": "detect_binary_flips",
                        "params": {"tag": tag},
                        "reasoning": f"Falling back to checking {tag} state changes due to LLM error."
                    }
            else:
                # Ultimate fallback
                return {
                    "tool": "detect_numeric_anomalies",
                    "params": {"tag": "FREEZER01.TEMP.INTERNAL_C"},
                    "reasoning": "Falling back to checking temperature anomalies due to LLM error."
                }
    
    def _format_relevant_tags(self, relevant_tags: List[Dict[str, Any]]) -> str:
        """
        Format relevant tags for LLM context.
        
        Args:
            relevant_tags: List of relevant tags with similarity scores
            
        Returns:
            Formatted string with tag information
        """
        if not relevant_tags:
            return "No relevant tags identified. Use appropriate tags from the following list of all available tags:\n" + \
                   "\n".join([f"- {tag['tag']}: {tag['description']}" for tag in self.glossary.list_all_tags()])
        
        formatted_text = "RELEVANT TAGS FOR THIS INVESTIGATION:\n"
        
        for i, tag in enumerate(relevant_tags, 1):
            tag_type = self._get_tag_type(tag['tag'])
            formatted_text += f"{i}. {tag['tag']} ({tag_type}, {tag.get('unit', 'no unit')})\n"
            formatted_text += f"   Description: {tag.get('description', 'No description')}\n"
        
        return formatted_text
    
    def _get_tag_type(self, tag_name: str) -> str:
        """
        Determine if a tag is numeric or binary.
        
        Args:
            tag_name: Name of the tag
            
        Returns:
            'numeric' or 'binary'
        """
        # Check tag metadata
        tag_meta = get_tag_metadata(tag_name)
        if tag_meta and 'value_type' in tag_meta:
            return tag_meta['value_type']
        
        # Fallback based on tag name
        if 'STATUS' in tag_name:
            return 'binary'
        else:
            return 'numeric'
    
    def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            
        Returns:
            Tool execution result
        """
        try:
            # Check if tool exists
            if tool_name not in TOOL_REGISTRY:
                return {"error": f"Unknown tool: {tool_name}"}
            
            # Get the tool function
            tool_fn = TOOL_REGISTRY[tool_name]
            
            # If this is the find_interesting_window tool, store the window after execution
            if tool_name == "find_interesting_window":
                # Execute the tool
                result = tool_fn(**params)
                
                # Store the window
                if 'window' in result and isinstance(result['window'], dict):
                    window = result['window']
                    if 'start_time' in window and 'end_time' in window:
                        self.window_start = window['start_time']
                        self.window_end = window['end_time']
                        self.window_identified = True
                        logging.info(f"Identified optimal window: {self.window_start} to {self.window_end}")
                
                return result
            
            # For subsequent tools, use the identified window if available
            if self.window_identified and (tool_name != "calculate_impact" and tool_name != "create_event_sequence"):
                # Only add the window params if they're not already specified
                if 'start_time' not in params or params['start_time'] is None:
                    params['start_time'] = self.window_start
                if 'end_time' not in params or params['end_time'] is None:
                    params['end_time'] = self.window_end
                
                logging.info(f"Using identified window for {tool_name}: {self.window_start} to {self.window_end}")
            
            # Execute the tool with potentially modified parameters
            result = tool_fn(**params)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": f"Tool execution failed: {e}"}
    
    def _build_evidence_context(self) -> str:
        """
        Build a context string from accumulated evidence.
        
        Returns:
            Formatted evidence context string
        """
        if not self.evidence:
            return "No evidence gathered yet."
        
        context = "EVIDENCE GATHERED SO FAR:\n"
        
        for idx, evidence in enumerate(self.evidence, 1):
            context += f"\nSTEP {idx}:\n"
            context += f"Tool: {evidence['tool']}\n"
            context += f"Parameters: {json.dumps(evidence['params'])}\n"
            context += f"Reasoning: {evidence['reasoning']}\n"
            
            # Format results based on tool type
            result = evidence['result']
            tool = evidence['tool']
            
            if tool == "detect_numeric_anomalies":
                anomalies = result.get('anomalies', [])
                context += f"Results: Found {len(anomalies)} anomalies in {result.get('tag', '')}\n"
                
                for anomaly in anomalies[:3]:  # Show top 3
                    context += f"  • {anomaly.get('timestamp', '')}: {anomaly.get('value', '')}, z-score={anomaly.get('z_score', ''):.2f}\n"
                    
            elif tool == "detect_binary_flips":
                changes = result.get('changes', [])
                context += f"Results: Found {len(changes)} state changes in {result.get('tag', '')}\n"
                
                for change in changes[:3]:  # Show top 3
                    context += f"  • {change.get('timestamp', '')}: {change.get('description', '')}\n"
                    
            elif tool == "test_causality":
                context += f"Results: Causality test between {result.get('cause_tag', '')} and {result.get('effect_tag', '')}\n"
                context += f"  • Correlation: {result.get('best_correlation', 0.0):.3f}\n"
                context += f"  • Lag: {result.get('best_lag_minutes', 0.0):.1f} minutes\n"
                context += f"  • Causal confidence: {result.get('causal_confidence', 0.0):.2f}\n"
                
            else:
                # Generic result formatting
                context += f"Results: {json.dumps(result, indent=2)[:500]}...\n"
        
        return context
    
    def _format_tool_descriptions(self) -> str:
        """
        Format tool descriptions for LLM prompt.
        
        Returns:
            Formatted tool descriptions string
        """
        descriptions = "AVAILABLE TOOLS:\n"
        
        descriptions += """
1. detect_numeric_anomalies(tag, start_time=None, end_time=None, threshold=None)
   Detects anomalies in numeric tags like temperature or power.

2. detect_binary_flips(tag, start_time=None, end_time=None)
   Detects state changes in binary tags like door status or compressor status.

3. detect_change_points(tag, start_time=None, end_time=None, sensitivity=2.0)
   Detects significant change points in numeric time series data.

4. test_causality(cause_tag, effect_tag, window_start=None, window_end=None, max_lag_minutes=10)
   Tests causal relationship between two tags, looking for correlations with time lag.

5. calculate_impact(event_type, duration_minutes, severity=1.0, price_per_kwh=0.12)
   Calculates business impact of an operational event in dollars.

6. create_event_sequence(primary_tag, related_tags, start_time=None, end_time=None)
   Creates a chronological sequence of events across multiple tags.

7. find_interesting_window(primary_tag, start_time=None, end_time=None, window_hours=2)
   Finds the most interesting time window for focused investigation.
"""
        return descriptions
    
    def _generate_final_report(self, query: str) -> Dict[str, Any]:
        """
        Generate a final investigation report.
        
        Args:
            query: The original investigation query
            
        Returns:
            Final report with root cause, timeline, and business impact
        """
        # Build context from all evidence
        evidence_context = self._build_evidence_context()
        
        system_prompt = """You are an AI manufacturing detective presenting your final investigation report.
Summarize the root cause, timeline, and business impact based on the evidence gathered.

Focus on creating a clear narrative that explains:
1. What happened (the sequence of events)
2. Why it happened (the root cause)
3. The business impact in dollars
4. Recommendations for preventing similar issues

Write in clear business language that a plant manager would understand.
"""
        
        user_prompt = f"""
INVESTIGATION QUERY: "{query}"

INVESTIGATION STEPS: {self.current_step}
FINAL CONFIDENCE: {self.current_confidence:.1%}

{evidence_context}

Based on all the evidence, generate a final report with:
1. A clear root cause statement (1-2 sentences)
2. A chronological timeline of events
3. Business impact calculation
4. 2-3 recommendations for preventing this issue

Your response must be a valid JSON object with these fields:
- root_cause: String explaining the primary cause
- timeline: Array of {{"time": "timestamp", "description": "event description"}}
- business_impact: Object with cost analysis including "total_cost", "energy_cost", "product_risk" all as numbers, and "severity" as string
- recommendations: Array of recommendation strings
"""
        
        try:
            # Call the LLM to generate the final report
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            
            # Extract JSON response
            report_text = response.choices[0].message.content
            
            # Parse JSON
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', report_text)
            if json_match:
                report = json.loads(json_match.group(1))
            else:
                # Try to find JSON without markdown formatting
                json_start = report_text.find('{')
                json_end = report_text.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    report = json.loads(report_text[json_start:json_end])
                else:
                    raise ValueError("Failed to extract JSON from LLM response")
            
            # Add confidence score to report
            report['confidence'] = self.current_confidence
            report['steps'] = self.current_step
            
            # Ensure business_impact has required fields
            if 'business_impact' not in report or not isinstance(report['business_impact'], dict):
                report['business_impact'] = {
                    'total_cost': 0.0,
                    'energy_cost': 0.0,
                    'product_risk': 0.0,
                    'severity': 'low',
                    'description': 'Unable to calculate impact'
                }
            else:
                # Ensure all required fields exist
                impact = report['business_impact']
                if 'total_cost' not in impact:
                    impact['total_cost'] = 0.0
                if 'energy_cost' not in impact:
                    impact['energy_cost'] = 0.0
                if 'product_risk' not in impact:
                    impact['product_risk'] = 0.0
                if 'severity' not in impact:
                    impact['severity'] = 'low'
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
            # Return a fallback report
            return {
                "root_cause": "Investigation inconclusive due to reporting error.",
                "timeline": [{"time": "N/A", "description": "Error generating timeline"}],
                "business_impact": {
                    "total_cost": 0.0, 
                    "energy_cost": 0.0,
                    "product_risk": 0.0,
                    "severity": "low",
                    "description": "Unable to calculate impact"
                },
                "recommendations": ["Review system logs for more information"],
                "confidence": self.current_confidence,
                "steps": self.current_step,
                "error": str(e)
            }
    
    def save_case(self, filename: Optional[str] = None) -> str:
        """
        Save the investigation case to a JSON file.
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            Path to the saved file
        """
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Generate default filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/case_{timestamp}.json"
        
        # Create case data
        case_data = {
            "evidence": self.evidence,
            "steps": self.current_step,
            "confidence": self.current_confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(case_data, f, indent=2)
        
        logger.info(f"Case saved to {filename}")
        return filename


def main():
    """
    Demo function to test the root cause agent.
    """
    print("🕵️ Manufacturing Copilot - Root Cause Agent Demo")
    print("=" * 60)
    
    # Sample queries
    demo_queries = [
        "Why did the freezer temperature spike yesterday afternoon?",
        "What caused the compressor to run more than usual yesterday?",
        "Was the increase in power consumption related to door activity?"
    ]
    
    # Create agent
    agent = RootCauseAgent(max_steps=4, confidence_threshold=0.85)
    
    # Run demo with first query
    query = demo_queries[0]
    print(f"Investigating: '{query}'")
    print()
    
    report = agent.investigate(query, verbose=True)
    
    # Save case
    case_file = agent.save_case()
    print(f"\nCase saved to {case_file}")
    

if __name__ == "__main__":
    main() 