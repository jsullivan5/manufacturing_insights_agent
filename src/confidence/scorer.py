#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Confidence Scoring Module

Provides deterministic confidence scoring for investigation evidence.
Transforms tool results into measurable confidence levels based on 
statistical significance, temporal sequences, and causal strength.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import yaml

# Configure logging
logger = logging.getLogger(__name__)

# Default weights if config not found
DEFAULT_WEIGHTS = {
    'temporal_sequence': 0.35,  # Events in correct order
    'correlation_strength': 0.25,  # |r| with decay based on lag
    'anomaly_severity': 0.20,   # z-score & duration
    'evidence_consistency': 0.20  # Multiple evidence paths
}

# Load weights from config if available
CONFIG_PATH = "config/confidence_weights.yaml"
try:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            WEIGHTS = yaml.safe_load(f)
            logger.info(f"Loaded confidence weights from {CONFIG_PATH}")
    else:
        WEIGHTS = DEFAULT_WEIGHTS
        logger.warning(f"Config not found at {CONFIG_PATH}, using default weights")
except Exception as e:
    logger.error(f"Error loading weights config: {e}")
    WEIGHTS = DEFAULT_WEIGHTS


def score_temporal_sequence(evidence: List[Dict[str, Any]]) -> float:
    """
    Score evidence based on temporal sequence of events.
    
    Higher scores when:
    - Door events precede temperature events
    - Temperature events precede compressor events
    - Events occur in physically plausible sequence
    
    Args:
        evidence: List of evidence items from tools
        
    Returns:
        Score from 0.0 to 1.0
    """
    # Extract events with timestamps
    events = []
    
    for item in evidence:
        tool_name = item.get('tool')
        result = item.get('result', {})
        
        # Look for binary state changes (doors, compressors)
        if tool_name == 'detect_binary_flips' and 'changes' in result:
            tag = result.get('tag', '')
            for change in result.get('changes', []):
                if change.get('to_state') == 1:  # State changed to ON/OPEN
                    events.append({
                        'timestamp': change.get('timestamp'),
                        'tag': tag,
                        'type': 'state_change_on',
                        'value': 1.0
                    })
        
        # Look for anomalies (temperature spikes)
        elif tool_name == 'detect_numeric_anomalies' and 'anomalies' in result:
            tag = result.get('tag', '')
            for anomaly in result.get('anomalies', []):
                events.append({
                    'timestamp': anomaly.get('timestamp'),
                    'tag': tag,
                    'type': 'anomaly',
                    'value': anomaly.get('value', 0.0)
                })
    
    # Sort events by timestamp
    events.sort(key=lambda x: x.get('timestamp', ''))
    
    # Check if we have enough events for sequence analysis
    if len(events) < 2:
        return 0.0
    
    # Look for key sequences
    score = 0.0
    
    # Check for door → temperature sequence
    door_events = [e for e in events if 'DOOR' in e.get('tag', '') and e.get('type') == 'state_change_on']
    temp_events = [e for e in events if 'TEMP' in e.get('tag', '') and e.get('type') == 'anomaly']
    
    if door_events and temp_events:
        # Find the first door open event
        first_door = min(door_events, key=lambda x: x.get('timestamp', ''))
        # Find temperature events after door opened
        temp_after_door = [t for t in temp_events if t.get('timestamp', '') > first_door.get('timestamp', '')]
        
        if temp_after_door:
            # Door → Temperature sequence found!
            score += 0.5
            logger.info("Found door → temperature sequence")
    
    # Check for temperature → compressor sequence
    compressor_events = [e for e in events if 'COMPRESSOR' in e.get('tag', '')]
    
    if temp_events and compressor_events:
        # Find the first temperature anomaly
        first_temp = min(temp_events, key=lambda x: x.get('timestamp', ''))
        # Find compressor events after temperature anomaly
        comp_after_temp = [c for c in compressor_events if c.get('timestamp', '') > first_temp.get('timestamp', '')]
        
        if comp_after_temp:
            # Temperature → Compressor sequence found!
            score += 0.5
            logger.info("Found temperature → compressor sequence")
    
    return score


def score_correlations(evidence: List[Dict[str, Any]]) -> float:
    """
    Score evidence based on correlation strength and lag.
    
    Higher scores when:
    - Strong correlations (|r| > 0.7)
    - Short lags (< 5 minutes)
    - Statistically significant correlations
    
    Args:
        evidence: List of evidence items from tools
        
    Returns:
        Score from 0.0 to 1.0
    """
    best_score = 0.0
    
    for item in evidence:
        tool_name = item.get('tool')
        result = item.get('result', {})
        
        # Look for correlation results
        if tool_name == 'test_causality':
            correlation = abs(result.get('best_correlation', 0.0))
            lag_minutes = abs(result.get('best_lag_minutes', 100.0))
            
            # Calculate correlation component (0-0.6)
            corr_component = 0.0
            if correlation >= 0.8:
                corr_component = 0.6
            elif correlation >= 0.6:
                corr_component = 0.4
            elif correlation >= 0.4:
                corr_component = 0.2
            
            # Calculate lag component (0-0.4)
            lag_component = 0.0
            if lag_minutes <= 5:
                lag_component = 0.4
            elif lag_minutes <= 10:
                lag_component = 0.2
            
            # Combined score for this correlation
            score = corr_component + lag_component
            best_score = max(best_score, score)
            
        # Legacy correlation results
        elif tool_name == 'get_correlations':
            for corr in result.get('details', []):
                correlation = abs(corr.get('correlation', 0.0))
                
                # Simple scoring for legacy correlations
                if correlation >= 0.8:
                    score = 0.7
                elif correlation >= 0.6:
                    score = 0.5
                elif correlation >= 0.4:
                    score = 0.3
                else:
                    score = 0.1
                    
                best_score = max(best_score, score)
    
    return best_score


def score_anomalies(evidence: List[Dict[str, Any]]) -> float:
    """
    Score evidence based on anomaly severity and duration.
    
    Higher scores when:
    - High z-scores (> 5)
    - Sustained anomalies (> 10 minutes)
    - Multiple anomalies in sequence
    
    Args:
        evidence: List of evidence items from tools
        
    Returns:
        Score from 0.0 to 1.0
    """
    max_z_score = 0.0
    anomaly_count = 0
    max_duration = 0.0
    
    for item in evidence:
        tool_name = item.get('tool')
        result = item.get('result', {})
        
        # Modern anomaly detection
        if tool_name == 'detect_numeric_anomalies':
            anomalies = result.get('anomalies', [])
            anomaly_count += len(anomalies)
            
            for anomaly in anomalies:
                z_score = abs(anomaly.get('z_score', 0.0))
                max_z_score = max(max_z_score, z_score)
        
        # Legacy anomaly detection
        elif tool_name == 'detect_anomalies':
            anomalies = result.get('details', [])
            anomaly_count += len(anomalies)
            
            for anomaly in anomalies:
                z_score = abs(anomaly.get('z_score', 0.0))
                max_z_score = max(max_z_score, z_score)
        
        # Consecutive anomalies or change points
        elif tool_name in ['detect_consecutive_anomalies', 'detect_change_points']:
            for period in result.get('details', []):
                duration = period.get('duration_minutes', 0.0)
                max_duration = max(max_duration, duration)
    
    # Calculate score components
    z_score_component = 0.0
    if max_z_score >= 5.0:
        z_score_component = 0.5
    elif max_z_score >= 3.0:
        z_score_component = 0.3
    elif max_z_score >= 2.0:
        z_score_component = 0.1
    
    # Count component
    count_component = min(0.2, anomaly_count * 0.05)
    
    # Duration component
    duration_component = 0.0
    if max_duration >= 15:
        duration_component = 0.3
    elif max_duration >= 5:
        duration_component = 0.2
    elif max_duration >= 1:
        duration_component = 0.1
    
    return z_score_component + count_component + duration_component


def score_consistency(evidence: List[Dict[str, Any]]) -> float:
    """
    Score evidence based on consistency across multiple tools.
    
    Higher scores when:
    - Multiple tools confirm the same findings
    - Evidence from different tool types
    - Tags from different categories show related patterns
    
    Args:
        evidence: List of evidence items from tools
        
    Returns:
        Score from 0.0 to 1.0
    """
    # Count unique tool types used
    tool_types = set()
    tag_categories = set()
    tag_pairs = set()
    
    for item in evidence:
        tool_name = item.get('tool', '')
        tool_types.add(tool_name)
        
        # Extract tags from params
        params = item.get('params', {})
        result = item.get('result', {})
        
        for param_name, param_value in params.items():
            if param_name.endswith('_tag') and isinstance(param_value, str):
                tag_categories.add(param_value.split('.')[1] if '.' in param_value else '')
                
                # For causality tests, record the tag pair
                if tool_name == 'test_causality':
                    cause = params.get('cause_tag', '')
                    effect = params.get('effect_tag', '')
                    if cause and effect:
                        tag_pairs.add((cause, effect))
    
    # Calculate score components
    tool_diversity = min(1.0, len(tool_types) / 4.0)  # Max score at 4+ tools
    category_diversity = min(1.0, len(tag_categories) / 3.0)  # Max score at 3+ categories
    pair_diversity = min(1.0, len(tag_pairs) / 2.0)  # Max score at 2+ tag pairs
    
    # Weight the components
    score = 0.5 * tool_diversity + 0.3 * category_diversity + 0.2 * pair_diversity
    
    return score


def calculate_confidence(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate overall confidence score from accumulated evidence.
    
    Combines scores from multiple factors using configurable weights.
    
    Args:
        evidence: List of evidence items from tools
        
    Returns:
        Dictionary with confidence score and component scores
    """
    # Calculate component scores
    score_components = {
        'temporal_sequence': score_temporal_sequence(evidence),
        'correlation_strength': score_correlations(evidence),
        'anomaly_severity': score_anomalies(evidence),
        'evidence_consistency': score_consistency(evidence)
    }
    
    # Apply weights to calculate final score
    final_score = sum(
        score * WEIGHTS.get(component, DEFAULT_WEIGHTS.get(component, 0.0))
        for component, score in score_components.items()
    )
    
    # Cap at 1.0
    final_score = min(final_score, 1.0)
    
    logger.info(f"Calculated confidence: {final_score:.2f} from components: {score_components}")
    
    return {
        'confidence': final_score,
        'components': score_components
    }


def get_confidence_report(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a human-readable confidence report.
    
    Args:
        evidence: List of evidence items from tools
        
    Returns:
        Dictionary with confidence score and explanation
    """
    result = calculate_confidence(evidence)
    confidence = result['confidence']
    components = result['components']
    
    # Generate explanation
    explanations = []
    
    if components['temporal_sequence'] > 0.3:
        explanations.append("Events occurred in the expected causal sequence")
    
    if components['correlation_strength'] > 0.4:
        explanations.append("Strong correlations between related tags")
    
    if components['anomaly_severity'] > 0.3:
        explanations.append("Significant anomalies detected")
    
    if components['evidence_consistency'] > 0.5:
        explanations.append("Consistent evidence from multiple sources")
    
    if not explanations:
        explanations.append("Insufficient evidence for high confidence")
    
    return {
        'confidence': confidence,
        'confidence_pct': int(confidence * 100),
        'explanation': ". ".join(explanations) + ".",
        'components': {k: round(v, 2) for k, v in components.items()}
    }


def main():
    """Demo function to test confidence scoring."""
    print("🧠 Manufacturing Copilot - Confidence Scoring Demo")
    print("=" * 60)
    
    # Create example evidence with realistic tool results
    evidence = [
        {
            'tool': 'detect_numeric_anomalies',
            'params': {'tag': 'FREEZER01.TEMP.INTERNAL_C'},
            'result': {
                'tag': 'FREEZER01.TEMP.INTERNAL_C',
                'anomalies': [
                    {'timestamp': '2025-05-25T14:35:00', 'value': -15.2, 'z_score': 3.5, 'description': 'High spike'}
                ]
            }
        },
        {
            'tool': 'detect_binary_flips',
            'params': {'tag': 'FREEZER01.DOOR.STATUS'},
            'result': {
                'tag': 'FREEZER01.DOOR.STATUS',
                'changes': [
                    {'timestamp': '2025-05-25T14:30:00', 'from_state': 0, 'to_state': 1, 'duration_minutes': 15.0},
                    {'timestamp': '2025-05-25T14:45:00', 'from_state': 1, 'to_state': 0, 'duration_minutes': None}
                ]
            }
        },
        {
            'tool': 'test_causality',
            'params': {'cause_tag': 'FREEZER01.DOOR.STATUS', 'effect_tag': 'FREEZER01.TEMP.INTERNAL_C'},
            'result': {
                'best_correlation': 0.75,
                'best_lag_minutes': 5.0,
                'direction': 'positive'
            }
        }
    ]
    
    print("Computing confidence score for example evidence...")
    result = get_confidence_report(evidence)
    
    print(f"\nConfidence Score: {result['confidence_pct']}%")
    print(f"Explanation: {result['explanation']}")
    print("\nComponent Scores:")
    for component, score in result['components'].items():
        print(f"• {component}: {score:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Confidence scoring demo complete!")


if __name__ == "__main__":
    main() 