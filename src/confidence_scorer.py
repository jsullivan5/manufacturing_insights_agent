import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def score_evidence(
    evidence: List[Dict[str, Any]],
    current_step_count: int,
    decay_base: float,
    penalize_duplicate_attempt: bool = False
) -> float:
    """
    Calculates a deterministic confidence score based on the accumulated evidence.
    Refined for better progression.

    Args:
        evidence: List of evidence dictionaries from tool calls.
        current_step_count: The current step number in the investigation.
        decay_base: The base for exponential decay of confidence over steps (e.g., 0.98).
        penalize_duplicate_attempt: Whether to apply a penalty for a duplicate tool call attempt.

    Returns:
        A confidence score between 0.0 and 1.0.
    """
    if not evidence:
        return 0.0

    base_score = 0.0
    found_window = False
    found_anomaly_in_window = False
    found_binary_flip_in_window = False
    found_causality = False
    anomaly_detected_this_run = False
    causality_tested_this_run = False

    # --- Basic Bonuses: Increased consistency bonus --- 
    base_score += min(0.20, len(evidence) * 0.04) # Increased from 0.15 and 0.03

    for i, item in enumerate(evidence):
        tool_name = None
        tool_result = {}

        if item.get("role") == "assistant" and item.get("function_call"):
            fc = item["function_call"]
            tool_name = fc.name if hasattr(fc, 'name') else fc.get("name")
            # Result for this call will be in the *next* evidence item if role is function
            if (i + 1) < len(evidence) and evidence[i+1].get("role") == "function" and evidence[i+1].get("name") == tool_name:
                content_str = evidence[i+1].get("content")
                if content_str and isinstance(content_str, str):
                    try: tool_result = json.loads(content_str)
                    except json.JSONDecodeError: logger.warning(f"Non-JSON content for func result: {tool_name}")
        elif item.get("role") == "function": # Direct function result (less common with new evidence structure)
            tool_name = item.get("name")
            content_str = item.get("content")
            if content_str and isinstance(content_str, str):
                try: tool_result = json.loads(content_str)
                except json.JSONDecodeError: logger.warning(f"Non-JSON content for func result: {tool_name}")

        if not tool_name:
            logger.debug(f"Could not determine tool name for evidence item index {i}")
            continue

        if tool_name == "find_interesting_window" and isinstance(tool_result, dict) and tool_result.get("data", {}).get("window") and tool_result.get("data", {}).get("window", {}).get("start_time"):
            found_window = True
            base_score += 0.10 # Keep window bonus modest, real value is enabling other tools
        
        if tool_name == "detect_numeric_anomalies" and isinstance(tool_result, dict) and tool_result.get("data", {}).get("anomalies"):
            anomaly_detected_this_run = True
            # P0.1: Raised anomaly bonus
            bonus = 0.30 
            if found_window: 
                bonus += 0.05 # Extra if within a focused window
            base_score += bonus
            if found_window: found_anomaly_in_window = True # Assume if window is found, this anomaly is in it for now
        
        # Consolidated check for detect_binary_flips tool results
        # if tool_name == "detect_binary_flips" and isinstance(tool_result, dict):
        #     actual_tool_data = tool_result.get("data", {})
        #     if isinstance(actual_tool_data, dict): # Ensure actual_tool_data is a dict before using .get()
        #         flips_found = actual_tool_data.get("changes") and len(actual_tool_data.get("changes", [])) > 0
        #         continuous_event_found = bool(actual_tool_data.get("continuous_high_event"))

        #         if flips_found or continuous_event_found:
        #             base_score += 0.10 # Base bonus for any binary event
        #             if found_window: # Additional bonus if within a focused window
        #                 base_score += 0.05
        #                 found_binary_flip_in_window = True
                    
        #             # Logging for clarity
        #             if flips_found and continuous_event_found:
        #                 logger.debug(f"Binary event for {actual_tool_data.get('tag')}: Found both flips and continuous high event. Setting found_binary_flip_in_window = True.")
        #             elif flips_found:
        #                 logger.debug(f"Binary event for {actual_tool_data.get('tag')}: Found flips. Setting found_binary_flip_in_window = True.")
        #             elif continuous_event_found:
        #                 logger.debug(f"Binary event for {actual_tool_data.get('tag')}: Found continuous high event. Setting found_binary_flip_in_window = True.")
        #     else:
        #         logger.warning(f"For tool {tool_name}, expected 'data' to be a dictionary, but it was not. Tool result: {tool_result}")

        elif tool_name == "detect_binary_flips" and isinstance(tool_result, dict):
            # Accept both storage conventions
            candidate = tool_result.get("data")
            if not isinstance(candidate, dict):
                candidate = tool_result      # fall back to top-level dict

            flips_found = bool(candidate.get("changes"))
            cont_high  = bool(candidate.get("continuous_high_event"))

            if flips_found or cont_high:
                base_score += 0.15
                if found_window:
                    base_score += 0.05
                    found_binary_flip_in_window = True

        elif tool_name == "test_causality" and isinstance(tool_result, dict):
            # accept both shapes
            candidate = tool_result.get("data")
            if not isinstance(candidate, dict):
                candidate = tool_result   # fall back to top-level

            causal_confidence = float(candidate.get("causal_confidence", 0.0))
            best_lag_minutes  = abs(float(candidate.get("best_lag_minutes", 1e6)))
            correlation_strength = abs(float(candidate.get("best_correlation", 0.0)))

            if causal_confidence > 0.6 and correlation_strength > 0.5 and best_lag_minutes < 20:
                base_score += (causal_confidence * 0.5) + 0.1
                found_causality = True
            elif causal_confidence > 0.4:
                base_score += causal_confidence * 0.25
            
    # P0: Updated Causal Chain Bonus
    if found_window and found_anomaly_in_window and found_binary_flip_in_window and found_causality:
        base_score += 0.25 # Increased from 0.20 and now includes binary_flip check
    elif found_window and (found_anomaly_in_window or found_binary_flip_in_window) and found_causality:
        base_score += 0.15 # Lesser bonus if one of the event types is missing but causality still found

    # New specific bonuses based on your request
    if found_anomaly_in_window and found_binary_flip_in_window: # Both occurred in the window
        base_score += 0.20
        logger.debug("Applied +0.15 bonus for co-occurring numeric anomaly and binary flip in window.")
    elif found_anomaly_in_window: # Only numeric anomaly in window (and not the combined case above)
        base_score += 0.05
        logger.debug("Applied +0.05 bonus for numeric anomaly in window.")

    # --- Penalties ---
    if penalize_duplicate_attempt:
        base_score -= 0.15 # Slightly increased penalty

    # --- Apply Decay --- 
    effective_score = base_score * (decay_base ** max(0, current_step_count -1))
    final_score = max(0.0, min(effective_score, 0.99))

    logger.debug(f"Confidence: {final_score:.3f} (Base: {base_score:.3f}, Step: {current_step_count}, Decay: {decay_base}, Penalty: {penalize_duplicate_attempt})")
    return final_score 