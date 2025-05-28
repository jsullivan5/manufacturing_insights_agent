# src/llm_orchestrator.py
import os
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Callable, Tuple, Set, Literal
import re
import copy # For deep copying evidence items
from decimal import Decimal # Added as per feedback
from concurrent.futures import ThreadPoolExecutor # Added for background I/O
import argparse # Add argparse import

import openai
import pydantic
import orjson # For faster JSON operations, especially for logging
import tiktoken
import pandas as pd

import dotenv
dotenv.load_dotenv()

from src.config import Settings
from src.glossary import TagGlossary
from src.tools.data_loader import get_data_time_range
from src.tools.tag_intel import get_tag_metadata
from src.tools.atomic_tools import (
    detect_numeric_anomalies,
    detect_binary_flips,
    detect_change_points,
    test_causality,
    calculate_impact,
    create_event_sequence,
    find_interesting_window,
    parse_time_range
)
from src.tool_models import (
    FindInterestingWindowArgs,
    DetectNumericAnomaliesArgs,
    DetectBinaryFlipsArgs,
    TestCausalityArgs,
    CalculateImpactArgs,
    CreateEventSequenceArgs,
    FinishInvestigationArgs,
    ParseTimeRangeArgs,
    BaseToolResult
)
from src.confidence_scorer import score_evidence
from src.tools.build_report import build_event_timeline, build_business_impact
from src.logger import setup_logging

logger = logging.getLogger(__name__)

# --- Custom Exceptions (as per checklist) ---
class MissingWindowError(Exception):
    pass

class StaleWindowError(Exception):
    pass

# --- Module Level Utility --- (as per checklist)
def truncate_json_or_text(data: Any, max_length: int = 150, max_array_elements: int = 2, max_string_length: int = 75, max_dict_keys: int = 10) -> str:
    """Truncates complex data structures for concise display or logging."""
    if isinstance(data, dict):
        items = list(data.items())[:max_dict_keys]
        truncated_dict = {k: truncate_json_or_text(v, max_length, max_array_elements, max_string_length, max_dict_keys) for k, v in items}
        json_str = json.dumps(truncated_dict)
    elif isinstance(data, list):
        truncated_list = [truncate_json_or_text(item, max_length, max_array_elements, max_string_length, max_dict_keys) for item in data[:max_array_elements]]
        json_str = json.dumps(truncated_list) + ("..." if len(data) > max_array_elements else "")
    elif isinstance(data, str):
        json_str = data[:max_string_length] + ("..." if len(data) > max_string_length else "")
    else:
        try:
            json_str = json.dumps(data)
        except TypeError:
            json_str = str(data)

    if len(json_str) > max_length:
        return json_str[:max_length-3] + "..."
    return json_str

class LLMOrchestrator:
    """
    LLM-driven agent that investigates anomalies in manufacturing data using OpenAI function calling.
    """

    BASE_SYS_PROMPT = """You are RootCauseGPT, an industrial forensics LLM.

TOOLS: You can call any of the JSON-typed functions provided.
STATE: I will give you a JSON blob each turn containing:
  - confidence: current 0-1 score
  - last_result: the JSON result of the previous function
  - window: {start, end} if already discovered
GOAL: Raise confidence to >=0.9 by discovering a causal chain.

INITIAL STEP:
- If the query contains temporal language (e.g., "yesterday afternoon", "on 24 May at 14:30", "last weekend"), first call parse_time_range. Provide its output (start_time, end_time) as the initial investigation_window. Both start_time and end_time are optional in parse_time_range; if the LLM cannot determine a precise one, it can omit it or provide a sensible default. Output must be ISO-8601 UTC.

STRATEGY:
  1. If no window yet (either from parse_time_range or a previous find_interesting_window call) -> call find_interesting_window on the most relevant numeric tag. If a window was determined by parse_time_range, use its start_time and end_time as arguments for the *search range* within find_interesting_window, if the tool supports it, to narrow its focus.
  2. Detect anomalies in the primary tag within the established investigation_window. If no anomalies found, consider retrying detect_numeric_anomalies with a lower threshold (e.g., 2.0 or 2.5).
  3. Search related tags (door, compressor...) and detect anomalies or flips within the investigation_window.
  4. Call test_causality on promising cause/effect pairs within the investigation_window.
  5. Adjust thresholds or choose new tags if confidence is stagnant. If initial window is unhelpful and confidence is low after a few steps, consider calling find_interesting_window again, perhaps with a different primary_tag or window_hours (this may override a window from parse_time_range if necessary). When confidence >=0.9 or you have no more meaningful actions, call function "finish_investigation".
  6. CRITICAL for finish_investigation: You MUST populate the 'event_timeline_summary' with at least 3 chronological events and the 'business_impact_summary' with all its required keys (total_cost_usd, energy_cost_usd, product_risk_usd, severity_level). Failure to provide these complete fields will result in an error and a retry.

  EXAMPLE of the required JSON when you finally call `finish_investigation`:

  {
    "summary_version": "0.1",
    "root_cause_statement": "…one-sentence conclusion…",
    "event_timeline_summary": [
      {"time": "2025-05-27T18:00:00Z", "description": "Investigation started"},
      {"time": "2025-05-27T18:05:42Z", "description": "Detected freezer door left open"},
      {"time": "2025-05-27T18:10:30Z", "description": "Investigation concluded"}
    ],
    "business_impact_summary": {
      "total_cost_usd": 12.5,
      "energy_cost_usd": 7.0,
      "product_risk_usd": 5.5,
      "severity_level": "medium",
      "details": "Door open caused a temperature excursion"
    },
    "recommendations": ["Install door alarm", "Brief staff on procedure"],
    "final_confidence_score": 0.92
  }

REMINDER: One tool call per turn. Use start_time/end_time from the established investigation_window for tools that require it. Choose tools deterministically. Return ISO-8601 UTC timestamps only.
"""

    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        
        # Setup logging as early as possible
        # You can customize levels and formatters here if needed,
        # or rely on defaults in setup_logging / environment variables.
        setup_logging(default_level=self.settings.log_level.upper()) 

        logger.setLevel(self.settings.log_level.upper()) # Ensure orchestrator's own logger respects the level
        
        # Rely on Pydantic to load the prefixed env var into settings.openai_api_key
        if not self.settings.openai_api_key:
            # This means MCP_OPENAI_API_KEY was not found by Pydantic
            raise ValueError("OpenAI API key not found. Ensure MCP_OPENAI_API_KEY is set in your .env file or as an environment variable.")
        self.openai_client = openai.OpenAI(api_key=self.settings.openai_api_key)
        
        try: self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e: logger.warning(f"Tiktoken load failed: {e}"); self.tokenizer = None
        self.tag_glossary = TagGlossary(glossary_path=self.settings.tag_glossary_path)
        self.tools: Dict[str, Callable] = self._register_tools()
        self.tool_argument_models: Dict[str, type[pydantic.BaseModel]] = self._load_tool_argument_models()
        self.tool_schemas: List[Dict] = self._generate_and_cache_tool_schemas()
        self.evidence_writer_executor = ThreadPoolExecutor(max_workers=1)
        self._reset_run_state()

    def _reset_run_state(self):
        """Resets state variables for a new investigation run."""
        self.current_run_id: Optional[str] = None
        self.evidence: List[Dict] = []
        self.current_confidence: float = 0.0
        self.investigation_window: Optional[Dict[str, str]] = None
        self.current_step_count: int = 0
        self.estimated_cost_usd: float = 0.0
        self.executed_tool_signatures: Set[str] = set()
        self.stale_confidence_counter: int = 0
        self.previous_confidence_for_staleness_check: float = 0.0
        logger.info("Orchestrator run state has been reset.")

    def _register_tools(self) -> Dict[str, Callable]:
        """Maps tool names to their callable Python functions."""
        # finish_investigation is handled differently, not a direct callable tool in the same way
        # Its arguments are provided by the LLM when it decides to finish.
        return {
            "parse_time_range": parse_time_range,
            "find_interesting_window": find_interesting_window,
            "detect_numeric_anomalies": detect_numeric_anomalies,
            "detect_binary_flips": detect_binary_flips,
            "test_causality": test_causality,
            "calculate_impact": calculate_impact,
            "create_event_sequence": create_event_sequence,
        }

    def _load_tool_argument_models(self) -> Dict[str, type[pydantic.BaseModel]]:
        """Maps tool names to their Pydantic argument models."""
        return {
            "parse_time_range": ParseTimeRangeArgs,
            "find_interesting_window": FindInterestingWindowArgs,
            "detect_numeric_anomalies": DetectNumericAnomaliesArgs,
            "detect_binary_flips": DetectBinaryFlipsArgs,
            "test_causality": TestCausalityArgs,
            "calculate_impact": CalculateImpactArgs,
            "create_event_sequence": CreateEventSequenceArgs,
            "finish_investigation": FinishInvestigationArgs, # For schema generation and final parsing
        }

    def _generate_openai_schema(self, tool_name: str, pydantic_model: type[pydantic.BaseModel]) -> Dict[str, Any]:
        """Converts a Pydantic model to an OpenAI function schema dictionary."""
        if not hasattr(pydantic_model, 'model_json_schema'):
            raise ValueError(f"Model {pydantic_model.__name__} is not a Pydantic BaseModel.")
        
        schema = pydantic_model.model_json_schema()
        
        # Construct the description including purpose and version
        # Pydantic model docstring is the primary source for purpose/returns
        purpose_returns_doc = pydantic_model.__doc__ if pydantic_model.__doc__ else "No description provided."
        # Clean up potential multi-line docstring for better JSON representation
        purpose_returns_doc_cleaned = re.sub(r'\s+', ' ',
                                     purpose_returns_doc.strip().replace('\n', ' '))

        # For FinishInvestigationArgs, its summary_version is specific to its output structure
        schema_version_str = self.settings.default_tool_openai_schema_version
        if tool_name == "finish_investigation" and hasattr(pydantic_model, 'summary_version'):
            # This is a slight conceptual override; summary_version is part of the args model itself
            # For OpenAI schema description, we can refer to the overall tool schema version
             pass # Using the default_tool_openai_schema_version for the schema description

        description_with_version = f"{purpose_returns_doc_cleaned} (OpenAI Schema Version: {schema_version_str})"
        
        # The parameters part of the schema is what Pydantic generates as the top-level schema
        parameters = {"type": schema.get("type", "object"), "properties": schema.get("properties", {}), "required": schema.get("required", [])}
        
        return {
            "name": tool_name,
            "description": description_with_version,
            "parameters": parameters,
        }

    def _generate_and_cache_tool_schemas(self) -> List[Dict[str, Any]]:
        """Generates OpenAI function schemas from Pydantic models and manages caching."""
        cache_path = self.settings.tool_schemas_cache_path
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    schemas = json.load(f)
                logger.info(f"Loaded tool schemas from cache: {cache_path}")
                # Ensure find_interesting_window is first if loaded from cache
                # This is a simple sort; a more robust way might involve explicit ordering if tools grow
                schemas.sort(key=lambda s: 0 if s['name'] == 'find_interesting_window' else 1)
                return schemas
        except Exception as e:
            logger.warning(f"Failed to load schemas from cache ({cache_path}): {e}. Regenerating...")

        generated_schemas = []
        tool_arg_models = self._load_tool_argument_models() # Ensure this is called if not already an attribute

        # Ensure find_interesting_window is processed first for ordering
        priority_tool = "find_interesting_window"
        if priority_tool in tool_arg_models:
            model = tool_arg_models[priority_tool]
            generated_schemas.append(self._generate_openai_schema(priority_tool, model))
        
        for tool_name, model in tool_arg_models.items():
            if tool_name == priority_tool:
                continue # Already processed
            generated_schemas.append(self._generate_openai_schema(tool_name, model))
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(generated_schemas, f, indent=2)
            logger.info(f"Saved generated tool schemas to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save schemas to cache ({cache_path}): {e}")
        
        return generated_schemas
    
    # TODO: Add streaming tools
    # TODO: Implement chart generation tools

    # --- Placeholder for other methods from the checklist ---
    # _get_tag_type, _build_chat_history, _truncate_value, _compact_state, 
    # _ask_gpt_for_next_action, _calculate_tool_signature, _execute_tool, 
    # _initial_tag_inference, _generate_final_report, run

    def _get_tag_type(self, tag_name: str) -> Optional[str]:
        metadata = get_tag_metadata(tag_name)
        return metadata.get('value_type') if metadata else None

    def _compact_state(self) -> Dict[str, Any]:
        last_result_summary = "No previous tool results."
        if self.evidence:
            last_evidence_item = self.evidence[-1]
            # Last item is usually the function result; previous is assistant's call
            tool_name = "Unknown"
            tool_result_content = "N/A"
            if last_evidence_item.get("role") == "function":
                tool_name = last_evidence_item.get("name", "Unknown")
                tool_result_content = last_evidence_item.get("content", "N/A")
            elif len(self.evidence) > 1 and self.evidence[-2].get("role") == "assistant" and self.evidence[-2].get("function_call"):
                tool_name = self.evidence[-2]["function_call"].get("name", "Unknown")
                # Result is in the last item
                if last_evidence_item.get("role") == "function": # Should always be the case here
                     tool_result_content = last_evidence_item.get("content", "N/A")
            
            try:
                # content might be a JSON string or already a dict if processed internally
                result_data = json.loads(tool_result_content) if isinstance(tool_result_content, str) else tool_result_content
                last_result_summary = f"Tool: {tool_name}, Result: {truncate_json_or_text(result_data)}"
            except json.JSONDecodeError:
                last_result_summary = f"Tool: {tool_name}, Result: {truncate_json_or_text(tool_result_content)}"
            except Exception:
                 last_result_summary = f"Tool: {tool_name}, Result: Error processing result for compact state."

        return {
            "confidence": round(self.current_confidence, 3),
            "last_result_summary": last_result_summary,
            "current_investigation_window": self.investigation_window,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "steps_taken": self.current_step_count
        }

    def _build_chat_history(self, query: str, evidence_override: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Builds the chat history for LLM interaction."""
        system_prompt_content = f"Current UTC date is {self.today_iso}. {self.BASE_SYS_PROMPT}"
        messages = [{"role": "system", "content": system_prompt_content}, {"role": "user", "content": query}]
        current_evidence = evidence_override if evidence_override is not None else self.evidence
        
        for item in current_evidence:
            role = item.get("role")
            if role == "assistant" and item.get("function_call"):
                fc = item["function_call"]
                current_fc_val = fc.model_dump(exclude_unset=True) if hasattr(fc, 'model_dump') and callable(fc.model_dump) else fc
                messages.append({"role": "assistant", "content": item.get("content"), "function_call": current_fc_val})
            elif role == "function" and item.get("name") and item.get("content") is not None:
                 messages.append({"role": "function", "name": item["name"], "content": str(item["content"])})
        return messages

    def _summarize_evidence_for_final_report(self, max_tokens_for_summary: int) -> str:
        """Creates a concise string summary of the evidence log for the final report prompt."""
        if not self.evidence:
            return "No evidence was gathered during this investigation."
        if not self.tokenizer:
            logger.warning("Tokenizer not available, cannot accurately summarize evidence by token count. Returning full evidence (potentially truncated by API limit).")
            # Fallback: just join first few items if tokenizer missing, as a very basic safety
            return "\n".join([f"Step {i+1}: {truncate_json_or_text(e, max_length=200)}" for i, e in enumerate(self.evidence[:5])])

        summaries = []
        current_tokens = 0
        separator = "\n---\n" # Separator between evidence items
        separator_tokens = len(self.tokenizer.encode(separator))

        for i, item in enumerate(self.evidence):
            # Create a one-line summary for each evidence item (step)
            # Focusing on assistant function calls and their results
            summary_line = f"Step {item.get('step', i//2 + 1)}: " # Approximate step if not explicitly stored per item
            if item.get("role") == "assistant" and item.get("function_call"):
                fc = item["function_call"]
                tool_name = fc.name if hasattr(fc, 'name') else fc.get("name", "Unknown")
                tool_args = fc.arguments if hasattr(fc, 'arguments') else fc.get("arguments", "{}")
                summary_line += f"LLM called {tool_name} with args {truncate_json_or_text(tool_args, max_length=100)}."
            elif item.get("role") == "function":
                tool_name = item.get("name", "Unknown")
                content_str = str(item.get("content", ""))
                summary_line += f"Tool {tool_name} returned: {truncate_json_or_text(content_str, max_length=150)}."
            else:
                continue # Skip items not directly representing a call or result for brevity
            
            line_tokens = len(self.tokenizer.encode(summary_line))
            
            if current_tokens + line_tokens + (separator_tokens if summaries else 0) > max_tokens_for_summary:
                logger.warning(f"Evidence summary reached token limit ({max_tokens_for_summary}). Truncating history for final prompt.")
                break
            
            if summaries: # Add separator if not the first summary item
                current_tokens += separator_tokens
            summaries.append(summary_line)
            current_tokens += line_tokens
       
        return separator.join(summaries) if summaries else "No relevant actions found in evidence for summary."

    def _calculate_tool_signature(self, tool_name: str, tool_args: Dict) -> str:
        """Creates a consistent string signature for a tool call."""
        # Sort args by key to ensure consistent signature
        sorted_args = json.dumps(tool_args, sort_keys=True)
        return f"{tool_name}::{sorted_args}"

    def _initial_tag_inference(self, query: str) -> str:
        relevant_tags = self.tag_glossary.search_tags(query, top_k=3)
        for tag_info in relevant_tags:
            if self._get_tag_type(tag_info['tag']) == 'numeric':
                if tag_info.get('similarity_score', 0.0) >= self.settings.min_initial_tag_similarity_threshold:
                    logger.info(f"Initial inference: Using tag '{tag_info['tag']}' based on query similarity.")
                    return tag_info['tag']
        logger.info(f"Initial inference: Falling back to default numeric tag '{self.settings.default_numeric_tag}'.")
        return self.settings.default_numeric_tag

    def _update_token_cost(self, usage: Optional[openai.types.CompletionUsage], model_name_for_pricing: str):
        if not usage or not self.tokenizer: return
        try:
            price_in = self.settings.get_token_price(model_name_for_pricing, "input")
            price_out = self.settings.get_token_price(model_name_for_pricing, "output")
            cost = float(usage.prompt_tokens * price_in + usage.completion_tokens * price_out)
            self.estimated_cost_usd += cost
            logger.info(f"Tokens: P={usage.prompt_tokens},C={usage.completion_tokens}. Cost: Call=${cost:.4f}, Total=${self.estimated_cost_usd:.4f} (Model: {model_name_for_pricing})")
        except ValueError as e: logger.warning(f"Cost calc failed for {model_name_for_pricing}: {e}")

    def _ask_gpt_for_next_action(self, query: str) -> openai.types.chat.ChatCompletionMessage:
        messages = self._build_chat_history(query) # History is built from self.evidence
        messages.append({"role": "assistant", "content": json.dumps(self._compact_state())}) # Prompt LLM with current state
        
        logger.debug(f"Sending messages to LLM for next action: {messages}")
        try:
            response = self.openai_client.chat.completions.create(
                model=self.settings.planning_model_name,
                messages=messages,
                functions=self.tool_schemas,
                function_call="auto",
                temperature=self.settings.planning_temperature,
            )
            self._update_token_cost(response.usage, self.settings.planning_model_name)
            return response.choices[0].message
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise # Re-raise to be handled by the run loop or higher level

    def _log_evidence_to_file(self, log_payload: Dict[str, Any]):
        try:
            evidence_file_path = os.path.join(self.settings.evidence_log_dir, f"evidence_{log_payload['run_id']}_{log_payload['step']}_{log_payload['tool_name']}.json")
            os.makedirs(self.settings.evidence_log_dir, exist_ok=True)
            with open(evidence_file_path, "wb") as f:
                f.write(orjson.dumps(log_payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY))
            logger.info(f"Logged evidence to {evidence_file_path}")
        except Exception as e: logger.error(f"Failed to log evidence to file: {e}")

    def _get_tag_data_bounds(self, tag: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Fetches the actual start and end datetime for a given tag from data_loader."""
        try:
            bounds = get_data_time_range(tag=tag)
            start = bounds.get('start')
            end = bounds.get('end')
            
            # Ensure they are timezone-aware (UTC) if not already
            # pandas Timestamps from to_datetime might be timezone-aware if source was, or naive
            if start and isinstance(start, datetime) and start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            elif start and not isinstance(start, datetime):
                 start = pd.to_datetime(start, utc=True).to_pydatetime() # Convert if not datetime

            if end and isinstance(end, datetime) and end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            elif end and not isinstance(end, datetime):
                end = pd.to_datetime(end, utc=True).to_pydatetime() # Convert if not datetime
                
            return start, end
        except Exception as e:
            logger.warning(f"Could not retrieve data bounds for tag {tag} via get_data_time_range: {e}")
            return None, None

    # --- Tag Validation --------------------------------------------------
    def _ensure_valid_tag(self, tag_name: str, fallback_query: str) -> str:
        """
        Validate *tag_name* against the glossary.

        • If the tag already exists, return it unchanged.  
        • Otherwise perform a semantic lookup using *fallback_query*.  
        • Prefer a replacement that shares the same suffix segment
          (everything after the last '.') because this often indicates the
          measurement type, e.g. `STATUS`, `POWER_KW`, `TEMP.INTERNAL_C`.
        • If no candidate shares the suffix, fall back to the highest‑scoring
          candidate returned by the glossary.

        Raises
        ------
        ValueError
            If the glossary cannot provide any suitable fallback.
        """
        # Fast path – exact match
        if self.tag_glossary.get_tag_info(tag_name):
            return tag_name

        # Desired suffix for heuristic match
        suffix: Optional[str] = tag_name.split('.')[-1] if '.' in tag_name else None

        # Semantic fallback search
        candidates = self.tag_glossary.search_tags(fallback_query, top_k=5)
        if not candidates:
            raise ValueError(
                f"Tag '{tag_name}' not found in glossary and no fallback match available."
            )

        # Try to keep the same suffix (value type / signal intent)
        best_tag = None
        if suffix:
            for cand in candidates:
                if cand["tag"].split('.')[-1] == suffix:
                    best_tag = cand["tag"]
                    break

        # Fallback to top candidate if suffix heuristic failed
        if best_tag is None:
            best_tag = candidates[0]["tag"]

        logger.warning(
            "Tag '%s' not found – substituting with '%s' returned by glossary semantic search.",
            tag_name, best_tag,
        )
        return best_tag

    def _validate_and_clamp_window(
        self,
        tag: str, 
        start_time_str: Optional[str],
        end_time_str: Optional[str],
        tool_name_for_error: str,
        allow_no_initial_window: bool = False 
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Validates the proposed window against actual data for the tag.
        Clamps the window if it extends beyond available data.
        Raises ValueError if the window is entirely outside available data.
        Returns (clamped_start_iso_str, clamped_end_iso_str) or (None, None) if inputs are None and allowed.
        """
        if not start_time_str and not end_time_str and allow_no_initial_window:
            logger.info(f"No start/end time provided for {tool_name_for_error} on {tag}, and it's allowed. Proceeding without specific window.")
            return None, None # Allow tools like find_interesting_window to operate on full range if no initial window

        if not start_time_str or not end_time_str:
            raise ValueError(f"Tool '{tool_name_for_error}' on tag '{tag}' requires both start_time and end_time, but one or both are missing: start='{start_time_str}', end='{end_time_str}'. The LLM must provide these, possibly via parse_time_range first.")

        try:
            # Ensure input strings are parsed as timezone-aware UTC datetimes
            req_start_dt = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            if req_start_dt.tzinfo is None: req_start_dt = req_start_dt.replace(tzinfo=timezone.utc)
            
            req_end_dt = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
            if req_end_dt.tzinfo is None: req_end_dt = req_end_dt.replace(tzinfo=timezone.utc)

        except ValueError as ve:
            raise ValueError(f"Invalid ISO 8601 format for start_time or end_time for {tool_name_for_error} on {tag}: {ve}. Dates must be like YYYY-MM-DDTHH:MM:SSZ")

        if req_start_dt >= req_end_dt:
            raise ValueError(f"start_time ({start_time_str}) must be before end_time ({end_time_str}) for {tool_name_for_error} on {tag}.")

        actual_data_start, actual_data_end = self._get_tag_data_bounds(tag)

        if not actual_data_start or not actual_data_end:
            raise ValueError(f"Could not get actual data bounds for tag '{tag}'. Cannot validate window for {tool_name_for_error}.")

        # Check if the requested window is entirely outside the available data range
        if req_end_dt <= actual_data_start or req_start_dt >= actual_data_end:
            raise ValueError(
                f"Requested window ({start_time_str} to {end_time_str}) for {tool_name_for_error} on tag '{tag}' "
                f"is entirely outside the available data range ({actual_data_start.isoformat().replace('+00:00','Z')} to {actual_data_end.isoformat().replace('+00:00','Z')})."
            )

        # Clamp the window
        clamped_start_dt = max(req_start_dt, actual_data_start)
        clamped_end_dt = min(req_end_dt, actual_data_end)

        if clamped_start_dt >= clamped_end_dt: # After clamping, window might become invalid
             raise ValueError(
                f"Requested window ({start_time_str} to {end_time_str}) for {tool_name_for_error} on tag '{tag}' "
                f"has no overlap with available data after clamping ({actual_data_start.isoformat().replace('+00:00','Z')} to {actual_data_end.isoformat().replace('+00:00','Z')})."
            )

        logger.info(f"Window for {tool_name_for_error} on {tag}: Requested [{start_time_str} to {end_time_str}], Clamped to [{clamped_start_dt.isoformat().replace('+00:00','Z')} to {clamped_end_dt.isoformat().replace('+00:00','Z')}]")
        return clamped_start_dt.isoformat().replace("+00:00", "Z"), clamped_end_dt.isoformat().replace("+00:00", "Z")

    def _execute_tool(self, tool_name: str, tool_args_str: str, run_id: str, step_num: int) -> Dict[str, Any]:
        logger.info(f"Attempting tool: {tool_name} with args: {tool_args_str}")
        actual_tool_result_data: Any = None
        error_msg: Optional[str] = None
        execution_status: Literal["success","error","skipped_duplicate"] = "success"
        parsed_args_for_log = {}
        validated_args_model = None
        current_args_for_tool_call = {}
        
        try:
            parsed_args_for_log = orjson.loads(tool_args_str)
            arg_model_pydantic = self.tool_argument_models.get(tool_name)
            assert arg_model_pydantic, f"No Pydantic argument model found for tool: {tool_name}"
            validated_args_model = arg_model_pydantic(**parsed_args_for_log)
            tag_fields = ["tag", "primary_tag", "cause_tag", "effect_tag"]
            for f in tag_fields:
                if f in parsed_args_for_log and parsed_args_for_log[f]:
                    parsed_args_for_log[f] = self._ensure_valid_tag(
                        parsed_args_for_log[f],
                        self.original_query        # always supply a fallback
                    )
            # then rebuild validated_args_model with the fixed tags
            validated_args_model = arg_model_pydantic(**parsed_args_for_log)
            current_args_for_tool_call = validated_args_model.model_dump()

            # ---- Tag existence guard‑rail ----
            for key in ("primary_tag", "tag", "cause_tag", "effect_tag"):
                if key in current_args_for_tool_call and current_args_for_tool_call[key]:
                    current_args_for_tool_call[key] = self._ensure_valid_tag(
                        current_args_for_tool_call[key],
                        self.original_query        # <- fallback here too
                    )

            tool_sig = self._calculate_tool_signature(tool_name, current_args_for_tool_call)
            is_duplicate = False
            if tool_name != "parse_time_range" and tool_sig in self.executed_tool_signatures:
                 is_duplicate = True
            # Allow parse_time_range only once at the beginning (step 1 ideally, or very early) effectively.
            # Or if its arguments changed significantly (handled by general tool_sig check).
            elif tool_name == "parse_time_range" and tool_sig in self.executed_tool_signatures and self.current_step_count > 1:
                 is_duplicate = True 
            
            if is_duplicate:
                execution_status = "skipped_duplicate"
                actual_tool_result_data = {"message": "Duplicate tool call skipped as similar parameters already used."}
                logger.info(f"Skipping duplicate call for tool: {tool_name}.")
            else:
                self.executed_tool_signatures.add(tool_sig) # Add signature if not a duplicate
                tool_fn = self.tools.get(tool_name)
                assert tool_fn, f"Tool function for {tool_name} not in registry."
                
                # Window Injection & Validation Logic
                # Tools that require a time window for their primary operation:
                window_dependent_tools = ["detect_numeric_anomalies", "detect_binary_flips", "test_causality", "create_event_sequence"]
                # find_interesting_window can accept a scan range, which also needs validation/clamping.
                tool_needs_window_validation = tool_name in window_dependent_tools or tool_name == "find_interesting_window"

                if tool_needs_window_validation:
                    # For find_interesting_window, its start_time/end_time define the *scan range*.
                    # For other tools, it's the *analysis window*.
                    llm_start_time = current_args_for_tool_call.get("start_time")
                    llm_end_time = current_args_for_tool_call.get("end_time")

                    target_tag_for_window = current_args_for_tool_call.get("primary_tag") if tool_name == "find_interesting_window" else current_args_for_tool_call.get("tag")
                    if not target_tag_for_window and tool_name == "test_causality": # test_causality uses cause_tag for bounds
                        target_tag_for_window = current_args_for_tool_call.get("cause_tag")
                    assert target_tag_for_window, f"Could not determine target tag for window validation for tool {tool_name}"

                    # If LLM provides start/end, validate & clamp. If not, and it's not FIW, try to inject investigation_window.
                    if llm_start_time and llm_end_time:
                        clamped_start, clamped_end = self._validate_and_clamp_window(target_tag_for_window, llm_start_time, llm_end_time, tool_name)
                        current_args_for_tool_call["start_time"] = clamped_start
                        current_args_for_tool_call["end_time"] = clamped_end
                    elif tool_name in window_dependent_tools: # Not FIW, and LLM didn't provide window args
                        if self.investigation_window and self.investigation_window.get("start_time") and self.investigation_window.get("end_time"):
                            logger.info(f"Injecting established investigation window into {tool_name} args: {self.investigation_window}")
                            clamped_start, clamped_end = self._validate_and_clamp_window(
                                target_tag_for_window, 
                                self.investigation_window["start_time"], 
                                self.investigation_window["end_time"], 
                                tool_name
                            )
                            current_args_for_tool_call["start_time"] = clamped_start
                            current_args_for_tool_call["end_time"] = clamped_end
                        else:
                            raise MissingWindowError(f"Tool {tool_name} requires a window, but none was provided by LLM and no investigation window is set.")
                    # For find_interesting_window, if no start/end provided by LLM, it's allowed to scan full range (handled by _validate_and_clamp_window with allow_no_initial_window=True if we choose that path)
                    # Current _validate_and_clamp_window needs start/end unless allow_no_initial_window. Let's assume FIW if called with no times will set them from full data range based on its own logic.
                    # The _validate_and_clamp_window call for FIW handles the case where LLM *does* provide start/end for scan range.
                    # If LLM provides NO times for FIW, _validate_and_clamp_window isn't called here for it. Its internal parse_time_reference will use full range.
                
                actual_tool_result_data = tool_fn(**current_args_for_tool_call)
                
                if tool_name == "parse_time_range" and isinstance(actual_tool_result_data, dict):
                    pt_start = actual_tool_result_data.get("start_time")
                    pt_end = actual_tool_result_data.get("end_time")

                    if not pt_start and not pt_end:
                        # LLM gave nothing – expand around 'yesterday' 14:30 by ±1 h
                        logger.info("parse_time_range returned no start/end time. Defaulting to yesterday 14:30 +/- 1 hour.")
                        pivot = datetime.now(timezone.utc).replace(hour=14, minute=30, second=0, microsecond=0) - timedelta(days=1)
                        pt_start = (pivot - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
                        pt_end   = (pivot + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
                        # Update actual_tool_result_data so it's logged correctly and used for window setting
                        actual_tool_result_data["start_time"] = pt_start
                        actual_tool_result_data["end_time"] = pt_end
                        actual_tool_result_data["defaulted_by_orchestrator"] = True

                    if pt_start and pt_end:
                        # Validate the window from parse_time_range against a reference tag (e.g. default numeric tag)
                        # to ensure it's not completely outside all data.
                        # This uses the default_numeric_tag as a general proxy for data availability.
                        ref_tag_for_validation = self.settings.default_numeric_tag 
                        # _initial_tag_inference could also be used if query available here, but might be too much
                        try:
                            clamped_pt_start, clamped_pt_end = self._validate_and_clamp_window(ref_tag_for_validation, pt_start, pt_end, "parse_time_range (validation)")
                            self.investigation_window = {"start_time": clamped_pt_start, "end_time": clamped_pt_end}
                            logger.info(f"Investigation window SET by parse_time_range (and validated/clamped): {self.investigation_window}")
                        except ValueError as ve_pts:
                            logger.warning(f"Window from parse_time_range ({pt_start} to {pt_end}) was invalid or outside all data ranges: {ve_pts}. Investigation window NOT set.")
                            # LLM will get this error if parse_time_range was directly called.
                            # If we forced it, this error means the forced call yielded unusable window.
                            actual_tool_result_data = {"error": str(ve_pts), "original_start": pt_start, "original_end": pt_end } # Pass error back
                            execution_status = "error"
                            error_msg = str(ve_pts)

                    elif pt_start or pt_end:
                        logger.info(f"parse_time_range returned partial window: start={pt_start}, end={pt_end}. Investigation window not fully set.")
                    # If parse_time_range returns None for both, investigation_window remains None.
                elif tool_name == "find_interesting_window" and isinstance(actual_tool_result_data, dict) and actual_tool_result_data.get("window") and isinstance(actual_tool_result_data["window"], dict):
                    fiw_start = actual_tool_result_data["window"].get("start_time")
                    fiw_end = actual_tool_result_data["window"].get("end_time")
                    if fiw_start and fiw_end:
                        self.investigation_window = {"start_time": fiw_start, "end_time": fiw_end}
                        logger.info(f"Investigation window UPDATED by find_interesting_window: {self.investigation_window}")
                # execution_status remains "success" if no exceptions were raised before this point

        except ValueError as ve_clamp_or_parse: # Catches errors from _validate_and_clamp_window or initial orjson.loads
            execution_status = "error"; error_msg = str(ve_clamp_or_parse); actual_tool_result_data = {"error_detail": str(ve_clamp_or_parse)}
            logger.warning(f"ValueError during arg processing or window validation for {tool_name}: {ve_clamp_or_parse}")
        except (MissingWindowError, StaleWindowError, AssertionError) as e_win:
            execution_status="error"; error_msg=str(e_win); actual_tool_result_data={"error_detail": str(e_win)}
            logger.warning(f"{type(e_win).__name__} for {tool_name}: {e_win}")
        except pydantic.ValidationError as ve_pydantic:
            execution_status="error"; error_msg=f"Invalid tool arguments: {ve_pydantic.errors()}"; actual_tool_result_data={"error_detail": str(ve_pydantic)}
            logger.error(f"Pydantic validation error for {tool_name} args: {tool_args_str}. Errors: {ve_pydantic.errors()}")
        except Exception as e_gen:
            execution_status="error"; error_msg=str(e_gen); actual_tool_result_data={"error_detail": str(e_gen)}
            logger.exception(f"General error executing {tool_name}: {e_gen}")
        
        log_payload_args = validated_args_model.model_dump() if validated_args_model else parsed_args_for_log
        log_payload = {"run_id":run_id, "step":step_num, "tool_name":tool_name, "arguments_str":tool_args_str, "parsed_arguments": log_payload_args, "result":actual_tool_result_data, "status":execution_status, "error_message":error_msg, "timestamp_utc":datetime.now(timezone.utc).isoformat()}
        final_log_payload = copy.deepcopy(log_payload)
        self.evidence_writer_executor.submit(self._log_evidence_to_file, final_log_payload)
        
        if execution_status == "success":
            return {"status": "success", "data": actual_tool_result_data}
        elif execution_status == "skipped_duplicate":
            return {"status": "skipped_duplicate", "data": actual_tool_result_data}
        else: 
            return {"status": "error", "data": {"error": error_msg or "Tool execution failed", "raw_result_on_error": actual_tool_result_data}}

    def run(self, query: str) -> Dict[str, Any]:
        self._reset_run_state()
        self.current_run_id = str(uuid.uuid4())
        self.original_query = query # Store for forced parse_time_range
        self.today_iso = datetime.now(timezone.utc).isoformat() # Store current UTC timestamp
        logger.info(f"Run ID: {self.current_run_id} Query: '{query}' Today: {self.today_iso}")
        final_status_for_report = "completed_max_steps" # Default unless loop breaks for other reasons

        try:
            for step_human in range(1, self.settings.max_steps + 1):
                self.current_step_count = step_human
                logger.info(f"--- Step {self.current_step_count}/{self.settings.max_steps} ---")
                
                tool_name_to_execute: Optional[str] = None
                tool_args_str_to_execute: Optional[str] = None
                llm_call_content: Optional[str] = None # Content from assistant message if any
                llm_function_call_object: Optional[Any] = None # Function call object from assistant

                # Initial step: guide or force parse_time_range
                if self.current_step_count == 1 and not self.investigation_window:
                    try: 
                        llm_msg_first_step = self._ask_gpt_for_next_action(query)
                        llm_call_content = llm_msg_first_step.content
                        llm_function_call_object = llm_msg_first_step.function_call
                    except Exception as api_err: 
                        logger.error(f"API error on first step: {api_err}. Investigation halting.") 
                        final_status_for_report = "halted_api_error"
                        return self._generate_final_report(query, final_status_for_report, error_details=str(api_err))

                    if llm_function_call_object and llm_function_call_object.name != "parse_time_range":
                        logger.warning(f"LLM chose '{llm_function_call_object.name}' as first step. Overriding to 'parse_time_range'.")
                        tool_name_to_execute = "parse_time_range"
                        tool_args_str_to_execute = json.dumps({"query": self.original_query}) # Modified line
                        # Adjust evidence to reflect the override
                        self.evidence.append({
                            "role":"assistant", 
                            "content": "Orchestrator override: Forcing parse_time_range.", 
                            "function_call": {"name": tool_name_to_execute, "arguments": tool_args_str_to_execute}
                        })
                    elif llm_function_call_object and llm_function_call_object.name == "parse_time_range":
                        tool_name_to_execute = llm_function_call_object.name
                        tool_args_str_to_execute = llm_function_call_object.arguments
                        # If LLM calls parse_time_range but with empty args, inject query
                        if tool_args_str_to_execute == "{}":
                            logger.info("LLM called parse_time_range with empty args. Injecting original query.")
                            tool_args_str_to_execute = json.dumps({"query": self.original_query})
                        self.evidence.append({"role":"assistant", "content": llm_call_content, "function_call": llm_function_call_object })
                    else: 
                        logger.warning(f"LLM did not make a function call on first step. Response: {llm_call_content}");
                        final_status_for_report = "halted_no_initial_call"
                        # This is a critical failure for the first step, generate report and exit.
                        return self._generate_final_report(query, final_status_for_report, llm_direct_response=llm_call_content)
                else: # Subsequent steps
                    try: 
                        llm_msg_subsequent_step = self._ask_gpt_for_next_action(query)
                        llm_call_content = llm_msg_subsequent_step.content
                        llm_function_call_object = llm_msg_subsequent_step.function_call
                    except Exception as api_err: 
                        logger.error(f"API error on step {self.current_step_count}: {api_err}. Investigation halting.") 
                        final_status_for_report = "halted_api_error"
                        return self._generate_final_report(query, final_status_for_report, error_details=str(api_err))
                    
                    if llm_function_call_object:
                        tool_name_to_execute = llm_function_call_object.name
                        tool_args_str_to_execute = llm_function_call_object.arguments
                        self.evidence.append({"role":"assistant", "content": llm_call_content, "function_call": llm_function_call_object })
                    else: 
                        logger.warning(f"LLM did not suggest a tool for step {self.current_step_count}. Response: {llm_call_content}");
                        final_status_for_report = "halted_no_function_call_mid_run"
                        break # Exit loop, proceed to final report generation at the end of 'run'
                
                # Execute the chosen/overridden tool
                logger.info(f"Executing: {tool_name_to_execute} with args: {tool_args_str_to_execute}")
                
                if tool_name_to_execute == "finish_investigation":
                    try:
                        parsed_args = FinishInvestigationArgs(**orjson.loads(tool_args_str_to_execute)).model_dump()
                        logger.info("LLM called finish_investigation. Finalizing.")
                        final_status_for_report = "completed_by_finish_call"
                        # This is a successful completion, return directly.
                        return {**parsed_args, "status":final_status_for_report, "steps":self.current_step_count, "cost_usd":round(self.estimated_cost_usd,4)}
                    except Exception as e: 
                        logger.error(f"Parse finish_investigation failed: {e}"); 
                        final_status_for_report = "halted_finish_parse_error"
                        # This is a critical error in finalization, generate report and exit.
                        return self._generate_final_report(query, final_status_for_report, error_details=str(e))
                
                tool_execution_package = self._execute_tool(tool_name_to_execute, tool_args_str_to_execute, self.current_run_id, self.current_step_count)
                execution_status = tool_execution_package.get("status", "error")
                actual_tool_result = tool_execution_package.get("data")
                
                # Use orjson to handle numpy types when serializing for evidence log
                try:
                    serialized_result_content = orjson.dumps(actual_tool_result, option=orjson.OPT_SERIALIZE_NUMPY).decode()
                except Exception as ser_err:
                    logger.error(f"Error serializing tool result for {tool_name_to_execute} with orjson: {ser_err}. Falling back to standard json.")
                    serialized_result_content = json.dumps(actual_tool_result, default=str) # Fallback, might still fail for complex unhandled types
                
                self.evidence.append({"role":"function", "name":tool_name_to_execute, "content":serialized_result_content})
                
                penalize_dup = execution_status == "skipped_duplicate"
                if execution_status == "error" and isinstance(actual_tool_result, dict) and actual_tool_result.get("error"):
                    logger.warning(f"Tool {tool_name_to_execute} executed with error: {actual_tool_result.get('error')}")
                elif execution_status != "skipped_duplicate" and execution_status != "success":
                    logger.error(f"Tool {tool_name_to_execute} returned unexpected status: {execution_status}")

                current_confidence_before_update = self.current_confidence
                self.current_confidence = score_evidence(self.evidence, self.current_step_count, self.settings.default_confidence_decay_base, penalize_duplicate_attempt=penalize_dup)
                logger.info(f"Confidence: {current_confidence_before_update:.3f} -> {self.current_confidence:.3f}")

                if abs(self.current_confidence - current_confidence_before_update) < self.settings.min_confidence_delta_for_staleness:
                    self.stale_confidence_counter += 1; logger.info(f"Confidence stale. Counter: {self.stale_confidence_counter}/{self.settings.max_stale_steps}")
                else: self.stale_confidence_counter = 0
                self.previous_confidence_for_staleness_check = current_confidence_before_update
                
                if self.current_confidence >= self.settings.confidence_threshold: 
                    logger.info("Confidence threshold reached.");
                    final_status_for_report = "completed_confidence_threshold"
                    break 
                if self.stale_confidence_counter >= self.settings.max_stale_steps: 
                    logger.warning("Max stale steps reached.");
                    final_status_for_report = "halted_stale_confidence"
                    break
            
                if self.estimated_cost_usd > float(self.settings.max_runaway_cost_usd):
                    logging.warning(f"Cost guard at ${self.estimated_cost_usd:.4f} (Limit: ${self.settings.max_runaway_cost_usd}). Halting.");
                    final_status_for_report = "halted_cost_limit"
                    break
           
            # If loop completes or breaks, generate final report based on final_status_for_report
            logger.info(f"Loop finished. Status: {final_status_for_report}. Investigation concluding.")
            return self._generate_final_report(query, status_override=final_status_for_report)
        finally:
            logger.info("Shutting down evidence writer executor.")
            self.evidence_writer_executor.shutdown(wait=True)

    def _generate_final_report(self, query: str, status_override: Optional[str] = None, error_details: Optional[str] = None, llm_direct_response: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Generating final report. Status: {status_override or 'LLM_summary_expected'}, Model: {self.settings.final_report_model_name}")
        
        current_status = status_override or "unknown_halt_reason"
        if status_override and status_override not in ["completed_confidence_threshold", "completed_max_steps", "completed_by_finish_call"]:
            logger.warning(f"Generating fallback report due to halt status: {current_status}")
            final_report_data = {
                "root_cause_statement": f"Investigation {current_status}. {error_details or llm_direct_response or 'Review logs for details.'}",
                "event_timeline_summary": [{"time": datetime.now(timezone.utc).isoformat(), "description": f"Investigation concluded due to: {current_status}"}],
                "business_impact_summary": {"total_cost_usd": round(self.estimated_cost_usd,4), "energy_cost_usd": "N/A", "product_risk_usd": "N/A", "severity_level": "unknown"},
                "recommendations": ["Review agent logs and evidence for detailed step-by-step analysis."],
                "final_confidence_score": self.current_confidence 
            }
            return {
                **final_report_data,
                "status": current_status,
                "orchestrator_confidence": self.current_confidence,
                "total_steps": self.current_step_count,
                "total_cost_usd": round(self.estimated_cost_usd, 4)
            }

        summarized_evidence_str = self._summarize_evidence_for_final_report(self.settings.max_tokens_final_report_evidence_summary)
        
        user_final_prompt = (
            "Based on all the evidence provided, please summarize the investigation. "
            "Call the 'finish_investigation' function with your comprehensive findings. "
            "IMPORTANT: You MUST provide ALL fields defined in the 'finish_investigation' function schema. "
            "The 'event_timeline_summary' MUST contain at least 3 chronological events detailing the trigger, response, and resolution if identified. "
            "The 'business_impact_summary' MUST include 'total_cost_usd', 'energy_cost_usd', 'product_risk_usd', and 'severity_level'. "
            "Provide 1-3 actionable 'recommendations'. "
            "Include your 'final_confidence_score' (0.0-1.0) in these conclusions."
        )
        if status_override:
            user_final_prompt = f"The investigation loop has concluded with status: {status_override}. {user_final_prompt}"
        
        final_messages = [
            {"role": "system", "content": self.BASE_SYS_PROMPT}, 
            {"role": "user", "content": f"Original Query: {query}\n\nSummarized Investigation Evidence:\n{summarized_evidence_str}"},
            {"role": "user", "content": user_final_prompt}
        ]
        
        finish_schema = next((s for s in self.tool_schemas if s["name"] == "finish_investigation"), None)
        assert finish_schema, "CRITICAL: finish_investigation schema not found!"

        try:
            logger.debug(f"Sending messages to LLM for final report ({self.settings.final_report_model_name}): {json.dumps(final_messages[-2:], indent=2)}")
            response = self.openai_client.chat.completions.create(
                model=self.settings.final_report_model_name,
                messages=final_messages,
                functions=[finish_schema],
                function_call={"name": "finish_investigation"},
                temperature=self.settings.temperature_final_report,
            )
            self._update_token_cost(response.usage, self.settings.final_report_model_name)
            llm_final_call = response.choices[0].message

            if llm_final_call.function_call and llm_final_call.function_call.name == "finish_investigation":
                final_args_str = llm_final_call.function_call.arguments
                logger.info(f"LLM provided finish_investigation arguments: {final_args_str}")

                # ---------- Pre‑validate & complete ---------- #
                raw_args = orjson.loads(final_args_str)

                # Ensure timeline has ≥3 events
                if "event_timeline_summary" not in raw_args or len(raw_args.get("event_timeline_summary", [])) < 3:
                    raw_args["event_timeline_summary"] = build_event_timeline(self.evidence)

                # Ensure we have a business‑impact section
                impact_ev = next(
                    (
                        ev.get("data")
                        for ev in reversed(self.evidence)
                        if ev.get("role") == "function" and ev.get("name") == "calculate_impact"
                    ),
                    None,
                )
                if "business_impact_summary" not in raw_args:
                    raw_args["business_impact_summary"] = build_business_impact(impact_ev)

                # ---------- Strict schema validation ---------- #
                report_data = FinishInvestigationArgs(**raw_args).model_dump()
                final_status = status_override or "completed_llm_summary"
                return {
                    **report_data,
                    "status": final_status,
                    "orchestrator_confidence_before_summary": self.current_confidence,
                    "total_steps": self.current_step_count,
                    "total_cost_usd": round(self.estimated_cost_usd, 4)
                }
        except Exception as e:
            logger.exception("Error during final report LLM call.")
            return {"error": "Exception during final report generation.", "details": str(e), "status": "halted_final_report_api_call"}
        pass # Ensure class definition ends cleanly

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the LLM Orchestrator for manufacturing root cause analysis.")
    parser.add_argument(
        "--query", 
        type=str, 
        default="Why did the freezer temperature spike yesterday afternoon?", 
        help="The natural language query to investigate."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Override maximum investigation steps from settings."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="Override confidence threshold from settings."
    )
    cli_args = parser.parse_args()

    print("Testing LLMOrchestrator.")
    if not os.path.exists(".env"):
        print("Creating a dummy .env file for testing. Please replace with your actual MCP_OPENAI_API_KEY.")
        with open(".env", "w") as f:
            f.write("MCP_OPENAI_API_KEY=your_key_here\n")
            f.write("MCP_DEFAULT_NUMERIC_TAG=FREEZER01.TEMP.INTERNAL_C\n")
            f.write("MCP_PLANNING_MODEL_NAME=gpt-4o-mini\n")
            f.write("MCP_FINAL_REPORT_MODEL_NAME=gpt-4o\n")
    
    try:
        settings = Settings()
        # Override settings from CLI if provided
        if cli_args.max_steps is not None:
            settings.max_steps = cli_args.max_steps
        if cli_args.confidence_threshold is not None:
            settings.confidence_threshold = cli_args.confidence_threshold
            
        if not settings.openai_api_key or settings.openai_api_key == "your_key_here":
            print("WARNING: OpenAI API key (MCP_OPENAI_API_KEY) appears to be a placeholder or not loaded correctly from .env.")
            print("A live test run against OpenAI will likely fail.")
            print("Please ensure MCP_OPENAI_API_KEY is correctly set in your .env file.")
        
        orchestrator = LLMOrchestrator(settings)
        print("LLMOrchestrator initialized successfully.")
        print(f"Using planning model: {settings.planning_model_name}, final report model: {settings.final_report_model_name}")
        print(f"Max steps: {settings.max_steps}, Confidence threshold: {settings.confidence_threshold}")

        print(f"\nAttempting a test run with query: '{cli_args.query}'")
        if settings.openai_api_key and settings.openai_api_key != "your_key_here":
            report = orchestrator.run(cli_args.query)
            print("\nTest Run Report:")
            try:
                print(orjson.dumps(report, option=orjson.OPT_INDENT_2).decode())
            except Exception:
                print(json.dumps(report, indent=2, default=str))
        else:
            print("Skipping live test run as OpenAI API key is not configured (or is placeholder value).")
            print("You can still review initialization and schema generation logs above (if any). Ensure .env is correct for full test.")

    except ValueError as ve:
        print(f"ValueError during setup or run: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() 