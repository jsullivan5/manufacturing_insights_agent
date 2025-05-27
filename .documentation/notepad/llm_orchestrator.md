# Developer Checklist: src/llm_orchestrator.py

## I. Imports
- [ ] `os`, `json`, `logging`
- [ ] `datetime`, `timedelta`
- [ ] `typing` (Dict, List, Any, Optional, Callable, Tuple, Set)
- [ ] `uuid` (for `run_id`)
- [ ] `openai`
- [ ] `pydantic`
- [ ] `orjson` (for evidence logging)
- [ ] `tiktoken` (or other tokenizer for prompt length checks)
- [ ] `src.config.Settings`
- [ ] `src.glossary.TagGlossary`
- [ ] `src.tools.tag_intel.get_tag_metadata`
- [ ] All core tool functions from `src.tools.atomic_tools`
- [ ] Specific Pydantic argument models and a base `ToolResultModel` from `src.tool_models`
- [ ] Consider the following so That lets every tool return a standardized flag the scorer can inspect.
```python
class ToolResultModel(BaseModel):
    schema_version: str = Field("0.1", frozen=True)
    status: Literal["success", "error"] = "success"
```
- [ ] `src.confidence_scorer.score_evidence`

## II. Global Constants & System Prompt (within `LLMOrchestrator` or from `Settings`)
- [ ] `SYS_PROMPT`: Multi-line system prompt string.
    -   Role: RootCauseGPT, industrial forensics LLM.
    -   Tools: Mention JSON-typed functions.
    -   State: Explain `confidence`, `last_result`, `window`.
    -   Goal: Raise confidence to >= (settings.confidence_threshold).
    -   Strategy:
        1.  If no window -> `find_interesting_window` on relevant numeric tag (listed first in schemas).
        2.  Detect anomalies in that tag within the window.
        3.  Search related tags, detect anomalies/flips.
        4.  Call `test_causality` on promising pairs.
        5.  Adjust thresholds or new tags if stagnant.
        6.  Call `finish_investigation` when done.
    -   Reminder: "One tool call per turn. Use `start_time`/`end_time` from `window` if present. Choose tools deterministically. Return ISO-8601 UTC timestamps only."

## III. LLMOrchestrator Class Definition
- [ ] `class MissingWindowError(Exception): pass`
- [ ] `class StaleWindowError(Exception): pass`
- [ ] `class LLMOrchestrator:`

### A. `__init__(self, settings: Settings)`
    - [ ] Store `self.settings: Settings = settings` (includes model names, temps, token limits, `max_window_age_days`, `default_confidence_decay_base` e.g., 0.98)
    - [ ] `self.openai_client = openai.OpenAI(api_key=self.settings.openai_api_key)`
    - [ ] `self.tokenizer = tiktoken.get_encoding("cl100k_base")` (or as per chosen model)
    - [ ] `self.tag_glossary = TagGlossary(glossary_path=self.settings.tag_glossary_path)`
    - [ ] `self.tools: Dict[str, Callable] = self._register_tools()`
    - [ ] `self.tool_argument_models: Dict[str, type[pydantic.BaseModel]] = self._load_tool_argument_models()`
    - [ ] `self.tool_schemas: List[Dict] = self._generate_and_cache_tool_schemas()`
    - [ ] `_reset_run_state()` call

### B. Per-Run State Reset Method (`_reset_run_state`)
    - [ ] As previously defined (run_id, evidence, confidence, window, step_count, cost, executed_signatures, stale_counter, prev_confidence).

### C. Tool and Schema Management
    - [ ] `_register_tools(self) -> Dict[str, Callable]`.
    - [ ] `_load_tool_argument_models(self) -> Dict[str, type[pydantic.BaseModel]]`.
    - [ ] `_generate_and_cache_tool_schemas(self) -> List[Dict]`:
        -   Cache path from `self.settings`. Load if exists.
        -   Else, generate: Inject `purpose_and_version` (e.g., "Purpose: ... SchemaVersion: X.Y") into schema description.
        -   Ensure `find_interesting_window` schema is first. Save to cache.

### D. Helper Methods
    - [ ] `_get_tag_type(self, tag_name: str) -> Optional[str]`.
    - [ ] `_build_chat_history(self, query: str, current_run_evidence: List[Dict]) -> List[Dict]`.
    - [ ] `_truncate_value(self, value: Any, max_length: int = 150, max_array_elements: int = 2, max_string_length: int = 75, max_dict_keys: int = 10) -> Any`:
        -   Clips long strings, arrays, and dictionary key counts.
    - [ ] `_compact_state(self) -> Dict` (uses `_truncate_value`).
    - [ ] `_check_prompt_length(self, messages: List[Dict]) -> bool`: (Optional, for advanced cost control before API call using `self.tokenizer`).
    - [ ] `_ask_gpt_for_next_action(...)`:
        -   `temperature=self.settings.planning_temperature`.
    - [ ] `_calculate_tool_signature(...)`.
    - [ ] `_execute_tool(self, tool_name: str, tool_args_str: str) -> Dict`:
        -   Parse, validate args.
        -   Signature check. If duplicate:
            -   `self.current_confidence = score_evidence(..., penalize_duplicate_attempt=True)`
            -   Return `{"status": "skipped_duplicate", ...}`.
        -   **Window Injection & Validation**:
            -   If tool requires window and `self.investigation_window` is `None` and args don't provide window: raise `MissingWindowError`.
            -   If `self.investigation_window` is used:
                -   Parse `self.investigation_window['start_time']` to datetime.
                -   If `(datetime.utcnow().replace(tzinfo=timezone.utc) - window_start_dt).days > self.settings.max_window_age_days`: raise `StaleWindowError`.
            -   Inject `start_time`, `end_time` from `self.investigation_window` if applicable. Assert not `None` if tool needs them.
        -   Call tool. Log evidence with `orjson`.
    - [ ] `_initial_tag_inference(...)`.
    - [ ] `_generate_final_report(...)`:
        -   Ensure `finish_investigation` Pydantic model includes `summary_version: str = "0.1"`.

### E. Main `run` Method
    - [ ] `run(self, query: str) -> Dict`:
        -   `self._reset_run_state()`.
        -   Loop:
            -   `llm_message = self._ask_gpt_for_next_action(...)`.
            -   If `llm_message.function_call`:
                -   ... (existing logic for tool call, evidence appending) ...
                -   `tool_result = self._execute_tool(...)`.
                -   If `tool_result.get("status") == "skipped_duplicate"`: continue loop.
                -   ... (append result to evidence) ...
                -   `new_confidence = score_evidence(self.evidence, self.current_step_count, self.settings.confidence_decay_factor, penalize_duplicate_attempt=False)`.
                -   Apply decay: `new_confidence *= (self.settings.default_confidence_decay_base ** self.current_step_count)` (Ensure decay applied *after* bonuses from `score_evidence`).
                -   ... (staleness check, confidence update, break conditions) ...
            -   Cost Guard: If `self.estimated_cost_usd > self.settings.max_runaway_cost_usd`: `logging.warning(f"Runaway cost guard triggered...")`, break.
        -   Return `self._generate_final_report(...)`.

## IV. Utility Functions (Module Level or Static)
- [ ] `def truncate_json_or_text(...)` (as refined).

## V. Logging & Error Handling
- [ ] Consistent `logging`. Critical errors/guards as `logging.WARNING` or `logging.ERROR`.
- [ ] Robust `try-except` for API calls, tool execution, JSON parsing.

This checklist should provide a good roadmap for implementing `llm_orchestrator.py`.
