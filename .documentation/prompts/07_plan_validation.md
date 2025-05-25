## 🧠 Goal
Ensure the system **gracefully handles invalid user queries** that reference:
- Tags that don't exist in the glossary or dataset
- Invalid time ranges (e.g., start time after end time)

Add a validation step immediately after the LLM generates the `AnalysisPlan`. If the plan is invalid:
- Set `can_continue: false` and include a short, helpful `error_reason`
- Return a message early in the `llm_interpret_query()` pipeline and **exit gracefully**
- Do not continue with data loading, analysis, or chart generation

## ✅ Tasks
- [ ] Create a new function called `validate_analysis_plan(plan: AnalysisPlan) -> Tuple[bool, str]`
  - If `plan.primary_tag` is not in `get_available_tags()`, return `False, "Unknown tag: ..."`.
  - If `plan.start_time >= plan.end_time`, return `False, "Invalid time range: ..."`.
- [ ] Call this function inside `llm_interpret_query()` after the plan is parsed
- [ ] If `can_continue` is false, return a friendly markdown error like: