# Initial Context

1 — Quick technical read-out (based on the last run)

Signal	What we see	Good / Needs work
Confidence trace	0 → 0.08 (window) → 0.157 (numeric anomalies) → 0.192 → 0.376 (binary flips) → 0.609 (causality) → 0.597 / 0.585	✅ Chain bonus is working; model hit > 0.6 and never stalled.
Halt reason	None – finished all 7 allotted steps, then produced finish_investigation.	✅ No stale-confidence halt.
Finish payload	Has root_cause_statement, but timeline & impact are still “No specific … provided”.	⚠️ The P0 “required timeline + impact” checklist item not enforced yet.
Checklist bonus (+0.25)	Fired: window ✓, anomaly ✓, binary flip ✓, causality ✓.	✅ Working.
Causality data path	Scorer now handles both {data:{…}} and flat dict – we saw the bonus.	✅ Fixed.
Binary continuous-high	detect_binary_flips now yields continuous_high_event; scorer counts it.	✅ Fixed.
Numeric detector	External baseline used; anomalies found on first pass.	△ still static σ; adaptive logic not started.
Costs / token breakdown	Final report still shows $0.0 because impact not supplied.	⚠️ Needs business-impact calc + report integration.

2 — Status vs. final_polish.md

Checklist item	Status	Notes / next micro-step
P0 - Adaptive numeric anomaly detector	Not started	Currently fixed σ, though external baseline is good.
P0 - Required timeline + impact in finish_investigation	Partially – model not forced to supply them	Add Pydantic required fields + system-prompt note → orchestrator rejects finish lacking ≥3 events.
P0 - Confidence bonus for complete chain	Done	Working as seen (jump to 0.609).
P1 - Tag-substring checks removed	Not started	atomic_tools still has if 'DOOR' in tag.
P1 - Config-driven thresholds	Not started	Still hard-coded.
P1 - generate_chart tool	Not started	
P2 - Unified CLI / ASCII timeline / repo clean	Not started	UX polish.
P3 - Sparkle items	Not started	

So we’re green on the causal-chain logic, yellow on numeric adaptivity, and red on timeline/impact enforcement & polish items.

⸻

3 — Context hand-off summary (for another model or dev)

Project: “Manufacturing Copilot” root-cause agent for freezer data.
Goal: Given a natural-language query (e.g. “What caused the temp spike at 14:30 yesterday?”) the orchestrator plans tool calls, gathers evidence, scores confidence, and ends with a structured finish_investigation JSON.

Current pipeline
	1.	parse_time_range infers ISO window (yesterday 14:00-15:00).
	2.	find_interesting_window keeps or refines the window (binary & numeric heuristics).
	3.	detect_numeric_anomalies (external baseline ± rolling fallback).
	4.	detect_binary_flips (diff-based; adds continuous_high_event fallback).
	5.	test_causality (optional 3-min smoothing for binary cause; uses updated cross_corr).
	6.	Scorer awards bonuses: window (0.10) + anomalies (0.30) + flips (0.10) + causality (0.??) + chain bonus (0.25).
Now consistently > 0.6, avoiding stale-halt.
	7.	At max 7 steps orchestrator calls finish_investigation.

Recent fixes
	•	Serializer handles NumPy types (orjson OPT_SERIALIZE_NUMPY).
	•	Scorer tolerant to {data:{…}} vs flat dict for flips & causality.
	•	continuous_high_event ensured when door already open at window start.
	•	Confidence bonuses tuned -> no more confidence stagnation.
	•	Causality block now accepts smoothed binary series.

Remaining blockers / risks
	•	LLM often omits event_timeline_summary & business_impact_summary.
– Need Pydantic required=True + validator default → reject or auto-stub.
	•	Adaptive numeric σ not implemented; still static 2.5/3.0.
	•	Hard-coded tag-string phrases; no metadata usage.
	•	Thresholds & config externalisation pending.

Suggested next micro-tasks
	1.	Enforce timeline/impact:
Update FinishInvestigationArgs → fields required, min length validator → orchestrator retries on validation error.
	2.	Adaptive anomaly detector:
If first pass finds < N anomalies, auto-halve threshold or switch to rolling Z-score; expose σ in tool args.
	3.	Move ‘DOOR’/’COMPRESSOR’ wording into tag_intel.get_tag_metadata() to make tools generic.
	4.	Optional polish: generate_chart tool & CLI wrapper once accuracy items done.

Total remaining engineering effort ≈ 2-3 focused hours for P0 completion, plus polish as time allows.

⸻

You’re now at a stable, non-halting state with 0.95 final confidence, but timeline/impact still empty.
Lock in the P0 checklist, then decide how much of P1-P3 to tackle before demo.

# Remediation steps

Below is a concrete, code-level action list that resolves every “Needs work / ⚠️ / △” item from the Quick Technical Read-out and will close the remaining P0 gaps.  Nothing here touches P1-P3 polish yet—we stay laser-focused on finishing the “green bar” for your demo.

--------------------------------------------------------------------
1. Finish payload lacks timeline & impact  
--------------------------------------------------------------------
Problem  
• `FinishInvestigationArgs` still marks `event_timeline_summary` and `business_impact_summary` as optional, so the LLM can omit them.  
• The system prompt tells the model to include them, but the orchestrator happily accepts a partial payload.

Fix  
A. Make the two fields required and add stricter validation.  
```python
# src/tool_models.py  (FinishInvestigationArgs)
event_timeline_summary: List[Dict[str, Any]] = Field(..., min_length=3,
    description="Chronological events covering trigger, response, resolution.")
business_impact_summary: Dict[str, Any] = Field(...,
    description="Must include total_cost_usd, energy_cost_usd, product_risk_usd, severity_level.")
```

B. Tighten the validator so the orchestrator aborts & logs an error if either:
   • timeline has < 3 events  
   • any key is missing in the impact dict.  (Raise `ValueError` so the orchestrator will treat it as `status="error"` and force the LLM to retry next step.)

C. In `LLMOrchestrator.BASE_SYS_PROMPT` append:
```
A valid finish_investigation MUST include ≥3 timeline events and a fully-populated business_impact_summary.
Omitting them will be rejected.
```

Effect  
• The run will not terminate at step 7 until the model supplies both structures, or the orchestrator times out.  
• Any finish payload that skips them will bounce back as a validation error, lowering confidence slightly and giving the model another attempt.

--------------------------------------------------------------------
2. Adaptive numeric anomaly detector (still static σ)  
--------------------------------------------------------------------
Goal  
Auto-tighten the threshold if the first pass returns “0 anomalies” so the LLM doesn’t waste a second step.

Implementation sketch (keep it deterministic, “tool neutral & atomic”):

A. Add a new kwarg to `detect_numeric_anomalies`: `auto_tune_sigma: bool = True`.

B. After computing `detected_anomalies_raw`, insert:

```python
if auto_tune_sigma and not detected_anomalies_raw and current_threshold > 1.5:
    # halve threshold (bounded) and recalc once
    new_thresh = max(1.5, current_threshold / 2)
    logger.info(f"No anomalies at σ={current_threshold}. Retrying at σ={new_thresh}.")
    return detect_numeric_anomalies(
        tag, parsed_start_time, parsed_end_time,
        threshold=new_thresh, baseline_window_hours=baseline_window_hours,
        auto_tune_sigma=False  # prevent infinite loop
    )
```

C. Expose `threshold` and `auto_tune_sigma` in `DetectNumericAnomaliesArgs`.

D. Remove the special-case 2.5/3.0 in the orchestrator prompt: the LLM can now rely on the detector’s built-in adaptivity.

--------------------------------------------------------------------
3. Business-impact numbers always zero  
--------------------------------------------------------------------
Root issue  
`calculate_impact` is never invoked automatically; when the model does call it, the orchestrator does not inject its output into the final report.

Solution  
A. Add a cheap post-processing hook in `_generate_final_report`:

```python
# before sending prompt to final LLM
# 1) scan evidence for any calculate_impact result
impact_candidates = [
    json.loads(item["content"])
    for item in self.evidence
    if item["role"]=="function" and item["name"]=="calculate_impact"
]
if impact_candidates:
    top_impact = impact_candidates[-1]  # use most recent
    self.current_estimated_impact = top_impact  # stash for later
```

B. When building the `business_impact_summary` fallback inside the validator, if `self.current_estimated_impact` exists, use its keys instead of zeros.

C. Encourage the LLM to call `calculate_impact` right after detecting a high-severity event by:
   • Adding a 0.05 confidence bonus when a `calculate_impact` result is logged.  
   • Brief note in system prompt: “After causality is established, call calculate_impact on the highest-severity event before finishing.”

--------------------------------------------------------------------
4. Cost / token breakdown shows $0.0  
--------------------------------------------------------------------
Once step 3 is in place, reuse `self.current_estimated_impact["total_cost"]` in the final report generator:

```python
business_impact_summary.setdefault("total_cost_usd",
    self.current_estimated_impact.get("total_cost", 0.0))
```

No new tool calls required—the orchestrator already tracks token cost in `self.estimated_cost_usd`.

--------------------------------------------------------------------
5. Minor housekeeping for logs & scoring  
--------------------------------------------------------------------
• Update `confidence_scorer.score_evidence` to award +0.05 when a successful `calculate_impact` call is present.  
• Add another small penalty −0.05 if the model tries to finish without timeline/impact—helps guide retries.  
• Log “Adaptive σ triggered” so you can demo the new behavior live.

--------------------------------------------------------------------
Testing path
1. Unit-test the new validator (fail < 3 events).  
2. Run `python -m src.llm_orchestrator --query "What caused…"` again; expect:  
   – confidence arc similar, but step 6 (after causality) the LLM calls `calculate_impact`.  
   – Step 7 provides timeline & impact, validator passes, final report has non-zero $$, confidence ≥ 0.9.

--------------------------------------------------------------------
Up-next (once P0 is locked)  
• Replace `'DOOR'`/`'COMPRESSOR'` substrings with metadata look-ups.  
• YAML-driven thresholds.  
• `generate_chart` tool + CLI flags.

That’s everything needed to turn the yellow cells green and hit the demo’s must-have list.
