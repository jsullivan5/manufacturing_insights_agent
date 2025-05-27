# Final Polish and Refinement Checklist

> Tasks ordered by importance and effort to facilitate incremental improvements for demonstration purposes.

---

## P0 — Core Accuracy Enhancements

- [ ] **Adaptive numeric anomaly detector**  
  - Baseline outside analysis window or rolling Z‑score  
  - Expose `threshold` so the LLM can lower σ on first pass
  - Make anomaly detector adaptive: If first pass finds 0 anomalies, auto-lower σ and/or use rolling-mean z-score. Return severity (# σ) so scorer can weight bigger spikes.
  - *Benefit:* Improves efficiency by reducing repeated retries (e.g., 2.5 → 2.0); aims to identify significant deviations (e.g., a 3°C jump) more directly.
- [ ] **Required timeline + impact in `finish_investigation`**  
  - Make `event_timeline_summary` & `business_impact_summary` *required* in the Pydantic model  
  - Add system‑prompt note: "Provide *all* keys; timeline needs ≥3 events"
  - Require timeline + impact in `finish_investigation`: Update `FinishInvestigationArgs` → `event_timeline_summary: List[EventItem]` (time, tag, desc, duration?) required. Add to SYS_PROMPT: "A valid finish call must contain 3+ timeline events covering trigger, response, resolution."
  - *Benefit:* Ensures comprehensive and structured reporting, providing a clear narrative of events.
- [ ] **Confidence bonus for complete causal chain**  
  - +0.15 when evidence has *(window ∧ anomaly ∧ binary flip ∧ causality)* in same window
  - Confidence bonus for complete chain: +0.15 when evidence has (window ∧ anomaly ∧ binary flip ∧ causality) in same window.
  - *Benefit:* More reliably guides the model towards the target confidence level based on evidence.

---

## P1 — Tool Robustness & Output Quality

- [ ] Remove tag‑substring checks (`\'DOOR\'`, `\'COMPRESSOR\'`) ← use tag metadata  
    - Remove tag-string if/else in atomic tools: Replace `if \'DOOR\' in tag` etc. with `meta = get_tag_metadata(tag)[\'category\']` → map to generic phrases ("binary state high/low"). Move any default thresholds to `config/thresholds.yaml`.
    - *Benefit:* Enhances maintainability and scalability by allowing new tags to be integrated without code modification.
- [ ] Move hard‑coded thresholds to `config/thresholds.yaml`
- [ ] **Chart tool** `generate_chart`
  - Args: `tag, start_time, end_time, overlay_tags?`  
  - Save PNG, return file path  
    - Chart generation tool (`generate_chart`): Args: `{tag, start_time, end_time, overlay_tags?}`. Tool saves PNG via matplotlib (no explicit colors) and returns filepath.
    - *Benefit:* Provides clear visual representation of data, enhancing analytical capabilities.
- [x] **LLM time parsing**  
  - Add `parse_time_range` tool (LLM fills `start_time/end_time`) for "yesterday afternoon" cases

---

## P2 — User Experience Polish

- [ ] Unified CLI (`mcp`) with sub‑commands  
  - `mcp rc "<query>"` – root‑cause  
  - `mcp chart TAG --last 4h` – quick chart  
  - `--verbose` flag hides debug by default
    - Unified CLI (`mcp`): `mcp rc "query"`, `mcp chart TAG --last 4h`, `mcp explain STEP`. Hide debug behind `--verbose`.
    - *Benefit:* Streamlines interaction and improves presentation clarity by reducing log verbosity by default.
- [ ] ASCII timeline fallback if LLM omits events
    - Pretty ASCII timeline in final report (decouple from LLM timeline) for fallback.
    - *Benefit:* Ensures a timeline is always present in the output.
- [ ] Clean repo: delete dead "blip" functions, run `ruff` + `black`
    - Clean repo: delete unused functions, move notebooks to `/sandbox`, run `ruff` + `black`.
    - *Benefit:* Improves code quality and maintainability.

---

## P3 — Additional Enhancements (Optional)

- [ ] Secondary‑correlation hunt after root cause (reward +0.05 each)
    - Secondary-correlation hunt: After root cause ≥0.9, let agent do 2 bonus steps: search other numeric tags for r > 0.5 within window; append "Additional contributing factors". Reward +0.05 per find.
- [ ] Cost breakdown (token & $$) in final summary  
    - Cost breakdown in summary (tokens + est-USD)
    - *Benefit:* Offers transparency regarding resource utilization.
- [ ] Live "thought streaming" with `rich.live`
    - Optional live-mode flag to stream thoughts (`rich.live`)
    - *Benefit:* Offers a dynamic view of the agent's decision-making process.

---

Estimated total effort: **≈6 focussed hours**

*Completing P0 items will ensure core functionality. P1 items enhance robustness. P2 and P3 items improve user experience and add advanced features.*