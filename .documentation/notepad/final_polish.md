

# 🏁 Home‑Stretch Checklist

> Ordered by **importance vs. effort** so you can stop at any point and still have a great demo.

---

## P0 — Must‑Have Accuracy

- [ ] **Adaptive numeric anomaly detector**  
  - Baseline outside analysis window or rolling Z‑score  
  - Expose `threshold` so the LLM can lower σ on first pass
  - Make anomaly detector adaptive• If first pass finds 0 anomalies, auto-lower σ and/or use rolling-mean z-score.• Return severity (# σ) so scorer can weight bigger spikes
  - Eliminates repeated 2.5 → 2.0 retries; finds the 3 °C jump first time.
- [ ] **Required timeline + impact in `finish_investigation`**  
  - Make `event_timeline_summary` & `business_impact_summary` *required* in the Pydantic model  
  - Add system‑prompt note: "Provide *all* keys; timeline needs ≥3 events"
  - Require timeline + impact in finish_investigation• Update FinishInvestigationArgs → event_timeline_summary: List[EventItem] (time, tag, desc, duration?) required.• Add to SYS_PROMPT: "A valid finish call must contain 3+ timeline events covering trigger, response, resolution."
  - Fills empty timeline & gives execs a CSI-style story.
- [ ] **Confidence bonus for complete causal chain**  
  - +0.15 when evidence has *(window ∧ anomaly ∧ binary flip ∧ causality)* in same window
  - Confidence bonus for complete chain+0.15 when evidence has (window ∧ anomaly ∧ binary flip ∧ causality) in same window.
  - Pushes model over 0.9 without random retries.

---

## P1 — Tool Robustness & Output

- [ ] Remove tag‑substring checks (`'DOOR'`, `'COMPRESSOR'`) ← use tag metadata  
    - Remove tag-string if/else in atomic tools• Replace if 'DOOR' in tag etc. with:meta = get_tag_metadata(tag)['category'] → map to generic phrases ("binary state high/low").• Move any default thresholds to config/thresholds.yaml.
    - Hard-code free, new tags drop in cleanly.
- [ ] Move hard‑coded thresholds to `config/thresholds.yaml`
- [ ] **Chart tool** `generate_chart`
  - Args: `tag, start_time, end_time, overlay_tags?`  
  - Save PNG, return file path  
    - Chart generation tool (generate_chart)Args: {tag, start_time, end_time, overlay_tags?}.Tool saves PNG via matplotlib (no colors explicit) and returns filepath.
    - Visual "wow"; cheap add.
- [x] **LLM time parsing**  
  - Add `parse_time_range` tool (LLM fills `start_time/end_time`) for "yesterday afternoon" cases


---

## P2 — User Experience Polish

- [ ] Unified CLI (`mcp`) with sub‑commands  
  - `mcp rc "<query>"` – root‑cause  
  - `mcp chart TAG --last 4h` – quick chart  
  - `--verbose` flag hides debug by default
    - Unified CLI (mcp)mcp rc "…", mcp chart TAG --last 4h, mcp explain STEP. Hide debug behind --verbose.
    - ready demo; no scrolling logs.
- [ ] ASCII timeline fallback if LLM omits events
    - Pretty ASCII timeline in final report (decouple from LLM timeline) for fallback.
    - Guarantees timeline even if LLM omits.
- [ ] Clean repo: delete dead "blip" functions, run `ruff` + `black`
    - Clean repodelete dead "blip" funcs, move notebooks to /sandbox, run ruff + black.
    - Avoid code-review nitpicks.

---

## P3 — Extra Sparkle (Nice‑to‑Have)

- [ ] Secondary‑correlation hunt after root cause (reward +0.05 each)
    - Secondary-correlation huntAfter root cause ≥0.9, let agent do 2 bonus steps: search other numeric tags for r>
    - 0.5: within window; append "Additional contributing factors". Reward +0.05 per find.
- [ ] Cost breakdown (token & $$) in final summary  
    - Cost breakdown in summary (tokens + est-USD)
    - Impresses with transparency.
- [ ] Live "thought streaming" with `rich.live`
    - Optional live-mode flag to stream thoughts (rich.live)
    - visually impressive

---

Estimated total effort: **≈6 focussed hours**

*Finish P0 for guaranteed success; P1 makes it future‑proof; P2‑P3 add wow‑factor.*