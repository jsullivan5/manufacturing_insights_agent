- [ ] Extend TagGlossary to include value_type (numeric/binary) and auto-computed expected_range (quantile based) for numeric tags
- [ ] Build data-pipeline job to compute baseline statistics (mean, std, 5th/95th quantiles) per tag at startup
- [ ] Create/standardize atomic tools:
  - detect_numeric_anomalies (wrap existing detect_spike)
  - detect_binary_flips
  - change_points (timestamp events)
  - lag_corr (wrap cross_corr)
  - delta (value change across window)
  - energy_cost (impact calculator)
- [ ] Implement tool_router that chooses tool based on tag metadata and investigation need
- [ ] Design deterministic confidence_scoring module combining |r|, lag, anomaly severity, sequence order
- [ ] Implement RootCauseAgent with GPT function-calling loop, state.evidence memory, max 6 steps, auto-stop at confidence ≥ 0.9
- [ ] Integrate agent into CLI (src/mcp.py or new command) with streaming thoughts & evidence log
- [ ] Add placeholder constants for $/kWh and product loss; expose via config/env.py
- [ ] Update README and inline docstrings to explain inspection pipeline
- [ ] (Optional) Lightweight Streamlit dashboard to visualize timeline & chart 