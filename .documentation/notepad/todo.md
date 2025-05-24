

# MCP CLI Demo – Implementation Plan

## Objective
Build a command-line prototype of a Manufacturing Copilot (MCP) that accepts natural language questions about operational performance (e.g., "Why did Freezer A use more power last night?"), processes synthetic time-series data, and returns insights using summarization, anomaly detection, and simple LLM-based reasoning.

## High-Level Flow
1. User enters question via CLI
2. System identifies relevant tags
3. Retrieves relevant data
4. Summarizes metrics and detects spikes
5. Correlates changes across related metrics
6. Returns human-readable insight and chart

---

## 1. CLI Shell

- Use Python’s `argparse` or a basic `input()` loop for MVP
- Command: `python mcp.py "Why did Freezer A use more power last night?"`
- Output: Summary paragraph + chart filepath

---

## 2. Synthetic Data Generation

### Goals
- Simulate realistic freezer operations across 3–5 days
- Include hourly resolution data
- Include an interpretable spike caused by plausible operational changes

### Metrics to Include
- `freezer_power_kwh`: Energy consumption (base ~300 ± 5 kWh/hour)
- `freezer_temp_c`: Internal temperature (target ~-18°C ± 0.5)
- `freezer_temp_setpoint_c`: Setpoint (e.g., constant -20°C)
- `compressor_cycles`: Integer cycles (Poisson-distributed, λ = 6)

### Anomaly Injection
- Introduce a spike in energy usage (e.g. +30 kWh) for 5 hours
- During spike, adjust:
  - Compressor cycles ↑
  - Setpoint ↓ (simulating overcooling)
- Add slight ambient temp drift to support plausibility

### Tools
- Use `pandas` and `numpy` to generate and inject data
- Save as CSV: `synthetic_freezer_data.csv`

---

## 3. Tag Glossary + Embeddings

- JSON file with tag metadata:
```json
{
  "freezer_power_kwh": {
    "description": "Hourly power usage for Freezer A",
    "category": "energy",
    "unit": "kWh"
  },
  ...
}
```
- Use OpenAI or SentenceTransformers to create embeddings for tag matching
- At runtime: embed user query, find top-N relevant tags via cosine similarity

---

## 4. Tool Functions

- `load_data(tag, start, end)`: Read CSV, filter by tag + time
- `summarize_metric(df)`: mean, std, min, max, trend
- `detect_spike(df)`: simple z-score or delta threshold
- `correlate_tags(primary_tag, others, window)`: rank tag deltas
- `generate_chart(df, tag)`: matplotlib line plot to PNG

---

## 5. Agent Loop

- Parse user query
- Match to relevant tags via embeddings
- Call tools in sequence:
  1. Load and summarize primary tag
  2. Detect spike and time range
  3. Summarize related tags during spike window
  4. Send summary to LLM for reasoning
  5. Return explanation and chart

---

## 6. Output

- Markdown-style summary:
  ```
  Freezer A energy usage spiked 22% on May 21 between 2–6am. 
  Compressor cycles increased and temperature setpoint dropped during the same window, indicating possible overcooling behavior.
  ```
- PNG chart (power vs time, spike annotated)

---

## Stretch Features

- Multi-freezer comparison
- Streamlit UI
- Baseline vs current trend comparison
- Save Q&A log to markdown file

---

## Next Steps

- [ ] Generate synthetic dataset
- [ ] Build CLI parser
- [ ] Build `summarize_data` and `detect_spike`
- [ ] Create chart renderer
- [ ] Add LLM reasoning layer
- [ ] Assemble full pipeline
- [ ] Demo against real use case