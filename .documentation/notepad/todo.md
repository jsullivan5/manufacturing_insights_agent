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

- Use Python's `argparse` or a basic `input()` loop for MVP
- Command: `python mcp.py "Why did Freezer A use more power last night?"`
- Output: Summary paragraph + chart filepath

---

## 2. Synthetic Data Generation ✅ COMPLETED

### Goals ✅
- Simulate realistic freezer operations across 3–5 days → **ACHIEVED: 7-day dataset**
- Include hourly resolution data → **ACHIEVED: 1-minute resolution (50,400 data points)**
- Include an interpretable spike caused by plausible operational changes → **ACHIEVED: 4 different anomaly types**

### Metrics to Include ✅
- `freezer_power_kwh`: Energy consumption (base ~300 ± 5 kWh/hour) → **IMPLEMENTED as FREEZER01.COMPRESSOR.POWER_KW**
- `freezer_temp_c`: Internal temperature (target ~-18°C ± 0.5) → **IMPLEMENTED as FREEZER01.TEMP.INTERNAL_C**
- `freezer_temp_setpoint_c`: Setpoint (e.g., constant -20°C) → **IMPLICIT in control logic**
- `compressor_cycles`: Integer cycles (Poisson-distributed, λ = 6) → **IMPLEMENTED as FREEZER01.COMPRESSOR.STATUS**

### Anomaly Injection ✅
- Introduce a spike in energy usage (e.g. +30 kWh) for 5 hours → **ACHIEVED: Multiple anomaly types**
- During spike, adjust: → **ACHIEVED with realistic physics**
  - Compressor cycles ↑
  - Setpoint ↓ (simulating overcooling)
- Add slight ambient temp drift to support plausibility → **ACHIEVED: Daily temperature cycles**

### Tools ✅
- Use `pandas` and `numpy` to generate and inject data → **IMPLEMENTED**
- Save as CSV: `synthetic_freezer_data.csv` → **COMPLETED: data/freezer_system_mock_data.csv**

### **COMPLETED FEATURES:**
- ✅ **AVEVA PI System Compatible Format**: Long format CSV with proper schema
- ✅ **Realistic Operational Physics**: Temperature response, compressor cycling, heat transfer
- ✅ **Shift-Based Operations**: Day (8 AM-8 PM) vs Night (8 PM-8 AM) patterns
- ✅ **Modular Anomaly Injection**: 4 different anomaly types can be toggled
  - Prolonged door open (19 minutes on 01/16 14:30)
  - Compressor failure (55 minutes on 01/18 02:15)  
  - Sensor flatline (4 hours on 01/19 16:00, marked as "Questionable" quality)
  - Power fluctuations (25 minutes on 01/20 11:45)
- ✅ **Data Quality Indicators**: Good vs Questionable quality flags
- ✅ **Comprehensive Verification**: Automated anomaly detection and visualization

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

- [x] Generate synthetic dataset
- [x] Build comprehensive data verification
- [x] Validate realistic operational patterns  
- [x] Confirm anomaly injection works correctly
- [x] Create visualization capabilities
- [ ] Build CLI parser
- [ ] Build `summarize_data` and `detect_spike`
- [ ] Create chart renderer
- [ ] Add LLM reasoning layer
- [ ] Assemble full pipeline
- [ ] Demo against real use case

**DATASET STATUS: ✅ PRODUCTION READY**
- 50,400 data points across 7 days
- AVEVA PI System compatible format
- 4 realistic anomaly scenarios for MCP demonstrations
- Automated verification and visualization capabilities