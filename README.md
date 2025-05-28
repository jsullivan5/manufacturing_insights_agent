# 🏭 Manufacturing Copilot (MCP)

**▶️ [Watch the Demo Video](https://drive.google.com/file/d/1IKtrB1PaNzqVZhUWzbaJhTUO0GjwGtrF/view?usp=sharing)**

> **AI‑powered root‑cause analysis, anomaly detection, and predictive maintenance for industrial time‑series data—delivered through natural‑language conversations.**

---

## 🎯 Objective

Manufacturing Copilot (MCP) turns raw PI‑System tags and other plant telemetry into expert‑level insights in seconds. It employs a sophisticated AI agent that iteratively investigates data anomalies. Given a natural language query (e.g., "Why did the freezer temperature spike?"), the MCP's orchestrator uses an LLM to formulate hypotheses, select appropriate analytical tools from a specialized toolkit, interpret their outputs (which are always structured JSON), and refine its understanding. This cycle continues, building an evidence chain and a confidence score, until a root cause is identified with high certainty or a maximum number of steps is reached. The agent then delivers a clear, actionable report. This process slashes unplanned downtime, breaks down knowledge silos, and frees engineers from hours of manual dashboard forensics.

---

## ❓ Why This Project Exists


| Pain Point (Today)        | MCP Solution (Tomorrow)                            |
|---------------------------|----------------------------------------------------|
| Slow root‑cause hunts     | Answers in minutes via auto‑planned LLM analysis   |
| Alert fatigue             | Context‑aware correlation & prioritised alerts     |
| Knowledge silos           | Embedded domain reasoning for every operator       |
| Reactive maintenance      | Predictive alerts before costly failure            |

*Replace/expand rows as needed.*

---

## 🌍 Global Roadmap & Governance (PoC → 50 Plants)

> **Vision:** “One analytics brain, many plants.”  
> **Approach:** start with a blueprint, then templatize, then federate.

| Roll‑out Wave | Scope              | Key Activities                                                                                                                         | Success KPIs                     |
|---------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| **0. Pilot**       | 1 plant, freezer cell | • Finalize tag standards<br>• Validate cost‑savings model                                                                            | • >95 % RCA accuracy<br>• \$/event |
| **1. Foundation**  | 5 flagship plants  | • Deploy **cloud landing zone** (Azure IoT + Data Lake)<br>• Publish **Analytics Template Package** (MLOps, dashboards, KPIs)<br>• Train local “analytics champions” | • Time‑to‑insight ↓ 70 %<br>• Energy/ton ↓ 5 % |
| **2. Expansion**   | +20 plants         | • Self‑service onboarding wizard<br>• Governance board approves new use‑cases<br>• Quarterly value reviews                             | • Adoption rate↑<br>• Cumulative \$ |
| **3. Full Scale**  | 50 + plants        | • Auto‑benchmark KPIs across sites<br>• Continuous model‑tuning pipeline<br>• OT/IT cybersecurity audits                              | • Global OEE ↑ 2 pts<br>• \$MM savings |

*Governance: Central CoE owns core models & security; local teams own data quality & actions.*

---

## 📈 OEE / Quality / Reliability Dashboards

```
┌────────────────────────── OEE Snapshot (Yesterday) ──────────────────────────┐
│ Availability  Performance  Quality   OEE                                    │
│     91.2 %        85.4 %     99.0 %   77.0 %   ▼ ‑2.1 pts vs. last week      │
└──────────────────────────────────────────────────────────────────────────────┘
```

> *Standardised KPI widgets (OEE, MTBF, MTTR, FPY) auto‑generated from PI/MES data and published to Power BI/Tableau.*  
> *Replace the ASCII block with a real screenshot 🔲 when available.*

---

## 🔄 Change‑Management & Capability Building

1. **Operator Tier** – 2‑hour hands‑on: *“Ask Copilot anything”*, alert handling.  
2. **Engineer Tier** – 1‑day deep dive: tag hygiene, custom tools, MLOps basics.  
3. **Leadership Tier** – exec dashboard & ROI clinic: value heat‑maps, budget alignment.

> **Enablement cadence:** monthly office hours + Teams community of practice.

---

## 💰 Quantified ROI — Quick Calculator

```python
# roi_calculator.py  (demo)
duration_min           = 20      # door-open minutes
energy_kwh_per_min     = 1.2
energy_cost_per_kwh    = 0.12
spoilage_risk_per_min  = 15

energy_cost   = duration_min * energy_kwh_per_min * energy_cost_per_kwh
spoilage_cost = duration_min * spoilage_risk_per_min

print(f"Energy cost:   ${energy_cost:,.0f}")
print(f"Spoilage risk: ${spoilage_cost:,.0f}")
```

> **Example:** a 20‑minute door‑open event costs **\$120 in energy** and exposes **\$300 in spoilage risk**—flagged instantly by Copilot.

---

## ☁️ Cloud & Security Posture

* **Architecture:** PI → **Edge Gateway** → Azure IoT Hub → ADLS Gen2 → Synapse/Databricks → Copilot API  
* **OT/IT Segregation:** one‑way DMZ data diode, MQTT over TLS, *no inbound traffic to plant floor*.  
* **Access Control:** RBAC via Entra ID; row‑level security on multi‑plant datasets.  
* **LLM Security:** Optionally deploy the Copilot’s large‑language‑model endpoint on **Azure OpenAI Service** to keep prompts & telemetry within Microsoft’s enterprise compliance boundary (no data leaves the tenant).
* **Compliance:** ISA‑95 tiers respected; SOC‑2 / ISO 27001 controls mapped.

> *Deploy once, then auto‑provision resource groups per plant with Terraform.*

---

---

## 🚀 Key Capabilities

- **Natural-language diagnostics** – *"Why did the freezer temperature spike yesterday?"*  
- **Multi-signal correlation** – tags, events, anomalies, lag analysis  
- **Expert-level recommendations** – concrete next actions with impact estimates  
- **Extensible toolchain** – plug-in atomic tools for custom analytics  
- **Live PI System connector (optional)** – real‑time streaming analytics

---

## 🛠️ Quick Start

```bash
# 1. Clone & create virtualenv
git clone https://github.com/your-org/manufacturing_mcp && cd manufacturing_mcp
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure OpenAI + optional PI credentials
cp .env.sample .env            # fill in OPENAI_API_KEY etc.

# 4. Generate demo data (optional)
python src/generate_freezer_data.py

# 5. Ask a question!
# Here are specific examples you can run to investigate pre-configured anomalies:

# Example 1: Investigate a freezer temperature spike (simulates door left open)
python src/mcp_cli.py "What caused the freezer temperature spike around 14:30 yesterday?"

# Example 2: Investigate a compressor stoppage (simulates compressor failure)
python src/mcp_cli.py "What caused the compressor to stop running around 02:15 on 05-23-2025?"
```

---

## 🔬 Deep Dive: Code Architecture

Manufacturing Copilot (MCP) is an LLM-driven system designed to investigate and diagnose anomalies in time-series data. Here's a breakdown of the core components:

*   **`src/llm_orchestrator.py` (Main Control Loop):** This is the brain of the MCP.
    *   It takes a natural language query from the user (e.g., "Why did the freezer temperature spike yesterday?").
    *   It uses an LLM (OpenAI's GPT models) in a loop to plan an investigation.
    *   At each step, the LLM decides which analytical tool to use from `atomic_tools.py` based on the current evidence and its understanding of the problem.
    *   It manages the evidence gathered, calculates a confidence score, and decides when to conclude the investigation.
    *   Finally, it generates a comprehensive report summarizing the findings, root cause, and business impact.

*   **`src/tools/atomic_tools.py` (Analytical Toolkit):** This module provides a suite of specialized functions that the orchestrator can call. These tools perform discrete analytical tasks on the data:
    *   `find_interesting_window`: Identifies the most relevant time window for analysis based on data variance or events.
    *   `detect_numeric_anomalies`: Spots unusual spikes or drops in numerical sensor data (e.g., temperature).
    *   `detect_binary_flips`: Detects changes in state for binary sensors (e.g., door open/closed, compressor on/off).
    *   `test_causality`: Analyzes potential cause-and-effect relationships between different sensor readings, considering time lags.
    *   `calculate_impact`: Estimates the business impact (e.g., cost of energy waste, product risk) of an identified event.
    *   `parse_time_range`: Interprets time-related phrases in the user's query to establish an initial analysis window.
    *   Each tool returns structured JSON, allowing the LLM to interpret the results and plan the next step.

*   **`src/glossary.py` (Tag Intelligence):** This module manages information about the PI System tags (the sensor data streams).
    *   It loads a `tag_glossary.csv` file containing metadata for each tag (description, units, category, normal operating ranges, etc.).
    *   It uses semantic search (via OpenAI embeddings and a Chroma vector database) to allow the system to find the most relevant tags based on natural language queries or descriptions. This helps the LLM connect user questions to the correct data streams.

*   **`src/config.py` (Configuration Management):** This file centralizes all configuration settings for the application.
    *   It uses `pydantic-settings` to load API keys, file paths, LLM model names, and operational parameters from environment variables and a `.env` file.
    *   This ensures a clean separation of configuration from code and supports different environments (e.g., development, production).

*   **`src/tool_models.py` (Data Schemas for Tools):** Defines the expected input arguments for each atomic tool using Pydantic models.
    *   This ensures that the LLM provides data in the correct format when requesting a tool execution, improving reliability.
    *   These models are also used to generate JSON schemas for the OpenAI function-calling API.

*   **`src/confidence_scorer.py` (Confidence Calculation):** This module implements the logic to calculate the `confidence_score` based on the accumulated evidence. The score reflects how certain the system is about its findings.

*   **`src/generate_freezer_data.py` (Demo Data Generation):** A utility script to create realistic mock time-series data for a freezer system.
    *   It simulates normal operation (temperature cycles, door openings, compressor activity) and injects various anomalies (prolonged door open, compressor failure, sensor malfunctions).
    *   This allows the MCP to be demonstrated without needing a live connection to a PI System. The output is `data/freezer_system_mock_data.csv`.

## 🏃 Running the Demo

To see the Manufacturing Copilot in action:

1.  **Ensure Prerequisites:**
    *   Python 3.11.9 (as specified in `.python-version`)
    *   Virtual environment activated.

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    *   Copy `.env.sample` to `.env`.
    *   Fill in your `OPENAI_API_KEY` in the `.env` file. Other settings can typically remain as default for the demo.
    ```bash
    cp .env.sample .env
    # Then edit .env to add your API key
    ```

4.  **Generate Demo Data:**
    This script creates `data/freezer_system_mock_data.csv` which contains simulated freezer sensor data with pre-injected anomalies.
    ```bash
    python src/generate_freezer_data.py
    ```

5.  **Run Investigations:**
    You can now ask the MCP to investigate events. The `src/mcp_cli.py` script is the command-line interface. Here are the test cases from `demo.txt`:

    *   **Scenario 1: Freezer Temperature Spike**
        This query targets an event where the freezer door was left open.
        ```bash
        python src/mcp_cli.py "What caused the freezer temperature spike around 14:30 yesterday?"
        ```

    *   **Scenario 2: Compressor Failure**
        This query targets a simulated compressor failure.
        ```bash
        python src/mcp_cli.py "What caused the compressor to stop running around 01:45 UTC on 05-24-2025? Consider data until 03:30 UTC"
        ```
        *(Note: The date "05-23-2025" in the query is relative to when the demo data was generated. The `generate_freezer_data.py` script sets up anomalies based on the current date when it's run. The `parse_time_range` tool and the LLM are designed to interpret such date references correctly against the data's actual timestamp range.)*

    Many of the individual Python modules in `src/` (like `glossary.py`, `atomic_tools.py`, `generate_freezer_data.py`) can also be run directly (e.g., `python src/glossary.py`) to see their specific functionalities in action or for debugging.

---

## 🤖 AI-Assisted Development & Process Notes

This project was developed with the assistance of AI coding tools, and some artifacts of that process are included for demonstration purposes:

*   **`.documentation/` Folder:**
    *   `prompts/`: May contain examples of prompts used to guide the LLM during development or for specific agent behaviors.
    *   `notepad/`: Contains planning documents, checklists, or scratchpad notes related to feature development, often generated in collaboration with an AI assistant. (For instance, `feature-name.md` files).
*   **`.cursor/rules/` Folder:** Contains custom instructions (`.mdc` files) used to guide the AI assistant (Gemini 2.5 Pro in this case) to adhere to specific coding standards, project goals, and architectural principles during development.

Typically, such development process artifacts might not be included in a final demo repository, but they are present here to showcase an AI-augmented development workflow and how an "AI pair programmer" can be leveraged effectively.

---

## 🔮 Future Enhancements

While MCP demonstrates a powerful new paradigm for manufacturing analytics, several areas offer exciting possibilities for future development:

*   **Advanced Temporal Reasoning:**
    *   **Contextual Date/Time Disambiguation:** Implement more robust heuristics within `parse_time_range` or a dedicated pre-processing step to better handle ambiguous user queries like "last Tuesday morning" or "during the night shift two days ago," especially when combined with the LLM's natural language understanding. This could involve more sophisticated relative date calculations and awareness of typical operational calendars (shifts, weekends).
    *   **LLM-Guided Time Window Refinement:** Allow the LLM to more actively guide the refinement of the `investigation_window` beyond just the initial `find_interesting_window` call, perhaps by suggesting shrinking or expanding the window based on intermediate findings, or even shifting it if initial anomalies prove to be red herrings.

*   **Dynamic Anomaly Detection Thresholds:**
    *   **Adaptive Baselines:** Enhance `detect_numeric_anomalies` to learn or dynamically adjust baseline statistics (mean, std dev) based on longer historical periods or different operational states (e.g., product changeovers, maintenance periods) rather than just a fixed lookback window.
    *   **Context-Aware Sensitivity:** Allow the `threshold` for anomaly detection to be influenced by tag metadata (e.g., known criticality, typical volatility) or even by the LLM's current hypothesis (e.g., "looking for subtle leading indicators" might use a more sensitive threshold).

*   **Orchestrator and Control Flow Hardening:**
    *   **Smarter Tool Parameterization:** Enable the LLM to suggest more nuanced parameters for tools based on prior results (e.g., suggesting a specific `max_lag_minutes` for `test_causality` if a related event was found at a certain offset).
    *   **Meta-Reasoning & Backtracking:** Introduce capabilities for the orchestrator to recognize unproductive investigation paths and explicitly backtrack or try alternative hypotheses if confidence stagnates for too long, rather than just relying on the `max_stale_steps` counter. This could involve the LLM reflecting on the evidence log and proposing a strategic shift.
    *   **Cost-Based Tool Selection:** If multiple tools could provide similar information, allow the LLM to consider estimated token costs (or even a proxy for computational cost) in its selection process, especially for less critical steps.

*   **Scalable Vector Database for Tag Glossary:**
    *   **Persistent & Scalable DB:** Migrate the `TagGlossary` from the in-memory ChromaDB to a more robust, persistent, and scalable vector database solution (e.g., PostgreSQL with pgvector, dedicated cloud-based vector DBs). This would support much larger tag sets and allow for easier updates and management of the tag embeddings.
    *   **Automated Embedding Updates:** Implement a pipeline to automatically update tag embeddings if their descriptions in the glossary source (e.g., `tag_glossary.csv` or a P&ID system) change.

*   **Enhanced Semantic Tag Search & Linking:**
    *   **Multi-Modal Tag Association:** Extend tag search to potentially incorporate information from P&ID diagrams directly (if OCR/Vision capabilities were added), linking visual context to tag metadata.
    *   **Relationship Inference:** Develop methods to infer relationships *between tags* (e.g., "Tag A is an input to process controlled by Tag B") directly from documentation or data patterns, and make this information available to the LLM.
    *   **Synonym & Abbreviation Handling:** Improve the semantic search to be more resilient to common industrial abbreviations or alternative naming conventions for similar types of equipment or measurements.

*   **Tool & Schema Evolution:**
    *   **Automated Tool Schema Versioning:** Implement a more automated way to manage and communicate `schema_version` changes for tools to the LLM, perhaps by including versioning directly in the dynamically generated `tool_schemas` if the LLM could reliably use it.
    *   **User-Defined Tools:** Create a framework where users could more easily define and register their own custom atomic tools with Pydantic schemas, making the MCP more extensible for specific plant needs.
