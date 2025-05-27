# Manufacturing Copilot - LLM-Powered Causal Analysis Agent

**Goal**: Develop an LLM-driven agent that performs causal analysis on manufacturing data.

## **Vision: LLM-Orchestrated Analytical Process**

**Illustrative Current-State Analysis:**
```
"Moderate correlation identified between door status and temperature."
```

**Targeted Analytical Output Example:**
```
System Log: Agent initiated analysis of temperature anomaly.

Analysis Step 1 (Confidence: 32%)
Agent Log: "Checking for operational events around the time of the spike."
Tool Call: `detect_change_points()` for door status.
Result: Door opening detected at 14:30:00.

Analysis Step 2 (Confidence: 67%)
Agent Log: "Door opening detected at 14:30. Assessing correlation with temperature."
Tool Call: `calculate_time_lagged_correlations()`.
Result: 0.191 correlation at 1-minute lag.

Analysis Step 3 (Confidence: 96%)
Agent Log: "Significant correlation found. Door open (19 minutes) likely contributed to 1.6°C rise."
Tool Call: `assess_business_impact()`.
Result: Estimated energy impact: $47, potential product quality risk.

Conclusion: Root cause identified based on evidence chain.
The system iteratively built confidence through sequential analysis.
```

---

## **Implementation Plan**

### **Phase 1: LLM Agent Orchestration (Est. 2-3 hours)**

#### **1.1 LLM Orchestrator Agent**
- **Core Concept**: An LLM orchestrates the analytical process, deciding which tools to call and when.
- **Key Behavior**: The system's confidence in a hypothesis increases as it gathers corroborating evidence.
- **Tools Available**: A suite of simple, single-purpose utility functions.

**Utility Tool Function Signatures (Illustrative):**
```python
def get_change_points(tag, time_window) -> List[Dict]:
    # Detects significant changes in data series.
    
def get_correlation(tag1, tag2, lag_minutes) -> float:
    # Calculates correlation between two data series.
    
def get_business_impact(event_type, duration) -> str:
    # Estimates potential business impact of an event.
```

#### **1.2 Iterative Analysis Loop**
- **LLM Decision**: Determines the next analytical step or tool.
- **Confidence Tracking**: Each step updates a quantitative confidence score.
- **Tool Selection**: Agent chooses tools based on prior findings and context.
- **Process Logging**: Log the LLM's reasoning steps for transparency.

### **Phase 2: Agent Behavior & Output Refinement (Est. 1-2 hours)**

#### **2.1 Agent Process Transparency**
- **System Narrative**: Log agent actions and inferences (e.g., "Initiating investigation...", "Anomaly detected..."). Maintain a neutral tone.
- **Progressive Confidence**: Clearly show how the confidence score (e.g., 32% → 67% → 96%) evolves with new evidence.
- **Tool Rationale**: Log the reasons for selecting each tool.
- **Key Findings**: Clearly indicate when significant evidence supports a hypothesis.

#### **2.2 Structured Output Generation**
- **Business-Relevant Information**: Include potential costs, risks, and actionable insights, not just technical metrics.
- **Confidence Score Presentation**: Display the confidence progression.
- **Reasoning Log**: Provide access to the LLM's thought process.
- **Final Summary**: Clearly state the identified root cause and associated business context.

---

## **Refocused TODO List**

### **LLM Agent Core**

#### **LLM Orchestrator Agent**
- [ ] Create `src/llm_orchestrator.py` (or similar) for the main agent.
- [ ] Implement `run_analysis()` (or similar) for the main analysis loop.
- [ ] Add confidence tracking and updating logic.
- [ ] Develop system prompts to guide the LLM's analytical process.
- [ ] Include logging for tool selection rationale.

#### **Utility Tools** (To replace or supplement `causal_analysis.py` if it's too monolithic)
- [ ] `get_change_points()` - Basic event detection (target: ≤20 LOC).
- [ ] `get_correlation()` - Simple correlation calculation (target: ≤15 LOC).
- [ ] `get_business_impact()` - Cost estimation (target: ≤10 LOC).
- [ ] `get_time_series_stats()` - Basic statistics (target: ≤10 LOC).

#### **Agent Integration**
- [ ] Integrate the agent with the primary application entry point.
- [ ] Add logging for LLM reasoning steps.
- [ ] Implement and display confidence progression.
- [ ] Add logging for tool call justifications.

### **Output and Presentation**

#### **Agent Process Logging**
- [ ] Log key decision points and inferences.
- [ ] Display progressive confidence scores.
- [ ] Log LLM reasoning steps with a clear prefix (e.g., "LLM_REASONING:").
- [ ] Clearly flag significant findings.
- [ ] Summarize business impact in clear terms.

#### **Demonstration/Reporting Elements**
- [ ] Show clear confidence progression.
- [ ] Quantify potential business impacts (e.g., energy costs, quality risks).
- [ ] Provide actionable insights or recommendations derived from the analysis.
- [ ] Prepare a demonstration script that showcases the system's problem-solving process.

### **Validation**

#### **Agent Behavior Tests**
- [ ] Test confidence score progression (low → high) based on evidence.
- [ ] Verify logical tool selection.
- [ ] Validate business impact calculations against defined parameters.
- [ ] Confirm clarity and neutrality of logged agent narrative.

---

## **Success Criteria (Refocused)**

### **System Performance & Demonstration**
- [ ] The LLM should guide the analytical process step-by-step.
- [ ] Confidence score should increase from initial low values (e.g., <40%) to high values (e.g., >90%) as evidence accumulates.
- [ ] The system should log its rationale for tool selection (e.g., "Anomaly detected in TagX, checking related TagY for correlations...").
- [ ] Key causal links should be clearly identified and reported.
- [ ] **The demonstration should highlight the system's automated analytical process, not just static data outputs.**

### **Output Quality**
- [ ] System outputs effectively communicate the identified issues and their context.
- [ ] Potential business impacts are quantified (e.g., in monetary terms where applicable).
- [ ] The system demonstrates a structured approach to root cause analysis.

---

## **Project Timeline (Illustrative)**

### **Phase 1 (Est. 2-3 hours)**
- Develop the LLM agent orchestrator.
- Create or adapt simple utility tools.
- Implement confidence progression and reasoning logging.

### **Phase 2 (Est. 1-2 hours)**
- Refine agent process logging and narrative.
- Add business impact calculation and reporting.
- Test and iterate on agent behavior and output.

### **Documentation and Demonstration Preparation**
- Prepare materials to clearly explain the system's workflow and capabilities.
- Focus on showcasing the automated, LLM-driven analytical process.

---

## **Key System Characteristics (Refocused)**

1. **LLM-Driven Analysis**: The LLM orchestrates the investigation, selecting tools and interpreting results.
2. **Progressive Confidence**: The system's confidence in its findings evolves based on accumulated evidence.
3. **Business-Contextualized Output**: Results are presented with relevant business implications.
4. **Automated Tool Orchestration**: The LLM determines the sequence of analytical tools.
5. **Transparent Reasoning**: The LLM's decision-making process is logged.
6. **Advanced Analytical Capability**: The system aims to provide a sophisticated approach to automated data analysis.

**The objective is to demonstrate an advanced, LLM-orchestrated analytical capability.**