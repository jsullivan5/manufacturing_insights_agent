## 🎯 **AI Detective Case Resolution TODO List**

### **🔧 Core Issues to Fix**

#### **1. Agent Stalling at 35% - Root Cause Analysis**
- [ ] **Problem**: Agent finds door anomalies but never connects them to temperature spikes
- [ ] **Missing Link**: No time-lagged correlation testing between cause → effect
- [ ] **Current Flow**: Generic stats → Global correlations → Door anomalies (but no causal link)
- [ ] **Need**: Window-based lag correlation to connect the dots

#### **2. Hard-coded Thresholds Problem**
- [ ] **Problem**: Using fixed `threshold = 1.0` for door sensors
- [ ] **Solution**: Dynamic threshold calculation per tag using MAD/IQR
- [ ] **Benefit**: Adaptive sensitivity based on actual data distribution

---

### **🛠 Implementation Priority (Ordered by Impact)**

#### **🏆 Priority 1: Window-Based Lag Correlation Tool**
- [ ] **Function**: `cross_corr(tag_a, tag_b, window, max_lag)` (~40 LOC)
- [ ] **Purpose**: Test specific time windows around anomalies with lags 0-10 minutes
- [ ] **GPT Prompt**: "Check door vs temp in 30-min window around spike, lags 0-10 min"
- [ ] **Impact**: Enables causal connection discovery → confidence jumps to 70%+

#### **🥈 Priority 2: Change-Point Detector Tool**
- [ ] **Function**: Reuse existing change-point code from `causal_analysis.py`
- [ ] **Purpose**: Produces explicit "14:30 temp shift" events GPT can pivot on
- [ ] **Impact**: Gives GPT specific timestamps to investigate
- [ ] **Effort**: Copy/paste existing code

#### **🥉 Priority 3: Evidence Memory System**
- [ ] **Implementation**: Store each tool call + JSON result in `state["evidence"]`
- [ ] **Purpose**: GPT can reference prior facts and build on previous findings
- [ ] **Impact**: Prevents redundant investigations, enables building narrative
- [ ] **Effort**: ~10 LOC in investigation loop

#### **4. Dynamic Threshold Calculator**
- [ ] **Function**: `dynamic_sigma(series)` using MAD/IQR (~15 LOC)
- [ ] **Purpose**: Compute adaptive threshold per tag vs hard-coded 1.0
- [ ] **Impact**: Better anomaly detection sensitivity per tag type

#### **5. Confidence Bump Rules**
- [ ] **Rule**: +30% confidence when cause precedes effect with correlation > 0.5
- [ ] **Rule**: +20% confidence when time lag makes physical sense
- [ ] **Impact**: Rewards finding actual causal relationships

---

### **🔄 Enhanced Investigation Loop**

#### **Current Loop** (3 steps, generic):
```
for step in range(3):
    plan = ask_gpt(query)
    result = exec_tool(plan)
    if confidence < 70%: continue
```

#### **Target Loop** (6 steps, evidence-driven):
```
for step in range(6):
    plan = ask_gpt(state)                      # GPT decides next probe
    result = exec_tool(plan["tool"], **plan["args"])
    state["evidence"].append({"plan": plan, "result": result})
    conf = score_confidence(state["evidence"]) # Dynamic confidence scoring
    log_step(plan, result, conf)
    if conf >= 90%: break  # Solved!
```

---

### **🧠 Enhanced GPT Prompts**

#### **System Prompt Addition**:
- [ ] **Tools Available**: `cross_corr`, `detect_change_points`, `dynamic_threshold`
- [ ] **Strategy Guidance**: "If global correlation fails, zoom into 1-hour window around anomalies"
- [ ] **Goal**: "Keep investigating until confidence ≥ 90% or 6 steps"

#### **Evidence-Driven Prompting**:
- [ ] Include previous findings in each step
- [ ] Guide GPT to test specific time windows
- [ ] Encourage lag correlation testing when anomalies found

---

### **🎯 Target Detective Flow (Post-Fix)**

#### **Expected Investigation Sequence**:
1. **Step 1** (25%): Basic stats → finds temperature range
2. **Step 2** (35%): Detect change-points on TEMP → finds 14:30 spike  
3. **Step 3** (78%): `cross_corr(DOOR.STATUS, TEMP, window=14:00-15:00, max_lag=10)` → r=0.88 @ 4min lag
4. **Step 4** (93%): Compute deltas during door opening period → **Resolution Criteria Met**

---

### **🚀 Quick Implementation Patches**

#### **Ready-to-Drop Code Snippets**:
- [ ] `dynamic_sigma()` function using MAD calculation
- [ ] `cross_corr()` function for lag correlation testing  
- [ ] Evidence state management in investigation loop
- [ ] Enhanced GPT prompts with tool guidance

#### **Integration Points**:
- [ ] Add new tools to `_execute_investigation_step()`
- [ ] Update GPT prompts to mention new capabilities
- [ ] Implement confidence scoring based on evidence strength

---

### **✅ Success Criteria**

#### **Agent Behavior**:
- [ ] Reaches 90%+ confidence on door opening scenarios
- [ ] Makes causal connections (not just finds isolated anomalies)
- [ ] Uses evidence from previous steps to guide next investigations
- [ ] Adapts thresholds per tag type automatically

#### **Demonstration Objectives**:
- [ ] **Illustrate confidence score progression** (e.g., 25% → 45% → 78% → 93% confidence)
- [ ] **Demonstrate identification of significant correlations** (e.g., 0.88 between door and temperature).
- [ ] **Present findings in business-relevant terms** (e.g., 'Door open event duration: 19 minutes; estimated energy impact: $47').
- [ ] **Highlight the system's capability to identify potential causal relationships** through data analysis.

Implementing lag correlation, evidence memory, and dynamic thresholds is projected to enable the agent to consistently reach higher confidence levels (e.g., >90%) based on the analytical process.
