# 🏭 Manufacturing Copilot (MCP) - AI-Powered Industrial Intelligence

> **Next-generation manufacturing analytics that transforms time-series data into expert-level operational insights**

Manufacturing Copilot demonstrates how Large Language Models can revolutionize industrial operations by providing instant root cause analysis, predictive maintenance recommendations, and context-aware process optimization—all through natural language queries.

---

## 🎯 **Why Manufacturing Copilot?**

Industrial operations generate massive amounts of time-series data, but extracting actionable insights requires deep domain expertise that's expensive and scarce. Manufacturing Copilot bridges this gap by embedding manufacturing intelligence directly into AI systems.

### **Critical Pain Points Solved:**
- **Slow Root Cause Analysis**: Hours of manual investigation → Minutes of AI-guided diagnosis
- **Alert Fatigue**: Noisy, context-free alarms → Intelligent, prioritized recommendations  
- **Knowledge Silos**: Expert insights trapped in individual experience → Democratized manufacturing intelligence
- **Reactive Maintenance**: Equipment failures surprise teams → Predictive insights prevent downtime

### **Next-Generation Capabilities:**
- **🧠 Natural Language Process Diagnosis**: "Why did the freezer temperature spike yesterday?"
- **🔍 Intelligent Anomaly Detection**: Context-aware pattern recognition with manufacturing domain knowledge
- **🔗 Multi-Signal Root Cause Analysis**: Automatic correlation discovery across equipment systems
- **📊 Expert-Level Recommendations**: Actionable maintenance and operational guidance
- **📐 Human-in-the-Loop Document Processing**: [AI-assisted P&ID diagram digitization](hybrid_ocr_gpt/README.md)

---

## 🚀 **How It Works: LLM-Powered Intelligence**

Manufacturing Copilot replaces deterministic rule engines with intelligent reasoning that adapts to context and provides expert-level analysis.

### **1. Intelligent Query Understanding**
```bash
python src/mcp.py "What caused the temperature problems yesterday?"
```
- **GPT-4 parses intent** → Understands manufacturing context and operational urgency
- **Semantic tag resolution** → Maps natural language to specific PI System tags
- **Dynamic analysis planning** → Creates multi-step investigation strategy

### **2. Multi-Step Analysis Execution**
```json
{
  "primary_tag": "FREEZER01.TEMP.INTERNAL_C",
  "analysis_steps": ["basic_statistics", "detect_anomalies", "correlate_tags", "generate_chart"],
  "reasoning": "Temperature issues require anomaly detection and correlation analysis to identify root causes..."
}
```

### **3. Expert-Level Insight Generation**
- **Root cause identification** → "Door left open + compressor inefficiency"
- **Operational impact assessment** → "Risk of product quality degradation"
- **Specific action items** → "Inspect door seals, schedule compressor maintenance"
- **Preventive recommendations** → "Implement door usage monitoring"

---

## 🎯 **Demo Queries That Showcase Intelligence**

```bash
# Comprehensive root cause analysis
python src/mcp.py "What caused the temperature problems in the freezer yesterday? I need a complete analysis with root cause and recommendations."

# Predictive maintenance insights
python src/mcp.py "Show me any unusual patterns in the compressor power consumption that might indicate upcoming failures"

# Energy efficiency optimization
python src/mcp.py "Why is the freezer using more energy than normal? What should we check?"

# Multi-system correlation analysis
python src/mcp.py "Give me a complete performance analysis of the freezer system with correlations and insights"
```

**Sample Expert Output:**
```
🔍 Root Cause Analysis
Temperature fluctuations caused by prolonged door opening (18 minutes) combined with 
compressor inefficiency during peak ambient temperature period.

⚡ Operational Impact  
Product quality risk due to temperature excursion above -16°C threshold.
Estimated 15% increase in energy consumption during recovery period.

🔧 Recommended Actions
1. Inspect door seals and alignment mechanisms immediately
2. Review staff door usage protocols during shift changes
3. Schedule compressor performance evaluation within 48 hours

🛡️ Preventive Measures
Implement door usage monitoring with 5-minute open-time alerts.
Consider compressor efficiency upgrade during next maintenance window.
```

---

## 🛠 **Setup & Installation**

### **Prerequisites**
- Python 3.11+
- OpenAI API key (GPT-4 access)

### **Quick Start**
```bash
# Clone and setup
git clone <repository-url>
cd manufacturing_mcp
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API access
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# Generate sample data (first time only)
python src/generate_freezer_data.py

# Test the system
python src/mcp.py "Show me freezer temperatures yesterday"
```

---

## 🎬 **Demo Modes**

### **🖥️ CLI Demo Mode**
Perfect for presentations and technical demonstrations:
```bash
python src/mcp.py --demo-mode "What caused the temperature problems yesterday?"
```
- Strategic pauses between analysis phases for narration
- Clear visual separation of GPT-4 reasoning, data analysis, and insights
- Interactive flow control for smooth presentations

### **🌐 Web Interface Demo**
**[Human-in-the-Loop P&ID Processing](hybrid_ocr_gpt/README.md)** - Demonstrates AI-assisted engineering document digitization:
```bash
cd hybrid_ocr_gpt/
python tag_reviewer.py
# Visit http://127.0.0.1:7860
```
Shows how Manufacturing Copilot could automatically process engineering drawings to build comprehensive tag databases with expert validation.

---

## 🔧 **Technical Architecture**

### **LLM-Powered Analysis Pipeline**
```
Natural Language Query → GPT-4 Analysis Planning → Tool Execution → Expert Insights
```

### **Core Technologies**
- **OpenAI GPT-4**: Natural language understanding and manufacturing domain reasoning
- **Chroma Vector DB**: Semantic tag search with OpenAI embeddings
- **Pandas + NumPy**: High-performance time-series data processing
- **Matplotlib**: Professional visualization with anomaly highlighting
- **Pydantic**: Type-safe data validation and structured LLM outputs
- **Gradio**: Interactive web interfaces for human-in-the-loop workflows

### **Manufacturing Domain Features**
- **PI System Integration**: Industry-standard AVEVA PI data format support
- **Statistical Anomaly Detection**: Z-score analysis with rolling window statistics
- **Multi-Modal Correlation Analysis**: Pearson, change-rate, and time-lagged correlations
- **Data Quality Assessment**: Good/Questionable/Bad quality indicators
- **Professional Reporting**: Publication-quality charts with anomaly highlighting

---

## 📊 **Sample Dataset & Scenarios**

The system includes **realistic 7-day freezer operation data** with injected anomalies for demonstration:

### **Data Characteristics**
- **50,400+ data points** across 5 PI tags at 1-minute resolution
- **Realistic physics simulation** with temperature cycling and equipment responses
- **Shift-based operational patterns** reflecting actual manufacturing schedules

### **PI Tags Included**
- `FREEZER01.TEMP.INTERNAL_C` → Internal freezer temperature monitoring
- `FREEZER01.COMPRESSOR.POWER_KW` → Power consumption analysis
- `FREEZER01.DOOR.STATUS` → Operational event tracking
- `FREEZER01.COMPRESSOR.STATUS` → Equipment state monitoring
- `FREEZER01.TEMP.AMBIENT_C` → Environmental condition tracking

### **Realistic Anomaly Scenarios**
- **Equipment Failures**: Compressor outages with temperature recovery patterns
- **Operational Issues**: Prolonged door openings during shift changes
- **Sensor Malfunctions**: Data quality degradation and flatline conditions
- **Power Quality Events**: Electrical fluctuations affecting equipment performance

---

## 🎯 **Business Value & Impact**

### **Immediate Operational Benefits**
- **10x Faster Root Cause Analysis**: Minutes instead of hours for complex investigations
- **24/7 Expert-Level Insights**: AI provides senior engineer analysis around the clock
- **Proactive Issue Detection**: Early warning systems prevent costly equipment failures
- **Democratized Manufacturing Intelligence**: Junior operators access expert-level guidance

### **Strategic Advantages**
| Traditional Approach | Manufacturing Copilot |
|---------------------|----------------------|
| ❌ Raw data dashboards | ✅ Contextual insights with recommendations |
| ❌ Manual correlation analysis | ✅ Automated multi-signal root cause detection |
| ❌ Reactive maintenance | ✅ Predictive maintenance recommendations |
| ❌ Expert knowledge silos | ✅ Democratized manufacturing intelligence |
| ❌ Alert fatigue | ✅ Intelligent, prioritized notifications |

---

## 🚀 **Vision & Roadmap**

### **Current Capabilities** ✅
- Natural language query processing with manufacturing domain expertise
- Multi-step analysis planning and execution
- Expert-level insight generation with actionable recommendations
- Professional visualization with anomaly highlighting
- Human-in-the-loop document processing prototype

### **Next Phase: Enterprise Integration** 🔄
- Real-time PI System connectivity and streaming analytics
- Multi-plant analysis with cross-facility correlation
- Custom manufacturing domain model fine-tuning
- Integration with existing CMMS and ERP systems

### **Future Vision: Autonomous Operations** 🎯
- Predictive maintenance with failure probability modeling
- Autonomous process optimization recommendations
- Real-time production efficiency optimization
- AI-driven quality control and yield improvement

---

## 📄 **Project Structure**

```
manufacturing_mcp/
├── src/
│   ├── mcp.py                    # Main CLI interface
│   ├── llm_interpreter.py        # 🧠 GPT-4 analysis engine
│   ├── glossary.py              # Semantic tag search
│   ├── generate_freezer_data.py  # Realistic data generator
│   └── tools/                   # Modular analysis functions
├── hybrid_ocr_gpt/              # 📐 P&ID processing demo
├── data/                        # Generated time-series data
├── charts/                      # Analysis visualizations
└── requirements.txt             # Python dependencies
```

---

## 🎉 **Experience the Demo**

Manufacturing Copilot represents the future of industrial AI—where domain expertise meets cutting-edge language models to deliver unprecedented operational intelligence.

**Get started:**
```bash
git clone <repository-url>
cd manufacturing_mcp
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key_here" > .env
python src/mcp.py "What's happening with my manufacturing system?"
```

**Ready to transform manufacturing operations with AI?** Let's build the future of intelligent manufacturing together! 🏭✨
