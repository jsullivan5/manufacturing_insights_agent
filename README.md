# 🏭 Manufacturing Copilot (MCP) - LLM-Powered Manufacturing Intelligence

> **GPT-4 powered manufacturing insights that think like a senior process engineer**

A revolutionary AI system that transforms manufacturing data analysis from raw numbers into expert-level insights, root cause analysis, and actionable operational recommendations - all through natural language queries.

![Manufacturing Copilot Demo](https://img.shields.io/badge/AI-GPT--4%20Powered-blue?style=for-the-badge&logo=openai)
![Python](https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge&logo=python)
![Manufacturing](https://img.shields.io/badge/Industry-Manufacturing-orange?style=for-the-badge&logo=factory)

---

## 🚀 **What Makes This Special**

Instead of traditional dashboards that show you data, **Manufacturing Copilot explains what the data means and tells you what to do about it** - just like a senior process engineer with 20+ years of experience would.

### **Traditional Approach:**
```
❌ "Temperature is 15.2°C at 14:30"
❌ "Power consumption increased 45%"  
❌ "3 anomalies detected"
```

### **Manufacturing Copilot Approach:**
```
✅ "Temperature spike caused by door usage patterns + compressor inefficiency"
✅ "Inspect door seals and check compressor performance immediately"
✅ "Implement door usage policy to prevent product quality issues"
✅ "Schedule predictive maintenance to avoid costly downtime"
```

---

## 🧠 **How It Works: LLM-Powered Intelligence**

### **1. Intelligent Query Understanding**
- **GPT-4 parses natural language** → No more complex query syntax
- **Manufacturing domain expertise** → Understands freezer systems, compressors, PI tags
- **Context-aware analysis** → Creates intelligent multi-step analysis plans

### **2. Multi-Step Analysis Execution**
```json
# GPT-4 creates analysis plan:
{
  "primary_tag": "FREEZER01.TEMP.INTERNAL_C",
  "analysis_steps": ["basic_statistics", "detect_anomalies", "correlate_tags", "generate_chart"],
  "reasoning": "Temperature problems require anomaly detection and correlation analysis..."
}
```

### **3. Expert-Level Insight Generation**
- **Root cause analysis** → Identifies why problems occurred
- **Operational impact assessment** → Explains business consequences  
- **Specific recommendations** → Actionable next steps for operators
- **Preventive measures** → How to avoid future issues

### **4. Professional Visualizations**
- **Anomaly highlighting** → Red-shaded problem periods
- **Trend analysis** → Statistical overlays and trend lines
- **Data quality indicators** → Good/Questionable/Bad data markers

---

## 🎯 **Demo Queries That Test The System**

```bash
# Root cause analysis with expert recommendations
python src/mcp.py "What caused the temperature problems in the freezer yesterday? I need a complete analysis with root cause and recommendations."

# Intelligent anomaly detection
python src/mcp.py "Show me any unusual patterns or anomalies in the compressor power consumption over the past week"

# Energy efficiency troubleshooting  
python src/mcp.py "Why is the freezer using more energy than normal? What should we check?"

# Comprehensive system analysis
python src/mcp.py "Give me a complete performance analysis of the freezer system with correlations and insights"
```

**Sample Output:**
```
1. **What Happened?**
Yesterday, the internal temperature of Freezer01 fluctuated between -18.8°C and -11.44°C, 
with an average temperature of -17.02°C. This is a significant deviation from the ideal 
operating temperature range.

2. **Root Cause Analysis**
The data analysis indicates a moderate correlation between the freezer door status and 
the internal temperature. This suggests that the door was opened more frequently or left 
open for extended periods, causing the temperature to rise.

3. **Operational Impact**
These temperature fluctuations can have a significant impact on product quality and safety. 
If the temperature rises above the safe storage level, it can lead to spoilage and waste.

4. **Recommended Actions**
Immediate actions should include a thorough inspection of the freezer door seals and latches 
to ensure they are functioning correctly. Staff should be reminded of the importance of 
minimizing door opening times.

5. **Preventive Measures**
Consider implementing a door alarm that will alert staff if the door is left open for an 
extended period. Regular maintenance and inspection of the compressor should be scheduled.
```

---

## 🛠 **Setup & Installation**

### **Prerequisites**
- Python 3.11+ 
- OpenAI API key (for GPT-4 access)
- Git

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd manufacturing_mcp
```

### **2. Create Virtual Environment**
```bash
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Set Up Environment Variables**
Create a `.env` file in the project root:
```bash
# Required: OpenAI API key for GPT-4 access
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Adjust logging level
LOG_LEVEL=INFO
```

### **5. Generate Sample Data (First Time Setup)**
```bash
# Generate 7 days of realistic freezer system data
python src/generate_freezer_data.py

# This creates:
# - data/freezer_data.csv (50,400+ data points)
# - data/tag_glossary.csv (PI tag descriptions)
# - Realistic anomalies and operational patterns
```

### **6. Test the Installation**
```bash
# Quick test with LLM-powered analysis
python src/mcp.py "Show me freezer temperatures yesterday"
```

---

## 🎬 **Running the Demo**

### **Interactive Demo Script**
```bash
# Run the complete job interview demo
python demo_script.py
```

### **Individual Queries**
```bash
# LLM-powered analysis (recommended!)
python src/mcp.py "What caused the temperature spike yesterday?"

# Verbose logging for debugging
python src/mcp.py "Analyze compressor performance" --verbose
```

### **Command Line Options**
```bash
python src/mcp.py [QUERY] [OPTIONS]

Options:
  --verbose, -v     Enable verbose logging
  --help           Show help message
```

---

## 📊 **Project Structure**

```
manufacturing_mcp/
├── src/
│   ├── mcp.py                    # Main CLI entry point
│   ├── llm_interpreter.py        # 🧠 GPT-4 powered analysis engine
│   ├── glossary.py              # Semantic tag search with embeddings
│   ├── generate_freezer_data.py  # Realistic data generator
│   └── tools/                   # Analysis tool functions
│       ├── anomaly_detection.py  # Z-score based spike detection
│       ├── correlation.py        # Multi-type correlation analysis
│       ├── visualization.py      # Professional chart generation
│       ├── data_loader.py        # PI System data loading
│       ├── metrics.py            # Statistical summarization
│       └── quality.py           # Data quality assessment
├── data/                        # Generated manufacturing data
│   ├── freezer_data.csv         # 7 days of time-series data
│   └── tag_glossary.csv         # PI tag descriptions
├── charts/                      # Generated visualizations
├── .documentation/              # Project documentation
├── demo_script.py              # Job interview demo script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔧 **Technical Architecture**

### **LLM-Powered Pipeline**
1. **Query Parsing** → GPT-4 understands intent and creates analysis plan
2. **Multi-Step Execution** → Runs analytical tools in intelligent sequence  
3. **Expert Insights** → GPT-4 analyzes results like a senior engineer
4. **Professional Output** → Actionable recommendations and visualizations

### **Core Technologies**
- **OpenAI GPT-4** → Natural language understanding and expert insights
- **Chroma Vector DB** → Semantic tag search with embeddings
- **Pandas + NumPy** → Time-series data processing and analysis
- **Matplotlib** → Professional visualization generation
- **Pydantic** → Data validation and structured outputs
- **Python 3.11+** → Modern async/await patterns and type hints

### **Manufacturing Domain Features**
- **PI System Integration** → Industry-standard data format support
- **Anomaly Detection** → Z-score based spike detection with rolling windows
- **Correlation Analysis** → Pearson, change, and time-lagged correlations
- **Data Quality Assessment** → Good/Questionable/Bad quality indicators
- **Professional Visualizations** → Charts with anomaly highlighting

---

## 📈 **Sample Data & Scenarios**

The system includes **realistic 7-day freezer operation data** with:

### **Data Volume**
- **50,400+ data points** across 5 PI tags
- **1-minute resolution** for detailed analysis
- **Realistic physics simulation** with temperature cycling

### **PI Tags Included**
- `FREEZER01.TEMP.INTERNAL_C` → Internal freezer temperature
- `FREEZER01.TEMP.AMBIENT_C` → Ambient room temperature  
- `FREEZER01.COMPRESSOR.POWER_KW` → Compressor power consumption
- `FREEZER01.COMPRESSOR.STATUS` → Compressor on/off status
- `FREEZER01.DOOR.STATUS` → Freezer door open/closed status

### **Injected Anomalies**
- **Prolonged door open** → 18-minute door event causing temperature rise
- **Compressor failure** → 55-minute compressor outage
- **Sensor flatline** → 4-hour sensor malfunction
- **Power fluctuations** → 25-minute power instability

---

## 🎯 **Business Value**

### **Immediate Benefits**
- **Faster Problem Resolution** → Minutes instead of hours for root cause analysis
- **Reduced Expert Dependency** → AI provides senior-level insights 24/7
- **Prevented Downtime** → Early detection of equipment issues
- **Optimized Operations** → Data-driven recommendations for efficiency

### **Competitive Advantages**
| Traditional Systems | Manufacturing Copilot |
|-------------------|---------------------|
| ❌ Shows raw data | ✅ Explains what data means |
| ❌ Requires expert interpretation | ✅ Provides expert-level analysis |
| ❌ Static dashboards | ✅ Dynamic insights |
| ❌ Manual root cause analysis | ✅ Automated problem diagnosis |
| ❌ High false positive alerts | ✅ Intelligent anomaly detection |

---

## 🚀 **Future Roadmap**

### **Phase 1: Current Demo** ✅
- LLM-powered query interpretation
- Expert-level insights and recommendations  
- Professional visualization with anomaly highlighting

### **Phase 2: Enterprise Integration** 🔄
- Real-time PI System connectivity
- Multi-plant analysis capabilities
- Custom manufacturing domain models

### **Phase 3: Predictive Intelligence** 🎯
- Predictive maintenance recommendations
- Production optimization suggestions
- Cost reduction opportunity identification

---

## 🤝 **Contributing**

This project demonstrates the future of AI in manufacturing. Contributions welcome!

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
ruff check src/

# Format code
black src/
```

---

## 📄 **License**

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🎉 **Ready to Revolutionize Manufacturing?**

This isn't just a demo - it's a glimpse of the future where AI provides expert-level manufacturing insights at scale.

**Get started:**
```bash
git clone <repository-url>
cd manufacturing_mcp
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key_here" > .env
python src/mcp.py "What's happening with my freezer system?"
```

**Questions? Issues? Want to see this in your manufacturing environment?**

Let's build the future of intelligent manufacturing together! 🏭🚀 