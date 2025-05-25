# Manufacturing Copilot (MCP) - Development Todo

## ✅ COMPLETED FEATURES

### Phase 1: Mock Data Generation ✅
- ✅ **Realistic freezer system data generator** (`src/generate_freezer_data.py`)
- ✅ **7-day dataset with 50,400 data points** in AVEVA PI System long format
- ✅ **5 synchronized PI tags** with realistic physics simulation
- ✅ **Modular anomaly injection system** (4 types of anomalies)
- ✅ **Data verification script** with automatic anomaly detection

### Phase 2: Semantic Tag Search ✅
- ✅ **Tag glossary system** using OpenAI embeddings and Chroma vector database
- ✅ **Semantic search** achieving 33-42% similarity scores for relevant queries
- ✅ **In-memory vector storage** for fast tag matching

### Phase 3: CLI Scaffold and Tool Functions ✅
- ✅ **Main CLI entry point** (`src/mcp.py`) with argparse
- ✅ **Modular tools architecture** in `src/tools/`
- ✅ **Data loading, metrics, and quality analysis tools**

### Phase 4: Natural Language Query Interpreter ✅
- ✅ **`QueryParams` Pydantic model** with validation
- ✅ **`parse_query()` function** using semantic search and dateparser
- ✅ **`interpret_query()` function** with markdown-formatted output
- ✅ **Natural language time parsing** ("last night", "yesterday", "Monday")
- ✅ **Intelligent fallbacks** for ambiguous queries
- ✅ **CLI integration** with default interpreter mode and legacy option

## 🔄 IN PROGRESS

### Advanced Analytics Tools
Working on implementing the remaining tool functions for comprehensive analysis:

1. ⬜ `detect_spike(df: pd.DataFrame, threshold: float = 3.0)`
   - Inputs: `df` (time-series DataFrame), `threshold` (z-score threshold for anomaly detection)
   - Processing: Calculate rolling statistics and identify data points exceeding threshold
   - Output: Return list of `(timestamp, value, z_score)` tuples for anomalous points
   - Testing: Verify detection of injected anomalies in synthetic data

2. ⬜ `correlate_tags(primary_df: pd.DataFrame, candidate_dfs: List[pd.DataFrame], window: Tuple[datetime, datetime])`
   - Inputs: `primary_df` (main metric DataFrame), `candidate_dfs` (list of related metric DataFrames), `window` (time range tuple)
   - Processing: Calculate correlation coefficients or aligned deltas between primary and candidate metrics within the spike window
   - Output: Return ranked list of `(tag_name, correlation_score)` indicating relevance to the primary anomaly
   - Testing: Confirm correlations reflect known relationships in synthetic data

3. ⬜ `generate_chart(df: pd.DataFrame, tag: str, highlights: Optional[List[Tuple[datetime, datetime]]] = None)`
   - Inputs: `df` (time-series DataFrame), `tag` (metric name string), `highlights` (optional list of anomaly time windows)
   - Processing: Plot time series data as PNG chart; annotate or highlight anomaly periods if provided
   - Output: Save chart PNG file and return filepath string
   - Testing: Visual inspection to ensure data and highlights are correctly rendered

## 📋 CURRENT CAPABILITIES

The Manufacturing Copilot can now:

### Natural Language Queries ✅
```bash
python src/mcp.py "Show me what happened with the freezer temperatures last night"
python src/mcp.py "What happened with the compressor yesterday?"
python src/mcp.py "Power consumption patterns yesterday"
```

### Intelligent Processing ✅
- **Semantic tag search**: Finds relevant PI tags from natural language
- **Time range parsing**: Understands "last night", "yesterday", "Monday", etc.
- **Automatic data loading**: No manual tag or time specification needed
- **Statistical analysis**: Mean, min, max, trend, change percentage
- **Data quality reporting**: Good/Questionable/Bad percentages

### Output Format ✅
```
✅ Summary for tag: FREEZER01.TEMP.INTERNAL_C
→ Time Range: May 22 11:59PM – May 23 11:59PM
→ Mean: -17.1°C | Min: -18.7°C | Max: -13.0°C | Trend: Rising
→ Data Points: 1,441 | Change: -0.9°C (+5.6%)
```

## 🎯 NEXT PRIORITIES

1. **Anomaly Detection Integration**: Implement `detect_spike()` and integrate with interpreter
2. **Multi-tag Correlation**: Build `correlate_tags()` for root cause analysis
3. **Visualization**: Add `generate_chart()` for matplotlib visualizations
4. **LLM Reasoning Layer**: Natural language explanations of findings
5. **End-to-end Pipeline**: Query → Tags → Data → Anomalies → Correlations → Insights

## 📊 PROJECT STATUS

- **Foundation**: ✅ Complete (Data, Search, CLI, Interpreter)
- **Core Analytics**: 🔄 In Progress (Anomaly detection, Correlation)
- **Advanced Features**: ⏳ Planned (Visualization, LLM reasoning)
- **Production Ready**: 🎯 Target (Full pipeline integration)