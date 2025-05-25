# Manufacturing Copilot - Query Interpreter Demo

## Overview

The Manufacturing Copilot now includes an intelligent query interpreter that can understand natural language questions about manufacturing operations and automatically:

1. **Find relevant tags** using semantic search
2. **Parse time ranges** from natural language (e.g., "last night", "yesterday", "Monday")
3. **Load and analyze data** automatically
4. **Provide concise summaries** in markdown format

## Key Features

### ✅ Implemented
- **Pydantic QueryParams model** with validation
- **Natural language time parsing** using dateparser
- **Semantic tag search** integration
- **Automatic data loading and summarization**
- **Robust error handling** and fallbacks
- **Clean markdown output** format

### 🎯 Query Examples

| Query | Tag Found | Time Range Parsed |
|-------|-----------|-------------------|
| "Show me freezer temperatures last night" | FREEZER01.TEMP.INTERNAL_C | Previous night (8PM-6AM) |
| "What happened with the compressor yesterday?" | FREEZER01.COMPRESSOR.STATUS | Full day yesterday |
| "Power consumption patterns yesterday" | FREEZER01.COMPRESSOR.POWER_KW | Full day yesterday |
| "Door activity from Monday morning" | FREEZER01.DOOR.STATUS | Monday 6AM-12PM |

## Usage

### Default Mode (Interpreter)
```bash
python src/mcp.py "Show me what happened with the freezer temperatures last night"
```

### Legacy Mode (Detailed Analysis)
```bash
python src/mcp.py "Show me freezer temperatures" --legacy
```

## Output Format

The interpreter provides concise, actionable summaries:

```
✅ Summary for tag: FREEZER01.TEMP.INTERNAL_C
→ Time Range: May 22 11:59PM – May 23 11:59PM
→ Mean: -17.1°C | Min: -18.7°C | Max: -13.0°C | Trend: Rising
→ Data Points: 1,441 | Change: -0.9°C (+5.6%)
```

## Technical Implementation

### QueryParams Model
```python
class QueryParams(BaseModel):
    tag: str = Field(..., description="PI System tag name to analyze")
    start: datetime = Field(..., description="Start time for data analysis")
    end: datetime = Field(..., description="End time for data analysis")
```

### Key Functions
- `parse_query(query: str) -> QueryParams`: Extracts structured parameters
- `interpret_query(query: str) -> str`: Full interpretation pipeline
- `_parse_time_range(query: str) -> Tuple[datetime, datetime]`: Time parsing logic

### Time Parsing Intelligence
- **Relative references**: "last night", "yesterday", "last week"
- **Specific times**: "Monday", "May 22", "2 PM"
- **Time periods**: "morning" (6AM-12PM), "afternoon" (12PM-6PM), "night" (8PM-6AM)
- **Fallback handling**: Defaults to last 24 hours if parsing fails

## Benefits

1. **Faster insights**: No need to specify tags or time ranges manually
2. **Natural interaction**: Ask questions in plain English
3. **Intelligent defaults**: Automatic fallbacks for ambiguous queries
4. **Consistent output**: Standardized markdown format
5. **Extensible**: Easy to add new time patterns and tag matching logic

## Next Steps

The interpreter provides a foundation for more advanced features:
- Anomaly detection integration
- Multi-tag correlation analysis
- Automated root cause suggestions
- Natural language explanations of findings 