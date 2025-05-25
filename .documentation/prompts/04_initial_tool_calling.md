## Task

Create a Python script named `interpreter.py` that interprets natural language queries for a Manufacturing Copilot (MCP) CLI.

### Context

- `search_tags(query: str)` returns the best matching tag name from a glossary
- `load_data(tag: str, start: datetime, end: datetime)` loads timeseries data
- `summarize_metric(df: pd.DataFrame)` summarizes it
- You may use `dateparser` to extract time ranges
- All modules (search, tools) are already implemented

### Requirements

1. Define a **Pydantic model** `QueryParams` with the following fields:
   - `tag: str`
   - `start: datetime`
   - `end: datetime`

2. Implement a function `parse_query(query: str) -> QueryParams`:
   - Use `search_tags` to resolve a tag name
   - Parse `start` and `end` times using `dateparser.search.search_dates` or similar
   - Return `QueryParams` object

3. Implement a function `interpret_query(query: str) -> str`:
   - Use `parse_query` to extract inputs
   - Load data using `load_data(tag, start, end)`
   - Summarize using `summarize_metric(df)`
   - Return a markdown-formatted string summarizing the result:
     ```
     ✅ Summary for tag: FREEZER01.TEMP.INTERNAL_C
     → Time Range: Jan 16 2:00am – Jan 17 6:00am
     → Mean: -17.9°C | Min: -18.5°C | Max: -16.7°C | Trend: Stable
     ```

4. Do **not** output raw time-series values or plots.
5. Be robust to partial or ambiguous time references.

### Intent

This script will allow the CLI to interpret and route user questions to the correct tag and time window, applying only `load_data` and `summarize_metric` for now, with schema validation via Pydantic.