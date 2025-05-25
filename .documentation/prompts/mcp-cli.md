# 🛠️ MCP CLI Scaffold – Cursor Prompt

I'd like to scaffold a simple CLI runner script called `mcp.py` that will serve as the entry point for my Manufacturing Copilot demo.

Here’s what I want it to do for now:

1. Accept a natural language query from the command line (e.g., `"Why did Freezer A use more power last night?"`)
2. Use the existing `glossary.py` module to embed the query and find the top relevant tag
3. Use `load_data()` from my tool functions to retrieve a DataFrame for that tag
4. For now, just print:
   - The matched tag
   - A quick summary (row count, time range, sample values)

Make the CLI ergonomic using `argparse`, and ensure the script runs from the project root like:

```bash
python src/mcp.py "Why did Freezer A use more power last night?"