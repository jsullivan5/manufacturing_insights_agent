# 📌 What to Do Now (Step-by-Step)

1.	Update interpret_query()
    - Incorporate conditional logic to route prompts to the correct tool(s)
    - Use the same prompt-parsing strategy you’re already using (e.g., with Pydantic)
2.	Tool Chain Logic
    - If anomalies are requested, run detect_spike()
    - If relationships are mentioned, run correlate_tags()
    - If “show me” is in the prompt, generate a chart with generate_chart()
3.	Prompt Response Strategy
    - Return both summary text and chart path
    - If possible, highlight causal inferences (e.g., “A temperature spike followed door opening by ~5 minutes”)