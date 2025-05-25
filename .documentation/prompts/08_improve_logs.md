## ✅ Log Formatting & CLI Output Polish Checklist

Make your CLI demo cleaner, more cinematic, and easier to narrate by doing the following:

### 🔕 Suppress or Silence Low-Value Logs
- [ ] Suppress HTTP logs for `openai.embeddings` and `chat.completions` (too verbose for CLI)
  - Use a more granular logging level (e.g. `DEBUG`) for those
  - Or selectively mute them via `logging.getLogger("httpx").setLevel(logging.WARNING)` or similar
- [ ] Silence telemetry messages from 3rd-party libraries unless in verbose/debug mode

### 🧠 Improve Phase Delineation and Clarity
- [ ] Add **visual section headers** with clear emoji icons:
  - `🔍 QUERY PARSING`, `📋 ANALYSIS PLAN`, `📈 EXECUTION`, `🤖 INSIGHTS`
- [ ] Add horizontal rule equivalents for CLI: `print("=" * 60)`
- [ ] Consider using colorized output via `colorama`:
  - Blue for headers
  - Grey/dim for secondary context (like LLM reasoning)
  - Green ✅ for success steps

### 📋 Surface Key Context for Narration
- [ ] After analysis plan, log in a readable format:
  - `Primary Tag: ...`
  - `Time Range: ...`
  - `Steps: ...`
  - `Reasoning:` (with slight indent or grey text)
- [ ] Consider using `rich.console` or `click.style` for styled output, if easy to add

---

**✅ Goal:** Create a clean CLI experience with well-spaced logs, contextual info for voiceover, and no log clutter.

When it feels like a smart assistant *telling a story*, you've nailed it.