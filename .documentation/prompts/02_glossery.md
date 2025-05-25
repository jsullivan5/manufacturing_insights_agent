## 🧠 Embedding Layer Setup – Tag Glossary with OpenAI + Chroma

Let's focus on the embedding/database layer.  
This is meant to create a simple but illustrative demo to show what’s possible with this toolchain,  
so I would like to keep it **light** and **in-memory**.

---

### ✅ Task

Build a Python module called `glossary.py` that:

1. Loads a tag glossary from a CSV file named `tag_glossary.csv`  
   (Columns: `tag`, `description`, `unit`)

2. Embeds the `description` field using OpenAI’s `text-embedding-3-small` model.

3. Stores vectors **in-memory** using **Chroma** (do not persist to disk).

4. Uses `python-dotenv` to read the OpenAI API key (`OPENAI_API_KEY`) from `.env`.

5. Exposes a function:
   ```python
   def search_tags(query: str, top_k: int = 3) -> List[Dict]:
       """Returns top_k semantically similar tags with similarity scores."""
    ```

6.	Includes a __main__ block that demonstrates loading the glossary and running a sample query.

---

Use standard Python libraries and keep the design easy to reason about.

---

This module enables fast, in-memory semantic search over tag metadata, allowing natural language queries to be translated into relevant PI tags for downstream analysis in the MCP CLI.