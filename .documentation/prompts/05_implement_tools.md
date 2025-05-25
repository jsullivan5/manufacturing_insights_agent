### 🧱 Recommended Development Order: Tool Functions First

Implement each tool function one at a time before enhancing the interpreter. This approach ensures reliability and simplifies downstream logic.

#### ✅ Why this order?

- 🔍 **Isolated Testing**: Validate each tool with `__main__` blocks before integrating.
- 🧠 **Simpler Interpreter**: Once tools are implemented, routing becomes a clean mapping.
- 🧪 **Reliable Output**: Easier to debug and refine if each function is known to work.