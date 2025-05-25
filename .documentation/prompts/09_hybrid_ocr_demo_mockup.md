## 🚀 Gradio Tag Glossary Mock UI

🎯 **Goal**: Build a simple Gradio app to simulate human-in-the-loop tag glossary generation using a single annotated region on a P&ID diagram image.

📂 **Directory**: `hybrid_ocr_gpt/`

🖼️ **Input Image**: Assume `diagram.png` is a pre-annotated image with a box around the tag `T1`.

🧪 **Mock Data**:
- Extracted Tag: `"T1"`
- GPT Suggested Description: `"Room 3 temperature sensor (°C) - monitors process area conditions."`

📋 **UI Requirements**:
- Display the image (`diagram.png`)
- Below the image, show:
  - Detected tag (`T1`)
  - Suggested description (editable `Textbox`)
  - Button: ✅ Accept or 📝 Modify
- When accepted or modified, display a confirmation message and log the final decision to the console (for now).

🧠 You can simulate this with Gradio + Python. Save the script in `hybrid_ocr_gpt/tag_reviewer.py`.

👩‍💻 **Example Interaction**:
> User sees `T1` on diagram with the prompt:  
> "Room 3 temperature sensor (°C)...", and edits it to  
> "Temperature probe inside Room 3 process chamber."

📎 Use this as a design mock for how GPT + OCR might be paired with a human glossary reviewer.

Build this as a fully working Gradio demo with a single tag — no backend logic or OCR needed for now.