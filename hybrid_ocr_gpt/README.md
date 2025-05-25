# 🏭 Manufacturing Copilot - Tag Glossary Reviewer

**Human-in-the-Loop Tag Description Review for Engineering Diagrams**

## 🎯 Overview

This demo illustrates a practical and collaborative approach to digitizing engineering diagrams for industrial settings. Instead of attempting full automation, it showcases how AI tools can assist subject matter experts (SMEs) by streamlining the tag description workflow.

By combining simulated tag detection (e.g., from a Process & Instrumentation Diagram or P&ID) with GPT-4-generated draft descriptions, the tool enables rapid iteration and refinement of equipment metadata — without solving the entire OCR problem.

The focus is on enabling **more efficient glossary creation**, not replacing the engineer’s judgment.

## 🚀 Demo Features

### 📐 Diagram + Tag Review Interface
- Displays a sample diagram image (`diagram.png`)
- Simulated tag bounding box overlay (e.g., `T1`)
- Mock tag detection (could represent OCR/vision output)

### 🧠 GPT-4 Aided Description Drafting
- **Smart Suggestions**: AI-generated tag descriptions based on tag names and context
- **Reasoning Display**: GPT-4’s explanatory rationale available for each suggestion
- **Editable Fields**: SME can adjust or overwrite suggestions

### 👨‍🔧 Human Review Workflow
- **Accept or Refine**: Review AI suggestions with a single click
- **Change History**: Log all modifications with timestamps
- **Audit Support**: Export final tag glossary and revision history
- **Confidence Metrics**: (Optional) Include simulated OCR confidence or model certainty

## 🛠️ How to Run

### Prerequisites
```bash
pip install gradio>=4.0.0
```

### Launch the Demo
```bash
cd hybrid_ocr_gpt/
python tag_reviewer.py
```

The interface will open at `http://127.0.0.1:7860`.

## 📋 Demo Workflow

1. **Load Diagram**: Open a sample engineering diagram (`diagram.png`)
2. **Tag Detected**: Simulated tag (`T1`) shown with a bounding box
3. **AI Description**: GPT-4 provides a draft description with reasoning
4. **Expert Review**: Accept, refine, or replace the draft description
5. **Log & Export**: See review history and export as needed

## 🎬 Example Interaction

**Detected Tag**: `T1`

**GPT-4 Suggestion**:  
> "Room 3 temperature sensor (°C) - monitors process area conditions for environmental control and safety compliance."

**Human Expert Refinement**:  
> "Temperature probe inside Room 3 process chamber - critical for batch temperature control and product quality assurance."

✅ **Final Description Accepted & Logged**

## 💡 Why This Matters

Many manufacturers have P&ID diagrams with valuable instrumentation metadata — but limited ways to digitize it efficiently. Fully solving OCR for industrial diagrams is extremely complex.

This demo presents a **middle path**:
- Use AI tools to assist SMEs rather than replace them
- Lower the manual burden of glossary building
- Retain control and accuracy through expert review

## 🔧 Architecture Snapshot

```
Engineering Diagram → Simulated Tag Detection → GPT-4 Suggestion → Human Review → Final Glossary
        ↓                         ↓                     ↓                   ↓               ↓
    diagram.png            'T1' Bounding Box      Draft Description      Edited Text     glossary.csv
```

## 🧩 Integrating with Manufacturing Copilot

This system complements broader manufacturing intelligence tools by:
- Populating tag metadata for semantic search
- Enabling faster onboarding of historical diagrams
- Supporting consistent naming and documentation practices

## 📈 Potential Enhancements

While this demo focuses on the review interface and simulated AI labeling, future versions could integrate real document processing pipelines using production-ready packages:

### 📦 OCR and Vision Tooling (for P&ID Diagrams)

- **Text Detection & OCR**  
  - [`PaddleOCR`](https://github.com/PaddlePaddle/PaddleOCR) – high-accuracy, multilingual OCR with GPU acceleration (via PaddlePaddle), good for engineering fonts
  - [`EasyOCR`](https://github.com/JaidedAI/EasyOCR) – lightweight, fast OCR, easy to deploy for prototyping
  - [`Tesseract`](https://github.com/tesseract-ocr/tesseract) – classical engine, less effective on dense industrial diagrams but widely used

- **Object/Tag Boundary Detection**  
  - [`YOLOv8`](https://github.com/ultralytics/ultralytics) or [`Detectron2`](https://github.com/facebookresearch/detectron2) – real-time instance segmentation models, excellent for identifying tag bounding boxes in noisy diagrams
  - [`LayoutParser`](https://github.com/Layout-Parser/layout-parser) – useful for extracting layout-aware elements from scanned documents, supports OCR fusion

- **Diagram Preprocessing (Optional)**  
  - OpenCV or skimage for contrast boosting, edge detection, and filtering non-text elements before OCR/vision pass

### 🛠 Integration Ideas
- Run OCR/tag detection once per uploaded diagram → populate tag candidates
- Feed cropped image + text into GPT with task-specific prompting for better description generation
- Allow manual override of bounding box + label (already supported in UI)

These tools, combined with a lightweight GPU server (or Azure VM with CUDA), would enable rapid prototyping of a fully integrated hybrid document digitization system.