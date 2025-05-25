#!/usr/bin/env python3
"""
Manufacturing Copilot - Tag Glossary Reviewer Demo

A Gradio-based human-in-the-loop interface for reviewing and refining
GPT-generated tag descriptions from P&ID diagrams. This demonstrates
how OCR + GPT + Human expertise can create high-quality tag glossaries.

This is a design mock showing how the Manufacturing Copilot could integrate
with document processing workflows to automatically build comprehensive
tag databases from engineering drawings.
"""

import gradio as gr
import os
from datetime import datetime
from typing import Dict, Any

# Mock data simulating OCR + GPT processing results
MOCK_TAG_DATA = {
    "tag_id": "T1",
    "confidence": 0.92,
    "bounding_box": {"x": 245, "y": 180, "width": 25, "height": 15},
    "gpt_description": "Room 3 temperature sensor (°C) - monitors process area conditions for environmental control and safety compliance.",
    "gpt_reasoning": "Based on the P&ID context and standard instrumentation symbols, this appears to be a temperature measurement device located in Room 3. The 'T' prefix indicates temperature measurement, and the location suggests process monitoring.",
    "suggested_tags": ["temperature", "sensor", "room3", "process", "monitoring"],
    "equipment_context": "Located near process equipment in Room 3, likely part of HVAC or process control system"
}

class TagGlossaryReviewer:
    """
    Human-in-the-loop tag glossary generation interface.
    
    Simulates the workflow where OCR extracts tags from P&ID diagrams,
    GPT generates intelligent descriptions, and human experts review
    and refine the results for maximum accuracy.
    """
    
    def __init__(self):
        """Initialize the tag reviewer with mock data."""
        self.current_tag = MOCK_TAG_DATA.copy()
        self.review_history = []
    
    def process_tag_review(self, description: str, action: str) -> tuple:
        """
        Process the human review decision for a tag description.
        
        Args:
            description: The final description (edited or original)
            action: The action taken ("accept" or "modify")
            
        Returns:
            Tuple of (status_message, updated_history, log_output)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine if description was modified
        was_modified = description.strip() != self.current_tag["gpt_description"].strip()
        actual_action = "modify" if was_modified else "accept"
        
        # Create review record
        review_record = {
            "timestamp": timestamp,
            "tag_id": self.current_tag["tag_id"],
            "original_description": self.current_tag["gpt_description"],
            "final_description": description.strip(),
            "action": actual_action,
            "was_modified": was_modified,
            "gpt_confidence": self.current_tag["confidence"]
        }
        
        # Add to history
        self.review_history.append(review_record)
        
        # Generate status message
        if actual_action == "accept":
            status_msg = f"✅ **Tag {self.current_tag['tag_id']} ACCEPTED** (No changes made)"
            console_msg = f"[{timestamp}] ✅ ACCEPTED: {self.current_tag['tag_id']} - '{description[:50]}...'"
        else:
            status_msg = f"📝 **Tag {self.current_tag['tag_id']} MODIFIED** (Human expertise applied)"
            console_msg = f"[{timestamp}] 📝 MODIFIED: {self.current_tag['tag_id']} - '{description[:50]}...'"
        
        # Create history display
        history_display = self._format_review_history()
        
        # Log to console for demo purposes
        print(console_msg)
        print(f"    Original: {self.current_tag['gpt_description']}")
        print(f"    Final:    {description}")
        print(f"    Action:   {actual_action.upper()}")
        print("-" * 80)
        
        return status_msg, history_display, console_msg
    
    def _format_review_history(self) -> str:
        """Format the review history for display."""
        if not self.review_history:
            return "No reviews completed yet."
        
        history_text = "## 📋 Review History\n\n"
        
        for i, record in enumerate(self.review_history, 1):
            action_emoji = "✅" if record["action"] == "accept" else "📝"
            history_text += f"**{i}. {action_emoji} {record['tag_id']}** ({record['timestamp']})\n"
            history_text += f"   - **Final**: {record['final_description']}\n"
            if record['was_modified']:
                history_text += f"   - **Original**: {record['original_description']}\n"
            history_text += f"   - **GPT Confidence**: {record['gpt_confidence']:.1%}\n\n"
        
        return history_text
    
    def reset_demo(self) -> tuple:
        """Reset the demo to initial state."""
        self.review_history = []
        return (
            MOCK_TAG_DATA["gpt_description"],  # Reset description
            "Demo reset. Ready for new review.",  # Status message
            "",  # Clear history
            ""   # Clear console log
        )


def create_tag_reviewer_interface():
    """Create the Gradio interface for tag glossary review."""
    
    reviewer = TagGlossaryReviewer()
    
    with gr.Blocks(
        title="Manufacturing Copilot - Tag Glossary Reviewer",
        theme=gr.themes.Soft(),
        css="""
        .tag-info { background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .gpt-reasoning { background-color: #f9f9f9; padding: 10px; border-left: 4px solid #4CAF50; margin: 10px 0; }
        .status-success { color: #4CAF50; font-weight: bold; }
        .status-modified { color: #FF9800; font-weight: bold; }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🏭 Manufacturing Copilot - Tag Glossary Reviewer
        
        **Human-in-the-Loop Tag Description Generation**
        
        This demo simulates how the Manufacturing Copilot processes P&ID diagrams:
        1. 🔍 **OCR** extracts tags from engineering drawings
        2. 🧠 **GPT-4** generates intelligent descriptions with manufacturing context
        3. 👨‍🔧 **Human Expert** reviews and refines for accuracy
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # P&ID Diagram Display
                gr.Markdown("## 📐 P&ID Diagram")
                diagram_image = gr.Image(
                    value=os.path.join(os.path.dirname(__file__), "diagram.png"),
                    label="Process & Instrumentation Diagram",
                    interactive=False,
                    height=400
                )
                
                # Tag Detection Info
                with gr.Group():
                    gr.Markdown("### 🎯 Detected Tag Information", elem_classes=["tag-info"])
                    
                    with gr.Row():
                        gr.Markdown(f"**Tag ID:** `{MOCK_TAG_DATA['tag_id']}`")
                        gr.Markdown(f"**OCR Confidence:** {MOCK_TAG_DATA['confidence']:.1%}")
                    
                    gr.Markdown(f"**Equipment Context:** {MOCK_TAG_DATA['equipment_context']}")
                    
                    with gr.Accordion("🧠 GPT-4 Reasoning", open=False):
                        gr.Markdown(
                            f"*{MOCK_TAG_DATA['gpt_reasoning']}*",
                            elem_classes=["gpt-reasoning"]
                        )
            
            with gr.Column(scale=3):
                # Review Interface
                gr.Markdown("## 👨‍🔧 Human Expert Review")
                
                # Description Editor
                description_input = gr.Textbox(
                    label="Tag Description",
                    value=MOCK_TAG_DATA["gpt_description"],
                    lines=3,
                    placeholder="Edit the GPT-generated description if needed...",
                    info="Review and modify the description to ensure accuracy and completeness."
                )
                
                # Action Buttons
                with gr.Row():
                    accept_btn = gr.Button("✅ Accept Description", variant="primary", scale=1)
                    modify_btn = gr.Button("📝 Accept with Modifications", variant="secondary", scale=1)
                    reset_btn = gr.Button("🔄 Reset Demo", variant="stop", scale=1)
                
                # Status Display
                status_output = gr.Markdown("Ready for review...", elem_classes=["status-success"])
                
                # Review History
                history_output = gr.Markdown("", label="Review History")
                
                # Console Log (for demo purposes)
                with gr.Accordion("🖥️ Console Log", open=False):
                    console_output = gr.Textbox(
                        label="System Log",
                        lines=3,
                        interactive=False,
                        placeholder="Review actions will be logged here..."
                    )
        
        # Event Handlers
        def handle_accept(description):
            return reviewer.process_tag_review(description, "accept")
        
        def handle_modify(description):
            return reviewer.process_tag_review(description, "modify")
        
        accept_btn.click(
            fn=handle_accept,
            inputs=[description_input],
            outputs=[status_output, history_output, console_output]
        )
        
        modify_btn.click(
            fn=handle_modify,
            inputs=[description_input],
            outputs=[status_output, history_output, console_output]
        )
        
        reset_btn.click(
            fn=reviewer.reset_demo,
            inputs=[],
            outputs=[description_input, status_output, history_output, console_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        
        ### 🚀 Vision: Automated Tag Glossary Generation
        
        This interface demonstrates how the Manufacturing Copilot could automatically process hundreds of P&ID diagrams to build comprehensive tag databases:
        
        - **📄 Document Processing**: Batch process engineering drawings and schematics
        - **🤖 AI-Powered Analysis**: Combine OCR, computer vision, and LLM reasoning
        - **👥 Expert Validation**: Human-in-the-loop review for critical accuracy
        - **📊 Quality Metrics**: Track confidence scores and review patterns
        - **🔄 Continuous Learning**: Improve AI models based on expert feedback
        
        **Built by James Sullivan** | Manufacturing Copilot Demo
        """)
    
    return interface


def main():
    """Launch the tag glossary reviewer demo."""
    print("🏭 Manufacturing Copilot - Tag Glossary Reviewer")
    print("=" * 60)
    print("🚀 Launching Gradio interface...")
    print("💡 This demo simulates human-in-the-loop tag description generation")
    print("📐 Using P&ID diagram with pre-annotated tag 'T1'")
    print("-" * 60)
    
    # Create and launch the interface
    interface = create_tag_reviewer_interface()
    
    # Launch with demo-friendly settings
    interface.launch(
        server_name="127.0.0.1",  # Local only for demo
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main() 