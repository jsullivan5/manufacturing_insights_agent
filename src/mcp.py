#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Command Line Interface

GPT-4 powered Manufacturing Copilot that provides expert-level manufacturing 
insights, root cause analysis, and operational recommendations using natural 
language queries.

Usage:
    python src/mcp.py "Why did Freezer A use more power last night?"
    python src/mcp.py "What caused the temperature spike on Monday?"
    python src/mcp.py "Show me any anomalies in the compressor system"
"""

import argparse
import sys
import logging
from typing import Optional

# Add the project root to the Python path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_interpreter import llm_interpret_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ManufacturingCopilot:
    """
    GPT-4 powered Manufacturing Copilot for intelligent manufacturing insights.
    
    Provides expert-level analysis and operational recommendations through
    natural language queries, just like a senior process engineer would.
    """
    
    def __init__(self):
        """Initialize the Manufacturing Copilot."""
        print("🏭 Manufacturing Copilot - GPT-4 Powered Analysis")
        print("=" * 60)
        
        # Check OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ Error: OPENAI_API_KEY not found in environment")
            print("💡 Please set your OpenAI API key in the .env file")
            sys.exit(1)
        
        print("✅ Ready for natural language queries!")
    
    def process_query(self, query: str, demo_mode: bool = False, use_detective: bool = False) -> None:
        """
        Process a natural language query and provide intelligent insights.
        
        Args:
            query: Natural language question about manufacturing operations
            demo_mode: If True, pause at key points for demo narration
            use_detective: If True, use AI Detective Agent for iterative investigation
        """
        try:
            result = llm_interpret_query(query, demo_mode=demo_mode, use_detective=use_detective)
            # The streaming output is already printed, so we just need to handle the final result
            if result != "Analysis completed successfully. See insights above.":
                print(result)
        except Exception as e:
            logger.error(f"Error processing query with LLM: {e}")
            print(f"❌ LLM Analysis Error: {e}")
            print("💡 Please check your OpenAI API key and try again")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manufacturing Copilot - GPT-4 Powered Manufacturing Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/mcp.py "Why did Freezer A use more power last night?"
  python src/mcp.py "What caused the temperature spike yesterday?"
  python src/mcp.py "Show me any anomalies in the compressor system"
  python src/mcp.py "Give me a complete analysis of freezer performance"
  
Demo Mode (for presentations/videos):
  python src/mcp.py --demo-mode "What caused the temperature problems yesterday?"
  
AI Detective Agent (for investigative reasoning):
  python src/mcp.py --detective "Why did the freezer temperature spike?"
  python src/mcp.py --detective --demo-mode "What caused the compressor failure?"
        """
    )
    
    parser.add_argument(
        "query",
        help="Natural language question about manufacturing operations"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--demo-mode",
        "-d",
        action="store_true",
        help="Enable demo mode with pauses for video narration"
    )
    
    parser.add_argument(
        "--detective",
        action="store_true",
        help="Use AI Detective Agent for iterative investigation"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run MCP
    try:
        mcp = ManufacturingCopilot()
        mcp.process_query(args.query, demo_mode=args.demo_mode, use_detective=args.detective)
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 