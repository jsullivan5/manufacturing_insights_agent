#!/usr/bin/env python3
"""
Manufacturing Copilot - Query Interpreter Demo Script

Demonstrates the natural language query interpretation capabilities
of the Manufacturing Copilot CLI with various example queries.
"""

import subprocess
import sys
import time

def run_query(query: str, description: str = None) -> None:
    """Run a query and display the results with formatting."""
    if description:
        print(f"\n{'='*60}")
        print(f"🎯 {description}")
        print(f"{'='*60}")
    
    print(f"\n💬 Query: '{query}'")
    print("-" * 50)
    
    try:
        # Run the MCP CLI with the query
        result = subprocess.run(
            [sys.executable, "src/mcp.py", query],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            # Extract just the result part (skip initialization)
            lines = result.stdout.split('\n')
            in_result = False
            for line in lines:
                if line.startswith('🔍 Query:'):
                    in_result = True
                    continue
                if in_result:
                    print(line)
        else:
            print(f"❌ Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ Query timed out")
    except Exception as e:
        print(f"❌ Error running query: {e}")
    
    print("\n" + "⏱️  Waiting 2 seconds before next query...")
    time.sleep(2)


def main():
    """Run the demo with various natural language queries."""
    print("🏭 Manufacturing Copilot - Query Interpreter Demo")
    print("=" * 60)
    print("This demo showcases the natural language query interpretation")
    print("capabilities of the Manufacturing Copilot CLI.")
    print("=" * 60)
    
    # Demo queries showcasing different capabilities
    demo_queries = [
        {
            "query": "Show me what happened with the freezer temperatures last night",
            "description": "Temperature Analysis with Time Parsing"
        },
        {
            "query": "What happened with the compressor yesterday?",
            "description": "Equipment Status Analysis"
        },
        {
            "query": "Power consumption patterns yesterday",
            "description": "Energy Usage Analysis"
        },
        {
            "query": "Door activity from Monday morning",
            "description": "Operational Pattern Analysis"
        },
        {
            "query": "Show me internal temperature readings",
            "description": "Specific Tag Identification"
        }
    ]
    
    # Run each demo query
    for i, demo in enumerate(demo_queries, 1):
        run_query(demo["query"], f"Demo {i}: {demo['description']}")
    
    print(f"\n{'='*60}")
    print("🎉 Demo Complete!")
    print("=" * 60)
    print("\n📋 Key Features Demonstrated:")
    print("✅ Natural language tag identification")
    print("✅ Intelligent time range parsing")
    print("✅ Automatic data loading and analysis")
    print("✅ Concise, actionable summaries")
    print("✅ Robust error handling and fallbacks")
    
    print("\n🚀 Try your own queries:")
    print("python src/mcp.py \"Your natural language question here\"")
    
    print("\n🔧 Advanced mode:")
    print("python src/mcp.py \"Your question\" --legacy")
    print("(Shows detailed step-by-step processing)")


if __name__ == "__main__":
    main() 