#!/usr/bin/env python3
"""
Manufacturing Copilot - Job Interview Demo Script

This script demonstrates the power of LLM-driven manufacturing insights
for potential employers. Shows real AI capabilities in manufacturing.

Run this to impress your interviewer! 🚀
"""

import subprocess
import time
import sys

def run_demo_query(query, description):
    """Run a demo query and display results."""
    print(f"\n{'='*80}")
    print(f"🎯 DEMO: {description}")
    print(f"{'='*80}")
    print(f"Query: '{query}'")
    print(f"{'='*80}")
    
    # Run the query
    result = subprocess.run([
        sys.executable, "src/mcp.py", query
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"❌ Error: {result.stderr}")
    
    print(f"{'='*80}")
    input("Press Enter to continue to next demo...")

def main():
    """Run the complete demo sequence."""
    print("""
🏭 MANUFACTURING COPILOT - LLM-POWERED DEMO
============================================

This demo showcases how GPT-4 can provide expert-level manufacturing insights,
root cause analysis, and operational recommendations - just like a senior 
process engineer would!

Key Features:
✅ Natural language query understanding
✅ Intelligent multi-step analysis planning  
✅ Expert-level manufacturing insights
✅ Actionable operational recommendations
✅ Professional visualization generation

Ready to see the magic? 🚀
    """)
    
    input("Press Enter to start the demo...")
    
    # Demo queries that showcase different capabilities
    demo_queries = [
        (
            "What caused the temperature problems in the freezer yesterday? I need a complete analysis with root cause and recommendations.",
            "Root Cause Analysis & Expert Recommendations"
        ),
        (
            "Show me any unusual patterns or anomalies in the compressor power consumption over the past week",
            "Anomaly Detection & Power Analysis"
        ),
        (
            "Why is the freezer using more energy than normal? What should we check?",
            "Energy Efficiency Troubleshooting"
        ),
        (
            "Give me a complete performance analysis of the freezer system with correlations and insights",
            "Comprehensive System Analysis"
        )
    ]
    
    print(f"\n🎬 Starting demo with {len(demo_queries)} scenarios...")
    
    for i, (query, description) in enumerate(demo_queries, 1):
        print(f"\n📋 Demo {i}/{len(demo_queries)}")
        run_demo_query(query, description)
    
    print("""
🎉 DEMO COMPLETE! 

What you just saw:
✅ GPT-4 understanding complex manufacturing queries
✅ Intelligent analysis planning and execution
✅ Expert-level insights and recommendations
✅ Professional chart generation with anomaly highlighting
✅ Actionable operational guidance

This is the future of manufacturing data analysis - AI that thinks like 
a senior process engineer and provides insights that drive real business value!

Ready to revolutionize your manufacturing operations? 🚀
    """)

if __name__ == "__main__":
    main() 