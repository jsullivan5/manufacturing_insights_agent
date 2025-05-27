#!/usr/bin/env python3
"""
Quick demo of the AI Detective Agent
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ai_detective import AIDetective

def main():
    print("🕵️ AI DETECTIVE AGENT DEMO")
    print("=" * 60)
    
    detective = AIDetective()
    
    # Test different types of queries
    test_cases = [
        "Why did the freezer temperature spike?",
        "What caused the temperature increase?", 
        "Investigate the temperature anomaly"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n🔍 TEST CASE {i}: '{query}'")
        print("-" * 50)
        
        try:
            case = detective.investigate_anomaly(query, demo_mode=False)
            
            print(f"\n✅ CASE SUMMARY:")
            print(f"   • Final Confidence: {case.final_confidence}%")
            print(f"   • Investigation Steps: {len(case.investigation_steps)}")
            print(f"   • Root Cause: {case.root_cause}")
            print(f"   • Business Impact: {case.business_impact}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 