#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Command Line Interface

Main entry point for the Manufacturing Copilot CLI. Accepts natural language
queries about manufacturing operations and provides data-driven insights using
semantic search, time-series analysis, and LLM reasoning.

Usage:
    python src/mcp.py "Why did Freezer A use more power last night?"
    python src/mcp.py "What caused the temperature spike on Monday?"
    python src/mcp.py "Is the compressor running normally?"
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional

# Add the project root to the Python path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.glossary import TagGlossary
from src.tools import load_data, summarize_metric, quality_summary
from src.tools.data_loader import get_data_time_range
from src.interpreter import interpret_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ManufacturingCopilot:
    """
    Main Manufacturing Copilot engine for processing natural language queries
    about manufacturing operations and generating data-driven insights.
    """
    
    def __init__(self, use_interpreter: bool = True):
        """Initialize the MCP with glossary and tools."""
        print("🏭 Manufacturing Copilot - Initializing...")
        
        self.use_interpreter = use_interpreter
        
        try:
            if not use_interpreter:
                # Initialize tag glossary for semantic search (legacy mode)
                print("   Loading tag glossary with semantic search...")
                self.glossary = TagGlossary()
                print(f"   ✅ Loaded {len(self.glossary.list_all_tags())} tags for analysis")
            else:
                print("   🧠 Using intelligent query interpreter...")
                print("   ✅ Ready for natural language queries!")
            
            print("   Ready for natural language queries!\n")
            
        except Exception as e:
            print(f"   ❌ Initialization failed: {e}")
            print("   Make sure OPENAI_API_KEY is set in your .env file")
            sys.exit(1)
    
    def process_query(self, query: str, time_window_hours: int = 24) -> None:
        """
        Process a natural language query and provide insights.
        
        Args:
            query: Natural language question about manufacturing operations
            time_window_hours: How many hours of recent data to analyze (default: 24, ignored in interpreter mode)
        """
        print(f"🔍 Query: '{query}'")
        print("=" * 60)
        
        if self.use_interpreter:
            # Use the new intelligent interpreter
            try:
                result = interpret_query(query)
                print(result)
                print()
                print("✅ Analysis complete! Next steps:")
                print("   • Run anomaly detection to find unusual patterns")
                print("   • Correlate with related tags for root cause analysis")
                print("   • Generate visualizations and detailed insights")
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"   ❌ Error: {e}")
        else:
            # Use the legacy detailed processing mode
            self._process_query_legacy(query, time_window_hours)
    
    def _process_query_legacy(self, query: str, time_window_hours: int = 24) -> None:
        """Legacy detailed query processing method."""
        try:
            # Step 1: Find relevant tags using semantic search
            print("1️⃣  Finding relevant tags...")
            tag_results = self.glossary.search_tags(query, top_k=3)
            
            if not tag_results:
                print("   ❌ No relevant tags found for your query")
                return
            
            # Use the top matching tag
            primary_tag = tag_results[0]
            print(f"   🎯 Primary tag: {primary_tag['tag']} (similarity: {primary_tag['similarity_score']*100:.1f}%)")
            print(f"   📝 Description: {primary_tag['description']}")
            print(f"   📊 Unit: {primary_tag['unit']}")
            
            # Show alternative tags
            if len(tag_results) > 1:
                print(f"   🔄 Alternative tags:")
                for i, result in enumerate(tag_results[1:], 2):
                    print(f"      {i}. {result['tag']} (similarity: {result['similarity_score']*100:.1f}%)")
            
            print()
            
            # Step 2: Load data for the primary tag
            print("2️⃣  Loading time-series data...")
            
            # Calculate time window based on available data
            try:
                # Get the actual data time range
                data_range = get_data_time_range(primary_tag['tag'])
                data_end = data_range['end']
                
                # Calculate start time based on requested hours from the end of available data
                start_time = data_end - timedelta(hours=time_window_hours)
                
                # Ensure we don't go before the data starts
                data_start = data_range['start']
                if start_time < data_start:
                    start_time = data_start
                    actual_hours = (data_end - start_time).total_seconds() / 3600
                    print(f"   ⚠️  Requested {time_window_hours} hours, but only {actual_hours:.1f} hours available")
                
            except Exception as e:
                logger.warning(f"Could not determine data range, using default: {e}")
                # Fallback to a reasonable default
                data_end = datetime.now()
                start_time = data_end - timedelta(hours=time_window_hours)
            
            tag_data = load_data(
                tag=primary_tag['tag'],
                start=start_time,
                end=data_end
            )
            
            print(f"   📈 Loaded {len(tag_data)} data points")
            print(f"   📅 Time range: {tag_data['Timestamp'].min()} to {tag_data['Timestamp'].max()}")
            print()
            
            # Step 3: Generate summary statistics
            print("3️⃣  Analyzing data patterns...")
            
            # Basic statistics
            stats = summarize_metric(tag_data)
            print(f"   📊 Statistics:")
            print(f"      • Mean: {stats['mean']:.2f} {primary_tag['unit']}")
            print(f"      • Range: {stats['min']:.2f} to {stats['max']:.2f} {primary_tag['unit']}")
            print(f"      • Change: {stats['change']:+.2f} {primary_tag['unit']} ({stats['change_pct']:+.1f}%)")
            print(f"      • Std Dev: {stats['std']:.2f} {primary_tag['unit']}")
            
            # Data quality
            quality = quality_summary(tag_data)
            if 'error' not in quality:
                print(f"   🔍 Data Quality:")
                print(f"      • Good: {quality['good_pct']:.1f}%")
                if quality['questionable_pct'] > 0:
                    print(f"      • Questionable: {quality['questionable_pct']:.1f}%")
                if quality['bad_pct'] > 0:
                    print(f"      • Bad: {quality['bad_pct']:.1f}%")
            
            print()
            
            # Step 4: Sample data preview
            print("4️⃣  Recent data sample:")
            recent_data = tag_data.tail(5)
            for _, row in recent_data.iterrows():
                timestamp = row['Timestamp'].strftime('%Y-%m-%d %H:%M')
                value = row['Value']
                quality = row['Quality']
                print(f"   📅 {timestamp}: {value:8.2f} {primary_tag['unit']} ({quality})")
            
            print()
            print("✅ Analysis complete! Next steps:")
            print("   • Run anomaly detection to find unusual patterns")
            print("   • Correlate with related tags for root cause analysis")
            print("   • Generate visualizations and detailed insights")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"   ❌ Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manufacturing Copilot - Natural Language Manufacturing Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/mcp.py "Why did Freezer A use more power last night?"
  python src/mcp.py "What caused the temperature spike?"
  python src/mcp.py "Is the compressor running normally?"
  python src/mcp.py "Show me door activity patterns" --hours 48
  python src/mcp.py "Show me freezer temperatures last night" --legacy
        """
    )
    
    parser.add_argument(
        "query",
        help="Natural language question about manufacturing operations"
    )
    
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of hours of data to analyze (default: 24, only used in legacy mode)"
    )
    
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy detailed processing mode instead of intelligent interpreter"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run MCP
    try:
        mcp = ManufacturingCopilot(use_interpreter=not args.legacy)
        mcp.process_query(args.query, args.hours)
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 