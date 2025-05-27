#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) - Root Cause Demo

Demonstrates the root cause agent's ability to investigate manufacturing anomalies
using intelligent tool selection and deterministic confidence scoring.

This showcases the agent's core capabilities:
- Metadata-driven tool selection
- Neutral atomic tools that return structured data
- Deterministic confidence scoring
- Business impact calculation
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.root_cause_agent import RootCauseAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure rich console
console = Console()

def run_demo(query: str = None, max_steps: int = 4, confidence_threshold: float = 0.85):
    """
    Run the demo with the given query.
    
    Args:
        query: The natural language query to investigate
        max_steps: Maximum number of investigation steps
        confidence_threshold: Confidence threshold for early stopping
    """
    # Default queries for demo if none provided
    demo_queries = [
        "Why did the freezer temperature spike yesterday afternoon?",
        "What caused the compressor to run more than usual yesterday?",
        "Was the increase in power consumption related to door activity?"
    ]
    
    # Use provided query or default to first demo query
    if not query:
        query = demo_queries[0]
    
    # Introduction
    console.print(Panel.fit(
        "[bold blue]Manufacturing Copilot - Root Cause Agent Demo[/bold blue]",
        subtitle="[italic]Showcasing metadata-driven causal analysis[/italic]"
    ))
    
    console.print("\n[bold]Demo query:[/bold]", query)
    console.print("[dim]The agent will investigate this query using neutral atomic tools.[/dim]")
    console.print("[dim]Confidence will increase with evidence of causality.[/dim]\n")
    
    # Create agent
    agent = RootCauseAgent(max_steps=max_steps, confidence_threshold=confidence_threshold)
    
    # Run investigation
    console.print("[bold green]Starting investigation...[/bold green]")
    console.print("-" * 80)
    
    # Track start time
    start_time = datetime.now()
    
    # Run investigation
    report = agent.investigate(query, verbose=True)
    
    # Track end time
    end_time = datetime.now()
    investigation_time = (end_time - start_time).total_seconds()
    
    # Print investigation summary
    console.print("\n[bold green]Investigation Complete![/bold green]")
    console.print(f"[dim]Time taken: {investigation_time:.1f} seconds[/dim]")
    console.print("-" * 80)
    
    # Print report details in a structured format
    console.print("[bold]Root Cause:[/bold]")
    console.print(Panel(report.get('root_cause', 'Unknown'), expand=False))
    
    # Print business impact
    impact = report.get('business_impact', {})
    console.print("\n[bold]Business Impact:[/bold]")
    impact_table = Table(show_header=True)
    impact_table.add_column("Metric", style="cyan")
    impact_table.add_column("Value", style="green")
    
    # Add impact metrics
    impact_table.add_row("Total Cost", f"${impact.get('total_cost', 0):.2f}")
    impact_table.add_row("Energy Cost", f"${impact.get('energy_cost', 0):.2f}")
    impact_table.add_row("Product Risk", f"${impact.get('product_risk', 0):.2f}")
    impact_table.add_row("Severity", impact.get('severity', 'Unknown'))
    
    console.print(impact_table)
    
    # Print timeline
    console.print("\n[bold]Event Timeline:[/bold]")
    timeline_table = Table(show_header=True)
    timeline_table.add_column("Time", style="cyan")
    timeline_table.add_column("Event", style="white")
    
    for event in report.get('timeline', []):
        timeline_table.add_row(event.get('time', 'Unknown'), event.get('description', 'Unknown'))
    
    console.print(timeline_table)
    
    # Print recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    for i, rec in enumerate(report.get('recommendations', []), 1):
        console.print(f"[cyan]{i}.[/cyan] {rec}")
    
    # Print confidence
    confidence_pct = int(report.get('confidence', 0) * 100)
    console.print(f"\n[bold]Final Confidence:[/bold] [{'green' if confidence_pct >= 85 else 'yellow'}]{confidence_pct}%[/{'green' if confidence_pct >= 85 else 'yellow'}]")
    
    # Save case
    case_file = agent.save_case()
    console.print(f"\n[dim]Case saved to {case_file}[/dim]")
    console.print("-" * 80)


def main():
    """
    Main function to run the demo with command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Manufacturing Copilot - Root Cause Agent Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/demo.py
  python src/demo.py --query "Why did the freezer temperature spike yesterday?"
  python src/demo.py --steps 6 --threshold 0.9
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        help="Natural language query to investigate"
    )
    
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=4,
        help="Maximum number of investigation steps (default: 4)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.85,
        help="Confidence threshold for early stopping (default: 0.85)"
    )
    
    args = parser.parse_args()
    
    try:
        run_demo(args.query, args.steps, args.threshold)
    except KeyboardInterrupt:
        console.print("\n[bold red]Demo cancelled by user.[/bold red]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error running demo: {e}[/bold red]")
        logger.exception("Demo failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 