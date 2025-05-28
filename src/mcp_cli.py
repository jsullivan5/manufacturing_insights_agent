#!/usr/bin/env python3
"""
Manufacturing Copilot (MCP) CLI

Command-Line Interface for interacting with the LLM Orchestrator to perform
root cause analysis on manufacturing data.
"""
import argparse
import json
import sys
import os
from typing import Dict, Any, List

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.padding import Padding
from rich.style import Style
from rich.theme import Theme

from src.llm_orchestrator import LLMOrchestrator
from src.config import Settings

# --- Rich Console Setup ---
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "panel_border": "blue",
    "table_header": "bold green",
    "cost": "yellow",
    "recommendation": "bold italic green",
    "timeline_event": "dodger_blue1",
    "status_good": "green",
    "status_bad": "red",
    "status_neutral": "yellow",
})
console = Console(theme=custom_theme)

def display_welcome_message() -> None:
    """Displays a flashy welcome message."""
    welcome_panel = Panel(
        Text("🏭 Manufacturing Copilot CLI 🏭", justify="center", style="bold magenta"),
        title="[bold green]MCP Root Cause Investigator[/bold green]",
        border_style="panel_border",
        padding=(1, 2)
    )
    console.print(welcome_panel)
    console.print("Starting investigation... This may take a few moments.", style="info")

def format_cost(cost: float) -> Text:
    """Formats cost with color based on value."""
    if cost > 50:
        style = "danger"
    elif cost > 10:
        style = "warning"
    else:
        style = "cost"
    return Text(f"${cost:.2f}", style=style)

def display_final_report(report: Dict[str, Any]) -> None:
    """
    Displays the final investigation report in a structured and visually appealing format.
    """
    console.rule("[bold green]📝 Investigation Report 📝[/bold green]", style="green")

    # --- Root Cause Statement ---
    root_cause_panel = Panel(
        Text(report.get('root_cause_statement', 'Not available.'), style="bold white"),
        title="[bold yellow]Root Cause[/bold yellow]",
        border_style="panel_border",
        expand=True
    )
    console.print(Padding(root_cause_panel, (1, 0)))

    # --- Event Timeline ---
    timeline_table = Table(title="[bold dodger_blue1]⏳ Event Timeline[/bold dodger_blue1]", show_header=True, header_style="table_header", border_style="dim blue")
    timeline_table.add_column("Time (UTC)", style="dim cyan", width=25)
    timeline_table.add_column("Description", style="white")

    event_timeline = report.get('event_timeline_summary', [])
    if not event_timeline:
        timeline_table.add_row(Text("No timeline events available.", style="italic dim white"), "")
    else:
        for event in event_timeline:
            timeline_table.add_row(event.get('time', 'N/A'), Text(event.get('description', 'N/A'), style="timeline_event"))
    console.print(Padding(timeline_table, (1,0)))

    # --- Business Impact ---
    impact_data = report.get('business_impact_summary', {})
    impact_table = Table(title="[bold orange1]💸 Business Impact[/bold orange1]", show_header=False, border_style="dim orange1", box=None)
    impact_table.add_column("Metric")
    impact_table.add_column("Value")

    impact_table.add_row(Text("Total Cost (USD)", style="white"), format_cost(impact_data.get('total_cost_usd', 0.0)))
    impact_table.add_row(Text("Energy Cost (USD)", style="white"), format_cost(impact_data.get('energy_cost_usd', 0.0)))
    impact_table.add_row(Text("Product Risk (USD)", style="white"), format_cost(impact_data.get('product_risk_usd', 0.0)))
    
    severity_level = impact_data.get('severity_level', 'unknown')
    severity_style = "status_neutral"
    if severity_level == "critical" or severity_level == "high":
        severity_style = "status_bad"
    elif severity_level == "low":
        severity_style = "status_good"
    impact_table.add_row(Text("Severity Level", style="white"), Text(severity_level.capitalize(), style=severity_style))
    
    details = impact_data.get('details', 'No details provided.')

    impact_panel_content = Table.grid(expand=True)
    impact_panel_content.add_row(impact_table)
    impact_panel_content.add_row(Padding(Text(f"Details: {details}", style="italic dim white"), (1,0,0,0)))
    
    impact_panel = Panel(
        impact_panel_content,
        title="[bold orange1]💸 Business Impact[/bold orange1]",
        border_style="panel_border",
        expand=True
    )
    console.print(Padding(impact_panel, (1, 0)))


    # --- Recommendations ---
    recommendations = report.get('recommendations', [])
    reco_content = Text()
    if not recommendations:
        reco_content = Text("No recommendations available.", style="italic dim white")
    else:
        for i, rec in enumerate(recommendations):
            reco_content.append(f"{i+1}. ", style="bold cyan")
            reco_content.append(rec + "\n", style="recommendation")

    recommendations_panel = Panel(
        reco_content,
        title="[bold chartreuse1]💡 Recommendations[/bold chartreuse1]",
        border_style="panel_border",
        expand=True
    )
    console.print(Padding(recommendations_panel, (1,0)))

    # --- Orchestrator Stats ---
    stats_table = Table(title="[bold bright_magenta]📊 Orchestrator Stats[/bold bright_magenta]", show_header=False, border_style="dim magenta", box=None)
    stats_table.add_column("Stat")
    stats_table.add_column("Value")

    final_confidence = report.get('final_confidence_score', report.get('orchestrator_confidence_before_summary', 0.0))
    confidence_style = "status_good" if final_confidence >= 0.75 else ("status_neutral" if final_confidence >= 0.5 else "status_bad")

    stats_table.add_row(Text("Final Confidence", style="white"), Text(f"{final_confidence:.2%}", style=confidence_style))
    stats_table.add_row(Text("Total Steps", style="white"), Text(str(report.get('total_steps', 'N/A')), style="magenta"))
    stats_table.add_row(Text("Estimated Cost (USD)", style="white"), format_cost(report.get('total_cost_usd', 0.0)))
    
    status = report.get('status', 'Unknown')
    status_style = "status_good" if "complete" in status.lower() else "status_bad"
    stats_table.add_row(Text("Status", style="white"), Text(status, style=status_style))


    console.print(Padding(stats_table, (1,0)))
    console.rule(style="green")

def main():
    parser = argparse.ArgumentParser(
        description="Manufacturing Copilot (MCP) - Root Cause Analysis CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "query",
        type=str,
        help="The natural language query to investigate (e.g., \"Why did the freezer temperature spike yesterday afternoon?\")"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to a JSON file containing a pre-canned run profile (query, settings overrides)."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Override maximum investigation steps from settings."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="Override confidence threshold from settings."
    )
    # Add more CLI arguments for other settings if needed

    args = parser.parse_args()

    display_welcome_message()

    query = args.query
    settings_overrides = {}

    if args.profile:
        try:
            with open(args.profile, 'r') as f:
                profile_data = json.load(f)
            query = profile_data.get('query', query) # Profile query overrides CLI arg if present
            settings_overrides = profile_data.get('settings', {})
            console.print(f"Loaded profile from: [bold cyan]{args.profile}[/bold cyan]", style="info")
        except Exception as e:
            console.print(f"Error loading profile file '{args.profile}': {e}", style="danger")
            sys.exit(1)

    # Apply CLI overrides (they take precedence over profile settings)
    if args.max_steps is not None:
        settings_overrides['max_steps'] = args.max_steps
    if args.confidence_threshold is not None:
        settings_overrides['confidence_threshold'] = args.confidence_threshold
    
    try:
        settings = Settings(**settings_overrides) # Initialize with potential overrides
        
        if not settings.openai_api_key or "your_key_here" in settings.openai_api_key :
            console.print(Panel(
                Text("OpenAI API Key (MCP_OPENAI_API_KEY) is missing or a placeholder.\nPlease set it in your .env file or as an environment variable.", style="bold red"),
                title="[bold red]Configuration Error[/bold red]",
                border_style="red"
            ))
            sys.exit(1)

        orchestrator = LLMOrchestrator(settings)
        
        # This is where you might implement a live update if the orchestrator supports it.
        # For now, we run it and then display the final report.
        # with Live(console=console, refresh_per_second=4) as live:
        # live.update(...) # Update with progress if possible
        
        final_report = orchestrator.run(query)
        display_final_report(final_report)

    except ValueError as ve:
        console.print(f"[danger]Configuration Error: {ve}[/danger]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[danger]An unexpected error occurred: {e}[/danger]")
        import traceback
        console.print(traceback.format_exc(), style="dim white")
        sys.exit(1)

if __name__ == "__main__":
    main()
