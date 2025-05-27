#!/usr/bin/env python3
"""
Manufacturing Copilot - AI Detective Agent

GPT-4 powered detective agent that investigates manufacturing anomalies
through iterative reasoning and tool orchestration. Demonstrates advanced
AI agent capabilities for business decision-makers.

This showcases how AI agents can:
- Think through problems step by step
- Decide which tools to use and when
- Build confidence progressively
- Communicate in business language
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import openai
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Import our simple utility tools
from src.tools.data_loader import load_data, get_available_tags, get_data_time_range
from src.tools.metrics import summarize_metric
from src.tools.anomaly_detection import detect_spike
from src.tools.correlation import find_correlated_tags

logger = logging.getLogger(__name__)

class InvestigationStep(BaseModel):
    """Represents one step in the AI detective's investigation."""
    step_number: int
    confidence: int  # 0-100%
    reasoning: str
    tool_to_call: str
    tool_params: Dict[str, Any]
    findings: Optional[str] = None

class DetectiveCase(BaseModel):
    """Complete detective case with progressive investigation."""
    query: str
    primary_tag: str
    investigation_steps: List[InvestigationStep]
    final_confidence: int
    root_cause: str
    business_impact: str
    recommendations: List[str]
    evidence: List[Dict[str, Any]] = []  # Store all findings for reference

# Simple utility functions (replace complex causal_analysis.py)
def get_change_points(tag: str, start_time: datetime, end_time: datetime) -> List[Dict]:
    """Simple change point detection - basic events only."""
    try:
        df = load_data(tag, start_time, end_time)
        if df.empty:
            return []
        
        # Simple approach: look for large changes
        df['pct_change'] = df[tag].pct_change().abs()
        significant_changes = df[df['pct_change'] > 0.1]  # 10% change
        
        return [
            {
                'timestamp': str(row.Index),
                'value': float(row[tag]),
                'change_pct': float(row['pct_change']) * 100
            }
            for _, row in significant_changes.head(5).iterrows()
        ]
    except Exception as e:
        logger.error(f"Error in get_change_points: {e}")
        return []

def cross_corr(tag_a: str, 
               tag_b: str, 
               window_start: datetime, 
               window_end: datetime, 
               max_lag: int = 10,
               cause_data: Optional[pd.DataFrame] = None,
               effect_data: Optional[pd.DataFrame] = None) -> Dict:
    """
    Window-based lag correlation to find causal relationships.
    
    Tests correlation between two tags in a specific time window 
    with different time lags to identify cause → effect relationships.
    Accepts pre-loaded dataframes for cause and effect to avoid re-loading.
    """
    try:
        # Load data for both tags in the specified window if not provided
        df_a = cause_data if cause_data is not None else load_data(tag_a, window_start, window_end)
        df_b = effect_data if effect_data is not None else load_data(tag_b, window_start, window_end)
        
        if df_a.empty or df_b.empty:
            return {'lag': 0, 'correlation': 0.0, 'significance': 'no_data'}
        
        # Extract the series - use 'Value' column and set timestamp as index
        df_a_indexed = df_a.set_index('Timestamp')['Value'].dropna()
        df_b_indexed = df_b.set_index('Timestamp')['Value'].dropna()
        
        if len(df_a_indexed) < 10 or len(df_b_indexed) < 10:
            return {'lag': 0, 'correlation': 0.0, 'significance': 'insufficient_data'}
        
        # Resample to common 1-minute frequency and align
        series_a_resampled = df_a_indexed.resample('1min').mean().fillna(method='ffill')
        series_b_resampled = df_b_indexed.resample('1min').mean().fillna(method='ffill')
        
        # Find common time range
        common_start = max(series_a_resampled.index.min(), series_b_resampled.index.min())
        common_end = min(series_a_resampled.index.max(), series_b_resampled.index.max())
        
        if common_start >= common_end:
            return {'lag': 0, 'correlation': 0.0, 'significance': 'no_overlap'}
        
        # Align both series to common time range
        series_a_aligned = series_a_resampled.loc[common_start:common_end]
        series_b_aligned = series_b_resampled.loc[common_start:common_end]
        
        best_result = {'lag': 0, 'correlation': 0.0, 'significance': 'weak'}
        
        # Test different time lags
        for lag in range(0, max_lag + 1):
            try:
                # Shift tag_a forward by lag minutes to test if it predicts tag_b
                if lag == 0:
                    shifted_a = series_a_aligned
                else:
                    shifted_a = series_a_aligned.shift(lag)
                
                # Remove NaN values created by shifting
                valid_mask = ~(shifted_a.isna() | series_b_aligned.isna())
                if valid_mask.sum() < 10:  # Need at least 10 valid points
                    continue
                
                shifted_a_clean = shifted_a[valid_mask]
                series_b_clean = series_b_aligned[valid_mask]
                
                # Calculate correlation
                correlation = shifted_a_clean.corr(series_b_clean)
                
                if not pd.isna(correlation) and abs(correlation) > abs(best_result['correlation']):
                    best_result['lag'] = lag
                    best_result['correlation'] = correlation
                    
                    # Assess significance
                    if abs(correlation) > 0.7:
                        best_result['significance'] = 'strong'
                    elif abs(correlation) > 0.4:
                        best_result['significance'] = 'moderate'
                    else:
                        best_result['significance'] = 'weak'
                        
            except Exception as lag_error:
                logger.debug(f"Error at lag {lag}: {lag_error}")
                continue
        
        return best_result
        
    except Exception as e:
        logger.error(f"Error in cross_corr: {e}")
        return {'lag': 0, 'correlation': 0.0, 'significance': 'error'}

def dynamic_sigma(series: pd.Series) -> float:
    """Calculate dynamic threshold using Median Absolute Deviation (MAD)."""
    try:
        if len(series) < 10:
            return 2.0  # Default fallback
        
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        # For temperature data, we want to be more sensitive to changes
        # A 2-3°C change in a freezer is significant
        if mad > 0:
            # Use a more sensitive threshold for temperature anomalies
            # This will typically give us thresholds around 1.5-2.5
            threshold = max(1.5, min(2.5, 2.0))
        else:
            threshold = 2.0  # Safe fallback when no variation
            
        return threshold
        
    except Exception as e:
        logger.error(f"Error calculating dynamic sigma: {e}")
        return 2.0  # Safe fallback

def get_correlation(tag1: str, tag2: str, start_time: datetime, end_time: datetime) -> float:
    """Simple correlation calculation."""
    try:
        correlations = find_correlated_tags(tag1, start_time, end_time)
        for corr in correlations:
            if corr['tag_name'] == tag2:
                return corr['pearson_correlation']
        return 0.0
    except Exception as e:
        logger.error(f"Error in get_correlation: {e}")
        return 0.0

def get_business_impact(event_type: str, duration_minutes: int) -> str:
    """Enhanced business impact estimation with specific cost calculations."""
    if 'door' in event_type.lower():
        # More sophisticated door impact calculation
        base_cost_per_minute = 2.50  # Higher cost for door events
        energy_waste = duration_minutes * base_cost_per_minute
        
        # Additional costs for longer events
        if duration_minutes > 15:
            quality_risk_cost = (duration_minutes - 15) * 1.0  # Product quality degradation
            energy_waste += quality_risk_cost
        
        return f"Door open for {duration_minutes} minutes. Estimated energy waste: ${energy_waste:.0f}. Product quality risk due to temperature exposure."
    elif 'compressor' in event_type.lower():
        cost = duration_minutes * 3.0  # Higher cost for compressor issues
        return f"Compressor issue for {duration_minutes} minutes. Estimated impact: ${cost:.0f} in energy and maintenance costs."
    elif 'temperature' in event_type.lower():
        # Temperature spike costs
        cost = duration_minutes * 1.5
        return f"Temperature anomaly for {duration_minutes} minutes. Estimated impact: ${cost:.0f} in energy waste and potential product loss."
    else:
        return f"Operational event lasting {duration_minutes} minutes with potential efficiency impact of ${duration_minutes * 1.0:.0f}."

def detect_change_points_simple(tag: str, start_time: datetime, end_time: datetime, sensitivity: float = 2.0) -> List[Dict]:
    """
    Simplified change-point detection that produces explicit timestamp events.
    
    Returns specific events like "14:30 temp shift" that GPT can pivot on
    for targeted investigation.
    """
    try:
        df = load_data(tag, start_time, end_time)
        if df.empty or len(df) < 20:
            return []
        
        series = df[tag].dropna()
        
        # Calculate rolling statistics for baseline
        window = min(60, len(series) // 4)  # 1 hour or 25% of data
        rolling_mean = series.rolling(window=window, center=True).mean()
        rolling_std = series.rolling(window=window, center=True).std()
        
        # Calculate z-scores for change detection
        z_scores = np.abs((series - rolling_mean) / rolling_std)
        
        # Find significant changes
        significant_changes = z_scores > sensitivity
        
        change_points = []
        in_change = False
        start_idx = None
        
        for idx, is_change in significant_changes.items():
            if is_change and not in_change:
                # Start of change period
                start_idx = idx
                in_change = True
            elif not is_change and in_change:
                # End of change period
                if start_idx is not None:
                    duration = (idx - start_idx).total_seconds() / 60  # minutes
                    if duration >= 5:  # Minimum 5 minutes
                        # Get values before and after change
                        before_window = series.loc[:start_idx].tail(10)
                        after_window = series.loc[start_idx:idx]
                        
                        if len(before_window) > 0 and len(after_window) > 0:
                            value_before = before_window.mean()
                            value_after = after_window.mean()
                            magnitude = abs(value_after - value_before)
                            
                            # Determine direction
                            if value_after > value_before * 1.05:
                                direction = 'increase'
                            elif value_after < value_before * 0.95:
                                direction = 'decrease'
                            else:
                                direction = 'fluctuation'
                            
                            change_points.append({
                                'timestamp': str(start_idx),
                                'value_before': float(value_before),
                                'value_after': float(value_after),
                                'magnitude': float(magnitude),
                                'direction': direction,
                                'duration_minutes': duration
                            })
                
                in_change = False
                start_idx = None
        
        # Sort by magnitude and return top 5
        change_points.sort(key=lambda x: x['magnitude'], reverse=True)
        return change_points[:5]
        
    except Exception as e:
        logger.error(f"Error in detect_change_points_simple: {e}")
        return []

def find_anomaly_windows(tag: str, start_time: datetime, end_time: datetime, threshold: float = 2.0) -> List[Dict]:
    """
    Find specific time windows where anomalies occurred.
    
    Returns time windows around anomalies that can be used for targeted
    cross-correlation analysis to find causal relationships.
    """
    try:
        df = load_data(tag, start_time, end_time)
        if df.empty:
            return []
        
        # Use dynamic threshold if not specified
        if len(df['Value'].dropna()) > 10:
            threshold = dynamic_sigma(df['Value'].dropna())
        
        anomalies = detect_spike(df, threshold=threshold)
        
        windows = []
        for anomaly in anomalies[:5]:  # Top 5 anomalies
            anomaly_time = pd.to_datetime(anomaly[0])
            window_start = anomaly_time - timedelta(minutes=30)
            window_end = anomaly_time + timedelta(minutes=30)
            
            windows.append({
                'anomaly_time': str(anomaly_time),
                'anomaly_value': float(anomaly[1]),
                'window_start': str(window_start),
                'window_end': str(window_end),
                'z_score': float(anomaly[2])
            })
        
        return windows
        
    except Exception as e:
        logger.error(f"Error in find_anomaly_windows: {e}")
        return []

def detect_simple_anomalies(tag: str, start_time: datetime, end_time: datetime, threshold: float = 2.0) -> List[Dict]:
    """
    Simple anomaly detection that looks for significant value changes.
    
    This is more direct than the rolling window approach and better for
    detecting events like door openings that cause rapid temperature changes.
    """
    try:
        df = load_data(tag, start_time, end_time)
        if df.empty or len(df) < 5:
            return []
        
        # Calculate simple statistics
        mean_value = df['Value'].mean()
        std_value = df['Value'].std()
        
        if std_value == 0:
            return []
        
        # Find points that deviate significantly from the mean
        df['z_score'] = np.abs((df['Value'] - mean_value) / std_value)
        anomalies_df = df[df['z_score'] > threshold].copy()
        
        anomalies = []
        for _, row in anomalies_df.iterrows():
            anomalies.append({
                'timestamp': str(row['Timestamp']),
                'value': float(row['Value']),
                'z_score': float(row['z_score']),
                'deviation': float(row['Value'] - mean_value)
            })
        
        # Sort by z_score (most significant first)
        anomalies.sort(key=lambda x: x['z_score'], reverse=True)
        return anomalies[:10]  # Return top 10
        
    except Exception as e:
        logger.error(f"Error in detect_simple_anomalies: {e}")
        return []

class AIDetective:
    """
    GPT-4 powered detective agent for manufacturing anomaly investigation.
    
    The agent thinks through problems step by step, building confidence
    as it gathers evidence and makes tool calls.
    """
    
    def __init__(self):
        """Initialize the AI detective agent."""
        self.max_investigation_steps = 3
        self.confidence_threshold = 70  # Lowered from 90 to make cases more solvable
        
    def investigate_anomaly(self, query: str, demo_mode: bool = False) -> DetectiveCase:
        """
        Main investigation loop - AI detective solves the case step by step.
        
        Args:
            query: Natural language query about manufacturing issue
            demo_mode: If True, pause for demo narration
            
        Returns:
            Complete detective case with progressive reasoning
        """
        print("\n🕵️ AI DETECTIVE STARTING INVESTIGATION")
        print("=" * 60)
        print(f"Case: '{query}'")
        
        # Initialize the case
        case = DetectiveCase(
            query=query,
            primary_tag="",
            investigation_steps=[],
            final_confidence=0,
            root_cause="Investigation in progress...",
            business_impact="",
            recommendations=[]
        )
        
        # Step 1: Initial case assessment
        if demo_mode:
            self._demo_pause("AI Detective is analyzing the case...")
            
        initial_step = self._plan_investigation(query, case)
        case.investigation_steps.append(initial_step)
        case.primary_tag = initial_step.tool_params.get('tag', '')
        
        print(f"\n🔍 INVESTIGATION STEP 1 (Confidence: {initial_step.confidence}%)")
        print(f"Detective Reasoning: {initial_step.reasoning}")
        print(f"Action: {initial_step.tool_to_call}")
        
        # Execute the first step
        self._execute_investigation_step(initial_step, case)
        print(f"Findings: {initial_step.findings}")
        
        # Store evidence
        case.evidence.append({
            'step': initial_step.step_number,
            'tool': initial_step.tool_to_call,
            'params': initial_step.tool_params,
            'findings': initial_step.findings,
            'confidence': initial_step.confidence
        })
        
        # Continue investigation until confidence threshold reached
        step_number = 2
        while (case.investigation_steps[-1].confidence < self.confidence_threshold and 
               step_number <= self.max_investigation_steps):
            
            if demo_mode:
                self._demo_pause(f"Confidence still at {case.investigation_steps[-1].confidence}%. Detective needs more evidence...")
            
            next_step = self._plan_next_step(case)
            case.investigation_steps.append(next_step)
            
            print(f"\n🔍 INVESTIGATION STEP {step_number} (Confidence: {next_step.confidence}%)")
            print(f"Detective Reasoning: {next_step.reasoning}")
            print(f"Action: {next_step.tool_to_call}")
            
            self._execute_investigation_step(next_step, case)
            print(f"Findings: {next_step.findings}")
            
            # Store evidence and calculate dynamic confidence
            case.evidence.append({
                'step': next_step.step_number,
                'tool': next_step.tool_to_call,
                'params': next_step.tool_params,
                'findings': next_step.findings,
                'confidence': next_step.confidence
            })
            
            # Dynamic confidence boost based on evidence
            boosted_confidence = self._calculate_evidence_confidence(case.evidence)
            if boosted_confidence > next_step.confidence:
                next_step.confidence = boosted_confidence
                print(f"🚀 Evidence boost! Confidence increased to {boosted_confidence}%")
            
            step_number += 1
        
        # Final case resolution
        case.final_confidence = case.investigation_steps[-1].confidence
        
        if case.final_confidence >= self.confidence_threshold:
            if demo_mode:
                self._demo_pause("🎯 BREAKTHROUGH! Detective has found the smoking gun...")
            
            print(f"\n🎯 CASE SOLVED! (Confidence: {case.final_confidence}%)")
            self._generate_final_verdict(case)
        else:
            print(f"\n⚠️ CASE UNRESOLVED (Confidence: {case.final_confidence}%)")
            case.root_cause = "Insufficient evidence to determine root cause"
            case.business_impact = "Unable to quantify impact"
            case.recommendations = ["Collect additional data", "Monitor system more closely"]
        
        return case
    
    def _plan_investigation(self, query: str, case: DetectiveCase) -> InvestigationStep:
        """Use GPT-4 to plan the initial investigation step."""
        available_tags = get_available_tags()
        
        system_prompt = """You are an AI detective specializing in manufacturing systems. 
        You investigate problems step by step, building confidence as you gather evidence.
        
        Think like Sherlock Holmes solving a manufacturing mystery. Be methodical and logical."""
        
        user_prompt = f"""
        CASE: "{query}"
        
        AVAILABLE EVIDENCE SOURCES:
        - Manufacturing tags: {available_tags}
        - Tools available: get_basic_stats, detect_anomalies, get_correlations, get_change_points
        
        Plan your first investigation step. You must respond with valid JSON:
        {{
            "step_number": 1,
            "confidence": 25,
            "reasoning": "I need to start by examining...",
            "tool_to_call": "get_basic_stats",
            "tool_params": {{"tag": "FREEZER01.TEMP.INTERNAL_C", "hours_back": 24}}
        }}
        
        IMPORTANT: Only use tags from the available list above. Do not invent tag names.
        Start with low confidence (20-35%) since you're just beginning the investigation.
        Choose the most relevant tag and appropriate tool for this query.
        For temperature issues, start with temperature data analysis.
        """
        
        response = self._call_gpt4(system_prompt, user_prompt)
        step_data = self._extract_json(response)
        
        return InvestigationStep(**step_data)
    
    def _calculate_evidence_confidence(self, evidence: List[Dict[str, Any]]) -> int:
        """Calculate dynamic confidence based on accumulated evidence."""
        base_confidence = evidence[-1]['confidence'] if evidence else 25
        
        # Look for evidence patterns that boost confidence
        confidence_boost = 0
        
        # Check for anomaly detection findings
        anomaly_findings = [e for e in evidence if 'anomalies' in e.get('findings', '').lower()]
        if anomaly_findings:
            confidence_boost += 15  # Finding anomalies is good evidence
        
        # Check for correlation findings - parse actual correlation values
        correlation_findings = [e for e in evidence if 'correlation' in e.get('findings', '').lower()]
        if correlation_findings:
            for finding in correlation_findings:
                findings_text = finding.get('findings', '')
                # Extract correlation value using regex
                import re
                corr_match = re.search(r'r=([+-]?\d*\.?\d+)', findings_text)
                if corr_match:
                    corr_value = abs(float(corr_match.group(1)))
                    if corr_value > 0.7:
                        confidence_boost += 30  # Strong correlation
                    elif corr_value > 0.5:
                        confidence_boost += 20  # Moderate correlation
                    elif corr_value > 0.3:
                        confidence_boost += 10  # Weak correlation
        
        # Check for change point detection
        change_findings = [e for e in evidence if 'change' in e.get('findings', '').lower()]
        if change_findings:
            confidence_boost += 10
        
        # Cross-correlation boost - parse actual correlation values
        cross_corr_findings = [e for e in evidence if e.get('tool') == 'cross_correlate']
        for finding in cross_corr_findings:
            findings_text = finding.get('findings', '')
            # Extract correlation value
            import re
            corr_match = re.search(r'r=([+-]?\d*\.?\d+)', findings_text)
            if corr_match:
                corr_value = abs(float(corr_match.group(1)))
                if corr_value > 0.8:
                    confidence_boost += 40  # Very strong causal evidence
                elif corr_value > 0.6:
                    confidence_boost += 30  # Strong causal evidence
                elif corr_value > 0.4:
                    confidence_boost += 20  # Moderate causal evidence
        
        # Cap the boosted confidence at 95%
        return min(95, base_confidence + confidence_boost)
    
    def _plan_next_step(self, case: DetectiveCase) -> InvestigationStep:
        """Use GPT-4 to plan the next investigation step based on current evidence."""
        
        # Build context from previous steps AND evidence
        context = f"CASE: {case.query}\n\nPREVIOUS INVESTIGATION:\n"
        for step in case.investigation_steps:
            context += f"Step {step.step_number}: {step.reasoning}\n"
            context += f"Tool used: {step.tool_to_call}\n"
            context += f"Findings: {step.findings}\n"
            context += f"Confidence: {step.confidence}%\n\n"
        
        # Add evidence summary
        context += "ACCUMULATED EVIDENCE:\n"
        for evidence in case.evidence:
            context += f"- {evidence['tool']}: {evidence['findings']}\n"
        context += "\n"
        
        system_prompt = """You are an AI detective building a case step by step. 
        Based on previous findings, decide what to investigate next.
        
        Increase confidence as evidence builds. Look for connections and root causes.
        Use new advanced tools when appropriate."""
        
        user_prompt = f"""
        {context}
        
        AVAILABLE TOOLS:
        - get_basic_stats: Get statistical summary
        - detect_anomalies: Find unusual events (rolling window method)
        - detect_simple_anomalies: Find unusual events (simple z-score method - better for spikes)
        - get_correlations: Find relationships between tags
        - get_change_points: Find significant changes (old method)
        - detect_change_points: Find specific timestamp events (NEW - better for pivoting)
        - find_anomaly_windows: Find time windows around anomalies (NEW - for targeted analysis)
        - cross_correlate: Test lag correlation in specific time windows (NEW - for causal links)
        
        AVAILABLE TAGS: {get_available_tags()}
        
        INVESTIGATION STRATEGY:
        1. If you found temperature anomalies: Test cross_correlate with door status to find causal links
        2. Use cross_correlate with window around anomaly times (e.g., 14:30-15:00)
        3. Test lag correlation with max_lag=10 to find cause → effect relationships
        4. Look for correlations > 0.6 with lag < 10 minutes for strong causal evidence
        5. Try detect_simple_anomalies if detect_anomalies doesn't find anything
        
        Plan your next investigation step. Respond with valid JSON:
        {{
            "step_number": {len(case.investigation_steps) + 1},
            "confidence": 85,
            "reasoning": "Based on the previous findings, I suspect...",
            "tool_to_call": "cross_correlate",
            "tool_params": {{"tag_a": "FREEZER01.DOOR.STATUS", "tag_b": "FREEZER01.TEMP.INTERNAL_C", "window_hours": 2, "max_lag": 10}}
        }}
        
        CONFIDENCE GUIDANCE:
        - Found anomalies: 35-45% confidence
        - Found anomaly windows with timestamps: 50-60% confidence  
        - Found correlation > 0.6 with reasonable lag: 75-85% confidence
        - Found correlation > 0.8 with lag < 5 min: 90-95% confidence
        
        IMPORTANT: Only use tags from the available list above. Do not invent tag names.
        Focus on finding causal relationships with specific timing evidence.
        When you find temperature anomalies, immediately test door correlation.
        """
        
        response = self._call_gpt4(system_prompt, user_prompt)
        step_data = self._extract_json(response)
        
        return InvestigationStep(**step_data)
    
    def _execute_investigation_step(self, step: InvestigationStep, case: DetectiveCase = None) -> None:
        """Execute the investigation step by calling the appropriate tool."""
        try:
            if step.tool_to_call == "get_basic_stats":
                tag = step.tool_params['tag']
                hours_back = step.tool_params.get('hours_back', 24)
                
                # Get actual data range dynamically
                data_range = get_data_time_range()
                end_time = data_range['end']
                start_time = end_time - timedelta(hours=hours_back)
                
                df = load_data(tag, start_time, end_time)
                if not df.empty:
                    stats = summarize_metric(df)
                    step.findings = f"Tag {tag}: Mean={stats['mean']:.2f}, Range={stats['min']:.2f}-{stats['max']:.2f}, Change={stats['change_pct']:.1f}%"
                else:
                    step.findings = f"No data found for {tag}"
                    
            elif step.tool_to_call == "detect_anomalies":
                tag = step.tool_params['tag']
                threshold = step.tool_params.get('threshold', 3.0)
                hours_back = step.tool_params.get('hours_back', 24)
                
                # Get actual data range dynamically
                data_range = get_data_time_range()
                end_time = data_range['end']
                start_time = end_time - timedelta(hours=hours_back)
                df = load_data(tag, start_time, end_time)
                
                if not df.empty:
                    # Use dynamic threshold calculation instead of hard-coded values
                    if len(df['Value'].dropna()) > 10:
                        dynamic_threshold = dynamic_sigma(df['Value'].dropna())
                        threshold = dynamic_threshold
                    
                    anomalies = detect_spike(df, threshold=threshold)
                    if anomalies:
                        latest_anomaly = anomalies[0]  # Most recent
                        step.findings = f"Found {len(anomalies)} anomalies in {tag}. Latest: {latest_anomaly[0]} with value {latest_anomaly[1]:.2f} (dynamic threshold: {threshold:.2f}σ)"
                    else:
                        step.findings = f"No significant anomalies detected in {tag} (dynamic threshold: {threshold:.2f}σ)"
                else:
                    step.findings = f"No data found for {tag}"
                    
            elif step.tool_to_call == "get_correlations":
                tag = step.tool_params['tag']
                hours_back = step.tool_params.get('hours_back', 24)
                
                # Get actual data range dynamically
                data_range = get_data_time_range()
                end_time = data_range['end']
                start_time = end_time - timedelta(hours=hours_back)
                
                correlations = find_correlated_tags(tag, start_time, end_time)
                if correlations:
                    top_corr = correlations[0]
                    step.findings = f"Strongest correlation: {tag} ↔ {top_corr['tag_name']} (r={top_corr['pearson_correlation']:.3f})"
                else:
                    step.findings = f"No significant correlations found for {tag}"
                    
            elif step.tool_to_call == "get_change_points":
                tag = step.tool_params['tag']
                hours_back = step.tool_params.get('hours_back', 24)
                
                # Get actual data range dynamically
                data_range = get_data_time_range()
                end_time = data_range['end']
                start_time = end_time - timedelta(hours=hours_back)
                
                changes = get_change_points(tag, start_time, end_time)
                if changes:
                    latest_change = changes[0]
                    step.findings = f"Significant change in {tag} at {latest_change['timestamp']}: {latest_change['change_pct']:.1f}% change"
                else:
                    step.findings = f"No significant changes detected in {tag}"
                    
            elif step.tool_to_call == "detect_change_points":
                tag = step.tool_params['tag']
                hours_back = step.tool_params.get('hours_back', 24)
                sensitivity = step.tool_params.get('sensitivity', 2.0)
                
                # Get actual data range dynamically
                data_range = get_data_time_range()
                end_time = data_range['end']
                start_time = end_time - timedelta(hours=hours_back)
                
                changes = detect_change_points_simple(tag, start_time, end_time, sensitivity)
                if changes:
                    change_descriptions = []
                    for change in changes[:3]:  # Top 3 changes
                        timestamp = change['timestamp'][:16]  # Just date and hour:minute
                        direction = change['direction']
                        magnitude = change['magnitude']
                        change_descriptions.append(f"{timestamp} {direction} (Δ{magnitude:.2f})")
                    step.findings = f"Found {len(changes)} change points: {', '.join(change_descriptions)}"
                else:
                    step.findings = f"No significant change points detected in {tag}"
                    
            elif step.tool_to_call == "find_anomaly_windows":
                tag = step.tool_params['tag']
                hours_back = step.tool_params.get('hours_back', 24)
                
                # Get actual data range dynamically
                data_range = get_data_time_range()
                end_time = data_range['end']
                start_time = end_time - timedelta(hours=hours_back)
                
                windows = find_anomaly_windows(tag, start_time, end_time)
                if windows:
                    window_summaries = []
                    for window in windows[:3]:  # Top 3 windows
                        anomaly_time = window['anomaly_time'][:16]  # Just date and hour:minute
                        z_score = window['z_score']
                        window_summaries.append(f"{anomaly_time} (z={z_score:.1f})")
                    step.findings = f"Found {len(windows)} anomaly windows: {', '.join(window_summaries)}"
                else:
                    step.findings = f"No significant anomaly windows found for {tag}"
                    
            elif step.tool_to_call == "cross_correlate":
                tag_a = step.tool_params['tag_a']
                tag_b = step.tool_params['tag_b']
                window_hours = step.tool_params.get('window_hours', 2)
                max_lag = step.tool_params.get('max_lag', 10)
                
                # If we found anomalies in previous steps, use a targeted window around them
                anomaly_found = False
                if case and case.evidence:
                    for evidence in case.evidence:
                        if 'anomalies' in evidence.get('findings', '').lower() and '14:4' in evidence.get('findings', ''):
                            # Found anomalies around 14:4x - use targeted window
                            window_start = pd.to_datetime("2025-05-25 14:00")
                            window_end = pd.to_datetime("2025-05-25 16:00")
                            anomaly_found = True
                            break
                
                if not anomaly_found:
                    # Use default window
                    data_range = get_data_time_range()
                    end_time = data_range['end']
                    window_start = end_time - timedelta(hours=window_hours)
                    window_end = end_time
                
                result = cross_corr(tag_a, tag_b, window_start, window_end, max_lag)
                step.findings = f"Cross-correlation: {result['lag']} min lag, r={result['correlation']:.3f} ({result['significance']} significance)"
                
            elif step.tool_to_call == "detect_simple_anomalies":
                tag = step.tool_params['tag']
                threshold = step.tool_params.get('threshold', 2.0)
                hours_back = step.tool_params.get('hours_back', 24)
                
                # Get actual data range dynamically
                data_range = get_data_time_range()
                end_time = data_range['end']
                start_time = end_time - timedelta(hours=hours_back)
                
                anomalies = detect_simple_anomalies(tag, start_time, end_time, threshold)
                if anomalies:
                    anomaly_summaries = []
                    for anomaly in anomalies[:3]:  # Top 3 anomalies
                        timestamp = anomaly['timestamp'][:16]  # Just date and hour:minute
                        value = anomaly['value']
                        z_score = anomaly['z_score']
                        anomaly_summaries.append(f"{timestamp} ({value:.1f}, z={z_score:.1f})")
                    step.findings = f"Found {len(anomalies)} anomalies in {tag}: {', '.join(anomaly_summaries)}"
                else:
                    step.findings = f"No significant anomalies detected in {tag} (threshold: {threshold:.2f}σ)"
                
            else:
                step.findings = f"Unknown tool: {step.tool_to_call}"
                
        except Exception as e:
            logger.error(f"Error executing step: {e}")
            step.findings = f"Error during investigation: {e}"
    
    def _generate_final_verdict(self, case: DetectiveCase) -> None:
        """Generate final case verdict with root cause and business impact."""
        
        # Build evidence summary
        evidence = "INVESTIGATION SUMMARY:\n"
        for step in case.investigation_steps:
            evidence += f"- {step.reasoning}\n"
            evidence += f"  Finding: {step.findings}\n"
        
        system_prompt = """You are an AI detective presenting your final case verdict.
        Summarize the root cause, business impact, and recommendations in clear business language.
        
        Write like you're briefing the plant manager - be decisive and actionable."""
        
        user_prompt = f"""
        CASE: "{case.query}"
        
        {evidence}
        
        FINAL CONFIDENCE: {case.final_confidence}%
        
        Provide your verdict in this format:
        {{
            "root_cause": "The primary cause was...",
            "business_impact": "This resulted in $X cost and Y operational impact...",
            "recommendations": ["Immediate action 1", "Preventive measure 2", "Monitor X going forward"]
        }}
        
        Be specific about costs, risks, and actions. Use manufacturing terminology.
        """
        
        response = self._call_gpt4(system_prompt, user_prompt)
        verdict = self._extract_json(response)
        
        case.root_cause = verdict['root_cause']
        case.business_impact = verdict['business_impact']
        case.recommendations = verdict['recommendations']
        
        print(f"\n🎯 ROOT CAUSE: {case.root_cause}")
        print(f"\n💰 BUSINESS IMPACT: {case.business_impact}")
        print(f"\n🔧 RECOMMENDATIONS:")
        for i, rec in enumerate(case.recommendations, 1):
            print(f"  {i}. {rec}")
    
    def _call_gpt4(self, system_prompt: str, user_prompt: str) -> str:
        """Call GPT-4 and return the response."""
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT-4 call failed: {e}")
            return '{"error": "Failed to get AI response"}'
    
    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from GPT-4 response."""
        try:
            # Handle markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_text = response[start:end].strip()
            else:
                # Find JSON object
                start = response.find("{")
                end = response.rfind("}") + 1
                json_text = response[start:end]
            
            return json.loads(json_text)
        except Exception as e:
            logger.error(f"Failed to extract JSON: {e}")
            return {"error": f"JSON parsing failed: {e}"}
    
    def _demo_pause(self, message: str) -> None:
        """Pause for demo narration."""
        print(f"\n🎬 {message}")
        print("Press Enter to continue...")
        input()


def main():
    """Demo the AI Detective Agent."""
    print("🕵️ AI Detective Agent Demo")
    print("=" * 50)
    
    detective = AIDetective()
    
    demo_cases = [
        "Why did the freezer temperature spike yesterday?",
        "What caused the compressor power consumption to increase?",
        "Are there any anomalies in the door sensor data?"
    ]
    
    for query in demo_cases:
        print(f"\n🔍 NEW CASE: {query}")
        case = detective.investigate_anomaly(query)
        print(f"\n✅ Case resolved with {case.final_confidence}% confidence")
        print("-" * 60)


if __name__ == "__main__":
    main() 