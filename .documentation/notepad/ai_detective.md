# AI Detective - Manufacturing Copilot

## Overview

The AI Detective is a frontier-tech demonstration of how an LLM-powered agent can investigate manufacturing time-series data, discover causal chains, and explain business impact - all without relying on hardcoded rules or domain-specific logic.

This document outlines the architecture and design principles behind the AI Detective and how it achieves true generalized intelligence for manufacturing analytics.

## Key Design Principles

1. **No Deterministic Hacks**
   - No hardcoded domain rules, fixed thresholds, or canned logic
   - No pre-determined "door → temp" relationships
   - No brittle sigma value optimization

2. **Neutral, Atomic Tools**
   - Tools provide objective analysis with no built-in conclusions
   - All tools return structured JSON for consistent reasoning
   - Tools never "tell" the agent what to think

3. **LLM-Driven Reasoning Loop**
   - GPT generates hypotheses, picks tools, and builds evidence
   - Confidence increases only with valid statistical proof
   - Causality requires evidence: cause precedes effect, |r| > 0.6, reasonable lag, physical plausibility

4. **Evidence-Based Confidence**
   - Each step builds a cumulative evidence chain
   - Confidence scores are deterministic based on statistical strength
   - Requires multiple corroborating signals to reach high confidence

## Architecture Components

### 1. Tag Intelligence (tag_intel.py)
- Extends glossary with automatic binary/numeric type detection
- Provides statistical baselines for normal operation
- Enables semantically-informed tool selection

### 2. Standardized Schemas (schemas.py)
- Defines consistent data structures for all tool outputs
- Enables objective confidence scoring
- Creates a common language for evidence

### 3. Confidence Scoring (scorer.py)
- Implements deterministic confidence calculation
- Considers temporal sequences, correlation strength, anomaly severity
- Requires evidence consistency across multiple signals

### 4. Atomic Tools (atomic_tools.py)
- Modular, statistically-sound analysis functions:
  - `detect_numeric_anomalies`: Finds outliers in continuous data
  - `detect_binary_flips`: Identifies state changes in binary data
  - `detect_change_points`: Pinpoints significant shifts in trends
  - `test_causality`: Evaluates cause-effect relationships with time lag
  - `calculate_impact`: Quantifies business impact in dollars
  - `create_event_sequence`: Builds a timeline of detected events
  - `find_interesting_window`: Identifies most relevant time periods

### 5. Root Cause Agent (root_cause_agent.py)
- Orchestrates the investigation process using LLM reasoning
- Uses semantic search to find relevant tags
- Selects appropriate tools based on context
- Builds evidence through sequential analysis
- Quantifies confidence with each step

## How It Works

1. **Initial Query Understanding**
   - User asks a question like "Why did the freezer temperature spike?"
   - System uses semantic search to find relevant tags
   - No hardcoded tag relationships or pre-determined paths

2. **Iterative Investigation**
   - Agent selects which tool to use based on context
   - Each step builds on previous evidence
   - Confidence increases as causal relationships are proven

3. **Evidence Collection**
   - Agent finds anomalies, correlations, and state changes
   - Statistical significance thresholds are dynamically determined
   - Each finding is stored as structured evidence

4. **Causal Verification**
   - Tests whether one event precedes another
   - Verifies correlation strength and directionality
   - Assesses physical plausibility based on domain understanding

5. **Business Impact Assessment**
   - Quantifies costs based on event duration and severity
   - Calculates energy waste and product risk
   - Provides actionable recommendations

## Example Investigation Flow

1. Detect temperature anomalies in freezer
2. Identify door opening events in the same timeframe
3. Test causal relationship between door status and temperature
4. Verify that door openings precede temperature spikes
5. Calculate business impact of the events
6. Generate recommendations to prevent future occurrences

## Running the Demo

To run the AI Detective demo:

```bash
python src/demo.py --query "Why did the freezer temperature spike yesterday afternoon?" --steps 6
```

Options:
- `--query`: The natural language question to investigate
- `--steps`: Maximum number of investigation steps (default: 3)
- `--demo-mode`: Enable narrative pauses for presentations

## Key Characteristics

The system is designed with the following characteristics, avoiding hardcoded domain rules:

1. **Data-Driven Discovery**: The system identifies relationships (e.g., "door left open → temp spike → compressor surge") through statistical analysis of the data, rather than relying on pre-programmed rules.

2. **Generalizable Architecture**: The underlying architecture is designed to be adaptable to various manufacturing domains (e.g., pharmaceuticals, semiconductors) with minimal code changes.

3. **Evidence-Based Confidence Metric**: The system includes a mechanism to quantify its findings' confidence based on the strength of the collected evidence, aiming for objective assessments.

4. **Business-Oriented Output**: Results are formulated to include business-relevant context, such as potential cost impacts, in addition to technical metrics.

## Future Directions

1. **Multi-Hop Reasoning**: Extend to discover deeper causal chains (A → B → C → D)
2. **Advanced Causality Testing**: Implement Granger causality and counterfactual testing
3. **Multivariate Analysis**: Consider interactions between multiple variables
4. **Recommendation Optimization**: Suggest optimal operational parameters 