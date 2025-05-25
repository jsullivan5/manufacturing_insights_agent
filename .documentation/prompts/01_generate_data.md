# Task

Generate realistic mock time series data simulating a freezer system monitored by the AVEVA PI System. The data will be used in a demo application to show how anomalies and causal relationships can be detected in industrial sensor data.

# Output Format

Produce a one-week dataset in **long format CSV** with the following columns:
- `Timestamp`
- `TagName` (e.g., FREEZER01.DOOR.STATUS)
- `Value`
- `Units`
- `Quality` (e.g., Good, Bad, Questionable)

# Tags to Include

Simulate the following PI tags:
- `FREEZER01.TEMP.INTERNAL_C` — Internal temp in °C
- `FREEZER01.DOOR.STATUS` — Boolean (0 = closed, 1 = open)
- `FREEZER01.COMPRESSOR.POWER_KW` — Float (power usage)
- (Optional) add 1–2 extra dummy tags if needed for realism

# Data Patterns

## Normal behavior
- Compressor cycles on when temp exceeds -16°C, off below -18°C
- Door opens ~5 times per shift during day, rarely at night
- Opening door causes a 5–10 min delay, then temp begins rising
- Compressor engages with a short delay once temp crosses threshold

## Shift differences
- Simulate two shifts: **Day (8am–8pm)** and **Night (8pm–8am)**
- Day shift has more door activity and compressor usage
- Slightly noisier readings in day to simulate variable traffic

# Anomalies (modularize in code)
Inject at least **3 clear anomalies** with realistic timing:
1. **Prolonged door open** (10–20 minutes) → temp spike → delayed compressor response
2. **Compressor fails to activate** → temp keeps rising
3. **Flatline sensor** — e.g., constant door closed for 6 hours with high temp
   - (Optional: a quality field set to "Questionable")

Make each anomaly standalone in code so additional ones can be toggled on/off.

# Goal

The resulting dataset should enable downstream code or humans to **infer cause and effect**, e.g.:

- “Door was left open → temp spiked → compressor failed to cool”
- “Compressor didn't turn on despite high temperature”

Keep timestamp granularity to ~1 min.

# Additional Context

I will provide background context from Perplexity about AVEVA PI format and expected schema.

Use Python with clear, well-commented structure for data generation.