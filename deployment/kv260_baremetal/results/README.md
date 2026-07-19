# Board results

This directory intentionally contains no claimed board measurements. Copy the
CSV-friendly UART output into `benchmark_template.csv`, one row per measured
run, and retain the UART log and screenshots alongside it. The latency interval
is `Start()` immediately before the control write through the first observed
`IsDone()`; it excludes cache maintenance, address programming, and output
decoding. HLS co-simulation cycles are not board latency.
