# Reciprocal precision software selection

Run: `reciprocal_precision_software_v1_20261002`  
Frozen control: `final_evidence_package_v1_20260930_final`

## 1. Baseline reproduction

The frozen candidate 3M flat frame reproduces 7.658487851 nm
fixed and 0.0 nm floating. RTL and the frozen
bit-accurate model have zero coefficient and slope/centroid mismatches, with
168 valid slopes, 10 coefficients,
and 9 final-pixel-to-done cycles.

## 2. Stage-by-stage error attribution

The dominant isolated replacement is `B_reciprocal_output_q18_quantization`: it reduces the flat
residual to 0.000000 nm and removes
100.000000% of baseline squared WFE.
The full ranked A-I table, coefficient vectors, and slope errors are in
`stage_error_attribution.csv`. The existing 65,536-entry LUT is already exactly
round-to-nearest Q18; the loss occurs because Q18 reciprocal quantization happens
before multiplication.

## 3. Precision sweep and architecture comparison

| Design | Flat 3M RMS nm | Mean FF nm | p95 FF nm | Estimated DSP delta | Estimated latency delta |
|---|---:|---:|---:|---:|---:|
| Baseline | 7.658488 | 2.033962 | 4.421340 | 0 | 0 |
| Normalized Q17 RNE | 0.014726 | 0.015277 | 0.030118 | 0 | 0 |
| Direct Q25 RNE | 0.036398 | 0.017350 | 0.045281 | +2 | 0 |
| NR1 Q25 RNE | 0.036398 | 0.017323 | 0.045000 | +4 | +2 |

All costs are structural estimates against a DSP48E1 25x18 multiplier; no
synthesis or achieved timing claim is made.

## 4. Selection before RTL

`normalized_q17_rne` is selected for experimental RTL. It keeps the reciprocal
mantissa at 18 bits, adds a 4-bit exponent, delays the binary-point shift until
after the existing 19x18 product, and rounds the shifted product to nearest.
It is the lowest-cost candidate and is numerically better than the wider Q25 and
NR1 Q25 candidates on the frozen 3M flat case. Newton-Raphson is not necessary.

This run is software-only. Experimental RTL verification belongs in a distinct
follow-on run; the baseline RTL and all frozen result directories remain unchanged.
