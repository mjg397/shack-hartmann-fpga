# Experimental reciprocal precision path

These modules implement the opt-in `normalized_q17_rne` experiment. They do not
replace or modify the default modules in `rtl/`.

The experiment represents a divisor as an 18-bit unsigned normalized Q17
reciprocal mantissa plus a 4-bit exponent. The centroid multiply remains 19x18;
rounding and the exponent shift happen after the full 37-bit product. The
simulation-only selector is `sim/tb_zynq_top_precision_experimental.v`.

The default runner continues to compile `rtl/shwfs_processing.v`,
`rtl/intensity_reciprocal_q18.v`, `rtl/reciprocal_u16_q18.v`, and
`rtl/slope_calculation.v`. The experimental reciprocal ROM is generated inside
the new result run and is not an active ROM.

See
[`evaluation/results/runs/reciprocal_precision_rtl_v1_20261002_final/REPORT.md`](../../evaluation/results/runs/reciprocal_precision_rtl_v1_20261002_final/REPORT.md)
for the bit-exact comparison, numerical results, and estimated resource cost.
