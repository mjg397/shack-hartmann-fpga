# Vivado reports required

No Vivado reports were found in this directory, so
`results/implementation_results.csv` and `.tex` contain no resource or timing
numbers. **Do not put resource or Fmax figures in the paper until these are
supplied** -- they cannot be derived from simulation.

Generate them from your implemented design and copy them here:

```tcl
# after opt_design / place_design / route_design
report_utilization      -file impl_utilization.rpt
report_timing_summary   -file impl_timing_summary.rpt
# optional, for comparison only
open_run synth_1
report_utilization      -file synth_utilization.rpt
```

Required (post-implementation preferred):

| File | Supplies |
|---|---|
| `impl_utilization.rpt` | LUT, LUTRAM, FF, BRAM, DSP counts and percentages |
| `impl_timing_summary.rpt` | WNS, TNS, target period, achieved Fmax |

Name them with `impl` in the filename so the parser prefers them over
synthesis estimates. Then re-run:

```bash
python evaluation/vivado_reports/parse_reports.py
```

## Constraint to state in the paper

The design targets 150.0 MHz (period 6.667 ns). The clock constraint
used for implementation must match, or the reported WNS is not comparable to
the latency figures in this evaluation.

## Note on which RTL to implement

The evaluation measures `sim/rtl/`, which contains three
correctness fixes not present in `rtl/`. Implementing `rtl/`
would characterise a design that produces incorrect coefficients. See
EVALUATION_REPORT.md.
