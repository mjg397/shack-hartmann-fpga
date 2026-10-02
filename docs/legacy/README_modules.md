# Module Verification Scaffold

This folder is a starting point for per-module verification.

Suggested order:

1. `streaming_emulator`
2. `Thresholding_CoG_intensity_accumulator`
3. `reciprocal_u16_q27`
4. `slope_calculation`
5. `ematrix_accumulator`
6. `shwfs_vivado_top`

The idea is:

- use Python for the expected behavior
- use Verilator for cycle-accurate RTL checks
- keep each module small enough that failures are easy to localize

Good rule of thumb:

- if the question is "what should the math be?", use a Python reference
- if the question is "what cycle should this happen on?", use Verilator

