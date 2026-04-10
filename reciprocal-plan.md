# Reciprocal Plan

## Context
- Goal: fast reciprocal model for FPGA-oriented divider path.
- Source reference: SEGGER Newton reciprocal method (fixed-point adaptation).
- Input contract: plain unsigned 16-bit integer input (not fixed-point).
- Output contract: unsigned Q0.16 reciprocal output.
- Chosen architecture: 256-entry LUT seed + one Newton refinement.
- Scope of this pass: C model only (no RTL/testbench edits yet).

## Confirmed Decisions
- Reciprocal seed LUT size: 256 entries.
- Reciprocal output format: Q0.16.
- Input interpretation: raw integer value in range 0..65535.
- Zero-input behavior: saturate output to 0xFFFF and raise divide-by-zero status flag.

## Implementation Plan
1. Define numeric contract and edge-case policy:
   - input as plain uint16
   - zero handling
   - CLZ-based normalization to [1, 2)
   - output rounding and saturation policy in Q0.16
2. Build fixed-point utility layer in C:
   - clz16
   - msb index helper
   - conversion helpers for Q0.16
3. Build 256-entry reciprocal seed LUT:
   - index from normalized mantissa bits
   - deterministic table values
4. Implement reciprocal pipeline:
   - normalize integer input to Q1.15
   - fetch seed from LUT
   - run one Newton step: x1 = x0*(2-ax0)
   - de-normalize by input power-of-two scaling
   - clamp/round to Q0.16
5. Add validation harness:
   - sweep v=1..65535
   - compare against double-precision reference 1.0/v
   - report max absolute LSB error, max/mean relative error
6. Add trace diagnostics for bring-up:
   - print normalization shift, LUT index, seed, Newton result, output for directed vectors
7. Define acceptance criteria:
   - explicit thresholds for max/mean error
   - edge behavior checks for 0, 1, powers of two, max input
8. Prepare HDL handoff notes:
   - expected 2-cycle partition (LUT stage + Newton stage)
   - required signal widths and golden vectors

## Current Code Status
- Implemented in src/reciprocal-model.c:
  - `reciprocal_result_t` result struct (q16 + status flags)
  - `clz16`, `msb_index_u16`
   - hard-coded 256-entry reciprocal seed LUT (Q0.16)
  - `reciprocal_newton_step_q16`
  - `reciprocal_u16_to_q16`
  - `run_validation` full sweep + directed traces
   - `main` single-input mode, full validation mode, and export modes:
      - `--emit-lut-vector`
      - `--write-lut-memh [path]`
      - `--write-golden [csv_path] [memh_path]`
- Notes:
  - Q0.16 cannot represent exactly 1.0, so max output uses 0xFFFF.
  - v=0 saturates and sets divide-by-zero.

## Verification Status
- Editor diagnostics: no syntax/lint errors in src/reciprocal-model.c.
- Runtime verification in this environment: blocked (no C compiler available: gcc/clang/cl not found).

## How To Run On Another Machine
1. Build:
   - `gcc src/reciprocal-model.c -O2 -Wall -Wextra -std=c11 -lm -o src/reciprocal-model.exe`
2. Full sweep validation:
   - `./src/reciprocal-model.exe`
3. Single input test:
   - `./src/reciprocal-model.exe 12345`

## Next Recommended Steps
1. Wire LUT and golden artifacts into Verilog testbenches (`$readmemh` and/or packed vector include).
2. Lock acceptance thresholds after first RTL-vs-model comparison run.
3. Add directed mismatch report mode for only non-exact Q16 cases if needed for debug focus.

## Artifact Outputs (Now Available)
- HDL vector text (stdout):
   - `./src/rm --emit-lut-vector`
- LUT memory file for `$readmemh`:
   - `./src/rm --write-lut-memh verilog/testbench/vectors/reciprocal_seed_lut_q16.memh`
- Golden vectors for testbench checking:
   - `./src/rm --write-golden verilog/testbench/vectors/reciprocal_golden_vectors.csv verilog/testbench/vectors/reciprocal_golden_io32.memh`
- Generated files in workspace:
   - `verilog/testbench/vectors/reciprocal_seed_lut_q16.memh`
   - `verilog/testbench/vectors/reciprocal_golden_vectors.csv`
   - `verilog/testbench/vectors/reciprocal_golden_io32.memh`
