# Implementation Audit

> **Historical audit.** Current simulation evidence and source/ROM hashes are
> in [`evaluation/results/runs/manuscript_v1_20260928_final/REPORT.md`](../../evaluation/results/runs/manuscript_v1_20260928_final/REPORT.md)
> and its `MANIFEST.json`. The current task deliberately retains the 18-bit
> reciprocal architecture; the Q27/Newton option discussed below was not
> restored or evaluated as a manuscript recommendation.

Reconciles the "known historical context" claims against the repository as it
actually stands. Primary source: `docs/EVALUATION_REPORT.md` §1-2, which
already performed this audit from RTL/`.mem` files, not from README prose.
This document adds nothing that contradicts that report; it restates the
relevant findings against each claim the validation prompt listed as unverified,
and records what this session additionally checked directly.

## Claim-by-claim reconciliation

| Historical claim | Status | Evidence |
|---|---|---|
| 256×256, 8-bit frames | **CONFIRMED** | `streaming_emulator.v` HSIZE/VSIZE; `hcipy_image_rotating.hex` is 65536 lines |
| 16×16 lenslets | **CONFIRMED** | `NUM_SUBAPETURES_SQRT`; bitmap is 256 bits = 16×16 |
| 10 non-piston Zernike modes | **CONFIRMED** | `NUM_MODES` = 10; `ematrix_x/y.mem` shaped for 10 modes |
| 120 ns @ 50 MHz | **DOES NOT DESCRIBE THE CURRENT DUT** | `docs/EVALUATION_REPORT.md` §5.1.2: this is the Gen A / DE1-SoC number (6 cycles @ 50 MHz), a stale generation not compiled in the current top |
| ~8 cycles @ 150 MHz (53.3 ns) | **NOT WHAT WAS MEASURED** | Measured tail latency is **9 cycles / 60.0 ns** at the recommended anchor (streaming-emulator output bus), or 7 cycles/46.67 ns at the earliest anchor (accumulator completion event) — never exactly 8/53.3 ns at any anchor tested. See §4.6 stage breakdown (2+3+2+2). Anchor choice matters and must be stated with any figure. |
| 168 vs 186 valid subapertures | **168 IS CURRENT AND VERIFIED, NOT 186** | Independently decoded `rtl/subaperture_bitmap.mem` this session: popcount = 168, a circle (r≤1) minus central obscuration (r≥0.15) sampled at 16 lenslet centers per axis — reproduces the bitmap exactly, 0 mismatched cells. 186 does not appear anywhere in the current `rtl/` tree; treat it as a stale/other-branch number until a source is found. |
| Thresholding implemented | **CONFIRMED, in the DUT generation only** | `Thresholding_CoG_intensity_accumulator.v` exists and is Gen B's accumulator (the evaluated DUT). The stale Gen A path uses plain `TCoG_intensity_accumulator.v` (no threshold) and is not the evaluated design — see Gen A/B table in `docs/EVALUATION_REPORT.md` §2.1. |
| Fixed-point formats | **TRACED, see report §2.4** | `sI` u16, `xI/yI` u19, reciprocal Q18 (19 bits, 18 as consumed at the `rec_intensity` port — 1 bit pruned), centroid Q18/24b, slope Q18/25b signed, E-matrix Q1.16/18b signed, Zernike out 25b signed. |
| Reciprocal method | **Q18 direct LUT, no Newton, in the evaluated DUT** | `intensity_reciprocal_q18` → `reciprocal_u16_q18.v`. This session additionally found that `reciprocal_u16_q18.v` computes an unused CLZ-normalization/`lut_idx` path (dead code, copy-pasted from the sibling `reciprocal_u16_q27.v`) — the actual ROM read is `recip_rom[v_u16]`, a flat 65536-entry direct-address table. Functionally irrelevant to correctness (synthesis strips it) but worth removing for clarity. A second, more precise Q27 reciprocal (256-entry seed LUT + 2 Newton-Raphson steps) exists in the tree (`reciprocal_u16_q27.v`, `newton_step_q27.v`) but is **not instantiated** by the evaluated top — see the ablation result below. |

## The one already-quantified, actionable defect

`docs/EVALUATION_REPORT.md` §4.3 (error ablation) and §5.1.1 already measured this:
the shipping Q18 LUT reciprocal is the **sole significant fixed-point error
source** in the whole pipeline (slope truncation, output slicing, E-matrix
quantization all contribute nothing measurable). Swapping in the Q27+Newton
reciprocal already sitting in the tree — no new design work — cuts Zernike
RMSE **2.34×** (1.899 nm → 0.811 nm) and would close most of the gap to the
project's own precision spec (`docs/RECIPROCAL_PRECISION_PROOF.md` requires
< 1e-4 px; shipping measures 6.7e-3 px, currently 67× short per report §5.1.1).
This is the highest-leverage, lowest-risk fix available before any new data
campaign, because it changes one module instantiation and the accuracy
evidence already exists.

## What this audit does NOT cover (see other validation/ documents)

- Real experimental data (§3 of the validation program) — no experimental
  Shack-Hartmann data exists in this repository; see `docs/validation/DATASET_CANDIDATES.csv`.
- FPGA implementation timing/resources (§8) — no Vivado project or report
  exists anywhere on this machine; see `HARDWARE_HANDOFF.md`.
- **UPDATE (this session): the 3 documented `rtl/` defects below have
  been ported directly into `rtl/`.** Previously this tree did not
  work as committed (~100% error on every mode); the evaluated DUT was only
  the corrected copy in `sim/rtl/`. Fixes applied in
  place: (1) `slope_calculation.v`'s `unsigned_mult` now slices
  `mult_out[23:0]` instead of `mult_out[35:12]`; (2)
  `intensity_reciprocal_q18.v` now delays the `xI`/`yI`/`centroids_done`
  side-band signals by `RECIP_LATENCY = 2` cycles to match the reciprocal's
  2-stage pipeline; (3) `slopes_ref_x.mem`/`slopes_ref_y.mem` were replaced
  with the validated Q18 versions (copied from `sim/rtl/`
  after independently reproducing an exact match via round-to-nearest
  right-shift-by-5 from the original Q4.23 source). Verified: `iverilog`
  elaborates the edited files cleanly, and a file-by-file diff against
  `sim/rtl/` now shows only comment wording and one
  `$readmemh` paths are now consistently bare and the private `sim/rtl/` tree
  has been removed. Strict staging/validation is covered by focused regression
  tests. A separate row-boundary streamer defect was also fixed: an 8-bit shift
  truncated every next-row BRAM address to zero, usually hidden by dark border
  pixels; the address is now explicitly `{next_row, 6'b0}` and covered by a
  nonzero-row-border regression. **Not done**: swapping in the Q27+Newton
  reciprocal (a separate, optional accuracy improvement, not one of the 3
  defects) and rerunning the RTL-equivalence regression pointed at
  `rtl/` itself — the 1,680/1,680 exact-match result in
  `docs/EVALUATION_REPORT.md` §4.1 was run against `sim/rtl/`,
  not `rtl/` directly; do that rerun before relying on this fix for a
  publication claim.
