# Normalized reciprocal precision experiment

Run: `reciprocal_precision_rtl_v1_20261002_final`  
Frozen control: `final_evidence_package_v1_20260930_final`  
Software selection: `reciprocal_precision_software_v1_20261002`

## 1. Baseline reproduction

The immutable control again reproduces 7.658487851 nm fixed and 0 nm floating
on the candidate 3M flat frame. Across all 19 frozen RTL
subset frames it remains exactly equal to its original bit-accurate model for
centroids, slopes, and coefficients. Counts are 168 slopes and 10 coefficients,
there are no unknown valid outputs, and latency remains
9 final-pixel-to-done cycles.

## 2. Stage-by-stage error attribution

The existing LUT contents exactly equal `round(2^18/d)`; LUT addressing,
integer multiplication, product slice, subtraction, slope format, E-matrix MAC,
and final slice are not material causes. Replacing only reciprocal-before-
multiply Q18 quantization with the exact rounded centroid quotient removes
100% of baseline squared flat WFE in the isolated test. The final slice removes
only 0.000966% in its non-additive isolated replacement.

| Rank | Isolated stage replacement | Modified flat RMS nm | Removed nm | Squared WFE removed |
|---:|---|---:|---:|---:|
| 1 | B: reciprocal output Q18 quantization | 0.000000 | 7.658488 | 100.000000% |
| 2 | I: final accumulator output slice | 7.658451 | 0.000037 | 0.000966% |
| 3 | A: reciprocal LUT algorithm/content | 7.658488 | 0.000000 | 0.000000% |
| 3 | C: numerator x reciprocal multiply | 7.658488 | 0.000000 | 0.000000% |
| 3 | D: post-multiply product slice | 7.658488 | 0.000000 | 0.000000% |
| 3 | E: centroid/reference subtraction | 7.658488 | 0.000000 | 0.000000% |
| 3 | F: centroid-to-slope conversion | 7.658488 | 0.000000 | 0.000000% |
| 3 | G: slope output slice | 7.658488 | 0.000000 | 0.000000% |
| 3 | H: E-matrix multiply/accumulate | 7.658488 | 0.000000 | 0.000000% |

## 3. Precision sweep

| Design | Flat 3M RMS nm | Mean FF nm | p95 FF nm | DSP delta estimate | Cycle delta | RTL bit-exact status |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 7.658488 | 2.033962 | 4.421340 | 0 | 0 | Yes, 19/19 frames |
| Normalized Q17 RNE | 0.014726 | 0.015277 | 0.030118 | 0 | 0 | Yes, 19/19 frames |
| Direct Q25 RNE | 0.036398 | 0.017350 | 0.045281 | +2 | 0 | Software bit model only |
| NR1 Q25 RNE | 0.036398 | 0.017323 | 0.045000 | +4 | +2 | Software bit model only |

The selected implementation removes 99.999630% of the original
flat squared WFE and improves arithmetic discrepancy on every one of the 276
deterministic frames. Newton-Raphson is not necessary.

## 4. Candidate architecture comparison

The selected path represents `d=a*2^k` with an 18-bit unsigned Q17 mantissa
`round(2^17/a)` and a 4-bit unsigned exponent. Each 19-bit unsigned moment times
18-bit mantissa produces a 37-bit unsigned product. A 38-bit rounding adder and
exponent-controlled shift retain a 24-bit unsigned Q18 centroid. Ties round up.
All later formats, ROMs, references, widths, and slices are unchanged.

## 5. Selected implementation

`normalized_q17_rne` is the lowest-cost material improvement. It is more accurate
than Q25 direct and Q25 NR on the flat case, uses the existing centroid multiplier
shape, and adds no arithmetic pipeline stage. The experimental normalized ROM
hash is `86c064bceb2ee44db68271f86f490b46d557e7d959711a4ea4782080487e26dc`.

## 6. RTL changes

Only new files under `rtl/experimental/` and the opt-in simulation wrapper
`sim/tb_zynq_top_precision_experimental.v` were added. The default baseline RTL,
active ROMs, calibration, and prior result directories are unchanged.

## 7. Bit-exact verification

The experimental RTL matches its own bit-exact model on all
19 frozen frames: zero centroid, slope, or coefficient
mismatches; 168 slopes and 10 coefficients per frame; no unknown valid output.
The baseline simultaneously passes the same requirements.

## 8. Numerical results

Candidate flat residuals are 2.691966 nm at
1M photons and 0.014726 nm at 3M, versus
2.967408 and
7.658488 nm for baseline. Across the full
deterministic set, candidate mean/p95/max fixed-minus-floating errors are
0.015277/
0.030118/
0.039554 nm. Worst coefficient
error is 0.029679 nm; slope
RMS/max errors are 0.00002108/
0.00005620 px. There are zero wraps and zero
invalid centroids.

| Photons | Input RMS nm | Baseline truth residual nm | Candidate truth residual nm | Change nm |
|---|---:|---:|---:|---:|
| 1M | 20 | 7.456916 | 7.385678 | -0.071238 |
| 1M | 100 | 27.483716 | 27.518199 | +0.034483 |
| 1M | 400 | 40.949148 | 40.986142 | +0.036994 |
| 3M | 20 | 4.041304 | 1.434536 | -2.606768 |
| 3M | 100 | 9.025770 | 8.350469 | -0.675301 |
| 3M | 400 | 36.282969 | 36.314970 | +0.032001 |

Arithmetic error improves on every frame, but ground-truth residual worsens on
104/276 individual frames because the old coherent bias sometimes canceled
optical error. The worst individual increase is 2.105503 nm. At the six mixed
operating cells, three means improve and three increase only 0.0320-0.0370 nm.

## 9. Latency and resource cost

Both designs simulate at 9 cycles
from the final contributing pixel to done, with three accumulator-to-reciprocal,
two reciprocal-to-slope, and two slope-to-done boundary cycles. Throughput remains
one accepted pixel per cycle and one completed subaperture per cycle after fill.
Structural inference is 22 DSP48s for both designs: two centroid multipliers plus
20 E-matrix multipliers, delta zero. The candidate adds exponent registers and
two rounding/barrel-shift networks; the normalized ROM is 65,536x18 rather than
65,536x19. No Vivado run was performed, so LUT/FF/BRAM mapping and achieved timing
remain estimates, not implementation results.

## 10. Recommendation

Retain the current baseline as the default control. The normalized Q17 RNE path
is the recommended experimental successor if the precision floor justifies an
opt-in RTL variant. It reaches 0.014726 nm on the defining flat case with no
estimated DSP or latency increase. NR1 is numerically unnecessary and materially
more expensive. Any promotion into the default design should follow synthesis
and timing closure, which were outside this bounded simulation-only run.

Explicit decisions:

- Reciprocal-before-multiply Q18 quantization accounts for 100% of the isolated
  original flat squared error; the implemented candidate removes 99.999630%.
- Newton-Raphson was not necessary.
- Normalized Q17 RNE is the lowest-cost material improvement.
- Arithmetic discrepancy improves on every case. Ground-truth residual worsens
  in 104 individual cases through removal of fortuitous baseline cancellation;
  three aggregate mixed cells increase only 0.0320-0.0370 nm.
- The experimental design adds zero simulated cycles and zero estimated DSP48s.
- The baseline implementation is completely unchanged and reproducible.
