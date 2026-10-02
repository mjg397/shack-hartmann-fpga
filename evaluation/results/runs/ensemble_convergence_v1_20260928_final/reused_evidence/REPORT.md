# Flat-bias, convergence, and clipping investigation

Run: `blocker_investigation_v1_20260928_final`  
Source run: `manuscript_v1_20260928_final` (read-only)  
Scope: HCIPy, Python fixed-point model, and Icarus RTL simulation only.

## Executive findings

The 7.615181 nm flat result is real for the frozen ROMs and includes all ten
reported modes.  It is created at the reciprocal/centroid multiply, before
reference subtraction: the reference stores the rounded exact quotient, while
RTL multiplies the moment by a rounded reciprocal.  The exact frozen flat frame
produces Tilt X = +5.3159348 nm, Tilt Y = +5.3159348
nm, Coma X = +0.3361900 nm, and Coma Y =
+0.3361900 nm.  No tip/tilt or fitted bias was removed.

The exact frame agrees with RTL at every checked centroid, slope, and output
coefficient: 0 coefficient mismatches,
0 slope/centroid mismatches, and
0 unknown valid values.  RTL residual:
7.615181 nm.

The corrected convergence comparison uses identical coefficient vectors and a
single common scoring pupil.  The production edge setting is demonstrably
converged: **no**.  The production
propagation setting is demonstrably converged: **no**.
The highest setting is never credited for agreeing with itself.

## 1. Flat-wavefront stage trace

The stored Q18 reference equals `round(moment / sum × 2^18)` in every valid tile.
The RTL result is `moment × round(2^18 / sum)`, truncated to the existing
centroid width.  Fixed-minus-exact centroid error is
1078.93 LSB mean and
1672.36 LSB RMS in x (the y
values are identical to numerical precision).  Its correlation with reciprocal
rounding error is 0.99544.

Spatially, most fully illuminated interior lenslets have a +2040 LSB error,
while the obscuration, spiders, and pupil rim create coherent negative and
intermediate bands.  The y map is the transpose-symmetric counterpart of the x
map.  This is why the reconstruction projects strongly and equally onto x/y
tilt and, more weakly, x/y coma; it is not an independent constant added to the
ten reported coefficients.

The integer trace also confirms the actual widths and signs: unsigned 16-bit
intensity sums, 19-bit moments, a 19-bit Q18 reciprocal narrowed to the 18-bit
multiplier port, unsigned 24-bit Q18 centroids, signed 25-bit Q18 slopes, signed
18-bit Q1.16 matrix words, a signed 52-bit accumulator, and signed output slice
`[42:18]`.  The flat frame exercises no moment, accumulator, or output wrap.

`flat_stage_trace.csv` contains all 168 valid tiles: thresholded intensity sums,
moments, exact and LUT reciprocals, both centroid conventions, stored references,
and signed reference-subtracted slopes.  `flat_stage_arrays.npz` also preserves
the raw and thresholded 65,536-pixel frames.

The original floating matrix and dequantized matrix both reproduce the same
coherent tilt/coma pattern when fed the RTL centroids.  The final integer MAC
changes the dominant tilt coefficients by less than 0.00002 nm.  This rules out
matrix quantization and the reconstruction MAC as the source.

### Noise metrics (fresh seed, same 3e6-photon operating point)

| Path | Mean per-frame RMS | RMS of mean wavefront | RMS frame variability |
|---|---:|---:|---:|
| Primary float | 3.563 nm | 0.589 nm | 3.763 nm |
| Fixed | 3.708 nm | 0.680 nm | 3.881 nm |

Noise changes tile sums and therefore changes the reciprocal rounding pattern;
it does not constitute subtraction of the deterministic flat.  Mean per-frame
RMS, RMS of the mean reconstruction, and frame variability are reported
separately.

### Why the earlier report said approximately zero

The historical `docs/OPTICAL_EVALUATION.md` headline used the exact-floating
flat value (0.000033 nm) but called it generic “flat-wavefront bias.”  Its own
archived `evaluation/results/optical/campaign_summary.json` records the fixed
value as 7.615181 nm.  This investigation resolves a labeling ambiguity; it does
not revise the archived arithmetic result.

## 2. Numerical convergence audit

The prior comparison had two confounders: it normalized each common random
direction separately on each setting's pupil and scored on that changing pupil.
Both are removed here.  Refined detector intensity is block-summed over the
physical production pixels before normalization, threshold, and ADC rounding.
Detector outer edges, block centers, and lenslet registration are recorded in
`convergence_physical_checks.csv`; each setting retains its own flat reference
and ±30 nm interaction calibration.

| Study | Setting | Mean residual (nm) | Change vs previous | Every paired case passes 5% and 0.5 nm |
|---|---|---|---|---|
| pupil_edge_supersampling | 2 | 5.863 | — | — |
| pupil_edge_supersampling | 4 | 5.004 | 0.859 nm / 14.7% | no |
| pupil_edge_supersampling | 6 | 5.195 | 0.191 nm / 3.8% | no |
| pupil_edge_supersampling | 8 | 4.295 | 0.899 nm / 17.3% | no |
| pupil_edge_supersampling | 10 | 5.685 | 1.390 nm / 32.4% | no |
| pupil_edge_supersampling | 12 | 4.748 | 0.937 nm / 16.5% | no |
| pupil_edge_supersampling | 16 | 4.488 | 0.260 nm / 5.5% | no |
| pupil_edge_supersampling | 20 | 4.571 | 0.084 nm / 1.9% | no |
| propagation_grid_with_detector_integration | 1 | 5.004 | — | — |
| propagation_grid_with_detector_integration | 2 | 6.230 | 1.226 nm / 24.5% | no |
| propagation_grid_with_detector_integration | 3 | 5.727 | 0.503 nm / 8.1% | no |
| propagation_grid_with_detector_integration | 4 | 6.080 | 0.353 nm / 6.2% | no |
| propagation_grid_with_detector_integration | 5 | 5.437 | 0.643 nm / 10.6% | no |

The production pupil uses HCIPy's mean of binary aperture sub-samples directly
as electric-field amplitude.  Fractional edge pixels therefore represent an
amplitude average (their propagated intensity contribution is its square), not
an area/intensity fill fraction.  Thin spiders and curved boundaries move among
these fractional samples as supersampling changes; their transmission sums and
squared sums are preserved per setting in the physical-check table.

The continuous-detector diagnostic in `convergence_summary.csv` separates
optical/grid dependence from threshold/ADC stair-steps.  Interaction matrices,
singular values, condition numbers, flat spots, and integrated flux are all
preserved.  Claims remain conditional whenever the selected production row or
the final successive pair fails the frozen criterion.

The mechanism is identifiable rather than merely inferred from an irregular
aggregate.  In the continuous diagnostic, propagation residual decreases at
every factor from 1 through 5, and every representative edge case passes the
frozen criterion from 16× to 20×.  The production-integer results keep large,
case-dependent jumps even though these frames have no ADC clipping.  Therefore
the propagation parity-looking sequence is an ADC-rounding/5-ADU-threshold
staircase in the per-setting calibration and test centroids.  The edge study
also has a genuine aperture-boundary/spider sampling dependence below 12×;
integer detector quantization prevents that continuous tail stability from
carrying over to production.  Matrix condition numbers remain of order six, so
ill-conditioning is not the explanation.

## 3. ADC-clipping ablation

Production and diagnostic paths use the same optical input, Poisson sample,
read-noise sample, full-well clipping, gain, integer rounding, and 5 ADU
threshold.  The diagnostic removes only the upper ADC clamp and is not an RTL
input path.

| Photons/pupil | Expected e⁻/valid lenslet | Nominal QE·photons/168 | Production float (nm) | No-ADC-clip diagnostic (nm) | Clipped pixels |
|---|---|---|---|---|---|
| 3e+06 | 13573 | 14286 | 5.95 | 5.95 | 0.000% (0/1572864) |
| 5e+06 | 22622 | 23810 | 7.17 | 6.45 | 0.188% (2962/1572864) |
| 8e+06 | 36196 | 38095 | 21.77 | 6.51 | 0.910% (14312/1572864) |
| 1.2e+07 | 54293 | 57143 | 30.90 | 15.53 | 0.990% (15570/1572864) |

“Expected e⁻/valid lenslet” is measured by summing normalized expected detected
photoelectrons inside the 168 valid tiles.  The nominal allocation is shown
separately and is not described as collected valid-lenslet flux.  Clipping
fractions use all detector pixels as the stated denominator; valid-tile counts
and denominators are also in the raw and summary tables.

At 8e6 photons, removing only the ADC clamp reduces the paired floating residual
from 21.77 nm to 6.51 nm, directly establishing clipping as the dominant tested
cause there.  At 12e6 photons the diagnostic still leaves 15.53 nm, despite no
full-well saturation.  ADC clipping is therefore a major contributor, not the
sole high-flux mechanism.  The remainder includes non-clipping signal-level
effects (the fixed threshold/reference and optical/calibration nonlinearity);
this one-factor ablation does not partition those remaining causes further.

## 4. Corrections and historical validity

- No arithmetic width, threshold, gain, active ROM, or RTL architecture was
  changed.  The 7.615181 nm result is retained with tilt included.
- `regen_reference.py` now explicitly distinguishes exact-quotient and RTL
  reciprocal reference conventions.  Its historical/default exact behavior is
  unchanged, and no new reference was applied.
- Historical optical residual tables and the 35-case RTL equivalence counts
  remain numerically valid because their code, frames, and ROMs were not changed.
- The old generic “zero flat bias” wording is superseded: zero applied only to
  the exact-floating path.
- The old convergence table is superseded by this paired/common-grid audit.
- The old noise-table electron column was a nominal allocation.  Its residuals
  remain valid, but the column must not be cited as measured valid-tile flux.

## 5. Claim-to-evidence

| Claim | Status | Evidence |
|---|---|---|
| Frozen RTL creates the 7.615181 nm flat residual at reciprocal/centroid arithmetic | Established for exact frozen frame | stage CSV/NPZ, ablations, exact RTL trace |
| RTL equals the bit-accurate model | Established for finite tested cases | old 35 cases plus this exact flat; not formal proof |
| Production optical grid is numerically converged | Not established | paired common-grid convergence tables |
| ADC clipping is a major, but not sole, high-flux contributor in this detector model | Established for paired simulated cases | clipping raw/summary tables; not a camera measurement |
| Physical-bench equivalence, FPGA timing, power, or resources | Untested | outside this simulation-only scope |

## 6. Reproduction

From the repository root:

```bash
sim/venv/bin/python -m unittest evaluation.tests.test_rom_loading evaluation.tests.test_reference_conventions -v
sim/venv/bin/python evaluation/optical/blocker_investigation.py \
  --config evaluation/configs/blocker_investigation_v1.json \
  --out evaluation/results/runs/blocker_investigation_v1_20260928_final_reproduction
```

The output directory must not already exist.  `MANIFEST.json`, the compressed
tracked-working-tree patch, relevant untracked source copies, frozen source-run
calibration/ROMs, dependency versions, commands, and SHA-256 hashes provide the
reproduction snapshot.
