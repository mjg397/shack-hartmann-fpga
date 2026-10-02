# Focused optical-forward-model and convergence-method investigation

Run: `forward_model_investigation_v2_20260928_final`  
Source: `ensemble_convergence_v1_20260928_final` (preserved)  
Scope: HCIPy/Python diagnostics; no RTL, ROM, gain, threshold, geometry, or
acceptance-tolerance change.

## Executive conclusions

The previous noisy **framewise prediction-difference** test was confounded.
Its all-grid shared-minimum construction preserved every setting's marginal
detector distribution, but an unrelated third grid changed the common Poisson
component and caused independent excess draws even for identical expectations.
That metric and its 0/36 decision are therefore **superseded**, not deleted.
Per-setting ensemble residuals remain interpretable; paired mean differences
retain their mean estimand but their uncertainty was coupling-dependent and the
affected production comparisons were repeated pair-locally.

The physical optical setup is preserved across refinement. Wavelength, OPD
coefficients, aperture dimensions, magnification, lenslet centres/pitch/focal
length, detector edges, propagation distance, and orientation are invariant;
every lenslet owns exactly the expected `(16×factor)^2` samples and block-
integrated detector centres align within 1e-15 m. All grids are even and omit a
sample exactly at zero in the same symmetric convention.

The large deterministic propagation sensitivity is not a geometry change. It
is already present before thresholding: for the three nonflat production→factor-5
cases, normalized expected-electron images differ by mean relative L2
0.14583 and continuous centroids by 0.03672 pixel RMS. The unchanged
threshold changes that to 0.02400 pixel and integer ADC sampling to
0.02720 pixel; the frozen reconstructor maps the result to 15.741 nm
mean wavefront change. The simple-lenslet result identifies coarse sampling of
an approximately 31 µm FWHM diffraction spot by 20.833 µm production pixels,
followed by discrete physical-pixel integration, as the initiating numerical
sensitivity. Threshold/ADC discontinuities and calibration then amplify or
reshape it; they do not create the pre-threshold difference.

No production optical source, RTL, or calibration was changed. The justified
source fix is confined to the **test methodology**: the corrected pairwise
coupling uses only the two expectations, assigns streams in stable name order,
has exact Poisson/Gaussian marginals, and produces identical frames for
identical expectations.

## 1. Noise-coupling controls

| Control | Mean detector RMS (ADU) | Mean reconstructed difference [95% CI] (nm) |
|---|---|---|
| identical_pair_legacy_only | 0.00000 | 0.000 [0.000, 0.000] |
| identical_pair_legacy_embedded | 0.17829 | 2.388 [2.220, 2.555] |
| identical_pair_corrected | 0.00000 | 0.000 [0.000, 0.000] |

Insertion-order permutation produced 0
failures in 100 controls because the old implementation
sorted keys. Adding/removing the unrelated grid did change the joint draw.
The corrected pair repeated bit-exactly after an unrelated pair was drawn
(0 failures).
No empirical floor was subtracted.

## 2. Independent simple-lenslet validation

One hard-edged square lenslet uses the production HCIPy `MicroLensArray` and
`FresnelPropagator`, without noise, threshold, clipping, or ADC rounding. The
independent reference is the thin-lens focal-plane
`sinc²(a x/(λf)) sinc²(a y/(λf))` solution integrated with adaptive quadrature
over the exact detector-pixel edges. Tilt is checked against `x=fθ`.

| Factor | Samples/lenslet | Image rel. L2 | Tilt centroid error (nm) | Captured power | First-zero-box energy |
|---|---|---|---|---|---|
| 1 | 16 | 0.14446 | -214.525 | 0.999582626 | 0.81866 |
| 2 | 32 | 0.03291 | -64.091 | 0.999497321 | 0.81750 |
| 3 | 48 | 0.01324 | -24.605 | 0.999430944 | 0.81710 |
| 4 | 64 | 0.00687 | -14.813 | 0.999368458 | 0.81690 |
| 5 | 80 | 0.00407 | -11.220 | 0.999307660 | 0.81682 |

The analytic first zero is 35 µm and FWHM is 31.006 µm. Captured flux and the
finite-field spot width are reported separately from normalized image shape.

## 3. Full-model physical audit

`physical_geometry_audit.csv` records every physical invariant, grid centre and
edge, MLA assignment count, Fresnel padding/branch condition, and orientation.
The Fresnel propagator uses its documented 2× zero padding and 2× transfer-
function sampling; no wraparound setting changes with factor 1–5.

The pupil aperture values are coherent **cell-average amplitudes**. This is the
appropriate quadrature for a binary coherent field. Replacing them by square
roots of area fractions would preserve per-cell energy but change coherent
diffraction, so it was not done. `aperture_discretization_audit.csv` compares
edge supersampling to a factor-5 binary point quadrature block-averaged onto the
production cells.

## 4. Stage-by-stage propagation localization

All cases use the same coefficient vectors and common scoring pupil. Absolute
HCIPy output power is recorded before the existing detector model normalizes to
its stated photons-per-pupil definition, exposing rather than hiding finite-
field differences.

| Case | Captured flux Δ | A expected-e rel. L2 | B centroid RMS (px) | C threshold RMS (px) | D integer RMS (px) | E fixed wavefront Δ (nm) |
|---|---|---|---|---|---|---|
| flat | +0.0724% | 0.14332 | 0.03548 | 0.02263 | 0.02997 | 8.622 |
| single_tilt_x_100nm | +0.0774% | 0.14438 | 0.03753 | 0.02392 | 0.02548 | 29.044 |
| mixed_20nm_0 | +0.0731% | 0.14361 | 0.03536 | 0.02413 | 0.03005 | 7.258 |
| mixed_100nm_0 | +0.0787% | 0.14950 | 0.03728 | 0.02394 | 0.02607 | 10.922 |

Stage definitions: A integrated expected photoelectrons; B continuous centroid
without threshold; C continuous centroid after the unchanged 5-ADU threshold;
D deterministic integer detector plus floating centroid; E frozen fixed-point
reconstruction. Per-lenslet spatial differences are in
`forward_stage_lenslets.csv`; image and coefficient arrays are in
`forward_stage_arrays.npz`.

## 5. Calibration sensitivity (diagnostic only)

| Factor | Calibration policy | Mean residual (nm) | Max residual (nm) | Production? |
|---|---|---|---|---|
| 1 | common_factor5_diagnostic | 8.672 | 11.687 | diagnostic |
| 1 | frozen_dequantized_rom | 4.698 | 9.847 | diagnostic |
| 1 | frozen_fixed_production | 6.764 | 8.399 | yes |
| 1 | per_grid_diagnostic | 4.698 | 9.847 | diagnostic |
| 5 | common_factor5_diagnostic | 5.428 | 10.679 | diagnostic |
| 5 | frozen_dequantized_rom | 10.696 | 18.916 | diagnostic |
| 5 | frozen_fixed_production | 11.661 | 23.230 | yes |
| 5 | per_grid_diagnostic | 5.428 | 10.679 | diagnostic |

The factor-5 common calibration and every per-grid calibration are diagnostics.
They were not written to active ROMs. Their contrast with the frozen rows shows
how calibration can absorb a portion of the forward-model change without
making the underlying detector images equal.

## 6. Bounded corrected noisy confirmation

Only the affected production→supported-reference comparisons at 3M photons
were repeated: 20 and 100 nm RMS, 20 independent wavefronts, and five nested
exposures. The new estimands were frozen before running: paired mean residual,
the wavefront RMS of the five-exposure **mean coefficient difference** as an
estimate of expected-reconstruction change, the deterministic expected-frame
difference, and separately the within-wavefront noise-variability change.

| Comparison | Input (nm) | Reference residual (nm) | Residual Δ [95% CI] (nm) | Mean reconstruction Δ (upper) (nm) | Expected-frame Δ (nm) | Noise variability Δ (nm) | Margin (nm) | Decision |
|---|---|---|---|---|---|---|---|---|
| edge_4_to_20 | 20 | 4.460 | +0.011 [-0.091, +0.113] | 0.466 (upper 0.530) | 1.318 | -0.007 | 0.223 | fail |
| edge_4_to_20 | 100 | 6.971 | +0.016 [-0.091, +0.123] | 0.577 (upper 0.679) | 0.969 | -0.011 | 0.349 | fail |
| propagation_1_to_5 | 20 | 7.073 | -2.775 [-3.283, -2.267] | 6.584 (upper 6.857) | 8.065 | +0.247 | 0.354 | fail |
| propagation_1_to_5 | 100 | 13.021 | -6.199 [-7.373, -5.025] | 11.004 (upper 12.274) | 12.103 | -0.382 | 0.500 | fail |

Accepted cells: **0/4**.
All corrected frames were non-clipping: **True**.
The old framewise numbers are retained beside the replacement estimand in
`noisy_before_after.csv`; coupled framewise RMS is not claimed to be a
coupling-independent physical quantity.

## 7. Stability and remaining limits

No full-model production setting is promoted as converged unless both original
tolerances hold without a self-comparison. For propagation factor 4→5, the
nonflat-case maximum residual change is 3.032 nm and
maximum fixed reconstructed-wavefront change is 4.346 nm.
The corrected noisy decisions do not override deterministic failures.

Historical RTL/model equality, nine-cycle simulated latency, the 7.615181 nm
fixed-flat result including tip/tilt, broad frozen-grid residuals, and clipping
ablation do not depend on the flawed coupling and need no rerun. All historical
noisy per-setting marginals remain usable. Claims based on the old 0/36
framewise prediction-difference acceptance table must instead cite this run;
other grid-dependent optical predictions remain conditional on discretization.

Practical full-model propagation and pupil-edge convergence remain unresolved.
The bounded study does not establish a highest grid as truth, physical-bench
validity, or implemented-FPGA performance.

## 8. Claim-to-evidence and reproduction

See `claim_to_evidence.csv` and `MANIFEST.json`.

```bash
sim/venv/bin/python -m unittest \
  evaluation.tests.test_noise_coupling \
  evaluation.tests.test_rom_loading \
  evaluation.tests.test_reference_conventions -v
sim/venv/bin/python evaluation/optical/forward_model_investigation.py \
  --config evaluation/configs/forward_model_investigation_v1.json \
  --out evaluation/results/runs/forward_model_investigation_v2_20260928_final_reproduction
```

The output directory must not exist. Reciprocal optimization remains deferred.
