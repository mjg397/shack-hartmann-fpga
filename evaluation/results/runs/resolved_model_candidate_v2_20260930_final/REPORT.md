# Resolved-model candidate calibration evaluation

Run: `resolved_model_candidate_v2_20260930_final`  
Source: `forward_model_investigation_v2_20260928_final`  
Scope: HCIPy/Python only. No active ROM, RTL arithmetic, reciprocal, gain,
threshold, geometry, synthesis, implementation, board, or real-data change.

## Executive decision

Factor 8 is the lowest tested numerical setting that passes the
predeclared **continuous optical** gate against the independently finer factor
9; factor 9 is not credited for agreeing with itself. It is therefore
a defensible lower candidate for expected optical intensity and continuous
centroids, not a validated end-to-end production setting.

The one frozen factor-8 candidate calibration **fails** the unchanged
5% and 0.5 nm end-to-end rule when applied at factors 8 and 9.
The largest selection-case integer-frame reconstructed-wavefront change is
0.485 nm. Held-out deterministic cases pass
48/130; the new noisy
confirmation passes 1/8 cells.
Consequently no candidate E matrix was quantized, no candidate hardware ROM set
was staged, no RTL simulation was run, and the full manuscript campaign must
not yet be repeated.

This does not alter the historical 7.615181 nm noiseless fixed-flat result,
including tip/tilt. Its reciprocal-rounding explanation and deferred status are
reused without reinvestigation.

## 1. Physical pixels versus numerical samples

All factors retain the same 256x256 physical detector, physical pixel edges,
lenslet geometry, wavelength, OPD coefficients, photons-per-pupil definition,
and detector order. A factor *f* HCIPy image has `(256f)^2` quadrature-power
samples; each non-overlapping `f x f` block is summed into one physical pixel
before noise, full-well clipping, gain, ADC rounding, and thresholded centroiding.
No extra area factor is applied because HCIPy's `NoiselessDetector` samples are
already `Wavefront.power`, including grid weights.

The independent analytic square-lenslet comparison continues to improve: its
worst factor-8 normalized-image error is 0.001569. A second
analytic sinc-squared test varies only composite midpoint quadrature inside the
unchanged physical pixel boundaries (`analytic_pixel_quadrature.csv`).

## 2. Successive refinement

| Pair | max image rel. L2 | max continuous centroid px | max threshold centroid px | max integer centroid px | max changed ADU px |
|---|---|---|---|---|---|
| 5→6 | 0.001660 | 0.000675 | 0.002287 | 0.003008 | 285 |
| 6→7 | 0.001319 | 0.000511 | 0.000893 | 0.002225 | 340 |
| 7→8 | 0.001281 | 0.000780 | 0.001248 | 0.002270 | 212 |
| 8→9 | 0.000639 | 0.000651 | 0.001258 | 0.001685 | 136 |

The factor-8→9 continuous centroid gate of 0.0012 pixel maps to
0.500 nm across one lenslet by the
fixed geometry; final decisions still use reconstructed common-pupil RMS and
both original tolerances. The optical gate passed: `True`.

## 3. Separated numerical controls

At factor 7, the largest successive
pupil-edge-control image change is 4.967e-05. At factor
5, changing zero padding or transfer
sampling from the documented defaults changes normalized images by at most
8.441e-05. Entrance-pupil/phase-only, Fresnel, edge, analytic pixel
quadrature, and full propagated comparisons are stored separately. These checks
show that the remaining factor-8→9 sensitivity is not explained
by insufficient edge quadrature or the default Fresnel padding alone.

## 4. Frozen candidate calibration

The factor-8 floating reference is the exact thresholded quotient
from one noiseless integer flat frame. Its documented nearest-integer Q18
representation is saved separately but is not substituted into this floating
primary path because the hardware gate was not reached. The interaction matrix
uses independent +/-30 nm modal
probes; its pseudoinverse is saved in physical metre-per-slope-pixel units.
The same reference and inverse are used at both grids. Factor-9
recalibration is diagnostic only and never enters the primary rows.

| Case | factor-9 residual nm | reconstruction change nm | absolute residual change nm | margin nm | decision |
|---|---|---|---|---|---|
| flat | 0.177 | 0.177 | 0.177 | 0.009 | FAIL |
| single_tilt_x_100nm | 11.144 | 0.421 | 0.358 | 0.500 | pass |
| mixed_20nm_0 | 0.855 | 0.439 | 0.213 | 0.043 | FAIL |
| mixed_100nm_0 | 8.748 | 0.485 | 0.029 | 0.437 | FAIL |

For context, candidate accuracy improved over the historical active fixed
calibration on these four selection cases, but improvement is not convergence:

| Grid factor | Calibration | mean residual nm | max residual nm | role |
|---|---|---|---|---|
| 8 | frozen candidate | 5.337 | 11.501 | primary |
| 8 | per-grid diagnostic | 5.337 | 11.501 | diagnostic |
| 8 | historical active fixed | 11.622 | 20.305 | diagnostic |
| 9 | frozen candidate | 5.231 | 11.144 | primary |
| 9 | per-grid diagnostic | 5.389 | 11.775 | diagnostic |
| 9 | historical active fixed | 11.897 | 19.927 | diagnostic |

Continuous and thresholded rows in `candidate_selection_paired.csv` localize
the failure. The continuous optical changes are small but some are already at
the strict combined margin; ADC rounding then changes hundreds of pixels and
raises the mixed-100-nm reconstruction change. The cause is therefore a
combination of residual forward quadrature sensitivity and nonlinear detector
boundary crossings, with calibration mapping determining their wavefront
impact—not pupil-edge or Fresnel-padding error.

## 5. Fresh held-out and noisy confirmation

Fresh post-freeze mixed directions at 20, 100, and 400 nm RMS, four 75 nm single
modes, flat, and 1M/3M photons were evaluated. Every reported detector frame is
non-clipping: `True`.
Tip/tilt is retained throughout.

Held-out mixed-wavefront accuracy of the same frozen factor-8 calibration is:

| Photons | input RMS nm | factor-8 mean (max) residual nm | factor-9 mean (max) residual nm |
|---|---|---|---|
| 1M | 20 | 8.154 (max 12.683) | 8.082 (max 12.839) |
| 1M | 100 | 29.626 (max 38.408) | 29.955 (max 39.258) |
| 1M | 400 | 40.572 (max 53.069) | 40.490 (max 53.192) |
| 3M | 20 | 1.385 (max 2.321) | 1.366 (max 2.335) |
| 3M | 100 | 8.359 (max 11.290) | 8.358 (max 11.384) |
| 3M | 400 | 36.277 (max 51.119) | 36.308 (max 51.326) |

| Condition | photons | residual delta [95% CI] nm | cross-batch squared nm^2 | deterministic mean (upper) nm | margin nm | decision |
|---|---|---|---|---|---|---|
| flat | 1M | -0.030 [-0.067, +0.008] | 0.012 (upper 0.028) | 0.114 (upper 0.114) | 0.264 | pass |
| flat | 3M | +0.006 [-0.011, +0.023] | 0.005 (upper 0.009) | 0.177 (upper 0.177) | 0.137 | FAIL |
| mixed_20nm | 1M | -0.084 [-0.129, -0.038] | 0.028 (upper 0.059) | 0.564 (upper 0.776) | 0.399 | FAIL |
| mixed_20nm | 3M | -0.002 [-0.015, +0.011] | 0.006 (upper 0.010) | 0.409 (upper 0.460) | 0.146 | FAIL |
| mixed_100nm | 1M | -0.213 [-0.262, -0.165] | 0.071 (upper 0.098) | 0.881 (upper 1.040) | 0.500 | FAIL |
| mixed_100nm | 3M | +0.012 [-0.011, +0.034] | 0.008 (upper 0.013) | 0.415 (upper 0.481) | 0.423 | FAIL |
| mixed_400nm | 1M | -0.036 [-0.067, -0.004] | 0.013 (upper 0.026) | 0.489 (upper 0.622) | 0.500 | FAIL |
| mixed_400nm | 3M | -0.022 [-0.043, -0.001] | 0.011 (upper 0.019) | 0.506 (upper 0.610) | 0.500 | FAIL |

The noisy expected-reconstruction rule does not take the positive norm of only
five exposures as exact. Four independent five-exposure batches are used. The
coefficient-space cross-products of independent batch means are unbiased for
the squared norm of the expected reconstruction difference; their interval is
the acceptance statistic. The ordinary positive five-exposure norm remains in
the tables only as a descriptive comparator. Every noisy frame is reconstructed
individually; no reconstructed mean detector image is used.

## 6. Hardware gate and usable claims

Hardware gate passed: `False`. Candidate floating files and
hashes are preserved under `candidate_calibration/`, but hardware-format E ROMs
were deliberately not generated. The active ROM hashes are unchanged.

Still usable with the earlier qualifications:

- the historical exact RTL/model agreement and latency;
- the 7.615181 nm fixed-flat result including tip/tilt and its known reciprocal bias;
- the clipping ablation and its distinction between captured expected electrons
  and nominal QE x photons / lenslet;
- per-setting noisy marginals from earlier runs;
- factor 8 continuous expected-intensity/centroid predictions, only
  with the factor-8→9 sensitivity bounds reported here.

Not usable as unconditional manuscript numbers: discrete detector,
reconstruction, residual, or noisy convergence claims that assume a single
resolved production grid. The candidate calibration accuracy is reported as
observed and is not repaired by bias subtraction, fitted output scale, or
tip/tilt removal.

## 7. Recommendation and unresolved work

Use factor 8 as the minimum setting for exploratory continuous optical
predictions, and bracket every discrete/reconstructed result with factor
9. Do not adopt factor 9 merely because it is the largest tested
grid. End-to-end practical convergence remains unresolved specifically at ADC/
threshold boundary crossings and their calibration-weighted reconstruction.
An unbounded refinement sweep is not justified by this bounded failure.

The full manuscript campaign should **not** be rerun. Reciprocal precision
optimization remains a separate deferred architecture task; no work on it was
performed here.

## 8. Reproduction

```bash
sim/venv/bin/python -m unittest \
  evaluation.tests.test_resolved_model_candidate \
  evaluation.tests.test_noise_coupling \
  evaluation.tests.test_rom_loading \
  evaluation.tests.test_reference_conventions -v
sim/venv/bin/python evaluation/optical/resolved_model_candidate.py \
  --config evaluation/configs/resolved_model_candidate_v2.json \
  --out evaluation/results/runs/resolved_model_candidate_v2_20260930_final_reproduction
```

The output directory must not already exist. See `MANIFEST.json` for exact
source, input, calibration, result, and active-ROM hashes.
