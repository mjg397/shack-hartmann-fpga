# Final simulation-only Shack-Hartmann evidence package

Run: `final_evidence_package_v1_20260930_final`  
Frozen optical candidate source: `resolved_model_candidate_v2_20260930_final`

## Evidence-based decision

The repository now supports a complete **simulation-only Results section with
explicit numerical-sensitivity qualifications**. It does not support claims of
continuum convergence, physical-bench validation, achieved implementation
timing/resources/power, board operation, or closed-loop performance.

The candidate factor-8 hardware calibration was generated in an isolated run
directory and tested without changing the active ROMs. Across 19
flat, noisy-flat, signed-mode, mixed-mode, low-flux, and operating-range frames,
RTL coefficients and slope/centroid records match the bit-accurate model exactly.
There are no unknown valid outputs, counts are 168 slopes and
10 coefficients per frame, and FINAL_TILE_PIXEL arrival-to-done
latency is [9] cycles in simulation.

The original 5% and 0.5 nm combined convergence criterion remains failed. This
report does not replace it with a favorable threshold. Instead it reports the
four estimands separately and treats factor-8/factor-9 differences as observed
adjacent-grid sensitivity, not a rigorous continuum-error bound.

## 1. Metric and statistical audit

Every wavefront metric uses the same aperture-weighted, piston-removed common
pupil. The sampled-mode Gram matrix satisfies `delta.T @ G @ delta` = direct
wavefront RMS squared for representative basis and mixed vectors to maximum
absolute RMS error 1.323e-23 m.
Tip/tilt remains included.

The earlier independent-batch calculations were correct: residual subtests
passed 8/8, expected noisy
reconstruction subtests passed 8/8,
deterministic integer-frame subtests passed 1/8,
and the preserved combined result is 1/8.
Of the seven failures, 3 are
near-zero relative-margin failures and
4 have
substantial absolute deterministic changes.
Negative unbiased squared estimates are retained; confidence intervals use
wavefronts as independent units and compare squared quantities with squared
tolerances.

## 2. Claim and uncertainty specification

- Optical accuracy is the mean ground-truth residual with a two-sided 95%
  Student-t interval across 20 independent mixed wavefronts.
- Numerical sensitivity is the paired factor-8/factor-9 change under one frozen
  calibration. It is an observed sensitivity, not a continuum bound.
- Expected noisy reconstructed-wavefront change uses four independent batches
  of five exposures and Gram-weighted cross-products of independent batch means.
- Arithmetic discrepancy is candidate fixed-minus-original-floating output on
  identical integer frames. Matrix/reference quantization and remaining
  arithmetic quantization are separated as coefficient vectors, with their Gram
  squares and cross terms retained in machine-readable results.
- No independently justified alternate error budget is introduced. Claims whose
  precision is comparable to the observed numerical sensitivity remain unresolved.

## 3. Candidate hardware calibration and arithmetic

The candidate E scale is derived, not fitted:
`max(|E|)/(2^17-1) = 1.196458e-13 m/LSB`
(`2^-42.926`). Matrix words use the full signed 18-bit
range without saturation. References use the documented exact-quotient Q18
rounding. The reciprocal LUT and all RTL arithmetic are unchanged.

The deterministic 3M factor-8 candidate flat result, including tip/tilt, is
7.658488 nm fixed versus
0.000000 nm floating. This new
candidate flat bias is reported directly; the historical 7.615181 nm value
remains valid only for its historical frame and calibration.
On this candidate frame, the reference, matrix, and remaining arithmetic
component RMS values are 0.000203,
2.074e-10, and
7.658488 nm, respectively; signed cross
terms are retained in `deterministic_confirmation_raw.csv`.

## 4. Fresh bounded confirmation

### Deterministic expected-frame performance

| Photons | input RMS nm | floating residual [95% CI] nm | fixed residual [95% CI] nm | mean fixed-float nm | p95 fixed-float nm |
|---|---|---|---|---|---|
| 1M | 20 | 7.385 [6.151, 8.620] | 7.457 [6.200, 8.714] | 0.960 | 1.253 |
| 1M | 100 | 27.517 [24.750, 30.284] | 27.484 [24.743, 30.225] | 1.467 | 1.835 |
| 1M | 400 | 40.987 [36.613, 45.361] | 40.949 [36.472, 45.426] | 1.448 | 1.761 |
| 3M | 20 | 1.433 [1.161, 1.705] | 4.041 [3.390, 4.692] | 3.835 | 5.732 |
| 3M | 100 | 8.349 [7.298, 9.400] | 9.026 [7.826, 10.225] | 2.905 | 4.747 |
| 3M | 400 | 36.314 [32.286, 40.343] | 36.283 [32.156, 40.409] | 1.797 | 2.909 |

### Noisy performance

| Photons | input RMS nm | floating residual [95% CI] nm | fixed residual [95% CI] nm |
|---|---|---|---|
| 1M | 20 | 7.825 [7.493, 8.156] | 7.849 [7.487, 8.210] |
| 1M | 100 | 26.567 [24.034, 29.100] | 26.528 [24.045, 29.010] |
| 1M | 400 | 41.083 [36.686, 45.479] | 41.142 [36.611, 45.672] |
| 3M | 20 | 2.885 [2.774, 2.996] | 3.227 [3.054, 3.399] |
| 3M | 100 | 8.316 [7.456, 9.175] | 8.631 [7.738, 9.524] |
| 3M | 400 | 36.420 [32.410, 40.430] | 36.500 [32.443, 40.556] |

All fresh deterministic and noisy frames are non-clipping. Across the full
fresh deterministic set there are 0
invalid centroids, 0 input/moment
overflows, 0 accumulator
wraps, and 0 output-field wraps.
Coefficient biases, per-mode tails, signed modes, flat cases, and all per-case
values are provided in the CSV/NPZ artifacts.

### Numerical sensitivity: separate estimands, floating candidate path

| Condition | photons | A residual delta [95% CI] nm | B expected squared nm² | C variability delta [95% CI] nm | D deterministic delta (upper) nm | original combined |
|---|---|---|---|---|---|---|
| flat | 1M | -0.040 [-0.064, -0.016] | 0.0049 (upper 0.0179) | -0.049 [-0.077, -0.021] | 0.114 (upper 0.114) | pass |
| flat | 3M | +0.006 [-0.009, +0.021] | 0.0081 (upper 0.0121) | +0.001 [-0.014, +0.016] | 0.177 (upper 0.177) | FAIL |
| mixed_20nm | 1M | -0.090 [-0.134, -0.045] | 0.0323 (upper 0.0505) | -0.029 [-0.073, +0.016] | 0.481 (upper 0.624) | FAIL |
| mixed_20nm | 3M | +0.000 [-0.019, +0.019] | 0.0064 (upper 0.0091) | -0.009 [-0.026, +0.007] | 0.374 (upper 0.438) | FAIL |
| mixed_100nm | 1M | -0.200 [-0.235, -0.165] | 0.0464 (upper 0.0790) | -0.028 [-0.077, +0.021] | 0.783 (upper 1.013) | FAIL |
| mixed_100nm | 3M | -0.021 [-0.042, -0.001] | 0.0062 (upper 0.0127) | +0.002 [-0.014, +0.017] | 0.373 (upper 0.455) | FAIL |
| mixed_400nm | 1M | -0.043 [-0.076, -0.010] | 0.0105 (upper 0.0236) | -0.007 [-0.035, +0.022] | 0.639 (upper 0.831) | FAIL |
| mixed_400nm | 3M | -0.016 [-0.046, +0.014] | 0.0046 (upper 0.0153) | +0.016 [-0.007, +0.040] | 0.563 (upper 0.645) | FAIL |

On the fresh data, the preserved rule passes 1/8 combined cells for floating,
1/8 for dequantized, and 1/8 for fixed arithmetic. The floating/dequantized
paths pass A and B in all eight cells but D in only one. The fixed path passes
A in 7/8,
B in 6/8,
and D in 1/8.
All paths and decisions are retained in `noisy_confirmation_summary.csv`.

Similar aggregate factor-8/factor-9 residuals do not erase deterministic
per-wavefront or integer-boundary differences. Failures whose absolute changes
remain below 0.5 nm but miss a near-zero 5% margin are labeled separately in
the machine-readable audit from failures exceeding 0.5 nm. Neither category is
relabeled as converged.

## 5. Results draft

Under the repository's assumed 8 m obstructed/spidered pupil, 16x16 lenslet,
0.7 µm wavelength, 256x256 detector, fixed gain, 5-ADU threshold, and stated
Poisson/read-noise model, factor 8 passed the previously frozen continuous-
optical gate against factor 9 on the selection cases. The fresh confirmation
quantifies residual and arithmetic performance in the table above. Candidate
fixed-point arithmetic was verified exactly against RTL for the selected
integer frames, while its discrepancy from the floating optical reconstruction
remains explicitly measured. Factor-8/factor-9 differences are reported as
numerical sensitivity and prevent an unconditional continuum-convergence claim.

## 6. Evidence ready to cite

- Candidate RTL/bit-accurate equality for the 19 explicitly
  enumerated frames, with simulated event latency and no unknown valid outputs.
- Fresh factor-8 floating, dequantized, and fixed residuals at 20/100/400 nm and
  1M/3M photons, with 20 independent wavefronts and stated 95% intervals.
- Candidate fixed-minus-floating discrepancy, component/cross-term breakdown,
  coefficient bias, tail errors, and flat bias including tip/tilt.
- Factor-8/factor-9 mean residual, expected noisy reconstruction, noise
  variability, and deterministic-frame sensitivity as four separate results.
- Historical stress/clipping evidence and historical 7.615181 nm flat result,
  only with their original calibration/input labels.

## 7. Limitations

Essential missing evidence for broader claims is physical-sensor/bench data and
an independently justified continuum or numerical-error bound. Vivado timing,
resources, power, board behavior, and closed-loop performance were not measured
and must be excluded. The original combined convergence tests still fail.

Optional future improvements include reciprocal precision optimization and
alternative numerical propagators, but both are outside this final bounded
scope. They are not prerequisites for citing the verified simulation arithmetic
or the qualified optical results above.

## 8. Reproduction

```bash
sim/venv/bin/python -m unittest \
  evaluation.tests.test_final_evidence_package \
  evaluation.tests.test_resolved_model_candidate \
  evaluation.tests.test_noise_coupling \
  evaluation.tests.test_rom_loading \
  evaluation.tests.test_reference_conventions -v
sim/venv/bin/python evaluation/optical/final_evidence_package.py \
  --config evaluation/configs/final_evidence_package_v1.json \
  --out evaluation/results/runs/final_evidence_package_v1_20260930_final_reproduction
```

The output directory must not exist. `RESULTS_INDEX.json` maps every principal
table and figure to its source run, optical setting, calibration/ROM hashes,
arithmetic path, sample count, and uncertainty definition. `MANIFEST.json`
contains exact source, dependency, input, ROM, and result hashes.
