# Reproducible HCIPy and RTL-simulation evidence

Run: `manuscript_v1_20260928_final`  
Revision: `9483c72504b72dac2d8a925de63de3e0e8d995fc` (dirty state: `True`)  
Scope: simulation only; no synthesis, implementation, board, power, or experimental-data claims.

## Methods

The optical forward model is HCIPy propagation through the frozen obscured, spidered pupil, magnifier, 16×16 microlens array, and detector model. OPD RMS is aperture-weighted after removing the aperture-weighted piston. Modes 1–10 after piston in HCIPy's RMS-normalized Zernike basis are reconstructed. The primary floating baseline uses the unquantized matrix obtained from independent ±30 nm physical optical probes. A second floating path uses the dequantized stored 18-bit matrix; the RTL model then adds reciprocal, centroid, multiply-accumulate, slicing, and wrap semantics. All three paths consume identical detector integers.

The pupil is 8 m with a 1.2 m central obscuration and four 0.05 m spiders, sampled on a 256×256 base grid with 4× pupil-edge supersampling. OPD is converted to phase as 2π·OPD/λ at λ=700 nm. Detector x is array-column/local lenslet x; y is array-row/local lenslet y. The slope vector is all valid x slopes followed by all valid y slopes, matching the measured interaction matrix. Detector integration is represented by the propagated grid at the production setting and by explicit block integration in the refined-grid convergence study.

The RTL formats are: 16-bit intensity, 19-bit moments at the interface, 19-bit Q18 reciprocal narrowed to the existing 18-bit port, 24-bit Q18 centroid, signed 25-bit Q18 slope, signed 18-bit stored E matrix, 52-bit accumulator, and signed 25-bit output slice [42:18]. ROM signed values use checked nibble sign extension. Arithmetic uses the RTL's truncation and wrap behavior; no fitted output multiplier is used.

Photons mean photons incident over the full simulated pupil per frame. Quantum efficiency is 0.8; read noise is in electrons RMS; gain is fixed at 20.0 e/ADU; threshold is 5 ADU. Noise intervals use wavefront means as independent units, with repeated detector-noise exposures nested within each wavefront.

## Principal results

Noiseless flat residual: primary float 0.000033 nm; fixed 7.615181 nm. Noisy flat (n=100): primary float 3.391 nm, fixed 3.650 nm.

| Input RMS | n | Primary float residual | Fixed residual | Relative fixed residual | Matrix quant. | Arithmetic quant. | ADC-clipped px |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 nm | 60 | 4.03 nm | 4.27 nm | 21.3% | 0.000 nm | 1.328 nm | 0 |
| 50 nm | 60 | 4.76 nm | 5.01 nm | 10.0% | 0.000 nm | 1.362 nm | 0 |
| 100 nm | 60 | 6.49 nm | 6.75 nm | 6.7% | 0.000 nm | 1.480 nm | 0 |
| 200 nm | 60 | 12.63 nm | 12.65 nm | 6.3% | 0.001 nm | 1.331 nm | 19 |
| 400 nm | 60 | 24.77 nm | 24.83 nm | 6.2% | 0.001 nm | 1.353 nm | 96 |

Individual-mode primary-float gain range: 0.935–1.085 over 100 signed amplitude cases.

Corrected noise sweep: 16 cells, each with 20 independent wavefronts × 5 noise exposures. The table below shows the 3 e⁻ RMS read-noise branch; shot-only results are retained in `noise_summary.csv`.

| Photons over pupil/frame | Mean e⁻ / valid lenslet | Primary float residual | Fixed residual (95% CI) | ADC-clipped px / 100 frames |
|---:|---:|---:|---:|---:|
| 5e+05 | 2381 | 30.27 nm | 30.28 nm (28.62–31.94) | 0 |
| 1e+06 | 4762 | 24.53 nm | 24.63 nm (22.45–26.82) | 0 |
| 2e+06 | 9524 | 10.09 nm | 10.17 nm (9.30–11.10) | 0 |
| 3e+06 | 14286 | 5.76 nm | 5.97 nm (5.59–6.35) | 0 |
| 5e+06 | 23810 | 6.73 nm | 7.46 nm (6.91–7.99) | 10490 |
| 8e+06 | 38095 | 22.30 nm | 22.43 nm (21.40–23.47) | 60523 |
| 1.2e+07 | 57143 | 31.18 nm | 31.37 nm (30.02–32.77) | 65268 |
| 2e+07 | 95238 | 29.02 nm | 29.29 nm (28.55–29.95) | 68811 |

The fixed 20 e⁻/ADU gain is clean through 3×10⁶ photons/pupil/frame. ADC clipping begins at 5×10⁶ and performance worsens; this is an observed detector-model operating point, not a physical-camera specification.

Operating-range sweep: 10 levels × 12 paired directions.

| Input RMS | Fixed relative residual | Mean outer-2-pixel flux | Mean edge-centroid count | Valid-tile flux vs flat | Output-field use | ADC-clipped px |
|---:|---:|---:|---:|---:|---:|---:|
| 10 nm | 39.2% | 2.53% | 0.00 | 99.9% | 0.3% | 0 |
| 30 nm | 12.1% | 2.54% | 0.00 | 100.0% | 0.7% | 0 |
| 100 nm | 6.5% | 2.57% | 0.00 | 100.1% | 2.4% | 0 |
| 300 nm | 6.4% | 2.60% | 0.00 | 100.1% | 7.1% | 10 |
| 600 nm | 6.1% | 2.65% | 0.00 | 100.1% | 14.2% | 27 |
| 1000 nm | 6.3% | 3.21% | 0.00 | 100.1% | 23.6% | 29 |
| 1500 nm | 12.4% | 5.72% | 0.33 | 100.0% | 33.8% | 20 |
| 2000 nm | 28.4% | 8.81% | 0.75 | 99.4% | 40.1% | 20 |
| 3000 nm | 55.1% | 14.84% | 1.17 | 98.0% | 50.6% | 13 |
| 4000 nm | 71.1% | 20.31% | 0.92 | 96.6% | 61.9% | 8 |

Degradation starts in these paired mixtures near 1.5 µm RMS, where edge occupancy and edge-centroid counts rise while the output field remains below half scale. ADC clipping occurs earlier without comparable degradation, so the diagnostics support spot displacement/escape as a contributor rather than treating it as an untested assumption or the sole cause.

Out-of-model sweep: 5 levels × 25 wavefronts.

| Unmodelled / modelled RMS | Represented-mode wavefront error | Full-pupil residual |
|---:|---:|---:|
| 0% | 6.73 nm | 6.73 nm |
| 10% | 11.06 nm | 14.98 nm |
| 25% | 22.04 nm | 33.57 nm |
| 50% | 45.05 nm | 67.48 nm |
| 100% | 83.90 nm | 131.28 nm |

These two residual definitions are reported independently; no ordering is assumed from modal orthogonality.

Subpixel study: 41 positions spanning 0.0–1.0 pixel; bias RMS 0.01389 px, bias peak-to-peak 0.05804 px.

RTL: 35 cases, 350 coefficients, 0 coefficient mismatches, 0 slope/centroid mismatches, 0 unknown valid values, latency [9] simulated cycles.
Across the held-out mixed campaign there were zero invalid centroids, intensity/moment overflows, accumulator wraps, or output-field wraps. Per-mode coefficient bias, RMSE, p95 absolute error, and maxima are in `mixed_coefficient_summary.csv`. The wavefront error decomposition closed to floating-point roundoff and retains the signed cross term in every raw row.

## Convergence

Frozen tolerance: both ≤5.0% and ≤0.5 nm versus the finest tested setting.
Each setting receives its own flat reference and ±30 nm push-pull optical interaction calibration before the common held-out wavefront directions are scored, so this test does not confuse grid refinement with a reference/matrix mismatch.

**The production numerical settings do not pass the frozen convergence criterion. Quantitative optical residuals are therefore conditional on the 256×256, 4×-edge-supersampled discretization and should not be presented as numerically converged.**

| Study | Setting | Residual | Δ vs finest | Pass |
|---|---:|---:|---:|:---:|
| pupil_edge_supersampling | 2 | 8.181 nm | 45.90% / 2.574 nm | no |
| pupil_edge_supersampling | 4 | 5.871 nm | 4.70% / 0.264 nm | yes |
| pupil_edge_supersampling | 8 | 6.396 nm | 14.06% / 0.789 nm | no |
| pupil_edge_supersampling | 16 | 5.607 nm | 0.00% / 0.000 nm | yes |
| propagation_grid_with_detector_integration | 1 | 5.871 nm | 9.40% / 0.505 nm | no |
| propagation_grid_with_detector_integration | 2 | 8.254 nm | 53.82% / 2.888 nm | no |
| propagation_grid_with_detector_integration | 3 | 5.366 nm | 0.00% / 0.000 nm | yes |

## Why close float/fixed agreement does not validate the optics

The paired fixed–floating difference tests arithmetic implementation on the same slopes and calibration. Optical accuracy instead compares either reconstruction with an independently propagated known OPD. The arithmetic paths can agree exactly or nearly exactly while both inherit centroid bias, calibration/model mismatch, aliasing, detector loss, or an inadequate modal basis.

## Limitations

All optical frames are simulated. Detector parameters and optical geometry are assumed, not measured. The RTL test is finite simulation, not formal proof. Simulated cycle latency is not an achieved clock rate. No resource, timing-closure, throughput-in-seconds, power, physical-bench, or hardware-equivalence conclusion is supported.

## Claim-to-evidence table

| Claim | Status | Evidence |
|---|---|---|
| Optical calibration is self-consistent for the simulated bench | ESTABLISHED AT FROZEN DISCRETIZATION | noiseless flat, independent ±mode probes, held-out optical wavefronts; see convergence qualification |
| Floating reconstruction accuracy over 20–400 nm RMS | CONDITIONAL ON FROZEN NUMERICAL GRID | 300 held-out mixed wavefronts |
| Fixed-point discrepancy is separated from optical residual | ESTABLISHED FOR TESTED CASES | paired original-E, dequantized-E, and bit-accurate reconstructions on identical frames |
| RTL equals the bit-accurate model | ESTABLISHED FOR FINITE TEST SET | 35 cases, 350 coefficients |
| Physical-bench optical performance or hardware timing/power/resources | UNTESTED | no experimental frames, board run, synthesis, implementation, or timing closure in scope |

## Reproduction

From the repository root:

```bash
sim/venv/bin/python evaluation/optical/manuscript_evaluation.py --config evaluation/configs/manuscript_v1.json --out evaluation/results/runs/manuscript_v1_reproduction
```

The command refuses a non-empty output directory. Historical results are therefore not overwritten.
