# Reproducible HCIPy and RTL-simulation evidence

Run: `manuscript_v1_20260928`  
Revision: `9483c72504b72dac2d8a925de63de3e0e8d995fc` (dirty state: `True`)  
Scope: simulation only; no synthesis, implementation, board, power, or experimental-data claims.

## Methods

The optical forward model is HCIPy propagation through the frozen obscured, spidered pupil, magnifier, 16×16 microlens array, and detector model. OPD RMS is aperture-weighted after removing the aperture-weighted piston. Modes 1–10 after piston in HCIPy's RMS-normalized Zernike basis are reconstructed. The primary floating baseline uses the unquantized matrix obtained from independent ±30 nm physical optical probes. A second floating path uses the dequantized stored 18-bit matrix; the RTL model then adds reciprocal, centroid, multiply-accumulate, slicing, and wrap semantics. All three paths consume identical detector integers.

The pupil is 8 m with a 1.2 m central obscuration and four 0.05 m spiders, sampled on a 256×256 base grid with 4× pupil-edge supersampling. OPD is converted to phase as 2π·OPD/λ at λ=700 nm. Detector x is array-column/local lenslet x; y is array-row/local lenslet y. The slope vector is all valid x slopes followed by all valid y slopes, matching the measured interaction matrix. Detector integration is represented by the propagated grid at the production setting and by explicit block integration in the refined-grid convergence study.

The RTL formats are: 16-bit intensity, 19-bit moments at the interface, 19-bit Q18 reciprocal narrowed to the existing 18-bit port, 24-bit Q18 centroid, signed 25-bit Q18 slope, signed 18-bit stored E matrix, 52-bit accumulator, and signed 25-bit output slice [42:18]. ROM signed values use checked nibble sign extension. Arithmetic uses the RTL's truncation and wrap behavior; no fitted output multiplier is used.

Photons mean photons incident over the full simulated pupil per frame. Quantum efficiency is 0.8; read noise is in electrons RMS; gain is fixed at 20.0 e/ADU; threshold is 5 ADU. Noise intervals use wavefront means as independent units, with repeated detector-noise exposures nested within each wavefront.

## Principal results

Noiseless flat residual: primary float 0.000033 nm; fixed 7.615181 nm. Noisy flat (n=100): primary float 3.391 nm, fixed 3.650 nm.

| Input RMS | n | Primary float residual | Fixed residual | Matrix quant. | Arithmetic quant. |
|---:|---:|---:|---:|---:|---:|
| 20 nm | 60 | 4.03 nm | 4.27 nm | 0.000 nm | 1.328 nm |
| 50 nm | 60 | 4.76 nm | 5.01 nm | 0.000 nm | 1.362 nm |
| 100 nm | 60 | 6.49 nm | 6.75 nm | 0.000 nm | 1.480 nm |
| 200 nm | 60 | 12.63 nm | 12.65 nm | 0.001 nm | 1.331 nm |
| 400 nm | 60 | 24.77 nm | 24.83 nm | 0.001 nm | 1.353 nm |

Individual-mode primary-float gain range: 0.935–1.085 over 100 signed amplitude cases.

Corrected noise sweep: 16 cells, each with 20 independent wavefronts × 5 noise exposures. See `noise_summary.csv`; clipping totals are reported rather than hidden.

Operating-range sweep: 10 levels × 12 paired directions. Spot edge occupancy, flux retention, centroid-edge counts, ADC clipping, accumulator/output wraps, and representative frames are stored alongside residuals. This supports only case-specific degradation conclusions.

Out-of-model sweep: 5 levels × 25 wavefronts. Represented-mode and full-pupil residuals are both reported without assuming an ordering.

Subpixel study: 41 positions spanning 0.0–1.0 pixel; bias RMS 0.01389 px, bias peak-to-peak 0.05804 px.

RTL: 35 cases, 350 coefficients, 0 coefficient mismatches, 0 slope/centroid mismatches, 0 unknown valid values, latency [9] simulated cycles.

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
