# Frozen-reconstructor and noisy-ensemble convergence evaluation

Run: `ensemble_convergence_v1_20260928_final`  
Frozen source evidence: `blocker_investigation_v1_20260928_final`  
Scope: HCIPy and Python; no RTL rerun was needed because RTL, widths, reciprocal,
references, and E-matrix ROMs are byte-identical to the frozen source run.

## Executive result

The requested remaining tests completed. Forward-grid refinement was evaluated
first with the **same frozen production reference and E matrix** at every grid;
the already-completed per-grid recalibration remains a separately labeled
diagnostic. A new noisy ensemble then used 20 independent wavefronts with five
nested exposures in each of four predeclared non-clipping amplitude/flux cells.

The production settings do not satisfy deterministic convergence. Against the
highest tested settings, the six-case frozen-reconstructor study gives a mean
prediction change of 1.476
nm for edge 4→20 and
14.227 nm for propagation
1→5, with case maxima of 3.124
and 29.044 nm respectively.
No finest-grid self-comparison is counted as evidence.

The new ensemble test accepted 0 of
36 predeclared comparison/condition cells under the
frozen two-part equivalence rule. 17 cells would have appeared
acceptable from residual means alone but failed the reconstructed-wavefront
difference requirement; they are not called converged.

## 1. Frozen production reconstructor: deterministic forward sensitivity

Every setting uses the same source-run reference ROMs, integer E-matrix ROMs,
reciprocal LUT, fixed-point reconstruction, 3e6-photon noiseless detector, and
common 240-base/20× edge-sampled scoring pupil. Refined optical intensity is block-summed into
the physical 256×256 detector pixels before detector conversion, integer ADC
rounding, and the unchanged 5 ADU threshold.

| Study | Setting | Mean residual (nm) | Mean prediction Δ vs highest (nm) | Max prediction Δ (nm) | All six residual cases pass 5% and 0.5 nm |
|---|---|---|---|---|---|
| pupil_edge_supersampling | 4 | 5.567 | 1.476 | 3.124 | no |
| pupil_edge_supersampling | 12 | 5.516 | 0.501 | 0.547 | no |
| pupil_edge_supersampling | 16 | 5.342 | 0.719 | 2.497 | no |
| pupil_edge_supersampling | 20 | 5.531 | 0.000 | 0.000 | self—not evidence |
| propagation_grid_with_detector_integration | 1 | 5.567 | 14.227 | 29.044 | no |
| propagation_grid_with_detector_integration | 2 | 11.010 | 4.499 | 8.300 | no |
| propagation_grid_with_detector_integration | 3 | 12.540 | 2.814 | 4.972 | no |
| propagation_grid_with_detector_integration | 4 | 12.119 | 2.406 | 4.346 | no |
| propagation_grid_with_detector_integration | 5 | 12.915 | 0.000 | 0.000 | self—not evidence |

`deterministic_frozen_paired.csv` records, per case, normalized detector-intensity
changes, expected and integer ADU changes, fixed-centroid changes, coefficient
changes, reconstructed-wavefront changes, and ground-truth residual changes.
Similar aggregate residuals are therefore not treated as equivalent predictions.

## 2. Per-grid recalibration diagnostic (kept separate)

The per-grid results were not substituted into the production path. They are
byte-copied from the blocker run into `reused_evidence/` and summarized in
`per_grid_recalibration_diagnostic_summary.csv`. They answer how much sensitivity
can be absorbed by recalibration at each numerical grid; the frozen path above
answers what the actual fixed reconstructor predicts when only the forward model
is refined. Their differing curves are shown together only for diagnosis.

## 3. New noisy-ensemble convergence test

The design and acceptance rule were frozen in the configuration before this
run: 20 and 100 nm RMS; 1e6 and 3e6 photons/pupil/frame; 20 independent
wavefronts; five nested exposures. Both fluxes are non-clipping in every tested
frame: **True**; all frames remain below full
well: **True**.

Noise pairing is explicit, not seed reuse by itself. At each pixel and exposure,
all grids share `Poisson(min λ)` and each receives an independent
`Poisson(λ_grid−min λ)` increment; their sum has exactly the required Poisson
marginal. One shared Gaussian read-noise field has the required N(0,3² e⁻)
marginal on every grid. Wavefront means over the five exposures are the 20
independent units used for Student-t intervals.

Acceptance requires both: (1) the 95% paired interval for signed residual change
lies wholly inside ±`min(0.5 nm, 5% of supported-reference mean residual)`, and
(2) the upper 95% interval for paired reconstructed-wavefront RMS difference is
below that same margin. A nonsignificant difference is not a pass.

| Comparison | Type | Accepted cells | All cells | Max |mean residual Δ| (nm) | Max mean prediction Δ (nm) | Worst prediction upper/margin |
|---|---|---|---|---|---|---|
| edge_4_to_12 | successive_refinement | 0/4 | no | 0.077 | 2.632 | 12.82× |
| edge_12_to_16 | successive_refinement | 0/4 | no | 0.128 | 2.595 | 13.45× |
| edge_16_to_20 | successive_refinement | 0/4 | no | 0.092 | 2.654 | 13.22× |
| edge_4_to_20 | production_to_supported_reference | 0/4 | no | 0.130 | 2.457 | 12.22× |
| propagation_1_to_2 | successive_refinement | 0/4 | no | 15.548 | 31.359 | 68.78× |
| propagation_2_to_3 | successive_refinement | 0/4 | no | 3.741 | 9.718 | 22.49× |
| propagation_3_to_4 | successive_refinement | 0/4 | no | 1.997 | 6.465 | 14.69× |
| propagation_4_to_5 | successive_refinement | 0/4 | no | 0.749 | 6.707 | 15.97× |
| propagation_1_to_5 | production_to_supported_reference | 0/4 | no | 9.060 | 42.388 | 94.27× |

### Production settings versus supported references

| Comparison | Input RMS (nm) | Photons | Reference residual (nm) | Residual Δ mean [95% CI] (nm) | Prediction Δ mean (upper 95%) | Margin (nm) | Decision |
|---|---|---|---|---|---|---|---|
| edge_4_to_20 | 20 | 1M | 10.563 | -0.130 [-0.245, -0.015] | 1.824 (upper 1.982) | 0.500 | fail |
| edge_4_to_20 | 20 | 3M | 4.306 | -0.060 [-0.307, +0.186] | 2.457 (upper 2.630) | 0.215 | fail |
| edge_4_to_20 | 100 | 1M | 28.002 | +0.012 [-0.162, +0.186] | 2.441 (upper 2.650) | 0.500 | fail |
| edge_4_to_20 | 100 | 3M | 6.799 | +0.106 [-0.053, +0.265] | 2.372 (upper 2.544) | 0.340 | fail |
| propagation_1_to_5 | 20 | 1M | 9.151 | +1.282 [+0.573, +1.991] | 10.480 (upper 10.929) | 0.458 | fail |
| propagation_1_to_5 | 20 | 3M | 7.300 | -3.054 [-3.530, -2.578] | 7.258 (upper 7.480) | 0.365 | fail |
| propagation_1_to_5 | 100 | 1M | 18.953 | +9.060 [+6.392, +11.729] | 42.388 (upper 47.136) | 0.500 | fail |
| propagation_1_to_5 | 100 | 3M | 13.039 | -6.135 [-7.460, -4.809] | 11.330 (upper 12.499) | 0.500 | fail |

This ensemble result is independent of—and does not override—the failed
noiseless per-case test. Full per-frame, per-wavefront, and uncertainty records
are in `ensemble_raw.csv`, `ensemble_paired_raw.csv`, and
`ensemble_paired_summary.csv`.

## 4. Completed evidence reused without rerun

- The 7.615181 nm noiseless fixed-flat result, including tip/tilt, is preserved.
  It was not re-investigated or corrected away.
- The clipping ablation is reused. At 8e6 photons, production float residual was
  21.77 nm versus
  6.51 nm with
  only the upper ADC clamp removed. Expected valid-tile electrons remain
  distinct from nominal QE×photons/168.
- The prior 35-case RTL suite and exact flat replay remain valid. No RTL
  simulation was rerun because no RTL or calibration input changed.

## 5. Manuscript use and limitations

Still usable, with the prior qualifications: the frozen-grid 300-wavefront
mixed residuals (4.03/4.76/6.49/12.63/24.77 nm primary float and
4.27/5.01/6.75/12.65/24.83 nm fixed at 20/50/100/200/400 nm RMS), finite-case
RTL equality, simulated nine-cycle latency, the fixed-flat trace, and the paired
clipping result. These are simulation results at the stated discretization and
detector assumptions—not numerically grid-independent optical predictions.

Whether any refinement comparison is stable is reported condition by condition
in the ensemble tables. A comparison is not promoted to a general convergence
claim unless all four cells pass the predeclared rule. Practical convergence of
the production optical model remains unresolved wherever that rule or a
successive deterministic comparison fails.

No physical-bench equivalence, FPGA timing closure, resource, power, synthesis,
or implementation claim is supported.

## 6. Claim-to-evidence

| Claim | Status | Evidence |
|---|---|---|
| Frozen production predictions are insensitive to tested refinements | NOT ESTABLISHED | deterministic and ensemble paired tables |
| Per-grid recalibration is production calibration | NOT CLAIMED | diagnostic is explicitly separated |
| Noisy ensemble intervals establish agreement | CONDITION-SPECIFIC | `ensemble_paired_summary.csv`; frozen equivalence rule |
| 7.615181 nm fixed flat and active calibration are preserved | ESTABLISHED | ROM hashes, source flat summary, manifest |
| Clipping is a major but not sole high-flux effect in the assumed detector | COMPLETED PRIOR EVIDENCE | byte-copied clipping ablation |
| Physical bench or implemented FPGA performance | UNTESTED | outside scope |

## 7. Reproduction

```bash
sim/venv/bin/python -m unittest \
  evaluation.tests.test_rom_loading \
  evaluation.tests.test_reference_conventions -v
sim/venv/bin/python evaluation/optical/ensemble_convergence.py \
  --config evaluation/configs/ensemble_convergence_v1.json \
  --out evaluation/results/runs/ensemble_convergence_v1_20260928_final_reproduction
```

The output directory must not exist. `MANIFEST.json` freezes the exact source,
configuration, calibration inputs, reused evidence hashes, dependency versions,
working-tree patch, result hashes, and commands. Reciprocal optimization is
listed only in `DEFERRED_WORK.md` and was not started.
