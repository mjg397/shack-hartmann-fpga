# HCIPy optical validation campaign

> **HISTORICAL CALIBRATION REPORT.** The authoritative focused optical audit is
> [`evaluation/results/runs/forward_model_investigation_v2_20260928_final/REPORT.md`](../evaluation/results/runs/forward_model_investigation_v2_20260928_final/REPORT.md),
> which preserves and supersedes the affected noisy metric from
> [`ensemble_convergence_v1_20260928_final`](../evaluation/results/runs/ensemble_convergence_v1_20260928_final/REPORT.md),
> and reuses the preserved flat/clipping evidence from
> [`blocker_investigation_v1_20260928_final`](../evaluation/results/runs/blocker_investigation_v1_20260928_final/REPORT.md)
> and is based on the preserved broad campaign at
> [`manuscript_v1_20260928_final`](../evaluation/results/runs/manuscript_v1_20260928_final/REPORT.md).
> It preserves these calibration artifacts but supersedes this document's
> outstanding-noise statement, partial subpixel interval, 26-case RTL count,
> operating-range attribution, and “4× adequate” convergence conclusion.
> Numerical optical residuals are now explicitly conditional on the production
> grid because propagation-grid convergence missed the frozen tolerance.

Simulation-only optical evaluation of the Shack-Hartmann FPGA reconstruction
pipeline, with the calibration corrected and re-measured.

```bash
sim/venv/bin/python evaluation/optical/regen_reference.py --apply
sim/venv/bin/python evaluation/optical/regen_ematrix.py --apply
sim/venv/bin/python evaluation/optical/campaign.py
sim/venv/bin/python evaluation/optical/arithmetic_discrepancy.py
sim/venv/bin/python evaluation/optical/experiments_e4_e6_e7.py
sim/venv/bin/python evaluation/optical/rtl_optical_subset.py
sim/venv/bin/python evaluation/optical/make_optical_figures.py
```

Generated 2026-09-28. Baseline preserved immutably in
`evaluation/results/baseline_v1/` with source and ROM hashes.

> **Nothing here is experimental validation.** Every image is an HCIPy
> simulation. No hardware measurement, no recorded optical data. Clock and
> resource figures remain **pending** on a Vivado run.

---

## 1. Headline: before / after

Two calibration defects were found and fixed. Both were **calibration**, not
datapath — the RTL was never modified.

| Metric | Baseline | Corrected | Change |
|---|---|---|---|
| Flat-wavefront bias (noiseless) | 8.349 nm | **0.000 nm** | eliminated |
| Flat residual (noisy, n=100) | 9.134 ± 0.443 nm | **3.391 ± 1.342 nm** | −63 % |
| Mode recovery gain spread | 0.774 – 1.076 | **0.935 – 1.085** | 0.302 → 0.150 |
| Relative error @ 20 nm RMS | 49.2 % | **21.3 %** | −57 % |
| Relative error @ 50 nm | 22.7 % | **10.0 %** | −56 % |
| Relative error @ 100 nm | 16.8 % | **6.7 %** | −60 % |
| Relative error @ 200 nm | 16.3 % | **6.3 %** | −61 % |
| Relative error @ 400 nm | 14.9 % | **6.2 %** | −58 % |
| OPD residual @ 400 nm | 59.50 nm | **24.77 nm** | −58 % |

Both campaigns used the same held-out seeds, same 60 wavefronts per level, same
detector. The baseline was re-run with its original ROMs restored, so this is a
like-for-like comparison, not a comparison against a stale log.

## 2. What was wrong, and what fixed it

### 2.1 Reference centroids (fixed)

The shipped `slopes_ref_{x,y}.mem` did not correspond to this bench's flat
wavefront — 0.011 px RMS on undigitised images, appearing as a **constant
8.3 nm OPD bias on every frame**.

`regen_reference.py` regenerates it from the same bench, the same detector, the
same 5 ADU threshold and the same thresholded-CoG the hardware uses, quantised
to the RTL's actual format (Q18, signed 25-bit). Change from baseline:
0.0375 px RMS, 0.186 px max.

**Result: noiseless flat bias 8.349 → 0.000 nm.** Not reduced — eliminated,
which is what a self-consistent reference should do.

### 2.2 Reconstruction matrix (fixed)

Regenerating the reference did **not** move the mode-gain spread (0.774–1.076 →
0.779–1.083) or the ~15 % large-amplitude floor. So neither was a
reference-offset effect.

`regen_ematrix.py` measures the interaction matrix optically instead of assuming
it: each mode is probed at ±30 nm through the same bench and detector, and the
push-pull difference gives the slope response per metre. `E = pinv(M)`.

> This uses `pinv`, but of an **empirically measured optical response**, not of
> the matrix under test. Probe amplitudes and seeds are separate from every
> evaluation case.

Held-out comparison, both matrices expressed in metre-per-slope-pixel:

| Input RMS | Baseline E | Measured E | Change |
|---|---|---|---|
| 50 nm | 7.40 nm | 4.94 nm | **−33 %** |
| 100 nm | 13.74 nm | 6.56 nm | **−52 %** |
| 200 nm | 28.64 nm | 12.71 nm | **−56 %** |

Per-mode gain spread **0.523 → 0.000**; interaction condition number 6.4.

### 2.3 The output scale is now derived, not fitted

The scale and the matrix quantisation are coupled: `E_q = round(E/scale)` and
`metres = raw × scale`, so `scale` is a free parameter. The principled choice is
the one that uses the full signed 18-bit word without saturating:

```
scale = max|E| / (2^17 − 1) = 1.577616e-13 m/LSB = 2^-42.527
```

Derived from the matrix and the word width alone. Zero saturated words (the
first attempt, at the old empirical scale, saturated 64 of 1680 — the script
refuses to apply in that case). Output field use at 400 nm: **13.4 %** of the
25-bit field, so ample headroom.

### 2.4 A normalisation error I had wrong

The analytic tilt cross-check assumed edge-amplitude Zernike normalisation.
HCIPy's basis is **RMS-normalised**, so tip is `Z = 2(x/R)` and a coefficient
`a` gives slope `2a/R`, not `a/R`. My prediction was a factor of 2 low.

This surfaced in E7, where the measured spot shift was exactly 2× predicted.
Corrected in both `calibrate.py` and the E7 sweep. One detector pixel of spot
motion corresponds to **1562 nm** of tilt coefficient.

## 3. Corrected arithmetic discrepancy — an earlier claim was wrong

I previously reported fixed-point contributing "~0.2 nm, ≲2 % of total". **That
was wrong.** It came from a pooled ratio of two residuals, which can be nearly
equal while the underlying wavefronts differ.

Measured properly — `RMS(W_fixed − W_float)` computed **per frame on the
pupil**, no quadrature anywhere:

| Amplitude | Photons | float−true | fixed−true | **\|fixed−float\|** | p95 | max |
|---|---|---|---|---|---|---|
| 20 nm | 1e6 | 10.70 nm | 10.75 nm | **1.41 nm** | 1.78 | 1.94 |
| 20 nm | 3e6 | 4.05 nm | 4.46 nm | **1.24 nm** | 2.47 | 2.78 |
| 100 nm | 1e6 | 28.16 nm | 28.23 nm | **1.39 nm** | 1.93 | 2.06 |
| 100 nm | 3e6 | 7.23 nm | 7.20 nm | **1.47 nm** | 2.14 | 3.02 |
| 400 nm | 1e6 | 31.59 nm | 31.64 nm | **1.57 nm** | 1.93 | 2.03 |
| 400 nm | 3e6 | 24.01 nm | 24.07 nm | **1.21 nm** | 2.20 | 2.65 |

n=40 per cell. The discrepancy is **1.2–1.6 nm and essentially constant** across
amplitude and signal — the signature of a fixed quantisation floor.

**Reconciliation.** The earlier ~0.2 nm came from comparing fixed and float
*coefficients* on identical pixels; this measures the reconstructed *wavefront*
difference on the pupil. Different quantities. The wavefront figure is the one
that matters for an optical claim.

**Revised significance.** At the corrected calibration, total residual at 100 nm
/ 3e6 photons is 7.2 nm and arithmetic contributes 1.5 nm — roughly **20 %**,
not 2 %. Fixed point is now a material term precisely *because* the calibration
errors that used to dominate have been removed.

## 4. Remaining experiments

### E4 dynamic range

| Input RMS | Rel. error | Spots at window edge | Clipped px | 25-bit field use |
|---|---|---|---|---|
| 10 nm | 35.5 % | 0 | 0 | 0.2 % |
| 100 nm | 7.5 % | 0 | 0 | 2.2 % |
| 300 nm | **6.5 %** | 0 | 1 | 6.6 % |
| 1000 nm | 6.7 % | 0 | 1 | 22.9 % |
| 2000 nm | 29.2 % | 0.2 | 1 | 38.2 % |
| 4000 nm | 84.8 % | 0.4 | 0 | 57.2 % |

**Failure onset ≈ 2000 nm RMS, caused by spots leaving their lenslet window** —
not by arithmetic. No accumulator ever exceeded its bound, no dark lenslets, and
the 25-bit output field peaked at 57 % even at 4000 nm. Best accuracy
300–1000 nm. This is the tested envelope for this mode mixture, not a universal
limit.

### E6 unmodelled aberrations

100 nm of modelled content plus a varying fraction of modes **above** the
reconstructed basis, retained in the truth:

| Unmodelled | Full residual | Modelled-only residual | Recovered-mode error |
|---|---|---|---|
| 0 % | 6.73 nm | 6.73 nm | 2.14 nm |
| 25 % | 33.57 nm | 22.04 nm | 6.92 nm |
| 100 % | 131.28 nm | 83.90 nm | 26.31 nm |

Unmodelled content **aliases into the recovered modes**: with 100 nm of
unmodelled aberration, the error on modes that *are* reconstructed grows from
2.1 to 26.3 nm. The full residual exceeds the modelled-only residual throughout,
as it must.

### E7 subpixel sampling

21 positions across 0–0.4 px (the reachable range: a full pixel needs 1562 nm of
tilt, near the 2000 nm failure onset, so the spot never traverses a whole pixel
in normal operation).

| Quantity | Value |
|---|---|
| Centroid bias, mean | **−0.0035 px** |
| Peak-to-peak | **0.0357 px** |

The bias varies by ±0.018 px with sub-pixel spot position — roughly **5× the
fixed-point quantisation floor** (0.007 px). A real systematic, independent of
arithmetic.

### Optical sampling convergence

The final study recalibrates the flat reference and ±30 nm push-pull interaction
matrix independently at every numerical resolution before scoring six common
held-out wavefronts. At 4× pupil-edge supersampling the result is within the
frozen 5% / 0.5 nm tolerance of 16× (4.70%, 0.264 nm), but the production
256×256 propagation grid is not within tolerance of the finest tested 3× grid
(9.40%, 0.505 nm). The older two-point “4× is adequate” conclusion is therefore
withdrawn; see the final report and `convergence_*.csv`.

## 5. Detector clarifications

| Question | Answer |
|---|---|
| Clipped-pixel counts | **Per frame** internally; the sweep CSV **aggregates** over all realisations at a level |
| Flux across lenslets | 334–648 ADU at 100 nm/3e6 (ratio **1.9×**), median 624 — not uniform |
| Negative read-noise samples | Read noise is additive Gaussian and **can** go negative; clipped to 0 before the ADC, so there is a hard floor at 0 ADU |
| Detector offset | 0 ADU |
| Operating point origin | Scanned on a **flat** wavefront, then **corrected** using aberrated frames, which clip earlier |
| Zero clipping | An **observation over the tested cases**, not a guarantee — a sufficiently concentrated spot pattern can clip at any fixed gain |

## 6. RTL verification against the corrected ROMs

Both calibration ROMs changed, so the RTL was re-verified against the exact
files intended for hardware.

| Check | Result |
|---|---|
| Optical + directed subset | **35 cases, 350 coefficients, 0 mismatches** |
| Latency | **[9] cycles**, single value |
| Unknown valid values | **0** |

Simulation and the synthesis tree read the **same** `rtl/` directory, so they
cannot diverge.

**Scope.** Passing a finite set supports correctness on those tests. It is not a
proof of equivalence for all inputs.

## 7. Claim status

| Claim | Status |
|---|---|
| Flat-wavefront bias removed by reference regeneration | **VERIFIED** 8.349 → 0.000 nm |
| Measured E matrix halves reconstruction error | **VERIFIED** held-out, −33 to −56 % |
| Output scale derived, not fitted | **VERIFIED** max\|E\| → full 18-bit word |
| Fixed point adds 1.2–1.6 nm RMS on the pupil | **VERIFIED** paired, per frame |
| Degradation begins near 1500 nm in tested mixtures, with rising edge occupancy | **SUPPORTED FOR TESTED CASES** — contributor, not sole-cause proof |
| Full-period subpixel centroid bias | **VERIFIED** 0.01389 px RMS, 0.05804 px p-p over 41 positions |
| Production numerical grid converged | **NOT ESTABLISHED** — 9.40%, 0.505 nm vs finest tested propagation grid |
| RTL matches model on corrected ROMs | **VERIFIED FOR TEST SET** 350 coefficients |
| Latency 9 cycles | **VERIFIED** in simulation |
| Latency 60 ns | **ASSUMED** — 150 MHz target, no timing closure |
| Residual is a "10-mode basis fitting floor" | **WITHDRAWN** — it was matrix mismatch |
| Fixed point is ≲2 % of total error | **WITHDRAWN** — corrected to ~20 % |
| Resources, achieved clock | **PENDING** — no Vivado run |
| Experimental optical performance | **NOT ESTABLISHED** — simulation only |

## 8. Remaining work

1. **Vivado implementation** of the corrected `rtl/` — the only route to
   resource and timing numbers. The ROMs changed, so any prior build is stale.
2. **Regenerate `slopes_ref` from a noisy average** rather than one noiseless
   frame, which is what a real calibration procedure would do.
3. **Per-mode residual gain** — spread is down to 0.935–1.085 but not 1.0;
   worth checking whether the remainder is probe amplitude or nonlinearity.
4. **Numerical propagation convergence** — the production grid missed the
   frozen criterion; refine the production/calibration grid before making an
   unconditional optical-accuracy claim.
5. **Experimental data** — none compatible; see `EVALUATION_REPORT.md` §4.10.

## 9. Files

```
evaluation/optical/
    calibrate.py              freeze scale + analytic cross-check
    regen_reference.py        regenerate flat reference -> Q18 ROM
    regen_ematrix.py          measure interaction matrix, derive scale, write ROM
    campaign.py               E1, E2, E3, E5
    arithmetic_discrepancy.py paired fixed-vs-float on the pupil
    experiments_e4_e6_e7.py   E4, E6, E7, convergence
    rtl_optical_subset.py     RTL on optical frames
    make_optical_figures.py

evaluation/results/baseline_v1/   immutable pre-correction snapshot + MANIFEST.json
evaluation/results/optical/       corrected results, CSV + JSON
evaluation/results/figures/optical/  PDF + SVG + PNG
```
