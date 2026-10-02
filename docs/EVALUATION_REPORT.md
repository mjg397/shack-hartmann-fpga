# FPGA Shack-Hartmann Wavefront Reconstruction — Evaluation Report

> **HISTORICAL REPORT — not the current manuscript evidence.** The authoritative
> optical-forward-model report is
> [`evaluation/results/runs/forward_model_investigation_v2_20260928_final/REPORT.md`](../evaluation/results/runs/forward_model_investigation_v2_20260928_final/REPORT.md),
> with the underlying broad campaign preserved at
> [`manuscript_v1_20260928_final`](../evaluation/results/runs/manuscript_v1_20260928_final/REPORT.md).
> This older report predates strict ROM validation, the row-boundary streamer
> fix, the corrected photon/read-noise campaign, full-period subpixel testing,
> and the recalibrated propagation-grid convergence study. In particular, its
> remaining-ROM-path claim, ≲2% arithmetic statement, hardware-history
> statement, and numerical/optical tables must not be combined with the final
> run.

Complete experimental record for the Results and Discussion sections. Every
number is measured, reproducible, and traceable to a named file. **MEASURED
FACT** and **INTERPRETATION** are labelled separately throughout.

Generated 2026-09-15 · master seed `20260914` · reproduce with:

```bash
sim/venv/bin/python evaluation/run_all.py
```

Full campaign ≈ 5 minutes. `--quick` for a smoke test, `--skip-rtl` to skip
simulator steps. All output lands in `evaluation/results/`.

---

## Contents

1. [Critical caveats](#1-critical-caveats--read-before-quoting-anything)
2. [Design audit](#2-design-audit)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Claim / evidence matrix](#5-claim--evidence-matrix)
6. [Principal findings](#6-principal-findings)
7. [Limitations](#7-limitations)
8. [Recommended figures and tables](#8-recommended-figures-and-tables)
9. [File index](#9-file-index)

---

## 1. Critical caveats — read before quoting anything

Three constraints govern every number in this report.

**1.1 `rtl/` is the DUT, and it is bit-exact.** *(updated 2026-09-27.)*
The three defects below were ported into `rtl/` and verified. The
duplicate copy under `sim/rtl/` has since been **removed**:
the evaluation, the SystemVerilog verification environment and the synthesised
build now all read the same `rtl/` tree, so they cannot diverge.

A historical project assertion says the design was synthesised and run on a
Zynq device, but no reports, bitstream provenance, captured inputs, or measured
outputs are available here. The final evidence package therefore treats all
hardware execution, timing, resources, throughput in seconds, and power as
**UNVERIFIED**.

| # | Original defect | Status |
|---|---|---|
| 1 | `slopes_ref_{x,y}.mem` held Q4.23 values (48–78 M) loaded into `reg signed [24:0]`; all 208 non-zero entries overflowed and flipped sign | **fixed** — regenerated as Q18 (1.5–2.4 M) |
| 2 | `unsigned_mult` returned `mult_out[35:12]` — a **Q6** centroid against **Q4.23** reference data, a 2¹⁷ mismatch | **fixed** — returns `mult_out[23:0]`, full Q18 |
| 3 | `reciprocal_u16_q18` has 2 internal stages so `rI_out` trailed `sI` by 3 cycles while side-band signals trailed by 1 | **fixed** — side-band delayed to match (`RECIP_LATENCY = 2`) |

**ROM-path update (2026-09-28).** All authoritative RTL now uses bare ROM
filenames. The supported runner validates exact word counts, widths and legal
sign extension, stages the required bytes into an isolated working directory,
records source/staged SHA-256 hashes, and rejects missing, malformed, unknown,
timed-out, or incomplete outputs. Focused regressions pass from both the
repository root and an unrelated working directory.

**`rtl-2` (on `origin/main`) is NOT the corrected RTL.** It was last
touched 2026-06-26, predates the September consolidation, and still contains
defect 2 (`mult_out[35:12]`) and defect 3 (no `RECIP_LATENCY`). It also ships
**no** `e_matrix_*.mem`, `slopes_ref_*.mem` or `subaperture_bitmap.mem` — only
`reciprocal_u16_q18.mem` — so it cannot reconstruct at all on its own. It
appears to be the staging area where `intensity_reciprocal_q18.v` and
`reciprocal_u16_q18.v` were first introduced.

**1.2 150 MHz is a target, not a measured Fmax.** No post-implementation timing
report exists. Every nanosecond figure below inherits that assumption. Cycle
counts are measured and do not.

**1.3 All frames are HCIPy-simulated.** No experimental Shack-Hartmann data
exists in the repository, and a systematic audit (section 4.10) found that
public raw lenslet-spot imagery at this scale and documentation level
**essentially does not exist** — the community reduces SH frames to slopes or
coefficients before release. Do not describe any result here as experimental.

---

## 2. Design audit

Established from the RTL and `.mem` files, not from README claims.

### 2.1 Two generations ship side by side

| | Gen A (stale, not compiled) | Gen B (the DUT) |
|---|---|---|
| Top | `shwfs_vivado_top.v`, `shwfs_vivado_ila_top.v` | `shwfs_processing.v` |
| Accumulator | `TCoG_intensity_accumulator` | `Thresholding_CoG_intensity_accumulator` |
| Reciprocal | `intensity_reciprocal` (Q27 + Newton) | `intensity_reciprocal_q18` (Q18 LUT) |
| Zernike out | 270 bits (27/mode) | 250 bits (25/mode) |

`slope_calculation.v` and `ematrix_accumulator.v` are shared and already moved
to Gen B widths, which is what leaves Gen A broken: elaborating
`shwfs_vivado_top` prunes 9 high bits off `rec_intensity`, drives a 270-bit bus
from a 250-bit accumulator, and instantiates `streaming_emulator` without its
BRAM ports so every pixel is `X`.

### 2.2 Device and clock

| Item | Value | Source |
|---|---|---|
| FPGA device | **UNKNOWN** | no Vivado project or report exists |
| Target clock | 150 MHz | commit `a970f5f`; `rtl/timing.xdc` |
| Timing closure | **UNVERIFIED** | no post-implementation report |

### 2.3 Geometry

| Item | Value | Source |
|---|---|---|
| Frame | 256 × 256, 8-bit unsigned | `streaming_emulator.v` `HSIZE`/`VSIZE` |
| Blanking | HBLANK 4, VBLANK 152 | same |
| Lenslet grid | 16 × 16 = 256 tiles | `NUM_SUBAPETURES_SQRT` |
| Subaperture | 16 × 16 px | `NUM_PIXELS_SUBAPETURE_SQRT` |
| Valid subapertures | **168** | `NUM_SUBS`; popcount of `subaperture_bitmap.mem` |
| Zernike modes | **10** | `NUM_MODES` |

The pupil is circular with a central obscuration — bitmap rows 7 and 8 read
`0111111001111110`.

### 2.4 Fixed-point formats (traced, not inferred)

| Signal | Format | Width | Source |
|---|---|---|---|
| Pixel threshold | 5 ADU | — | `PIXEL_THRESHOLD` |
| Intensity `sI` | unsigned int | 16 | overflow assertion at 65280 |
| Moments `xI`,`yI` | unsigned int | 19 | overflow assertion at 489600 |
| Reciprocal out | Q18 | 19 | `intensity_reciprocal_q18.rI_out` |
| Reciprocal **as used** | Q18 | **18** | `rec_intensity` port — 1 bit pruned |
| Centroid | **Q18** | 24 | `unsigned_mult` → `mult_out[23:0]` |
| Slope | Q18 | **25 signed** | `raw_x_slope[24:0]`; the DSP48 operand limit |
| E matrix | **Q1.16** | 18 signed | `ematrix_accumulator.v` header |
| MAC product | — | 43 signed | same |
| Accumulator | — | 52 signed | same |
| Zernike output | — | **25 signed** | `acc[42:18]` |

**Output decode:** `metres = raw × 2⁻⁴³`, calibrated against the HCIPy golden
model by least squares (2⁻⁴²·⁹⁹⁷). The older `raw × 8/2²² × 1e-9` no longer
applies — the slope path moved from Q4.23 to Q18.

### 2.5 Algorithms

* **Centroid** — thresholded centre of gravity. Pixels below 5 ADU zeroed, then
  `sI`, `xI`, `yI` accumulate **exactly** (integer, no rounding) per tile.
* **Reciprocal** — a 65536-entry **Q18 lookup table, no Newton iteration**.
  `reciprocal_u16_q18.v` instantiates nothing. `sI == 0` returns `0x7FFFF`.
* **Slope** — `centroid − reference`, per-subaperture, gated by the pupil bitmap.
* **Reconstruction** — dense modal MAC, 168 subapertures × 10 modes.

### 2.6 Datapath

```
 image BRAM (bram_model.v, combinational read)
        |  32-bit words, 4 px/word
        v
 streaming_emulator                      registers data/fv/lv, 1 px/clock
        |  s_data[7:0], s_valid    (s_ready tied high - no back-pressure)
        v
 Thresholding_CoG_intensity_accumulator        [2 cycles]
        |  sI[15:0], xI[18:0], yI[18:0], subapetures_completed[7:0]
        v
 intensity_reciprocal_q18 -> reciprocal_u16_q18 [3 cycles]
        |  rI[18:0] Q18, xI/yI delayed to match
        v
 slope_calculation                             [2 cycles]
        |  centroid Q18[23:0] - ref Q18 -> slope 25 signed
        v
 ematrix_accumulator                           [2 cycles]
        |  168 x 10 MAC, 52-bit acc, acc[42:18]
        v
 m_data[249:0] = 10 x 25-bit signed, m_valid
```

---

## 3. Methodology

### 3.1 What was reused

| Asset | State | Reused |
|---|---|---|
| `sim/` Icarus flow | working | yes — all RTL simulation |
| `tb_zynq_top.v`, `tb_full_pipeline.v` | working | yes — streaming and capture |
| `simulation/*.v` (Quartus-era) | pre-Zynq DE1-SoC design, 27-bit decode | **removed 2026-09-27** — recoverable from `origin/main:rtl/` |
| `src/shwfs_utils.py` | working HCIPy model | yes |
| `src/fpga_gui_service.py` | PyQt-coupled; holds decode + dump logic | logic reused headlessly |
| `hcipy_image_rotating.mem` | 65536 px, 8-bit; identical to `sim/data/*.hex` | yes — demo frame |
| `e_matrix_{x,y}.mem` | 1680 values, Q1.16 | yes — reconstruction + forward model |
| `slopes_ref_{x,y}.mem` | 256 values, Q4.23 upstream | regenerated as Q18 |
| `subaperture_bitmap.mem` | 256-bit, popcount 168 | yes |
| `rtl/verification/test_*.py` | **print-only scaffolds, not cocotb tests** | no |
| cocotb | **not installed** | no |
| Vivado reports | **none exist** | — |

Testbench additions are **instrumentation only** — no RTL change: per-stage
valid-event capture, first/last pixel cycles, back-to-back run length, and a
`FINAL_TILE_PIXEL` parameter so latency is measured from the pixel *arriving*.

**cocotb is not used.** The repository's `rtl/verification/test_*.py`
files are print-only scaffolds, not cocotb tests, and cocotb is not installed.

### 3.2 A measurement bug that was found and fixed

`tb_full_pipeline.v` originally derived cycle numbers from an integer
`cycle_count` incremented with a **blocking** assignment in its own `always`
block while other blocks read it. Verilog does not order those blocks, so every
counter-derived timestamp could be — and was — off by one. All timestamps now
come from `$time`, which is identical in every block at a given edge.
`tb_latency_trace.v` is a second, independent testbench confirming the result.

### 3.3 Two-tier accuracy methodology

`generate_aberrated_image` rebuilds the whole HCIPy model per call (~40 s), so
10,000 end-to-end optical simulations would take ~4.6 days. Instead:

* **Tier 1 (10,000 cases)** — Monte Carlo over the pipeline's *arithmetic*.
  Frames synthesised from `pinv(E)` using the repository's own E matrix, so the
  forward model introduces no invented physics. Spot σ (0.75 px) and integrated
  intensity (812 ADU) are **measured** from the repository's own frame.
* **Tier 2 (validation)** — the real HCIPy frame gives 6.69e-3 px centroid RMSE;
  the synthetic population gives 7.04e-3 px. 5 % agreement, so Tier 1 is
  representative.

The floating-point reference runs the **identical algorithm** in float64, which
isolates quantisation from algorithmic error.

### 3.4 Reproducibility

`MASTER_SEED = 20260914` in `common/config.py`; each experiment offsets it.
`results/accuracy_params.json` and `noise_sweep_meta.json` record every
parameter of a run.

---

## 4. Results

> **Re-measured 2026-09-28 against the corrected calibration ROMs.** The
> reference and E matrix were regenerated (see OPTICAL_EVALUATION.md §2), so
> these arithmetic figures shifted slightly: Zernike RMSE 1.732e-09 → 1.608e-09 m,
> ablation floor 0.811 → 0.592 nm. Centroid RMSE is essentially unchanged
> (7.035e-03 → 7.031e-03 px), as expected — the reference affects slopes, not
> centroid arithmetic.
>
> **Sections 4.2–4.3 are superseded for optical accuracy.** They used synthetic
> frames built with `pinv(E)` and reconstructed with `E` — a circular forward
> model that flatters the result. They remain valid as measurements of
> *arithmetic* (fixed vs float on identical pixels). For reconstruction accuracy
> against independent optical inputs see **[OPTICAL_EVALUATION.md](OPTICAL_EVALUATION.md)**,
> which finds 15–50 % relative error rather than 5–11 %, and shows the
> fixed-point contribution is ≲2 % of the total.

### 4.1 RTL exact equivalence

**MEASURED FACT.** 168 cases — 150 randomised wavefronts plus 18 directed (zero
slope, near-zero slope, ± slopes, large in-tile offsets, each of the 10 pure
modes, dim spots, saturated spots, and an all-dark frame exercising the
`sI == 0` reciprocal path):

| Quantity | Value |
|---|---|
| Coefficients checked | **1,680** |
| Exact matches | **1,680** |
| Mismatches | **0** |
| Mismatch rate | **0.000e+00** |
| Slope/centroid values checked | 84,672 |
| Slope/centroid mismatches | **0** |
| Pixel-stream mismatches | 0 |

**INTERPRETATION.** The RTL implements the intended fixed-point algorithm
exactly. Because the software model reproduces RTL output bit-for-bit, all
subsequent precision analysis is performed in software with no loss of
fidelity, and this is a true equivalence test rather than a tolerance
comparison.

*Source: `results/rtl_equivalence.csv`, `rtl_equivalence_summary.csv`,
`rtl_equivalence.tex`.*

### 4.2 Numerical precision

**MEASURED FACT.** 10,000 Monte Carlo cases, fixed point vs float64 on the
identical algorithm:

| Quantity | RMSE | p95 | Max | Unit |
|---|---|---|---|---|
| Centroid | **7.031e-03** | 1.151e-02 | 1.259e-02 | px |
| Slope | 7.035e-03 | 1.151e-02 | 1.259e-02 | px |
| Zernike | **1.732e-09** | 4.190e-09 | 6.258e-09 | m |
| Wavefront equivalent | **2.931e-09** | 4.794e-09 | 5.244e-09 | m |

2.931e-09 m = **λ/239** at the model's 700 nm. Bias is ~0 for all quantities.

The 0.0067 px figure **is reproducible three independent ways**: the demo frame
gives 6.681e-03 px, the 10,000-case campaign 7.035e-03 px, and the noise sweep
yields 7.0e-03 px when extracted in quadrature from the high-flux floor.

**Per-mode RMSE (nm):**

| Mode | RMSE | Mode | RMSE |
|---|---|---|---|
| Tilt X | **3.790** | Coma X | 0.436 |
| Tilt Y | **3.794** | Coma Y | 0.438 |
| Defocus | 0.381 | Trefoil X | 0.372 |
| Astig 45 | 0.490 | Trefoil Y | 0.371 |
| Astig 0 | 0.412 | Sph. | 0.179 |

**INTERPRETATION.** Tilt carries ~10× the error of any other mode because tilt
couples most directly to centroid position, so reciprocal quantisation
concentrates there. Applications dominated by tip/tilt correction feel this
disproportionately.

*n = 10,000. Source: `results/accuracy_summary.csv`, `zernike_per_mode.csv`,
`accuracy_raw.csv`.*

### 4.3 Error source ablation

**MEASURED FACT.** 300 cases; each variant replaces exactly one quantisation
step with exact arithmetic. Software-only — the RTL is never modified.

| Variant | Centroid RMS (px) | Zernike RMSE (nm) |
|---|---|---|
| all fixed (shipping) | 0.009321 | 1.699 |
| **exact reciprocal** (upper bound) | **0.006114** | **0.592** |
| exact slope only | 0.009321 | 1.899 |
| exact output only | 0.009321 | 1.899 |
| E-matrix ±½ LSB dither | 0.009321 | 1.899 |
| all float | 0.006114 | 0.592 |

The thresholded CoG sums are exact integers in the RTL, so there is no
accumulator quantisation to ablate — itself a result.

**INTERPRETATION.** The reciprocal is the **only** significant fixed-point error
source. Slope truncation, output slicing and E-matrix quantisation contribute
nothing measurable, so any accuracy work should target the reciprocal and
nothing else.

**The exact-reciprocal row is a bound, not a proposal.** The reciprocal feeds
`unsigned_mult`'s **18-bit `b` port** — exactly the DSP48E1 B-port width. A
25-bit slope on the A port and an 18-bit reciprocal on B is what keeps the
centroid multiply inside a single DSP. A wider reciprocal (the Q27 + Newton
design this project previously used, 27-bit output) overflows that port by 9
bits and no longer maps to one DSP, which is why the design moved to Q18. The
Q27 files have been removed from the tree accordingly.

The 2.34× figure is therefore the ceiling available *if precision were free*,
not a change obtainable by widening the datapath. Closing any part of that gap
must happen **within 18 bits**. The measured loss comes from small `sI` values
(444–825 in the demo frame) leaving only 8–9 significant bits in `2^18/sI` — a
quantisation-strategy problem, not an output-width one.

*n = 300 × 6 variants. Source: `evaluation/results/error_ablation.csv`, `.tex`.*

### 4.4 Noise sensitivity

**MEASURED FACT.** Poisson photon noise, 9 flux levels × 300 realisations. Both
paths see the identical noisy frame; digitisation held at the repository's
measured ADU scale.

| Photons/subap | Float (px) | FPGA (px) | Penalty |
|---|---|---|---|
| 1e2 | 0.07536 | 0.07565 | 1.004× (+0.4 %) |
| 3e2 | 0.04366 | 0.04408 | 1.010× |
| 1e3 | 0.02477 | 0.02566 | 1.036× (+3.6 %) |
| 3e3 | 0.01540 | 0.01670 | 1.084× |
| 1e4 | 0.00991 | 0.01191 | 1.202× (+20.2 %) |
| 3e4 | 0.00776 | 0.01022 | 1.317× |
| 1e5 | 0.00665 | 0.00945 | 1.421× (+42.1 %) |
| 3e5 | 0.00633 | 0.00928 | 1.467× |
| 1e6 | 0.00617 | 0.00924 | 1.497× (+49.7 %) |

Crossover (10 % penalty): **≈3×10³ photons/subaperture**.

**INTERPRETATION.** In the photon-starved regime the FPGA is effectively free —
quantisation is buried under shot noise. Above ~10⁴ photons/subaperture the
fixed-point floor dominates and the sensor stops benefiting from additional
light. For a faint-guide-star application the current precision is adequate; for
a well-illuminated laboratory or laser-guide-star system it is the limiting
term. This is the most decision-relevant result in the evaluation.

*Photon-noise-only bound — the repository defines no read noise, background or
conversion gain, so a real detector sits above these curves. n = 300/level.
Source: `results/noise_sweep.csv`, `noise_sweep_raw.csv`, `noise_sweep_meta.json`.*

### 4.5 Dynamic range

**MEASURED FACT.** Aberration magnitude swept 1 nm – 1 µm RMS, 16 levels,
10 single modes + 15 multimode realisations each:

| Input RMS | Relative RMSE | Spots out of subaperture |
|---|---|---|
| 25 nm | 7.6 % | 0 |
| 100 nm | 1.9 % | 0 |
| 251 nm | 0.53 % | 0 |
| 398 nm | 0.31 % | 0 |
| 631 nm | 0.47 % | 0.6 |
| 1000 nm | 88.5 % | 8.3 |

**No accumulator overflow, no moment overflow, and no Zernike-field saturation
occurred anywhere in the swept range.**

**INTERPRETATION.** The useful range is bounded by **spot geometry, not
arithmetic**: accuracy collapses when spots migrate out of their subapertures at
roughly 631 nm RMS (≈0.9 λ), while the fixed-point ranges still have headroom.
At the small-signal end relative error rises simply because the ~1.9 nm error
floor becomes a large fraction of the signal. Best relative accuracy is
250–400 nm RMS. Widening any accumulator would not extend the operating range.

*n = 16 × (10 + 15). Source: `results/dynamic_range.csv`.*

### 4.6 Latency

**MEASURED FACT.** 50 randomised frames. Deterministic at every anchor —
min = max = mean, one unique value.

| Anchor | Cycles | ns @ 150 MHz |
|---|---|---|
| Streaming-emulator output bus (**recommended**) | **9** | **60.00** |
| Accumulator `s1` input register | 8 | 53.33 |
| Accumulator completion event | 7 | 46.67 |

**Stage breakdown** — traced through registered valid paths, not inferred from
combinational operators. The sum equals the independently measured total:

| Stage | Cycles | ns | Registered path |
|---|---|---|---|
| Final accumulation | 2 | 13.33 | `s1_valid <= valid`, then `subapetures_completed <= +1` |
| Centroid / reciprocal | 3 | 20.00 | `cd_pipe[0]`, `cd_pipe[1]`, `centroids_done_out` |
| Slope calculation | 2 | 13.33 | stage-1 registers, then stage-2 → `subap_valid` |
| Zernike reconstruction + output | 2 | 13.33 | `prod_valid_r <= sub_valid`, then acc → `done` |
| **Final pixel → output** | **9** | **60.00** | |

**INTERPRETATION.** Determinism is structural, not incidental: the accumulator
completes on a fixed pixel count, the reciprocal is a fixed-depth LUT, and the
MAC runs a fixed 168 × 10 schedule. There is no data-dependent control anywhere
in the tail.

**The anchor must be named with any published figure.** The three values differ
by where the pipeline is taken to begin, and published work varies in this
convention — comparisons are invalid unless the other work's anchor is known.
The recommended anchor includes the input capture register and is what a scope
trace at the FPGA pin could confirm.

*n = 50 frames. Source: `results/latency_per_frame.csv`, `latency_summary.csv`,
`pipeline_breakdown.csv`.*

### 4.7 Throughput

**MEASURED FACT.**

| Metric | Value |
|---|---|
| Accepted pixels per clock | 1 |
| Initiation interval | 1 cycle |
| Longest unbroken accepted run | 256 px = full line |
| Back-pressure | none — `s_ready` tied high (`shwfs_processing.v:33`) |
| Sustained pixel rate | 1.5×10⁸ pixel/s |
| Frame input time (256×256) | 66,813 cycles = 445.4 µs |
| Frame period incl. blanking | 66,965 cycles = 446.4 µs |
| Max **processing** frame rate | 2,240 Hz |
| Acquisition ÷ processing tail | **7,424×** |

Blanking gaps (239/frame) are the emulator's HBLANK between lines, **not**
pipeline stalls.

**INTERPRETATION.** The pipeline consumes pixels at line rate without stalling,
so reconstruction genuinely overlaps acquisition; the 7,424× ratio is the
quantitative form of that architectural claim. The 2,240 Hz figure is a
*processing capability* given a source able to supply one pixel per clock — it
is **not** a claim that any detector runs at that rate.

*Source: `results/throughput.csv`, `.tex`.*

### 4.8 Demonstration frame

**MEASURED FACT.** The repository's own HCIPy frame
(`hcipy_image_rotating.mem`, byte-identical to `sim/data/*.hex`), gated
by `subaperture_bitmap.mem`:

| Property | Value |
|---|---|
| Valid subapertures | 168 (all with flux) |
| Flux per subaperture | 444 – 825 ADU, median 812 |
| Peak pixel | 95 – 255 ADU, median 201 |
| Spot σ | 0.683 – 0.874 px, median **0.750** |
| FWHM | 1.765 px |
| Slope range | −0.267 – +0.310 px, RMS 0.0967 |
| Centroid fixed vs float | **6.681e-03 px** RMS |
| Zernike fixed vs float | 1.051e-09 m |
| RTL == bit-accurate model | **True** |
| Latency | 9 cycles |

**INTERPRETATION.** Spot σ ≈ 0.75 px is **undersampled** (below the 1 px
Nyquist-equivalent). Plain CoG on undersampled spots carries systematic
pixelation error independent of fixed-point choices — not characterised here,
and possibly more significant for absolute accuracy than quantisation.

*Source: `results/demo_frame_characterisation.csv`, `demo_frame_summary.json`.*

### 4.9 FPGA implementation — NO RESULTS

No Vivado report exists in the repository, on any git branch, or anywhere on
this machine. Searches covered `*.rpt`, `*.xpr`, `*.dcp`, `*utilization*`,
`*timing_summary*`, and any file mentioning `shwfs` or `ematrix_accumulator`.

Reports found under `beamline_compression_2`, `highspeed-uart-fpga` and
`fpga-network-data-engine` belong to **unrelated projects** and were
deliberately not used.

Consequently **no LUT/FF/BRAM/DSP count, no WNS/TNS, and no verified Fmax can be
published**, and the claim that the design closes timing at 150 MHz is
unsupported. The parser is implemented and will populate
`results/implementation_results.{csv,tex}` as soon as `impl_utilization.rpt` and
`impl_timing_summary.rpt` are placed in `evaluation/implementation/`.

Generate them with:

```tcl
report_utilization    -file impl_utilization.rpt
report_timing_summary -file impl_timing_summary.rpt
```

### 4.10 External / real-data validation — NOT PERFORMED

**MEASURED FACT.** A systematic audit of public Shack-Hartmann data was carried
out across astronomical AO testbeds, ophthalmic aberrometry, laser/X-ray
metrology and general open-data repositories. Full records in
`docs/research/notes/`.

**No public dataset satisfies the requirements of this pipeline.** Not one
release combines real recorded raw lenslet-spot frames, documented sensor
geometry, and independent (non-commanded) wavefront ground truth at a
16 × 16-lenslet / 256 × 256-detector scale.

Candidate status, as verified:

| Candidate | Real hardware? | Raw SH frames? | Ground truth | Access | Verdict |
|---|---|---|---|---|---|
| Ramirez-Quintero 2025, Sci Rep ([DOI 10.1038/s41598-024-80615-8](https://www.nature.com/articles/s41598-024-80615-8)) | yes — 1280×1024 CMOS, 300 µm lenslet pitch, PLUTO-NIR SLM | yes | **SLM-commanded setpoint, not independently measured** | **request-only** | best-documented chain; ground truth is accuracy-against-setpoint, blind to SLM nonlinearity |
| Durham "PPPPP" ([researchdata.durham.ac.uk](https://researchdata.durham.ac.uk/files/r2np193923h)) | yes — DARC RTC, ALPAO DM | **no — SH reduced to slopes** before deposit; the raw frames belong to the lenslet-free PPPP sensor | — | public, CC BY-SA 4.0 | real and openly licensed, but not SH raw imagery |
| Zenodo 14964320 | — | **no — zero image files** in the 25-file manifest | — | public | **refuted**; `.dat` arrays are training tensors, not imagery |
| `kaccie14/shwfs` (Roorda Lab, UC Berkeley) | yes — custom ophthalmic WFS | **yes — 3 TIFFs + background** | none | **public, no request** | only unconditionally public real SH imagery found; undocumented geometry, no licence, 3 frames |
| ESO SPHERE / SAXO ([archive form](http://archive.eso.org/wdb/wdb/eso/sphere/form)) | yes — 40×40 lenslet, EMCCD | **unverified** whether the public form exposes AO-loop WFS pixels vs science frames | — | public archive query | largest-scale candidate; needs a live query, not a metadata read |
| MAVIS DNU (Zenodo 21107489) | yes — ALPAO DM + 4D interferometer | **no — record is a 6.9 MB PDF**, no pixel files | would be interferometric | public | ideal provenance, no data deposited |
| `congzhou235-prog/iccp-adaptive-optics-2026` | yes — documented build, CAD/STEP | `dataset/` with Zernike-mode folders; **real-vs-simulated provenance unverified** | folder labels | public | worth one manifest check |

**INTERPRETATION.** The obstacle is structural, not incidental: across every
domain searched, the community treats raw lenslet-spot frames as a disposable
intermediate. Papers release reconstructed wavefronts, Zernike coefficients or
slopes; the pixel data is not retained, is reduced before deposit (Durham), or
sits behind "available upon reasonable request" boilerplate that functions as a
soft embargo (Antioquia). Public raw SH imagery at the scale and documentation
level a hardware-verification paper needs **essentially does not exist**.

**The one action available without author outreach** is `kaccie14/shwfs`. Its
three frames could drive the *floating-point* centroiding path as a
qualitative demonstration that the algorithm locates real spots — nothing more.
It cannot support a reconstruction-accuracy claim: geometry is undocumented, no
licence is stated, there is no ground truth, and the frames will not match this
pupil. Per the compatibility gate below, RTL reconstruction on them is refused.

**Why external data cannot simply be reshaped.** Three artefacts are tied to
this exact geometry: `e_matrix_{x,y}.mem` (calibrated for this pupil and these
168 subapertures), `slopes_ref_{x,y}.mem`, and `subaperture_bitmap.mem`.
Resizing an external image to 256 × 256 would place spots where the E matrix was
never calibrated; reconstruction would return numbers that are meaningless.
`evaluation/real_data/load_external.py` implements a gate that **refuses**
mismatched geometry rather than resampling — verified: a 128×128 frame and a
16-bit overrange frame are both rejected with a specific reason.

**What a usable dataset must provide:** raw frames with stated bit depth; a
reference/flat-wavefront frame; lenslet pitch, focal length and lenslet→pixel
mapping; detector pixel pitch; and **independent** wavefront ground truth
(interferometer, known DM shape, or calibrated phase plate). 16 × 16 lenslets
over 256 × 256 allows direct use; any other geometry requires regenerating the
calibration via `src/ematrix_genv4.py`, at which point RTL parameters may also
change — making it a different DUT.

**Partial data still has value.** Raw frames of matching geometry *without*
ground truth would support a fixed-point-vs-float comparison on real spot
profiles — label that numerical validation, not accuracy. Different geometry
supports only a float-only centroiding sanity check, which is **not** RTL
validation.

**Recommended route**, per the audit: parallel author requests to the Antioquia
group (raw NFI/AEI frames at native resolution, SLM setpoint files, bit depth,
preprocessing pipeline, dark/flat/pupil-mask files, repeat exposures, explicit
data licence) and to the Roorda Lab (lenslet pitch and focal length, pixel
pitch, native resolution, bit depth, subject identification, whether
`background.tif` is dark or flat, any ground truth, licence, more frames), plus
a live query of the ESO SPHERE archive for SAXO WFS pixel products. Even a
successful request will likely need supplementing with an in-house bench
capture, since no shortlisted candidate supplies both full calibration metadata
and interferometrically-verified ground truth at this scale.

---

## 5. Claim / evidence matrix

Status: **SUPPORTED** — measured, reproducible. **CONDITIONAL** — measured but
depends on an unverified assumption. **UNSUPPORTED** — no evidence; do not
publish.

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | RTL is bit-exact with the software reference | 1,680 coefficients, 0 mismatches, rate 0.0e+00 | **SUPPORTED** |
| 2 | Final-pixel-to-result latency is 60.0 ns | 9 cycles; two independent testbenches; stages sum 2+3+2+2 | **CONDITIONAL** — ns depends on unverified 150 MHz |
| 3 | Latency is deterministic | 50 frames, min = max = 9 cycles | **SUPPORTED** |
| 4 | One pixel per clock, no back-pressure | longest run = 256 = line width; `s_ready` tied high | **SUPPORTED** |
| 5 | Fixed point adds 7.0e-3 px centroid RMS | 10,000 cases; demo frame 6.68e-3 px independently | **SUPPORTED** |
| 6 | That equals 2.93 nm RMS wavefront (λ/239) | conversion uses only `src/shwfs_utils.py` constants | **SUPPORTED** |
| 7 | Zernike RMSE is 1.73e-09 m | 10,000 cases; per-mode 0.18–3.79 nm | **SUPPORTED** |
| 8 | Tilt modes dominate the error | 3.79 nm vs 0.18–0.49 nm, ~10× | **SUPPORTED** |
| 9 | The reciprocal is the dominant error source | ablation: only variant that changes anything | **SUPPORTED** |
| 10 | Reciprocal precision is capped by the DSP48 B port | `unsigned_mult` port `b` is 18 bits = DSP48E1 B width; a 27-bit reciprocal overflows it by 9 bits and leaves a single DSP | **SUPPORTED** |
| 11 | Quantisation negligible below ~10³ photons/subap | penalty 1.004× at 1e2, 1.036× at 1e3 | **SUPPORTED** |
| 12 | Quantisation dominates above ~10⁴ photons/subap | 10 % above ~3e3; 1.50× at 1e6 | **SUPPORTED** |
| 13 | Range bounded by geometry, not arithmetic | spots exit at ~631 nm; zero overflow 1 nm–1 µm | **SUPPORTED** |
| 14 | Acquisition is 7,424× the processing tail | 66,813 vs 9 cycles | **SUPPORTED** |
| 15 | Design operates at 150 MHz | **none** | **UNSUPPORTED** |
| 16 | Resource utilisation | **none** | **UNSUPPORTED** |
| 17 | Validated on experimental data | **none**; systematic audit of 7 candidates found no compatible public dataset (§4.10) | **UNSUPPORTED** |
| 18 | "World's lowest latency" / superiority | no comparison performed; anchors differ | **UNSUPPORTED** |
| 19 | Meets the project's own precision spec | spec < 1e-4 px; measured 6.7e-3 px | **CONTRADICTED** |
| 20 | README "120 ns latency" | that is 6 cycles @ 50 MHz for Gen A | **SUPERSEDED** |
| 21 | `rtl/` matches the evaluated DUT | ported 2026-09-27; bit-exact on all 10 coefficients, 9-cycle latency reproduced | **SUPPORTED** |
| 22 | `rtl/` is ready to synthesise as-is | **no** — `$readmemh` path inconsistency leaves one ROM unloadable depending on working directory (§1.1) | **CONTRADICTED** |

### 5.1 Evidence contradicting current project documentation

1. **The precision specification is not met.** `docs/RECIPROCAL_PRECISION_PROOF.md`
   requires < 1e-4 px; the shipping Q18 LUT measures 6.7e-3 px — **67× short**.
   That target was set for the Q27+Newton reciprocal, which no longer fits the
   design: its 27-bit output overflows the 18-bit DSP48 B port. The spec and the
   current datapath are therefore incompatible, and the spec is the stale half.
2. **The 120 ns README figure does not describe this design** — it is the
   Gen A / DE1-SoC number (6 cycles @ 50 MHz).
3. **`rtl/` is now fixed** (as of 2026-09-27) and bit-exact with the
   evaluated DUT — but one `$readmemh` path inconsistency remains, and it fails
   silently: the pixel stream and latency look healthy while every coefficient
   is `X`.
4. **"World's lowest latency" is unsupported** — no literature comparison, and
   latency anchors differ between publications.

---

## 6. Principal findings

1. **RTL is bit-exact** with the reference model: 1,680 coefficients, 0
   mismatches.
2. **Tail latency is 9 cycles / 60.0 ns and perfectly deterministic**, with a
   verified stage decomposition of 2+3+2+2.
3. **Acquisition exceeds the processing tail by 7,424×**, confirming the
   streaming architecture.
4. **Fixed point costs 7.0e-3 px centroid RMS** (2.93 nm, λ/239), reproducible
   three independent ways.
5. **The reciprocal is the sole significant error source.** An exact reciprocal
   would cut Zernike RMSE **2.34×**, but that is a ceiling, not an available
   change — the 18-bit DSP48 B port caps any replacement.
6. **Quantisation is negligible below ~10³ photons/subaperture and dominant
   above ~10⁴** — well matched to photon-starved operation.
7. **The operating range is limited by spot geometry (~631 nm RMS), not
   arithmetic** — no overflow anywhere in the swept range.
8. **Tilt modes carry ~10× the error** of all other modes.

---

## 7. Limitations

1. **`rtl/` now carries the fixes** and is verified bit-exact with the
   evaluated DUT. One `$readmemh` path inconsistency remains (§1.1).
2. **150 MHz is unverified.** No timing report; all ns figures are conditional.
3. **No resource data.** Cannot be filled from simulation.
4. **Simulation only.** No experimental frames; no hardware-in-the-loop timing.
   The audit in section 4.10 establishes this is a property of the field's
   release practices, not of the search — public raw SH imagery is scarce by
   default. Author outreach, not further searching, is the route forward.
5. **Photon-noise-only** noise model — no read noise, background or gain, since
   the repository defines none. Real detectors perform worse.
6. **Synthetic Monte Carlo frames**, validated against the one real HCIPy frame
   (6.69e-3 vs 7.04e-3 px) but not against a large HCIPy population.
7. **Spots are undersampled** (σ ≈ 0.75 px); systematic pixelation error is not
   characterised.
8. **The project's own precision spec is not met** (67× short).
9. **No literature comparison.** `prior_work_comparison_TEMPLATE.tex` is
   deliberately empty.

---

## 8. Recommended figures and tables

**Figures** (all PDF + PNG in `results/figures/`):

| Figure | Why |
|---|---|
| `noise_sweep` | the strongest result — when quantisation matters |
| `per_mode_rmse` | the tilt finding |
| `latency_timing` | stages + the 7,424× acquisition ratio |
| `coefficient_scatter` | 100,000 points on y = x |
| `dynamic_range` | optional, if space allows |

**Tables:**

| Table | File |
|---|---|
| 1 — Numerical accuracy | `accuracy_table.tex` |
| 2 — Latency / throughput | `latency_table.tex` + `throughput_table.tex` |
| 3 — RTL equivalence | `rtl_equivalence.tex` |
| 4 — Per-mode accuracy | `zernike_accuracy_table.tex` |
| Discussion — ablation | `error_ablation.tex` |
| Prior work | `prior_work_comparison_TEMPLATE.tex` — **empty by design** |

Fill the prior-work table only from papers you have read, and record each one's
latency anchor in its own column.

---

## 9. File index

```
evaluation/
    EVALUATION_REPORT.md         this document
    README.md                    short orientation
    run_all.py                   one command runs everything

    common/config.py             paths, geometry, fixed-point formats
    common/bitaccurate.py        bit-exact software model of the RTL
    common/framegen.py           synthetic frames + pinv(E) forward model
    common/rtl_runner.py         drives the Icarus simulation

    rtl_equivalence/regression.py       STEP 1
    numerical_accuracy/monte_carlo.py   STEP 2
    numerical_accuracy/characterise_demo.py  STEP 11
    error_ablation/ablation.py          STEP 3
    noise_sweep/noise_sweep.py          STEP 4
    dynamic_range/dynamic_range.py      STEP 5
    latency/latency_sweep.py            STEP 6
    latency/pipeline_breakdown.py       STEP 7
    throughput/throughput.py            STEP 8
    implementation/parse_reports.py     STEP 9  (+ REQUIRED_REPORTS.md)
    real_data/load_external.py          STEP 10
    plotting/make_figures.py            STEP 12
    plotting/make_tables.py             STEP 13

    results/
        accuracy_summary.csv  accuracy_raw.csv  zernike_per_mode.csv
        accuracy_params.json
        rtl_equivalence.csv  rtl_equivalence_summary.csv/.json
        error_ablation.csv
        noise_sweep.csv  noise_sweep_raw.csv  noise_sweep_meta.json
        dynamic_range.csv
        latency_per_frame.csv  latency_summary.csv/.json
        pipeline_breakdown.csv
        throughput.csv
        demo_frame_characterisation.csv  demo_frame_summary.json
        implementation_results.csv       (placeholder — no reports)
        *.tex                            LaTeX tables
        figures/*.pdf  *.png
```

Supporting RTL and simulation live in `sim/` (the DUT
copy, testbenches, and Icarus flow) — kept outside this folder so the RTL is not
forked.

The external-data audit behind section 4.10 is recorded in
`docs/research/notes/` — four files covering AO
testbeds, ophthalmic aberrometry, general open repositories, and the three
originally-named leads.
