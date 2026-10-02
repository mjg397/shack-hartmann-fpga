# Zynq SHWFS full-pipeline simulation

> The single-frame demo notes below are retained for bring-up provenance. The
> current frozen multi-case evidence, strict runner, corrected calibration, and
> derived (not fitted) output scale are documented in
> [`evaluation/results/runs/manuscript_v1_20260928_final/REPORT.md`](../evaluation/results/runs/manuscript_v1_20260928_final/REPORT.md).

Streams a real 256×256 image through the entire Zynq pipeline in Icarus Verilog and
compares the resulting Zernike coefficients against a live HCIPy golden model.

## Running

```bash
python3 -m venv venv
venv/bin/pip install hcipy numpy scipy matplotlib
./run.sh --ref        # --ref regenerates the HCIPy reference (~1 min)
./run.sh              # subsequent runs reuse the reference
```

Artifacts land in `build/`; waveforms in `build/waves/zynq_pipeline.vcd`.

## Structure

```
bram_model.v          image BRAM (combinational read -- see note below)
tb_zynq_top.v         sim-only Zynq top: BRAM + emulator + shwfs_processing
tb_full_pipeline.v    clock/reset, pixel-integrity check, coefficient capture
pack_image.py         65536 8-bit pixels -> 16384 32-bit words
gen_reference.py      HCIPy golden model + image equality gate
compare_hcipy.py      decode + comparison table
diagnose.py           fixed-point format diagnostics
../rtl/                authoritative DUT sources (no private simulation copy)
```

There is no `sim/rtl/` copy. The simulator compiles the authoritative `rtl/`
tree. All `$readmemh` calls use bare filenames; the supported evaluation runner
validates the exact ROM shapes and stages byte-identical files into its isolated
working directory. This makes ROM resolution independent of the caller's CWD.

## What is simulated

Only the current-generation chain:

```
bram_model -> streaming_emulator -> shwfs_processing
                                      ├─ Thresholding_CoG_intensity_accumulator
                                      ├─ intensity_reciprocal_q18
                                      ├─ slope_calculation
                                      └─ ematrix_accumulator
```

`shwfs_vivado_top.v` and `shwfs_vivado_ila_top.v` are **excluded**. They are a
previous generation: 27-bit/Q27 with `TCoG_intensity_accumulator` and
`intensity_reciprocal`, wiring a 27-bit reciprocal into `slope_calculation`'s
18-bit port (9 high bits pruned) and a 270-bit `zernike_out` onto a 250-bit
accumulator. They also instantiate `streaming_emulator` without its BRAM ports,
so every streamed pixel would be `X`.

## Verification built into the run

- **Pixel integrity** — the testbench checksums every streamed byte against the
  source `.mem`. This caught a one-word BRAM lag during bring-up that was
  invisible in the image's large zero regions.
- **Image equality gate** — `gen_reference.py` regenerates the detector image
  from HCIPy and diffs it against the streamed `.mem`. Currently **MATCH**, so
  the golden model and the RTL see the same scene and the comparison is valid.

## Current status

**Working.** Correlation against HCIPy **0.9926**, RMS error **3.56e-09 m**
(11.0% of the 3.25e-08 m signal). The five dominant modes agree to 3.8–11.7%,
comparable to the 5–15% in the historical `output_dumps/zernike_comparison.txt`.
Coma/Trefoil/Sph. show large *relative* error only because HCIPy puts them near
zero; their absolute error stays within ~1e-9 m.

Everything stays within **25 bits**, matching the DSP48 operand width.

## Defects found and fixed

All three were in the RTL, not the testbench. `diagnose.py` re-verifies each one
so a future change cannot silently reintroduce them.

1. **Reference-slope ROM overflowed.** `slope_calculation.v` declares
   `reg signed [24:0] slopes_ref_{x,y}_mem`, but `slopes_ref_*.mem` held Q4.23
   values of 48–78 million. All 208 non-zero entries truncated — 63,753,412
   (7.600 px) became −3,355,452, sign flipped.
   **Fix:** `make_slopes_ref_q18.py` regenerates them as Q18 (1.5–2.4 M) from
   the untouched Q4.23 originals in `rtl/`, preserving pixel values
   exactly (5.7273–9.2727 px) with round-to-nearest.

2. **Binary-point mismatch.** `unsigned_mult` returned `mult_out[35:12]`, a
   **Q6** centroid, against **Q4.23** reference data — confirmed empirically at
   exactly 2⁻¹⁷ against the host model.
   **Fix:** return `mult_out[23:0]`, keeping the full **Q18** centroid. The
   product is bounded by the 16-pixel subaperture (16·2¹⁸ = 2²²), so the upper
   bits were structurally zero and nothing is lost. Centroid, reference and
   slope now share Q18 and all fit 25 bits.

3. **Reciprocal pipeline misalignment.** `reciprocal_u16_q18` has two internal
   stages, so `rI_out` trailed `sI` by 3 cycles while `xI/yI/centroids_done`
   trailed by only 1. Every centroid was multiplied by a *stale* reciprocal —
   magnitudes stayed plausible but each value belonged to the wrong subaperture.
   Centroid correlation against the host model was **+0.044**.
   **Fix:** delay the side-band signals by the matching 2 cycles in
   `intensity_reciprocal_q18.v`. Correlation went to **+0.995**, max relative
   error 0.15%.

4. **First pixel of each row addressed word zero.** The horizontal-blanking
   transition computed the next-row BRAM address with an 8-bit left-shift
   expression, truncating it to zero. Dark detector borders masked the defect;
   a noisy border pixel exposed it through the strict pixel-order check.
   **Fix:** form the packed word address explicitly as the 14-bit
   `{next_row, 6'b0}` value. A regression now drives a nonzero first pixel on
   every row.

### The 25-bit Zernike headroom question

This resolved itself. It was a *consequence* of defect 2, not an independent
problem: with slopes at Q23 the accumulator needed values like 33,052,052 for
Tilt X, overflowing the ±16,777,216 field. At Q18 the same physical coefficient
is 32× smaller, so the largest raw value is now ~512,000 — about 3% of the
available range. `acc[42:18]` is fine as it stands, and no DSP48 operand exceeds
25 bits.

## Decode scale

The historical demo used a fitted 2⁻⁴³ decode. The frozen optical calibration
now derives the scale from the unquantized optically measured matrix by mapping
`max|E|` to the full signed 18-bit word:

```
slope (Q18) x E matrix (Q1.16) -> Q34,  sliced acc[42:18] -> Q16
meters = raw * 1.5776159884040182e-13
```

No evaluation outcome is fitted to obtain this value. The exact frozen artifact
and method are `derived_scale.json` and `ematrix_regen.json` in the final run.

## Still open

- `intensity_reciprocal_q18.rI_out` is 19 bits into an 18-bit `rec_intensity`
  port. Narrow impact — only reachable when `sI <= 1` — and the testbench counts
  occurrences (currently 0).
- **`bram_model` reads combinationally on purpose.** `streaming_emulator`
  registers the address for pixel N+1 during cycle N, so data must be ready the
  same cycle. Real Zynq BRAM is synchronous, so the emulator's
  `STATE_LINE_PRIME` ("wait one cycle") primes only the first word of each line;
  on silicon the same one-word lag would appear. The RTL needs an extra address
  pipeline stage to be correct on hardware.
- These fixes are in the authoritative `rtl/` tree used by simulation. That
  does not establish synthesis, timing closure, or board behavior.
