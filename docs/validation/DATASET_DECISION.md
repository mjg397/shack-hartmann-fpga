# Dataset Decision

Source: full research report at
`docs/research/Real Shack Hartmann sensor datasets.md`, backed by four independent
research passes in `docs/research/notes/`.
See `docs/validation/DATASET_CANDIDATES.csv` for the per-candidate detail this decision is
based on.

## Headline finding

**No publicly downloadable dataset currently satisfies the spec** (real,
physically-recorded raw SH lenslet spot images + documented geometry +
calibration frames + independently-verified ground truth, at anything close to
16×16/256×256 scale). This is a genuine negative result from a real search,
not a gap in the search. The field routinely reduces raw SH frames to slopes
or coefficients before any public release, or gates them behind "available on
request."

## Decision: pursue in this order

1. **`kaccie14/shwfs` (Roorda Lab GitHub demo) — use now, for free, as a smoke
   test only.** Already public, real instrument, zero request needed. It is
   3 frames + 1 background frame with no stated geometry, no license, no
   ground truth — **not sufficient for any accuracy claim**, but sufficient to
   sanity-check that the ingestion/compatibility-gate code in
   `evaluation/real_data/load_external.py` correctly parses a real camera
   frame format and correctly *refuses* it for full validation (its geometry
   almost certainly won't match the 168-subaperture E-matrix). Do this
   immediately; it costs nothing and exercises code we already have.

2. **Live-query the ESO SPHERE archive before requesting anything from
   anyone.** This is the single highest-value unresolved question from the
   research pass: does the public archive form
   (`archive.eso.org/wdb/wdb/eso/sphere/form`) expose SAXO's internal WFS
   pixel frames, or only science-camera output? If yes, this is by far the
   largest-scale, best-documented real SH sensor found (40×40 lenslets,
   on-sky since 2014, standard ESO calibration policy) and would leapfrog
   every other candidate. This needs a live, hands-on query — not another
   metadata read — and does not require contacting anyone.

3. **Request from Ramirez-Quintero's group (Univ. of Antioquia) — highest
   documentation quality, but ground truth is SLM-commanded, not
   interferometric.** Pursue in parallel with (2), since it needs a human
   to send a request (see `DATA_REQUEST_DRAFTS.md`). Note going in: even a
   full grant only gives commanded-setpoint ground truth, not independently
   verified wavefronts — plan the paper's claims accordingly (algorithm
   agreement against a real optical bench, not absolute wavefront accuracy
   against verified truth).

4. **Request from the MAVIS DNU / Univ. of Padova group as a stretch goal.**
   Conceptually the best ground-truth design found anywhere in the search —
   simultaneous DM-commanded *and* interferometrically-measured aberrations —
   but the public Zenodo record is a PDF only, so this requires asking
   whether the underlying bench data was ever deposited anywhere. Lower
   probability of yielding usable data than (3), but higher value if it does.

5. **Do not pursue further:** Zenodo 14964320 (definitively no images, fully
   verified by file manifest), and Durham "PPPPP" as a raw-image source (its
   SH content is slopes-only, not raw frames — it could still be asked about
   as a long-shot "do unreduced frames exist" question, but do not budget
   real effort here).

6. **Resolve before pursuing:** `congzhou235-prog/iccp-adaptive-optics-2026`
   — open `manifest.csv` and the README directly first. If the images turn
   out to be real captures (not simulated training data mislabeled by Zernike
   mode), this jumps up the list since it's already public and documents real
   hardware. This is a 10-minute inspection task, not a request — do it before
   spending any outreach effort on items 2-4.

## What this means for the paper, right now

Given the structural scarcity found, do not block the rest of the validation
program on real data arriving. Per `VALIDATION_STATUS.md`, everything that
can be done with simulated data is already done
(`docs/EVALUATION_REPORT.md`). The realistic framing for a submission
in the near term is: rigorous simulated-data validation (already complete) +
whatever real-instrument evidence materializes from items 1-4 above,
explicitly labeled by provenance per the evidence-labeling convention in
`IMPLEMENTATION_AUDIT.md` and the original validation program's §1. Do not
present request-pending or unresolved-provenance candidates as validated data
in any draft.
