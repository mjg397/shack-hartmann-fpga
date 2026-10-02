# Data Request Wish-Lists

These are precise lists of what to ask for, per `DATASET_DECISION.md` items
3-4. **These are not drafted emails and none of this has been sent.** Per
standing instructions, contacting authors requires your explicit go-ahead;
this file exists so that when you're ready, turning it into an actual email
is a five-minute task rather than a re-research task.

## 1. Ramirez-Quintero et al. 2025 (Universidad de Antioquia, Optics and Photonics group)

Paper: "Experimental wavefront sensing techniques based on deep learning
models using a Hartmann-Shack sensor for visual optics applications,"
*Scientific Reports* (2025), DOI 10.1038/s41598-024-80615-8. Data Availability
statement: "available from the authors upon reasonable request."

Ask for:
1. Raw camera frames for both the Noise-Free Images (NFI) and Artificial-Eye
   Images (AEI) sets, at native **1280×1024** resolution — not only the
   network-preprocessed crops used in the paper.
2. The corresponding SLM Zernike-coefficient setpoint file for every frame
   (the commanded ground truth).
3. Camera bit depth (not stated in the published text).
4. The exact preprocessing pipeline (crop/resize/normalization) used to reach
   the paper's neural-network input size — needed to confirm whether/how it
   relates to a 256×256-class geometry.
5. Any dark, flat, or background reference frames, plus a
   lenslet-registration map or pupil mask (none mentioned in the Methods).
6. Whether repeated exposures were captured per aberration setting (needed to
   separate sensor noise from model error).
7. Explicit reuse/license terms for the image data itself (the article's Open
   Access status covers text/figures, not necessarily the underlying images).

Context to include when this becomes a real message: state the intended use
(hardware verification of an FPGA SH reconstruction pipeline, 16×16 lenslets,
256×256-class detector, 10-mode Zernike reconstruction), that this is
academic/non-commercial, and that a small representative subset (a few
hundred frames spanning flat + single-mode + mixed-mode + repeats) would
already be very useful — not necessarily their full corpus.

## 2. Roorda Lab (`kaccie14/shwfs` GitHub demo maintainer)

Repository: https://github.com/kaccie14/shwfs — 3 raw TIFF frames + 1
background frame, no stated geometry, no license, no ground truth.

Ask for:
1. Full sensor geometry: lenslet pitch, lenslet focal length, camera pixel
   size/pitch, native resolution, bit depth.
2. Subject identification for each of the three sample frames (human eye vs.
   model/artificial eye vs. bench test target).
3. Clarification of whether `background.tif` is a true dark frame (no
   illumination) or a flat/reference frame (illuminated, no spots).
4. Any available aberration ground truth (Zernike coefficients or
   equivalent) for the three frames.
5. Explicit license/reuse permission (no LICENSE file currently exists in the
   repo).
6. Ideally, an expanded set of frames spanning multiple distinct aberration
   states, since the current three read as a data-reduction-pipeline demo
   rather than a validation corpus.

Note: this is a small ask (a GitHub maintainer, not necessarily the PI) — a
lightweight issue/message asking about license terms and geometry may get a
faster answer than a formal data request, and is a reasonable first move
before escalating to the lab directly.

## 3. MAVIS DNU / University of Padova group (stretch goal)

Zenodo record: "Laboratory tests of the MAVIS DNU Shack Hartmann Wavefront
Sensor," https://zenodo.org/records/21107489 — currently a 6.9 MB
conference-proceedings PDF only, no data files attached.

Ask for:
1. Whether the underlying bench pixel data (raw SH spot frames) from the
   reported ALPAO 97-15 DM + 4D Phasecam 4030 interferometer tests was ever
   retained/deposited, and if so, where.
2. If retained: geometry (lenslet count/pitch, detector resolution/bit
   depth), calibration frames, and the paired commanded-vs-interferometric
   aberration values for each test case — this pairing is the most valuable
   single property found across the entire search, since it would give
   independently-verified (not just commanded) ground truth.

## 4. Nazim Bharmal / Durham (long shot, low priority)

Dataset: "PPPP beam profiles and measured Shack Hartmann slopes from the
PPPPP experiment," DOI 10.15128/r2np193923h, CC BY-SA 4.0. The deposited SH
content is already-reduced slopes, not raw frames.

Ask for (only if items 1-3 above don't pan out): whether raw, pre-reduction
Shack-Hartmann camera frames from this experiment still exist and were simply
not part of the public deposit.
