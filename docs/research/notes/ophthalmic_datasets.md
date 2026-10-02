# Public Raw Shack-Hartmann Spot Image Datasets — Ophthalmic / Vision-Science Aberrometry

Scope: candidate datasets of REAL, physically-recorded SH lenslet-array spot images (human eye, model/artificial eye, or test target) from ophthalmic/vision-science aberrometers, suitable as optical-bench-substitute validation data for a 16x16-lenslet, 256x256-pixel-class SH centroiding/reconstruction pipeline.

## Question 1-8 combined per candidate (as found)

### Candidate A: kaccie14/shwfs (GitHub) — Roorda Lab (UC Berkeley) custom-built WFS raw spot images

**Q1 — Public download / working link:** YES, directly verified. Repository: https://github.com/kaccie14/shwfs. File listing (verified via direct fetch of the repo root) contains, among MATLAB analysis scripts (`Arizona.m`, `GWE.m`, `LensSystem.m`, `Roorda.m`, `minboundcircle.m`, `eyeModelTest.mlx`, `misc.mlx`, `roordaTest.mlx`, `strehl.mlx`, `README.md`):
- `R25M0004g.tif`
- `R25M0005g.tif`
- `R25M0006g.tif`
- `background.tif`

These are actual raw-format `.tif` image files sitting in the repo (not derived Zernike/wavefront data), directly downloadable from the GitHub file listing/raw links. This is the strongest candidate found in this research pass.

**Q2 — Physically recorded on real instrument:** Stated in the README (quoted from fetch): "All images are from Roordalab's custom-built WFS" — i.e., a real, physical Shack-Hartmann wavefront sensor built in Austin Roorda's vision-science lab at UC Berkeley (a lab well known for adaptive-optics ophthalmoscopy / ocular wavefront sensing research). NOT confirmed whether the specific 3 `R25M000Xg.tif` frames are from a human eye, a model/artificial eye, or a bench test target — the README does not specify the subject for these particular files, and this was not independently resolved from the repo's other analysis scripts within this pass (the `Arizona.m` / `eyeModelTest.mlx` files reference the "Arizona Eye Model," suggesting the analysis pipeline is built around a model-eye/human-eye ocular aberration context, but that is inferential, not a direct statement about the source of the three TIFFs).

**Q3 — Sensor/lenslet geometry:** UNCONFIRMED / not stated in the README as fetched. No resolution, bit depth, lenslet count, pixel pitch, or focal length was found in the README text retrieved. Actual TIFF headers were not inspected (per task constraint not to download large files); this would need to be checked by opening the TIFF metadata directly to get resolution/bit depth.

**Q4 — Reference/calibration data:** PARTIALLY YES — `background.tif` is present in the repo, which is very likely a background/dark reference frame given its filename, but the README does not explicitly document its use or confirm it is a dark frame vs. a flat/reference frame. No explicit lenslet map or pupil mask file was identified in the listing.

**Q5 — Label / ground truth provenance:** UNCONFIRMED. No stated Zernike/aberration ground truth or ophthalmic subject metadata accompanies the three sample TIFFs in what was retrieved. This looks like a small demo/sample dataset for a data-reduction pipeline rather than a full labeled dataset.

**Q6 — License/reuse terms:** No LICENSE file is visible in the repository listing (directly verified via file-tree fetch). Reuse terms are therefore unstated/ambiguous — this should be flagged for the user/report as "no explicit license; reuse permission unclear," typical of many small academic GitHub repos.

**Q7 — Repeated exposures for same setting:** UNCONFIRMED from README; three numbered frames (`R25M0004g`, `0005g`, `0006g`) suggest a short sequence, possibly repeated exposures of the same or a slowly-varying eye/target, but this is inferential from the sequential naming, not a documented fact.

**Q8 — File count/size:** Small: repo contains 3 sample spot-image TIFFs + 1 background TIFF + ~9 MATLAB/script files. Actual byte sizes were not inspected (task constraint: do not download; metadata such as size-in-bytes was not surfaced by the fetch tool). This is clearly a small demonstration dataset, not a large corpus.

**What was directly verified vs. unconfirmed:** Directly verified — the repo exists, is publicly reachable, and its file tree includes the four TIFF files and the "Roordalab's custom-built WFS" attribution in the README. Unconfirmed — sensor specs (resolution/bit depth/lenslet count/pitch/focal length), the specific subject (eye vs. model eye vs. target) of the three sample frames, presence of a lenslet/pupil map, ground-truth aberration values, license, and exact file sizes.

---

## Key Question 1: Is there an actual public download with raw camera frames (not just processed coefficients)?

### Takeaway
Only one directly-confirmed candidate was found with actual downloadable raw `.tif` SH spot-image files in this pass — the `kaccie14/shwfs` GitHub repo, attributed to the Roorda Lab's custom WFS at UC Berkeley. All other ophthalmic-SH papers surfaced either explicitly used simulated data, stated data is available only "upon reasonable request," or gave no data-availability statement at all.

### Cited Findings
- The GitHub repo `kaccie14/shwfs` contains raw `.tif` spot images (`R25M0004g.tif`, `R25M0005g.tif`, `R25M0006g.tif`) plus a `background.tif`, alongside MATLAB scripts for spot sorting, centroiding, and wavefront estimation, with the README stating "All images are from Roordalab's custom-built WFS" — [GitHub - kaccie14/shwfs](https://github.com/kaccie14/shwfs)
- "Large-Dynamic-Range Ocular Aberration Measurement Based on Deep Learning with a Shack–Hartmann Wavefront Sensor" states its data availability only as "Data are contained within the article," and its Hartmannograms are synthetic/simulated (generated via Fourier-optics simulation from statistical eye models), not raw camera frames from a real instrument — [PMC11086325](https://pmc.ncbi.nlm.nih.gov/articles/PMC11086325/)
- A Biomedical Optics Express paper on multi-layer SH wavefront sensing states "data underlying the results presented in this paper are available from the authors upon reasonable request" (request-based, not a direct public download) — cited in search results referencing Optica-journal data-availability boilerplate; primary paper is "Multi-layer Shack-Hartmann wavefront sensing in the point source regime," Akondi & Dubra, [BOE 12(1):409](https://opg.optica.org/boe/abstract.cfm?uri=boe-12-1-409)
- The "Educational Shack–Hartmann wavefront sensor" article (American Journal of Physics, Hassani & Khajehhassanirabori, 2025, DOI 10.1119/5.0230503) could not be accessed for its data-availability/supplementary-material section — the publisher page returned HTTP 403 Forbidden on fetch — [AIP Publishing page](https://pubs.aip.org/aapt/ajp/article/93/3/264/3335805/Educational-Shack-Hartmann-wavefront-sensor)
- No Zenodo, Figshare, OSF, or PhysioNet-style repository with real (non-simulated) raw ophthalmic SH lenslet spot images was found via direct search; the one Zenodo dataset surfaced (DOI 10.5281/zenodo.21227985, from a paper on a "virtual multi-eccentric Hartmann–Shack aberrometer") is explicitly described as synthetic/simulated MATLAB-generated data, not real camera frames — [Sensors 26(14):4419](https://doi.org/10.3390/s26144419)

### Inferences
- Public, freely-downloadable raw SH ophthalmic spot-image datasets appear to be genuinely scarce; most groups either keep raw frames private, offer them only on request, or publish only derived Zernike/wavefront-map products.
- The `kaccie14/shwfs` repo, while real and directly verified, is a small demo dataset (3 sample frames + 1 background) attached to a data-reduction/teaching codebase rather than a curated, documented research dataset — it may still be usable as a substitute validation sample if its geometry can be reverse-engineered from the TIFF metadata and the MATLAB scripts (`Arizona.m`, `Roorda.m`, `LensSystem.m`), which likely hard-code the lenslet pitch/focal length/pixel size used by the Roorda lab instrument.

### Gaps
- No license was found for the `kaccie14/shwfs` repo, so reuse terms for those specific TIFFs are unconfirmed and should be treated as "ask the author" before any redistribution.
- Could not determine whether the three sample TIFFs are from a human eye, an artificial/model eye, or a bench test target — this is important for the user's stated inclusion criteria and needs direct inspection of the TIFFs/scripts or contact with the repo author (GitHub handle `kaccie14`) to resolve.
- Did not exhaustively search Figshare's and OSF's own internal search interfaces directly (only via general web search), since those platforms' search results are not always well-indexed by general web search — a follow-up pass hitting figshare.com and osf.io search endpoints directly could surface additional candidates not found here.
- The AIP-hosted "Educational Shack–Hartmann wavefront sensor" paper could not be inspected (403 Forbidden) — this paper explicitly provides "computational codes... to analyze the obtained data" per its abstract and describes a low-cost sensor built from spare parts, so it is a plausible candidate for a test-target-based raw-data release, but this remains unconfirmed and should be re-checked (e.g., via an institutional/no-paywall mirror, or arXiv preprint if one exists) in a follow-up.
- Low-cost/DIY open-source aberrometer hardware projects (the Raspberry-Pi-based low-cost SHWFS mentioned in search results, and the ~$20K-under telescope-oriented sensor) were surfaced only in summarized search snippets, not directly fetched/verified for actual raw-data releases — these are unconfirmed leads, not verified candidates.

## Key Question 2: Physically recorded on a real instrument (human eye, model/artificial eye, or test target)?

### Takeaway
Only the `kaccie14/shwfs` repo was confirmed to contain images from a real physical instrument (Roorda Lab's "custom-built WFS"); every other ophthalmic SH dataset/paper surfaced in this pass was either explicitly simulated or did not release raw images at all, so their real-vs-simulated status for released data is moot.

### Cited Findings
- README text for `kaccie14/shwfs`: "All images are from Roordalab's custom-built WFS," directly identifying a real physical sensor built at Austin Roorda's UC Berkeley vision-science lab — [GitHub - kaccie14/shwfs](https://github.com/kaccie14/shwfs)
- The PMC11086325 deep-learning ocular SH paper explicitly generated its Hartmannograms via "Fourier optics simulations based on statistical models derived from 332 normal and 344 diseased eyes" — i.e., simulated, not physically recorded — [PMC11086325](https://pmc.ncbi.nlm.nih.gov/articles/PMC11086325/)
- The Zenodo dataset tied to the "virtual multi-eccentric Hartmann–Shack aberrometer" paper is described as synthetic datasets and MATLAB scripts used to generate simulated multi-eccentric images, not physical camera captures — [Sensors 26(14):4419 / Zenodo DOI 10.5281/zenodo.21227985](https://doi.org/10.3390/s26144419)

### Inferences
- None of the confirmed candidates in this pass involve a human eye with documented, released raw frames; the one real-instrument candidate (`kaccie14/shwfs`) has an unresolved subject type (eye vs. model eye vs. target).

### Gaps
- Subject type (human eye / model eye / test target) for the `kaccie14/shwfs` sample TIFFs is unresolved — flagged above and needs direct follow-up.

## Key Question 3: Sensor/lenslet geometry (resolution, bit depth, lenslet count, pixel pitch, focal length)

### Takeaway
No sensor geometry details were confirmed for the one real candidate found (`kaccie14/shwfs`); the only concretely-stated SH geometry uncovered in this pass belongs to a simulated (not real) dataset.

### Cited Findings
- PMC11086325 (simulated data) states: 20×20 lenslet array, 300 μm lenslet size, 6.5217 μm pixel pitch, downsampled Hartmannogram resolutions of 200×200 and 224×224 from an original 920×920, 6 mm pupil diameter, 840 nm wavelength — [PMC11086325](https://pmc.ncbi.nlm.nih.gov/articles/PMC11086325/) — NOTE: this is simulated data and does not qualify as a real-instrument candidate per the task's constraints, included here only as a reference point for typical ocular-SH geometry.
- `kaccie14/shwfs` README does not state resolution, bit depth, lenslet count, pixel pitch, or focal length for the Roorda-lab WFS — [GitHub - kaccie14/shwfs](https://github.com/kaccie14/shwfs)

### Gaps
- Actual TIFF header metadata (image dimensions, bit depth) for the `R25M000Xg.tif` files was not inspected, per the task constraint against downloading large files; a follow-up should open just the TIFF headers (a few KB read, not a full download) to get resolution/bit depth, and check the `LensSystem.m` / `Roorda.m` scripts for hard-coded lenslet pitch/focal-length constants.

## Key Question 4: Reference/calibration data included (flat/reference frames, dark frames, lenslet map, pupil mask)?

### Takeaway
The `kaccie14/shwfs` repo includes one `background.tif` file, plausibly a dark/background reference frame, but this is inferred from the filename only — no explicit lenslet map or pupil mask file was found, and no other candidate repository/dataset with real images was confirmed to include calibration data.

### Cited Findings
- File listing of `kaccie14/shwfs` includes `background.tif` alongside the three spot-image TIFFs — [GitHub - kaccie14/shwfs](https://github.com/kaccie14/shwfs)

### Gaps
- Whether `background.tif` is a dark frame (no illumination) or a flat/reference frame (uniform illumination, no spots) was not confirmed — the README does not explain its purpose. No lenslet-position map or pupil-mask file was identified in the repo's file tree during this pass.

## Key Question 5: Label/ground-truth provenance (nominally commanded vs. independently measured)?

### Takeaway
No ground-truth/aberration-label information was found associated with any real-instrument raw-image candidate in this pass.

### Gaps
- No aberration ground-truth (commanded SLM/phase-plate setpoints or independently measured Zernike values) was found associated with the `kaccie14/shwfs` sample images; this dataset appears to be released purely as sample raw frames for demonstrating a processing pipeline, not as a labeled validation/calibration set. This is a significant gap relative to the task's objective of validating a reconstruction pipeline against known ground truth.

## Key Question 6: License/reuse terms

### Takeaway
The one real-instrument candidate found (`kaccie14/shwfs`) has no visible license file, so reuse terms are unstated and should be treated as unresolved/ask-the-author.

### Cited Findings
- Direct file-tree fetch of `github.com/kaccie14/shwfs` shows no LICENSE file among the repo contents — [GitHub - kaccie14/shwfs](https://github.com/kaccie14/shwfs)

### Gaps
- No license terms confirmed for any real-instrument candidate dataset in this pass.

## Key Question 7: Repeated exposures for the same aberration/eye setting?

### Takeaway
Unconfirmed for all candidates; the sequential filenames in `kaccie14/shwfs` (`R25M0004g`, `0005g`, `0006g`) suggest a short burst/sequence but this was not documented as "repeated exposures of the same setting" anywhere retrieved.

### Gaps
- Whether the three sample frames represent repeated exposures of an identical aberration/eye state, or three different states/timepoints, is unresolved without either reading the MATLAB analysis scripts in detail or contacting the repo author.

## Key Question 8: Rough file count/size

### Takeaway
The only confirmed real-instrument candidate is small: 3 spot-image TIFFs + 1 background TIFF + roughly 9 supporting MATLAB/script files in a single GitHub repo — clearly a demo-scale sample set, not a large corpus, though exact byte sizes were not obtained (task constraint against downloading).

### Gaps
- Exact file sizes (bytes/MB) and image pixel dimensions for the TIFFs are unconfirmed; a lightweight follow-up (reading just file size metadata from the GitHub API or a HEAD request, without downloading full content) could resolve this without violating the "no large downloads" constraint.

## Overall summary / recommendation for report writer

Across this research pass, the search space for **real, physically-recorded, publicly-downloadable raw Shack-Hartmann ophthalmic spot images** turned out to be very sparse. The single directly-verified candidate is:

- **`kaccie14/shwfs`** (https://github.com/kaccie14/shwfs) — 3 raw `.tif` SH spot-image frames + 1 background/reference TIFF, explicitly attributed to "Roordalab's custom-built WFS" (Austin Roorda's UC Berkeley vision-science lab, a real ophthalmic/adaptive-optics research group). No license, no stated sensor geometry, no confirmed subject type (eye/model-eye/target), and no ground-truth aberration labels were found — all flagged as open follow-up items above.

Every other lead investigated in this pass (PMC11086325's deep-learning ocular SH paper, the Zenodo "virtual multi-eccentric Hartmann-Shack aberrometer" dataset, the BOE multi-layer SH paper, various patent/USPTO documents, and ResearchGate figure pages) either explicitly used **simulated** data, gated real data behind an **"available upon reasonable request"** policy with no public link, or provided **no raw images at all** — only figures, derived coefficients, or methodology descriptions. The "Educational Shack-Hartmann wavefront sensor" AJP paper (Hassani & Khajehhassanirabori, 2025, DOI 10.1119/5.0230503) is a plausible additional lead — it is described as a build-it-yourself sensor with "computational codes... to analyze the obtained data" — but its publisher page returned a 403 error on fetch and its data-availability status could not be confirmed in this pass; it merits direct follow-up (e.g., checking for an arXiv preprint, an author's personal/institutional page, or a Zenodo/GitHub companion repo).

No datasets were found that combine: (a) a real ophthalmic-class instrument, (b) publicly downloadable raw camera frames, (c) documented sensor geometry, (d) calibration frames, and (e) ground-truth aberration labels — the ideal profile the user described. The `kaccie14/shwfs` repo satisfies (a) and partially (c)/(d) is inferential; it does not satisfy (e). This suggests that, absent further leads, a substitute optical-bench validation dataset meeting the full spec (16x16-lenslet, 256x256-pixel-class, 10-mode Zernike ground truth) may not currently exist in the public domain for ophthalmic SH sensors specifically, and the search may need to broaden to astronomical/AO-community SH datasets (which are more often released with lab test-target data, per the VLT/VISIR raw-data reference found in search results) if ophthalmic provenance is not a hard requirement.
