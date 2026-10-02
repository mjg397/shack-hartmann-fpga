# General Open-Data Repositories for Real Shack-Hartmann Raw Spot Images (Non-Astronomy/AO-Testbed, Non-Ophthalmic)

Scope note: this covers Zenodo, Figshare, OSF, GitHub, and lab/university repositories for **conventional lenslet-array** Shack-Hartmann (SH) raw camera-frame releases in domains other than astronomy/AO-testbeds and ophthalmic aberrometry (laser/beam metrology, industrial/freeform optics metrology, X-ray/synchrotron metrology, open-hardware/DIY SH builds, general SH instrumentation papers, and ML-for-SH-centroiding training-data releases). Searches covered Zenodo, Figshare, OSF, GitHub code/topic search, arXiv, MDPI, and SPIE/Optica-adjacent literature. Overall finding: **no candidate was confirmed to be both (a) a conventional lenslet-array SH sensor and (b) an unambiguous public download of raw, physically-recorded camera frames** with full metadata; several near-misses were found and are detailed below, each with what was directly verified vs. not.

## Key Question 1: Is there an actual public download with a working link to raw camera frames (not just processed slopes/coefficients/figures)?

### Takeaway
No fully-confirmed "conventional lenslet-array SH raw frames, non-astronomy/non-ophthalmic" dataset was found with a verified working download of raw camera images. The closest candidates are (1) a GitHub repo (`kaccie14/shwfs`) that does bundle real `.tif` SH camera frames but originates from an ophthalmic-adaptive-optics lab (Roorda Lab), so it likely overlaps with the ophthalmic track rather than being a clean "other domain" hit; and (2) a Zenodo dataset tied to a laser-beam wavefront-sensor paper that, on inspection, uses a DMD-based amplitude-modulation sensor rather than a lenslet array, so it fails the "conventional lenslet-array only" constraint despite surfacing in lenslet/SH searches.

### Cited Findings
- `kaccie14/shwfs` (GitHub) — repository description and file listing confirm sample raw Shack-Hartmann camera images in TIFF format directly in the repo root: `R25M0004g.tif`, `R25M0005g.tif`, `R25M0006g.tif`, and a `background.tif` (dark/background frame); repo text states "All images are from Roordalab's custom-built WFS" — [https://github.com/kaccie14/shwfs](https://github.com/kaccie14/shwfs)
- Zenodo record 18696645, "Wavefront Sensor for Laser Beams Based on Reweighted Amplitude Flow Algorithm," is the article PDF only (`applsci-16-01942.pdf`, 1.1 MB) — no raw images in this record; sensor described uses a Digital Micromirror Device (DMD) as a programmable amplitude modulator, not a lenslet array — [https://zenodo.org/records/18696645](https://zenodo.org/records/18696645)
- A companion Zenodo record 18692566, "Dataset for article called 'Wavefront Sensor for Laser Beams Based on Reweighted Amplitude Flow Algorithm'," contains a single `Dataset.zip` (57.6 MB) described as "supplementary data related to the article" plus a text file describing file relationships — file-level contents (raw frames vs. processed) could not be determined from the landing page alone — [https://zenodo.org/records/18692566](https://zenodo.org/records/18692566)
- `congzhou235-prog/iccp-adaptive-optics-2026` (GitHub), an ICCP 2026 mini-course from PKU's Computational Scientific Imaging Lab titled "Build a Shack-Hartmann sensor and learn to measure wavefront aberrations," includes hardware assembly guides, CAD (STEP) files, and a Python tutorial package (`sh_tutorial_prog/`) containing a `calibration_pics/` folder and a `dataset/` folder with subfolders `first_10_zernike/`, `last_10_zernike/`, `mixed_zernike/`, and a `manifest.csv` — the subfolder naming (organized by which Zernike modes are represented) suggests this may be a curated/generated training set rather than a raw exposure dump, and this could not be confirmed either way (real captured frames vs. simulated/rendered) from the directory listing — [https://github.com/congzhou235-prog/iccp-adaptive-optics-2026](https://github.com/congzhou235-prog/iccp-adaptive-optics-2026)
- Educational instrumentation paper collection "Educational Shack-Hartmann wavefront sensor" on AIP Publishing's Figshare (Khajehhassanirabori & Hassani, 2025) was located via search but the landing page returned HTTP 403 Forbidden on direct fetch, so file contents could not be verified — [https://aip.figshare.com/collections/Educational_Shack-Hartmann_wavefront_sensor/7548900](https://aip.figshare.com/collections/Educational_Shack-Hartmann_wavefront_sensor/7548900)
- "Adaptive Shack-Hartmann wavefront sensor accommodating large wavefront variations" (Aftab, Choi, Liang, Kim; Optics Express 2018) — a freeform/general-optics-metrology-relevant SH instrumentation paper (Liang and Kim's group works on freeform optics metrology) — no associated Zenodo/Figshare data-deposit page could be located via search; only the paper abstract/PubMed/ResearchGate pages were found, with no data-availability link — [https://pubmed.ncbi.nlm.nih.gov/30650864/](https://pubmed.ncbi.nlm.nih.gov/30650864/)
- Github topic page `wavefront-sensing` was inspected; of the SH-tagged repos found (`yuuutosasa/ShackHartmannAnalysis`, `bah2026-ps9`, `jacotay7/makewfs`, `jacopoantonello/mshwfs`, `SergioBonaqueGonzalez/Shack-Hartmann-Simulator`), only `jacopoantonello/mshwfs` was previously known to bundle any SH `.mat` frame (`examples/data/sh_flat.mat`, `img.mat`), and `makewfs` explicitly simulates ("hyper realistic") frames rather than capturing real ones — none of these confirmed to be real-bench data outside astronomy contexts — [https://github.com/topics/wavefront-sensing](https://github.com/topics/wavefront-sensing), [https://github.com/jacopoantonello/mshwfs](https://github.com/jacopoantonello/mshwfs)
- `yuuutosasa/ShackHartmannAnalysis` demo directory produces output images (`moments_result.png`, `NCC_result.png`, `PC_result.png`) via `example.py`, which the repo text implies are example-generated rather than confirmed to originate from real captured raw frames — MIT licensed, incorporates code from Timothy Gebhard's `hswfs` (also MIT) — [https://github.com/yuuutosasa/ShackHartmannAnalysis](https://github.com/yuuutosasa/ShackHartmannAnalysis)
- MDPI Electronics paper "FPGA Implementation of Shack–Hartmann Wavefront Sensing Using Stream-Based Center of Gravity Method for Centroid Estimation" (2023, CC-BY) was located as directly relevant instrumentation work, but no data-availability/supplementary-dataset link was found in search results; the MDPI landing page itself (mdpi.com/2079-9292/12/7/1714) was not directly fetched to check its Data Availability Statement section — flagged as a gap — [https://www.mdpi.com/2079-9292/12/7/1714](https://www.mdpi.com/2079-9292/12/7/1714)
- X-ray Shack-Hartmann metrology paper "Shack–Hartmann wavefront sensors based on 2D refractive lens arrays and super-resolution multi-contrast X-ray imaging" (J. Synchrotron Rad. 2020, Karlsruhe Institute of Technology) is a genuine non-astronomy, non-ophthalmic conventional-lenslet-array SH application (hard X-ray beamline metrology using 2D refractive lens arrays as the lenslet equivalent); no public raw-image dataset repository (Zenodo/Figshare) was found associated with it in search results — [https://pmc.ncbi.nlm.nih.gov/articles/PMC7206553/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206553/)
- arXiv 2310.00862, "Shack-Hartmann wavefront sensing: A new approach to time-resolved measurement of stress intensity during dynamic fracture of small brittle specimens" (published in Mechanics of Materials, DOI 10.1016/j.mechmat.2024.105010) — a genuine mechanical-metrology (non-astronomy, non-ophthalmic) application of conventional SH sensing to dynamic fracture testing; the abstract page and (partially, due to size) the PDF were checked and contain no mention of a released dataset, GitHub repo, or data-repository link, and no sensor hardware specifics (lenslet pitch, camera resolution) were found in the retrievable excerpt — [https://arxiv.org/abs/2310.00862](https://arxiv.org/abs/2310.00862)
- arXiv 2607.25281, "Real-Time Megapixel Kilohertz Neuromorphic Shack-Hartmann Wavefront Sensor" — described in search-result summaries as combining a lenslet-based SHWFS with neuromorphic/event-based sensing for static optical metrology and dynamic turbulence measurement (throughput, dynamic-range, and centroid-error figures given); no confirmation of a released raw-frame dataset was found in the search snippets, and the full text was not fetched — flagged as a gap — [https://arxiv.org/abs/2607.25281](https://arxiv.org/abs/2607.25281)

### Inferences
- Genuine "other-domain" SH raw-data releases (laser/beam metrology, freeform/industrial optics, X-ray metrology, fracture mechanics) are exceptionally rare in general-purpose repositories; papers in these domains describe hardware and results but very rarely deposit raw camera frames, unlike some astronomy/AO groups that have begun depositing lab-test data on Zenodo (e.g., MAVIS DNU SHWFS lab tests, noted below but out of this track's scope).
- Several search hits that appeared promising because they matched "Shack-Hartmann" + "lenslet"/"wavefront sensor" text actually turned out on inspection to use non-lenslet wavefront-sensing modalities (DMD amplitude modulation, Fourier-plane holographic filters, pyramid-like sensors), underscoring that keyword search alone over-recalls non-conventional-SH methods; each such case had to be individually excluded per the task's lenslet-array-only constraint.

### Gaps
- MDPI Electronics 12(7):1714 (FPGA-based SH centroiding) Data Availability Statement was not directly checked (search only) — worth a direct fetch of the MDPI page itself.
- `congzhou235-prog/iccp-adaptive-optics-2026` dataset provenance (real captured images vs. rendered/simulated Zernike-labeled training data) is unresolved; would need to open `manifest.csv` or the README inside `sh_tutorial_prog/` directly.
- AIP Figshare "Educational Shack-Hartmann wavefront sensor" collection could not be accessed (403 Forbidden) — unresolved whether it contains raw images.
- Zenodo Dataset.zip 18692566 (laser-beam WFS paper) contents unverified at file level, though the parent paper's sensor is DMD-based, not lenslet-array, making it likely out of scope regardless.
- No SPIE Digital Library or Optica (OSA) journal pages were directly fetched (paywalled); any data-availability statements in SPIE proceedings on freeform-optics SH metrology remain unchecked.

## Key Question 2: Physically recorded on a real optical bench (not simulated/rendered)?

### Takeaway
Only `kaccie14/shwfs`'s bundled TIFFs were explicitly and directly confirmed as real-bench captures ("custom-built WFS"); everything else is either explicitly simulated, ambiguous, or unverified.

### Cited Findings
- `kaccie14/shwfs`: TIFF images explicitly described in the repo as coming from "Roordalab's custom-built WFS" (a real, physically built wavefront sensor) — [https://github.com/kaccie14/shwfs](https://github.com/kaccie14/shwfs)
- Zenodo 18696645 (laser-beam WFS paper record) states the authors "experimentally demonstrate wavefront reconstruction at 650 nm and at 2116 nm" and validated results "against a commercial lateral shearing interferometer," and integrated the sensor into "a closed-loop adaptive optics system using a deformable mirror" — i.e., real bench work, but with a DMD-based sensor, not a lenslet array — [https://zenodo.org/records/18696645](https://zenodo.org/records/18696645)
- `jacotay7/makewfs` (GitHub) explicitly simulates "hyper realistic wavefront sensor frames" — confirmed simulated, not real — [https://github.com/topics/wavefront-sensing](https://github.com/topics/wavefront-sensing)
- `congzhou235-prog/iccp-adaptive-optics-2026`: the SH sensor hardware itself is real and physically assembled (assembly guide, parts list, CAD/STEP files provided), but whether the specific `dataset/` folder's images are captures from that hardware or synthetic training examples organized by commanded Zernike mode could not be confirmed — [https://github.com/congzhou235-prog/iccp-adaptive-optics-2026](https://github.com/congzhou235-prog/iccp-adaptive-optics-2026)

### Inferences
- The presence of a real, physically documented SH sensor build (hardware guide, CAD files) does not guarantee that any bundled sample dataset is itself composed of raw physical captures rather than simulated stand-ins for teaching purposes; this distinction requires opening the actual data files, which was outside this task's constraints (no large-file downloads).

### Gaps
- Direct confirmation of the X-ray SH metrology work (KIT) and the fracture-mechanics SH work (Mechanics of Materials 2024) as bench-based is available from their titles/abstracts (both are explicitly experimental papers) but no associated dataset was found to check file-level provenance.

## Key Question 3: Sensor/lenslet geometry (detector resolution, bit depth, number of lenslets, pixel pitch, focal length)

### Takeaway
Essentially no geometry specifics were recoverable for any candidate within this task's search-only, no-large-download constraints; this is the weakest-covered question for this track.

### Cited Findings
- No detector resolution, bit depth, lenslet count, pixel pitch, or focal length figures were found in any of the fetched pages/repos for `kaccie14/shwfs`, the Zenodo laser-beam WFS records, `congzhou235-prog/iccp-adaptive-optics-2026`, or the X-ray/fracture-mechanics papers.
- The X-ray SH metrology paper title itself specifies the lenslet-array equivalent is a "2D refractive lens array" fabricated for hard X-rays (not a conventional visible-light glass/polymer lenslet array), which is a geometry-relevant qualitative detail but no pitch/count numbers were retrieved — [https://pmc.ncbi.nlm.nih.gov/articles/PMC7206553/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206553/)

### Gaps
- All quantitative sensor geometry for every candidate in this track is unconfirmed; would require opening full-text PDFs/READMEs directly, which was not done for most candidates due to tool-call budget and file-size constraints.

## Key Question 4: Reference/calibration data included (flat/reference frames, dark frames, lenslet map, pupil mask)?

### Takeaway
Only `kaccie14/shwfs` was confirmed to include an explicit reference/background frame alongside sample data frames; the ICCP course repo has a plausibly relevant `calibration_pics/` folder whose contents are unverified.

### Cited Findings
- `kaccie14/shwfs` includes `background.tif`, a background/dark-style reference frame, alongside the three sample data TIFFs — [https://github.com/kaccie14/shwfs](https://github.com/kaccie14/shwfs)
- `congzhou235-prog/iccp-adaptive-optics-2026`'s `sh_tutorial_prog/` package includes a `calibration_pics/` subdirectory (name implies calibration frames) and a `calibration.py` script, but the actual image contents of that folder were not enumerated in the fetch — [https://github.com/congzhou235-prog/iccp-adaptive-optics-2026](https://github.com/congzhou235-prog/iccp-adaptive-optics-2026)

### Gaps
- Whether `calibration_pics/` contains flat/reference frames, dark frames, or a lenslet/pupil map specifically is unconfirmed.
- No pupil-mask or lenslet-map files were identified for any candidate.

## Key Question 5: Label/ground-truth provenance (nominally commanded setpoints vs. independently measured, e.g., interferometer)

### Takeaway
Only the laser-beam DMD-based sensor work (out of scope as non-lenslet) was confirmed to cross-validate against an independent interferometer; the ICCP course dataset's folder names ("first_10_zernike," "last_10_zernike," "mixed_zernike") suggest labels are organized by commanded Zernike-mode ground truth, but whether these are independently verified or purely nominal setpoints is unconfirmed.

### Cited Findings
- Zenodo 18696645 (laser-beam WFS, DMD-based — excluded from the lenslet-only scope but noted for context): authors validated wavefront reconstruction "against a commercial lateral shearing interferometer" — an independently-measured ground truth — [https://zenodo.org/records/18696645](https://zenodo.org/records/18696645)
- `congzhou235-prog/iccp-adaptive-optics-2026` dataset folder structure (`first_10_zernike/`, `last_10_zernike/`, `mixed_zernike/`, `manifest.csv`) implies organization by which/how-many Zernike modes are commanded, consistent with nominally-commanded setpoint labeling, but this was not confirmed by reading the manifest — [https://github.com/congzhou235-prog/iccp-adaptive-optics-2026](https://github.com/congzhou235-prog/iccp-adaptive-optics-2026)

### Gaps
- No ground-truth provenance information was found for `kaccie14/shwfs`'s sample images (unclear what aberration, if any, was present when `R25M0004g.tif` etc. were captured).
- The `manifest.csv` in the ICCP repo was not opened; it likely contains the actual labels and would resolve this question directly.

## Key Question 6: License/reuse terms if stated

### Takeaway
Where licenses were found, they were permissive (CC-BY-4.0 for the Zenodo records, MIT for one GitHub repo); several candidates had no discoverable license.

### Cited Findings
- Zenodo 18696645 (laser-beam WFS paper record): Creative Commons Attribution 4.0 International (CC-BY-4.0) — [https://zenodo.org/records/18696645](https://zenodo.org/records/18696645)
- Zenodo 18692566 (companion dataset record): also CC-BY-4.0 — [https://zenodo.org/records/18692566](https://zenodo.org/records/18692566)
- `yuuutosasa/ShackHartmannAnalysis`: MIT License, with acknowledged reuse of Timothy Gebhard's `hswfs` code also under MIT — [https://github.com/yuuutosasa/ShackHartmannAnalysis](https://github.com/yuuutosasa/ShackHartmannAnalysis)
- MDPI Electronics 12(7):1714 (FPGA SH centroiding paper): confirmed via search-result summary to be "an open access article distributed under the terms of the Creative Commons Attribution (CC BY) license" (standard for all MDPI articles) — [https://www.mdpi.com/2079-9292/12/7/1714](https://www.mdpi.com/2079-9292/12/7/1714)
- `kaccie14/shwfs`: no LICENSE file or licensing terms found in the fetched content — [https://github.com/kaccie14/shwfs](https://github.com/kaccie14/shwfs)
- `congzhou235-prog/iccp-adaptive-optics-2026`: no license information found in the fetched README content — [https://github.com/congzhou235-prog/iccp-adaptive-optics-2026](https://github.com/congzhou235-prog/iccp-adaptive-optics-2026)

### Gaps
- License status for the AIP Figshare educational SH collection is unknown (page inaccessible).

## Key Question 7: Repeated exposures for the same aberration setting?

### Takeaway
Not confirmed for any candidate; no direct evidence either way was found within this task's scope.

### Cited Findings
- None found. No candidate's documentation explicitly stated whether multiple exposures/frames were captured per aberration setting.

### Gaps
- `kaccie14/shwfs` has three data TIFFs (`R25M0004g`, `R25M0005g`, `R25M0006g`) with sequential-looking naming that could indicate a repeated-exposure series of the same or stepped conditions, but this is speculative — filenames were not decoded and no README explaining the sequence was found.
- The ICCP course `manifest.csv` might indicate repeat counts per Zernike setting but was not opened.

## Key Question 8: Rough file count/size if discoverable

### Takeaway
Concrete sizes were only found for the two Zenodo records; GitHub repo data-folder sizes were not disclosed in any fetched page.

### Cited Findings
- Zenodo 18696645: single file, `applsci-16-01942.pdf`, 1.1 MB (article only, no raw data) — [https://zenodo.org/records/18696645](https://zenodo.org/records/18696645)
- Zenodo 18692566: single archive, `Dataset.zip`, 57.6 MB — [https://zenodo.org/records/18692566](https://zenodo.org/records/18692566)
- `kaccie14/shwfs`: exactly 4 image files identified (3 data TIFFs + 1 background TIFF); no size information given — [https://github.com/kaccie14/shwfs](https://github.com/kaccie14/shwfs)
- `congzhou235-prog/iccp-adaptive-optics-2026`: dataset organized into 3 subfolders by Zernike-mode grouping plus a manifest CSV; individual file counts/sizes not shown in the directory listing view — [https://github.com/congzhou235-prog/iccp-adaptive-optics-2026](https://github.com/congzhou235-prog/iccp-adaptive-optics-2026)

### Gaps
- No file-count/size data available for any other candidate.

## Brief note on out-of-scope candidates encountered (not duplicated in depth per task instructions)

- Astronomy/AO-testbed items surfaced repeatedly on Zenodo and were **not** pursued in depth per task scope: "Laboratory tests of the MAVIS DNU Shack Hartmann Wavefront Sensor" ([https://zenodo.org/records/21107489](https://zenodo.org/records/21107489)) — described as lab characterization of a multi-lenslet-array on-axis SHWFS for the MAVIS instrument; "MORFEO: optomechanical analyses of the Low-Order and Reference (LOR) NGS WFS Module" ([https://zenodo.org/records/21106141](https://zenodo.org/records/21106141)); "Shack-Hartmann sensing with arbitrarily shaped pupil" ([https://zenodo.org/records/3885508](https://zenodo.org/records/3885508)), an adaptive-microscopy AO Python-code release (arguably microscopy rather than pure astronomy, but is a code/protocol release, not a raw-image dataset, on inspection of the search summary).
- Ophthalmic-adjacent: `kaccie14/shwfs` (Roorda Lab, UC Berkeley — a group known for adaptive-optics ophthalmoscopy) is flagged here because it was found via general GitHub search, but it likely belongs with the ophthalmic-aberrometer research track rather than being a clean "other domain" example; included above for completeness since it was the only repo with directly confirmed real-bench raw TIFF frames.
- Non-lenslet wavefront sensors excluded per task constraints despite surfacing in "Shack-Hartmann"/"wavefront sensor" searches: DMD-based amplitude-flow sensor (Zenodo 18696645/18692566), holographic Fourier-plane wavefront sensors ("Recording, Implementation and Testing of Deep Holographic Fourier-plane Wavefront Sensors," [https://zenodo.org/records/21128332](https://zenodo.org/records/21128332)), and the "Bi-O-Edge" pyramid-like wavefront sensor ([https://zenodo.org/records/21128370](https://zenodo.org/records/21128370)).
