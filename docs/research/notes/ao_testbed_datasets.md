# Real Shack-Hartmann Lenslet-Array Spot Image Datasets from AO Testbeds/Telescopes

Scope note: candidates below were checked against the exclusion criteria (no pyramid WFS, no solar/extended-source SH, no curvature sensors, no purely synthetic data). Overall finding, stated up front: **no fully-verified, directly-downloadable public dataset of raw conventional-lenslet-array Shack-Hartmann camera frames from a real AO bench/telescope was found** that meets all criteria (working download link + confirmed real raw frames + stated lenslet geometry + calibration frames + provenance). Several partial/candidate leads were found and are detailed per question below, with what was directly verified vs. what remains unconfirmed clearly separated.

## Key Question 1: Is there an actual public download (Zenodo, Figshare, OSF, institutional/observatory archive, GitHub) with a working link to raw camera frames?

### Takeaway
Two Zenodo records and one large observatory archive (ESO) turned up as the strongest leads, but none was directly confirmed to expose downloadable raw SH camera-frame images meeting the full spec; most other searches surfaced only code repositories, simulators, or papers without an accompanying public raw-image dataset.

### Cited Findings
- Zenodo record "Laboratory tests of the MAVIS DNU Shack Hartmann Wavefront Sensor" is a working link (https://zenodo.org/records/21107489), but on inspection it is a single 6.9 MB conference-proceedings PDF, not a raw-image dataset; a follow-up attempt to fetch its files archive returned HTTP 404, and the record's own description gives no indication that raw camera frames are attached — [Zenodo record](https://zenodo.org/records/21107489)
- Zenodo record "Codebase for Wavefront Aberration Calibration Using Supervised and Deep Learning (SHWFS)" is a working link (https://zenodo.org/records/14964320) with real downloadable files: `Z_train.dat` (777.7 MB), `Z_test.dat` (129.6 MB), `W_train.dat` (24.5 MB), `W_test.dat` (4.1 MB), single-sample variants, CSV statistics files, and `Original_data.xlsx` — [Zenodo record](https://zenodo.org/records/14964320)
- The companion paper is "Calibration of wavefront aberrations using supervised learning and deep learning for a Shack–Hartmann wavefront sensor," Optical Engineering 64(5), 053101 (2025) by Jara Reyna et al. — [SPIE abstract page](https://www.spiedigitallibrary.org/journals/optical-engineering/volume-64/issue-5/053101/Calibration-of-wavefront-aberrations-using-supervised-learning-and-deep-learning/10.1117/1.OE.64.5.053101.short) (page could not be fetched for full text — returned empty content)
- ESO's Science Archive Facility (archive.eso.org) hosts all raw VLT/SPHERE data since October 2014, including a dedicated SPHERE data query form — [ESO Science Archive SPHERE form](http://archive.eso.org/wdb/wdb/eso/sphere/form), [ESO Archive Home](http://archive.eso.org/)
- No public data release could be found for the CANARY MOAO demonstrator (William Herschel Telescope) despite it using multiple SH configurations (7×7, 36-subaperture, 14×14 figure-sensor) — search of "CANARY AO testbed Shack-Hartmann data release open access" returned only papers using CANARY, not a data repository — [arXiv 1610.06313](https://arxiv.org/pdf/1610.06313)
- No public dataset found for HeNOS (Herzberg NFIRAOS Optical Simulator, University of Victoria), PAPYRUS (Observatoire de Haute-Provence — which in any case runs a Zernike/pyramid WFS in its published configuration, not a lenslet SH as primary sensor), or SEAL (Santa Cruz Extreme AO Lab, UC Santa Cruz — SH is one arm of a multi-WFS testbed but no public raw-frame archive was located)
- GitHub repositories found (ShackHartmannAnalysis, cacao-org/WFS_ShackHartmann, Shack-Hartmann-AO-Simulation, mshwfs, SH-CNN) are code/tool repositories; only ShackHartmannAnalysis was confirmed to ship an example image (`mainImage.png` in a `demo/` folder), and its real-vs-simulated origin could not be determined from the available page content — [GitHub: yuuutosasa/ShackHartmannAnalysis](https://github.com/yuuutosasa/ShackHartmannAnalysis), [GitHub: cacao-org/WFS_ShackHartmann](https://github.com/cacao-org/WFS_ShackHartmann)
- A Figshare item exists ("Visualization 1: Comparison of the plenoptic sensor and the Shack–Hartmann sensor," Optica Publishing Group) but is a supplementary visualization/video tied to an Applied Optics paper, not a raw-frame dataset — [Figshare item](https://opticapublishing.figshare.com/articles/media/Visualization_1_Comparison_of_the_plenoptic_sensor_and_the_Shack_Hartmann_sensor/4954262)

### Inferences
- The MAVIS DNU Zenodo record is best understood as a proceedings-paper mirror rather than a data deposit — the description text ("aberrations introduced by the DM can be measured simultaneously" by a 4D interferometer) documents a genuine real-bench experiment, but the actual pixel data does not appear to be attached to this specific Zenodo record.
- The SHWFS calibration Zenodo record (14964320) is the single most promising lead for an actual downloadable dataset by volume and structure (hundreds of MB of `.dat` files split into train/test), but without being able to open the paper or a README, it cannot be confirmed whether `Z_train.dat`/`Z_test.dat` contain real camera pixel arrays or simulated/synthetic SH images — this is a genuine open question, not a confirmed finding.
- ESO's archive is a large, credible candidate purely on infrastructure grounds (SPHERE/SAXO's SH sensor is a real, well-documented lenslet array), but availability of the *internal AO-loop WFS pixel frames* specifically (as opposed to science-camera exposures) through the public raw-data query form was not directly confirmed in this session.

### Gaps
- Could not open/parse the SPIE abstract page for the Jara Reyna et al. 2025 paper (fetch returned empty content), so the real-vs-simulated nature of the Zenodo 14964320 `.dat` files remains unverified.
- Could not verify by direct file listing whether ESO's SPHERE raw-data query form exposes SAXO WFS pixel frames (as distinct from science detector frames); this requires an actual query against archive.eso.org/wdb/wdb/eso/sphere/form, which was not performed (out of scope for "inspect metadata/listings only" given the paginated query-form nature of that interface).
- Institutional repositories for CANARY, HeNOS, PAPYRUS, SEAL, and other named testbeds were not found to host public raw SH datasets in the time available; a negative search result is not proof of absence, since some testbeds may host data behind institutional log-ins or on request only.

## Key Question 2: Physically recorded on a real bench/telescope (not simulated)?

### Takeaway
Only the MAVIS DNU record and the SPHERE/SAXO instrument are confirmed as physically-recorded, real-bench/real-telescope systems; the Zenodo SHWFS calibration dataset's data provenance (real vs. simulated) could not be confirmed, and most GitHub tool repositories give no clear provenance for their sample images.

### Cited Findings
- MAVIS DNU record: explicitly describes a laboratory bench at the University of Padova using "an ALPAO 97-15 Deformable Mirror (DM) and a 4D interferometer (Phasecam 4030)" with the multi-lenslet array WFS tested on-axis — a real physical bench — [Zenodo 21107489](https://zenodo.org/records/21107489)
- SPHERE/SAXO: a 40×40 lenslet Shack-Hartmann sensor with a "red-sensitive sub-electron noise EMCCD," physically operating on-sky at the VLT since 2014, per multiple sources — [ESO SPHERE archive](http://archive.eso.org/wdb/wdb/eso/sphere/form)
- The MNRAS 2022 paper "Deep phase retrieval for astronomical Shack-Hartmann wavefront sensors" (Guo, Wu, Li, Rao, Rao) reports "experiments on a 241-unit deformable-secondary-mirror AOS" — indicating real-bench/real-system validation, with an associated GitHub repo (SH-CNN) — but the repo's own data content (real vs. simulated) was not verified in this session — [MNRAS abstract](https://academic.oup.com/mnras/article/510/3/4347/6482856), [GitHub: IOE8AILab/SH-CNN](https://github.com/IOE8AILab/SH-CNN)
- The Zenodo SHWFS calibration record gives no explicit statement of real vs. simulated origin in its visible metadata — [Zenodo 14964320](https://zenodo.org/records/14964320)

### Inferences
- Real-bench provenance is confirmed at the level of the associated papers/instruments (MAVIS DNU bench, SPHERE/SAXO, the 241-actuator AOS in the MNRAS paper) but not confirmed at the level of specific downloadable pixel files for two of the three, because the actual data files could not be inspected.

### Gaps
- Real-vs-simulated status of the Zenodo 14964320 `.dat` files: unresolved.
- Real-vs-simulated status of GitHub sample/demo images (ShackHartmannAnalysis's `mainImage.png`, SH-CNN's any bundled examples): unresolved — page content did not clarify.

## Key Question 3: Sensor/lenslet geometry (detector resolution, bit depth, number of lenslets, pixel pitch, focal length)

### Takeaway
Only SPHERE/SAXO's lenslet count and detector type were confirmed with a citable specific number (40×40 lenslets, EMCCD); no source gave a full geometry spec (resolution, bit depth, pixel pitch, focal length together) for any candidate, including the MAVIS DNU bench.

### Cited Findings
- SPHERE/SAXO: "40x40 lenslet Shack-Hartmann sensor equipped with a red-sensitive sub-electron noise EMCCD," operating in visible light 500–900 nm at up to 1.2 kHz — [ESO SPHERE documentation / search summary](https://www.eso.org/sci/facilities/paranal/instruments/sphere/doc/VLT-MAN-SPH-14690-0430_v100_p2.pdf) (full PDF exceeded fetch size limit; figures taken from search-engine synthesis of the manual, not independently re-verified line-by-line)
- MAVIS DNU record: sensor described only as "a multi-lenslet array" functioning as an on-axis SH WFS, tested against an ALPAO 97-15 DM; no lenslet count, pixel pitch, detector resolution, or bit depth is given in the visible Zenodo metadata — [Zenodo 21107489](https://zenodo.org/records/21107489)
- Historical CANARY configurations: 7×7 SH WFS (SCAO mode), a 36-subaperture SH WFS, and a 14×14-subaperture SH figure sensor — geometry is stated but no downloadable data accompanies it — [arXiv 1610.06313](https://arxiv.org/pdf/1610.06313)

### Inferences
- None of the found candidates match the target 16×16/256×256-pixel-class, 168-valid-subaperture geometry exactly; SPHERE/SAXO's 40×40 array and CANARY's 7×7/36-subaperture/14×14 arrays are the closest documented geometries but at different scales.

### Gaps
- Full geometry (pixel pitch, lenslet focal length, detector bit depth) for SPHERE/SAXO was not independently confirmed from primary-source text since the user manual PDF exceeded the fetch tool's size limit; only a search-engine-synthesized summary was obtained.
- No geometry information was found for the Zenodo SHWFS calibration dataset (14964320).

## Key Question 4: Reference/calibration data included (flat/reference frames, dark frames, lenslet map, pupil mask)?

### Takeaway
No candidate was confirmed to include an explicit, itemized set of flat/dark/lenslet-map/pupil-mask calibration files; ESO's archive is the only source where such calibration files are structurally guaranteed to exist (ESO policy of pairing raw science frames with their calibrations) but this was not confirmed specifically for SAXO WFS frames.

### Cited Findings
- "All raw data from La Silla Paranal Observatory are stored together with the corresponding calibrations" — a general ESO archive policy statement, not confirmed specific to SAXO WFS pixel data — [ESO Archive Home](http://archive.eso.org/), search-engine synthesis of ESO documentation
- The MAVIS DNU Zenodo record's metadata gives no mention of dark/flat/pupil-mask files — [Zenodo 21107489](https://zenodo.org/records/21107489)
- The Zenodo SHWFS calibration record's file list (Z_train/Z_test, W_train/W_test, statistics CSVs, Original_data.xlsx) shows no explicitly named dark, flat, or pupil-mask files — [Zenodo 14964320](https://zenodo.org/records/14964320)

### Gaps
- Whether SPHERE/SAXO's public raw-data holdings include WFS-specific dark/flat/pupil-mask calibration products (as opposed to science-detector calibrations) is unconfirmed.
- No source for any candidate explicitly documents a lenslet-registration/reference map file.

## Key Question 5: Label/ground-truth provenance (nominally commanded DM/SLM setpoints vs. independently measured via interferometer/known phase plate)?

### Takeaway
The MAVIS DNU bench is the only candidate with confirmed dual provenance — both commanded DM setpoints and an independent 4D interferometer measurement recorded simultaneously — which is exactly the kind of independent ground truth the objective is looking for, but the underlying raw images were not confirmed to be publicly downloadable (see Q1/Q2).

### Cited Findings
- MAVIS DNU bench: "an ALPAO 97-15 Deformable Mirror (DM) and a 4D interferometer (Phasecam 4030) so the aberrations introduced by the DM can be measured simultaneously" — i.e., ground truth exists both as DM commands and as independent interferometric measurement — [Zenodo 21107489](https://zenodo.org/records/21107489)
- No other candidate (Zenodo 14964320, ESO/SPHERE, CANARY, HeNOS, SEAL) gave a citable statement about ground-truth provenance in the material retrieved this session.

### Gaps
- Ground-truth provenance for the Zenodo SHWFS calibration dataset (W_train/W_test presumably being Zernike-coefficient labels) is unconfirmed — unclear if labels are commanded SLM/DM values or independently measured.

## Key Question 6: License/reuse terms if stated

### Takeaway
Both inspected Zenodo records carry an explicit, permissive open license (CC-BY 4.0); license terms for the other candidates (ESO archive, GitHub repos) were not directly checked in this session.

### Cited Findings
- Zenodo 21107489 (MAVIS DNU): licensed under Creative Commons Attribution 4.0 International — [Zenodo 21107489](https://zenodo.org/records/21107489)
- Zenodo 14964320 (SHWFS calibration codebase): licensed under Creative Commons Attribution 4.0 International — [Zenodo 14964320](https://zenodo.org/records/14964320)

### Gaps
- ESO archive data-use policy/terms were not looked up in this session.
- License terms for GitHub repositories found (ShackHartmannAnalysis, cacao-org/WFS_ShackHartmann, SH-CNN, mshwfs, Shack-Hartmann-AO-Simulation) were not checked.

## Key Question 7: Repeated exposures for the same aberration setting?

### Takeaway
No source directly confirmed repeated-exposure structure for any candidate; this remains an open gap across the board.

### Cited Findings
- None found. No retrieved page or search snippet stated whether multiple raw frames were captured per fixed aberration/DM setting for any of the candidates (MAVIS DNU, Zenodo SHWFS dataset, SPHERE/SAXO, CANARY, HeNOS, SEAL).

### Gaps
- This question could not be answered for any candidate with the searches performed; would require opening the actual data files/README (MAVIS DNU PDF, SHWFS `.dat` file headers) or the SPIE paper text, none of which were accessible in this session.

## Key Question 8: Rough file count/size if discoverable

### Takeaway
Only the Zenodo SHWFS calibration record gives concrete, discoverable file sizes; all other candidates lack any confirmed file-count/size figures from this session's searches.

### Cited Findings
- Zenodo 14964320 file sizes: `Z_train.dat` 777.7 MB, `Z_test.dat` 129.6 MB, `W_train.dat` 24.5 MB, `W_test.dat` 4.1 MB, plus smaller single-sample variants, `Statistics_samples_x_y.csv` 533.7 kB, and `Original_data.xlsx` 21.7 kB — [Zenodo 14964320](https://zenodo.org/records/14964320)
- Zenodo 21107489 (MAVIS DNU): single PDF file, 6.9 MB — [Zenodo 21107489](https://zenodo.org/records/21107489)

### Gaps
- No file count/size information found for ESO/SPHERE SAXO WFS holdings, CANARY, HeNOS, SEAL, or any GitHub repository's sample data.

## Additional candidates checked and ruled out (exclusion criteria)

### Takeaway
Several superficially relevant "Shack-Hartmann" results were excluded per the task's stated constraints (solar/extended-source SH, pyramid/Zernike WFS) and are noted here so the report-writer knows they were considered and deliberately excluded, not missed.

### Cited Findings
- "Extended scene deep-phase-retrieval Shack-Hartmann wavefront sensors," A&A 2024 — explicitly for solar telescopes ("to improve the image quality of solar telescopes"); excluded as an extended-source/solar SH sensor per task constraints — [A&A article](https://www.aanda.org/articles/aa/full_html/2024/03/aa47960-23/aa47960-23.html)
- PAPYRUS testbed (Observatoire de Haute-Provence) — described as running with a "vector ZWFS" (Zernike wavefront sensor) in its second stage in the retrieved source; not confirmed as a conventional-lenslet-array SH testbed with public data — treated as out of scope — [A&A / search summary referencing PAPYRUS](https://www.aanda.org/articles/aa/full_html/2024/12/aa51449-24/aa51449-24.html)
- "LOOPS testbed" — uses "programmable Fourier-based wavefront sensors" (SLM-based, pyramid-like), not a conventional lenslet array; excluded — [arXiv 1903.06599](https://arxiv.org/pdf/1903.06599)

### Gaps
- None — these were successfully identified and excluded per the task's own criteria.

## Overall summary for the report-writer

### Takeaway
No dataset was found and fully verified to satisfy all eight criteria simultaneously (working public download of raw conventional lenslet-array SH camera frames, confirmed real-bench provenance, full stated geometry, calibration frames, ground-truth provenance, license, repeat-exposure structure, and file count/size). The two strongest leads for follow-up are: (1) the Zenodo "Codebase for Wavefront Aberration Calibration Using Supervised and Deep Learning (SHWFS)" record (14964320), which has real downloadable `.dat` files of substantial size under a CC-BY 4.0 license but unconfirmed real-vs-simulated provenance; and (2) ESO's SPHERE/SAXO holdings on archive.eso.org, which is a real, physically operating 40×40 lenslet SH sensor with raw data policy guaranteeing calibration pairing, but whose public exposure of WFS-specific (as opposed to science-camera) pixel frames was not directly confirmed via a live archive query in this session. The MAVIS DNU Zenodo record (21107489) is valuable as a description of a real bench with dual (commanded + interferometric) ground truth, but the underlying pixel data does not appear to be attached to that specific record — only the proceedings PDF is.

### Cited Findings
(See per-question sections above for all citations.)

### Inferences
- The overall scarcity of results suggests that most AO testbed groups either do not publicly release raw WFS pixel data (keeping it behind institutional access or reducing it to processed slopes/Zernike coefficients before any release), or that any released data is not indexed under obvious "Shack-Hartmann" + "dataset"/"data availability" search terms on Zenodo/Figshare/OSF/GitHub.

### Gaps
- A live, hands-on query of archive.eso.org's SPHERE raw-data form (to check for a WFS-frame product category) was not performed and is recommended as a next step.
- Direct inspection of the Zenodo 14964320 paper full text (SPIE Optical Engineering, paywalled/JS-rendered, could not be fetched) is recommended to resolve real-vs-simulated provenance.
- Direct outreach to CANARY (Durham University), HeNOS (University of Victoria/NRC Herzberg), SEAL (UC Santa Cruz), and PAPYRUS (OHP) teams may surface data-on-request arrangements not visible via public search.
