# Validation Status

Task ledger for the journal-validation program. Updated as work progresses.
Evidence labels follow the program's convention: optical simulation / RTL
simulation / post-route implementation estimate / measured FPGA replay
(synthetic) / measured FPGA replay (experimental) / live optical experiment.
Nothing below is a live optical experiment or a measured FPGA replay of any
kind — no board is present on this machine.

## Completion gates

| Gate | Status | Notes |
|---|---|---|
| A. Experimental dataset qualified | **BLOCKED — searched, none qualifies yet** | No experimental SH data in repo. Full candidate search complete: `docs/research/Real Shack Hartmann sensor datasets.md` + `docs/validation/DATASET_CANDIDATES.csv` + `DATASET_DECISION.md`. No public dataset satisfies the spec; best paths forward are a live ESO SPHERE archive query and two author requests (drafted, not sent — `DATA_REQUEST_DRAFTS.md`). |
| B. Calibration/reference validated | **DONE (simulated data only)** | `docs/EVALUATION_REPORT.md` §3-4; units/indexing sanity-checked, only against the repo's own HCIPy frame + synthetic Monte Carlo population, not real data. |
| C. RTL verified | **DONE** | 1,680/1,680 exact coefficient matches, 0 mismatches, RTL vs bit-accurate model, 168 directed + randomized cases. `docs/EVALUATION_REPORT.md` §4.1. Caveat: this is against the *corrected* copy in `sim/rtl/`, not `rtl/` as committed (see IMPLEMENTATION_AUDIT.md). |
| D. Optical campaign completed | **DONE for simulated inputs, NOT for experimental data** | 10,000-case Monte Carlo, noise sweep (9 flux levels × 300), dynamic-range sweep (16 levels), per-mode breakdown — all in `evaluation/results/`. No experimental-image campaign exists. |
| E. Board deployment verified | **BLOCKED — no hardware/Vivado access on this machine** | See `HARDWARE_HANDOFF.md` for exact commands to run once a board/Vivado project exists. |
| F. Reproducibility | **PARTIAL** | `evaluation/run_all.py` reruns the simulated-data campaign end to end (~5 min) and regenerates figures/tables. No equivalent exists yet for real-data ingestion (doesn't exist yet) or board capture (no board). |

## Section-by-section status (validation program §1-12)

| § | Topic | Status |
|---|---|---|
| 1 | Working principles / evidence labels | Adopted in this document and `IMPLEMENTATION_AUDIT.md` |
| 2 | Audit and freeze implementation | **DONE** — `IMPLEMENTATION_AUDIT.md`, reconciled against `docs/EVALUATION_REPORT.md` §1-2 |
| 3 | Find/qualify real datasets | **DONE — negative result.** Four-track search (named leads, astronomy/AO, ophthalmic, other domains) complete. No public raw-SH-image dataset satisfies the spec. Full detail: `docs/research/Real Shack Hartmann sensor datasets.md`, `docs/validation/DATASET_CANDIDATES.csv`, `DATASET_DECISION.md`. |
| 4 | Ingest data without distortion | **NOT STARTED**, but one free action available now: run `kaccie14/shwfs`'s 3 public real-instrument frames through `evaluation/real_data/load_external.py` as a smoke test of the compatibility gate (expect it to correctly refuse, since geometry is undocumented/unlikely to match). Full ingestion depends on §3's requests (ESO query, author asks) landing usable data. |
| 5 | Independent reference levels (A/B/C/D) | **A, B, C already exist** for simulated data (`evaluation/common/bitaccurate.py`, `rtl_equivalence/regression.py`); **D (board replay) blocked**, no hardware |
| 6 | Functional/numerical RTL verification | **DONE for the cases already run** — 168 directed+random cases, exact-match gate. Exhaustive-denominator reciprocal sweep exists separately in `rtl/verification/reciprocal_lut_sizing.py` / `docs/RECIPROCAL_PRECISION_PROOF.md`. Coverage of every bullet in §6 (all reset/stall/gap patterns) not independently re-audited this session. |
| 7 | Optical validation campaign | **DONE for simulated/synthetic inputs** (flat, per-mode, mixed-mode, noise sweep, dynamic range) — `docs/EVALUATION_REPORT.md` §4.2-4.5. Not done: real-image campaign, unmodeled-mode sensitivity, repeated-exposure statistics (no real exposures exist). |
| 8 | Hardware replay and timing | **BLOCKED** — no board, no Vivado report anywhere on this machine (searched `*.rpt/*.xpr/*.dcp/*utilization*/*timing_summary*`, confirmed in `docs/EVALUATION_REPORT.md` §4.9). `HARDWARE_HANDOFF.md` has exact commands for when access exists. |
| 9 | Metrics/statistics | **DONE for simulated data** — `docs/EVALUATION_REPORT.md` §4.2-4.5, with RMSE/p95/max, per-mode breakdown, noise-level sweep. |
| 10 | Architectural evidence / lit. comparison | **PARTIAL** — reciprocal-precision ablation done and is the strongest architectural result already in hand (§4.3, 2.34× RMSE reduction available for free). No literature comparison table exists yet (`prior_work_comparison_TEMPLATE.tex` is deliberately empty per report §8). |
| 11 | Deliverables | **IN PROGRESS** — this file, `IMPLEMENTATION_AUDIT.md`, `docs/validation/DATASET_CANDIDATES.csv`, `HARDWARE_HANDOFF.md` created this session. Remaining: `DATASET_DECISION.md`, `DATA_REQUEST_DRAFTS.md`, `CLAIM_EVIDENCE_MATRIX.md` (report §5 already has the substance of this — needs light reformatting), `VERIFICATION_AND_EXPERIMENTAL_SETUP.md`, `REPRODUCE.md` (mostly covered by `evaluation/README.md` already). |
| 12 | Completion gates | See table above. |

## Immediate next actions (smallest remaining human/agent steps)

1. **Free accuracy win, no new data needed**: instantiate `reciprocal_u16_q27` in
   place of `reciprocal_u16_q18` in the evaluated top and re-run
   `evaluation/run_all.py` to confirm the 2.34× RMSE reduction in RTL (not just
   in the ablation model). This is the single highest-value, lowest-risk task
   available right now.
2. Dataset research is complete (see §3 above). Next concrete steps, in
   order, are in `DATASET_DECISION.md`: (a) smoke-test the compatibility gate
   with `kaccie14/shwfs`'s public frames — free, no approval needed; (b) live
   query the ESO SPHERE archive form to check whether it exposes SAXO WFS
   pixel frames — free, no approval needed; (c) open
   `congzhou235-prog/iccp-adaptive-optics-2026`'s `manifest.csv` to resolve
   real-vs-simulated provenance — free; (d) decide whether to actually send
   the two author requests drafted in `DATA_REQUEST_DRAFTS.md` — needs you.
3. Decide whether to pursue Vivado implementation (needs the actual tool +
   part/board) to unblock Gate E — this requires you, since it's a real
   EDA-tool + hardware step outside what I can do here.
4. **DONE this session**: the 3 documented `rtl/` defects (Q4.23
   reference-slope overflow, `unsigned_mult` Q6/Q18 width mismatch,
   reciprocal 2-cycle side-band misalignment) have been ported directly into
   `rtl/`. See `IMPLEMENTATION_AUDIT.md` for exactly what changed and
   what verification was done (`iverilog` elaboration + file-diff against the
   validated DUT copy). Still outstanding: rerun the RTL-equivalence
   regression pointed at `rtl/` itself (it was only ever run against
   `sim/rtl/`) before treating this as fully verified.
