# Documentation

All standalone documents live here. Folder `README.md` files stay with their
code and only orient that folder.

## Start here

| Document | What it covers |
|---|---|
| **[Final simulation-only evidence package](../evaluation/results/runs/final_evidence_package_v1_20260930_final/REPORT.md)** | Authoritative metric audit, isolated factor-8 candidate ROMs, final bounded optical/noisy confirmation, exact candidate RTL replay, figures, and provenance. |
| [Normalized reciprocal precision experiment](../evaluation/results/runs/reciprocal_precision_rtl_v1_20261002_final/REPORT.md) | Opt-in arithmetic experiment preserving the baseline: A-I attribution, fixed-point sweep, normalized-Q17 RTL, exact dual-path verification, and estimated cost. |
| [Resolved-model candidate evaluation](../evaluation/results/runs/resolved_model_candidate_v2_20260930_final/REPORT.md) | Frozen source candidate: bounded factor-5-through-9 study, separated numerical controls, and held-out/noisy rejection. |
| [Focused forward-model investigation](../evaluation/results/runs/forward_model_investigation_v2_20260928_final/REPORT.md) | Preserved analytic lenslet validation, physical-grid audit, stage localization, and corrected pair-local noise-coupling evidence used by the current run. |
| [Frozen-reconstructor and ensemble convergence report](../evaluation/results/runs/ensemble_convergence_v1_20260928_final/REPORT.md) | Preserved deterministic and noisy convergence evidence; its comparison-set-dependent framewise noisy metric is superseded by the focused investigation. |
| [Blocker-resolution report](../evaluation/results/runs/blocker_investigation_v1_20260928_final/REPORT.md) | Preserved flat-bias trace, exact RTL replay, deterministic convergence failures, and clipping ablation reused by the current report. |
| [Source manuscript simulation report](../evaluation/results/runs/manuscript_v1_20260928_final/REPORT.md) | Preserved broad HCIPy/RTL campaign; its flat wording and convergence qualification are superseded by the blocker-resolution report. |
| [EVALUATION_REPORT.md](EVALUATION_REPORT.md) | **Historical.** Earlier arithmetic and RTL evaluation; stale claims are identified at the top. |
| [OPTICAL_EVALUATION.md](OPTICAL_EVALUATION.md) | **Historical.** Corrected-calibration campaign retained for provenance; superseded by the final manuscript run. |

## Design

| Document | What it covers |
|---|---|
| [SIGNAL_TRANSFER.md](SIGNAL_TRANSFER.md) | signal-level description of the datapath |
| [RECIPROCAL_PRECISION_PROOF.md](RECIPROCAL_PRECISION_PROOF.md) | **HISTORICAL.** Justifies a Q27 + Newton reciprocal targeting < 1e-4 px. That design was dropped: its 27-bit output overflows the 18-bit DSP48 B port. The shipping Q18 LUT measures 6.7e-3 px. See EVALUATION_REPORT.md §4.3 for why the gap cannot be closed by widening. |

## Implementation

| Document | What it covers |
|---|---|
| [REQUIRED_VIVADO_REPORTS.md](REQUIRED_VIVADO_REPORTS.md) | exactly which Vivado reports to supply to fill the empty resource/timing table |

## Validation programme

| Document | What it covers |
|---|---|
| [validation/VALIDATION_STATUS.md](validation/VALIDATION_STATUS.md) | task ledger |
| [validation/IMPLEMENTATION_AUDIT.md](validation/IMPLEMENTATION_AUDIT.md) | historical claims vs the repository as it stands |
| [validation/HARDWARE_HANDOFF.md](validation/HARDWARE_HANDOFF.md) | what is blocked on hardware/Vivado |
| [validation/DATASET_DECISION.md](validation/DATASET_DECISION.md) | which external dataset to pursue, and why |
| [validation/DATA_REQUEST_DRAFTS.md](validation/DATA_REQUEST_DRAFTS.md) | per-source wish-lists (nothing sent) |
| [validation/DATASET_CANDIDATES.csv](validation/DATASET_CANDIDATES.csv) | per-candidate detail |

## Research

| Document | What it covers |
|---|---|
| [research/Real Shack Hartmann sensor datasets.md](research/Real%20Shack%20Hartmann%20sensor%20datasets.md) | synthesis: no public raw SH imagery exists at the needed scale |
| [research/notes/](research/notes/) | the four underlying research passes — AO testbeds, ophthalmic, open repositories, the three named leads |

## Legacy

| Document | Status |
|---|---|
| [legacy/README_modules.md](legacy/README_modules.md) | **superseded.** Proposed per-module checks against Gen A modules (`reciprocal_u16_q27`, `shwfs_vivado_top`). Replaced by the SystemVerilog environment in `rtl/verification/`. |

## Where the code documentation lives

| Folder | Its README covers |
|---|---|
| `evaluation/` | running the measurement campaign and the single-frame tester |
| `rtl/verification/` | running the SystemVerilog verification environment |
| `sim/` | the Icarus simulation flow |
