# Evaluation

The authoritative simulation-only manuscript evidence package is
**[`results/runs/final_evidence_package_v1_20260930_final/REPORT.md`](results/runs/final_evidence_package_v1_20260930_final/REPORT.md)**,
with its machine-readable inventory in
[`RESULTS_INDEX.json`](results/runs/final_evidence_package_v1_20260930_final/RESULTS_INDEX.json).
It audits the separated convergence metrics, stages and verifies an isolated
factor-8 candidate hardware calibration, runs the final bounded confirmation,
and records exact RTL/model agreement without changing active ROMs. Its frozen
candidate source is
[`resolved_model_candidate_v2_20260930_final`](results/runs/resolved_model_candidate_v2_20260930_final/REPORT.md).
The immediate source investigation is
[`forward_model_investigation_v2_20260928_final`](results/runs/forward_model_investigation_v2_20260928_final/REPORT.md),
and its preserved source ensemble is
[`ensemble_convergence_v1_20260928_final`](results/runs/ensemble_convergence_v1_20260928_final/REPORT.md),
and its read-only blocker source is
[`blocker_investigation_v1_20260928_final`](results/runs/blocker_investigation_v1_20260928_final/REPORT.md),
which contains the fixed-flat trace and clipping ablation. The underlying broad
campaign is
[`manuscript_v1_20260928_final`](results/runs/manuscript_v1_20260928_final/REPORT.md),
whose unaffected broad optical and 35-case RTL results remain preserved.
`docs/EVALUATION_REPORT.md` and
`docs/OPTICAL_EVALUATION.md` are retained as historical reports because their
older arithmetic, convergence, ROM-path, and RTL-count conclusions are useful
provenance but are not the current manuscript evidence.

The latest opt-in arithmetic experiment is the
[`normalized_q17_rne` reciprocal-precision report](results/runs/reciprocal_precision_rtl_v1_20261002_final/REPORT.md).
It preserves the final evidence package and default RTL as the control, adds a
separate experimental path, and verifies both implementations bit-exactly.

Reproduce the frozen campaign without overwriting the completed run:

```bash
sim/venv/bin/python evaluation/optical/manuscript_evaluation.py \
  --config evaluation/configs/manuscript_v1.json \
  --out evaluation/results/runs/manuscript_v1_reproduction
```

Reproduce the blocker investigation:

```bash
sim/venv/bin/python evaluation/optical/blocker_investigation.py \
  --config evaluation/configs/blocker_investigation_v1.json \
  --out evaluation/results/runs/blocker_investigation_v1_reproduction
```

Reproduce the bounded frozen-reconstructor/noisy-ensemble study:

```bash
sim/venv/bin/python evaluation/optical/ensemble_convergence.py \
  --config evaluation/configs/ensemble_convergence_v1.json \
  --out evaluation/results/runs/ensemble_convergence_v1_reproduction
```

Reproduce the focused forward-model investigation:

```bash
sim/venv/bin/python evaluation/optical/forward_model_investigation.py \
  --config evaluation/configs/forward_model_investigation_v1.json \
  --out evaluation/results/runs/forward_model_investigation_v2_reproduction
```

Reproduce the resolved-model candidate evaluation:

```bash
sim/venv/bin/python evaluation/optical/resolved_model_candidate.py \
  --config evaluation/configs/resolved_model_candidate_v2.json \
  --out evaluation/results/runs/resolved_model_candidate_v2_reproduction
```

Reproduce the final bounded evidence package without overwriting it:

```bash
sim/venv/bin/python evaluation/optical/final_evidence_package.py \
  --config evaluation/configs/final_evidence_package_v1.json \
  --out evaluation/results/runs/final_evidence_package_v1_reproduction
```

Focused ROM/runner regressions:

```bash
sim/venv/bin/python -m unittest \
  evaluation.tests.test_final_evidence_package \
  evaluation.tests.test_resolved_model_candidate \
  evaluation.tests.test_noise_coupling \
  evaluation.tests.test_rom_loading \
  evaluation.tests.test_reference_conventions -v
```

The older broad arithmetic campaign remains available as:

```bash
sim/venv/bin/python evaluation/run_all.py
```

Generated output lands in `results/`. `run_all.py` is the historical campaign;
`--quick` for a smoke test, `--skip-rtl` to skip simulator steps.

### Testing a single frame

`run_frame.py` runs ONE image through the real RTL, from any source:

```bash
python evaluation/run_frame.py --demo                  # the repo's own frame
python evaluation/run_frame.py --hcipy                 # generate via HCIPy (~40 s)
python evaluation/run_frame.py --synth --seed 7        # fast forward-model frame
python evaluation/run_frame.py --frame f.mem           # any stored frame
python evaluation/run_frame.py --frame real.npy --external   # external dataset
```

It reports equivalence (RTL vs bit-accurate model, must be exact), quantisation
error, latency, and accuracy against ground truth when the source supplies one.
`--external` applies the compatibility gate and refuses mismatched geometry.

**The DUT is `rtl/`** — the same tree that is synthesised. There is no
second copy.

Two results are **not** produced because the underlying data does not exist —
FPGA resource/timing (no Vivado report anywhere on this machine) and
experimental-data validation (no compatible dataset). Both are documented in the
report rather than estimated.
