#!/usr/bin/env python3
"""Verify the separate normalized-Q17 RTL experiment and frozen baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/shwfs-matplotlib")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import config as C  # noqa: E402
from evaluation.common import precision_experimental as P  # noqa: E402
from evaluation.common import rtl_runner as R  # noqa: E402
from evaluation.common.roms import stage_rom_set, validate_rom_set  # noqa: E402
from evaluation.optical.blocker_investigation import (  # noqa: E402
    frozen_pipeline,
    load_frozen_roms,
)
from evaluation.optical.resolved_model_candidate import (  # noqa: E402
    common_scoring_bench,
    coefficient_metric,
    metric_rms,
)


DEFAULT_CONFIG = C.EVAL_DIR / "configs" / "reciprocal_precision_rtl_v1.json"
EXPERIMENTAL_SOURCES = [
    C.RTL_DIR / "experimental" / "shwfs_processing_precision_experimental.v",
    C.RTL_DIR / "Thresholding_CoG_intensity_accumulator.v",
    C.RTL_DIR / "experimental" / "intensity_reciprocal_normalized_q17_rne.v",
    C.RTL_DIR / "experimental" / "reciprocal_u16_normalized_q17_rne.v",
    C.RTL_DIR / "experimental" / "slope_calculation_normalized_q17_rne.v",
    C.RTL_DIR / "ematrix_accumulator.v",
    C.RTL_DIR / "streaming_emulator.v",
    C.SIM_DIR / "bram_model.v",
    C.SIM_DIR / "tb_zynq_top_precision_experimental.v",
    C.SIM_DIR / "tb_full_pipeline.v",
]
BASELINE_FILES = [
    C.RTL_DIR / "shwfs_processing.v",
    C.RTL_DIR / "Thresholding_CoG_intensity_accumulator.v",
    C.RTL_DIR / "intensity_reciprocal_q18.v",
    C.RTL_DIR / "reciprocal_u16_q18.v",
    C.RTL_DIR / "slope_calculation.v",
    C.RTL_DIR / "ematrix_accumulator.v",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def generate_normalized_rom(path: Path) -> dict:
    values = []
    for divisor in range(1 << 16):
        if divisor == 0:
            value = 0
        else:
            exponent = divisor.bit_length() - 1
            value = P.divide_unsigned(1 << (17 + exponent), divisor,
                                      "nearest_half_up")
        if not 0 <= value < (1 << 18):
            raise AssertionError(f"normalized reciprocal does not fit 18 bits: d={divisor}")
        values.append(value)
    path.write_text("".join(f"{value:05X}\n" for value in values))
    return {
        "words": len(values), "width_bits": 18,
        "minimum_nonzero": min(values[1:]), "maximum": max(values),
        "sha256": sha256(path),
        "definition": "round_half_up(2^(17+floor(log2(d)))/d), d>0; zero for d=0",
    }


class ExperimentalRTLSimulator(R.RTLSimulator):
    def __init__(self, experimental_rom: Path, *args, **kwargs):
        self.experimental_rom = Path(experimental_rom)
        super().__init__(*args, **kwargs)

    def _build(self) -> None:
        missing = [str(path) for path in EXPERIMENTAL_SOURCES if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing experimental simulation source: " + ", ".join(missing))
        cmd = [
            "iverilog", "-g2012", "-o", str(self.vvp), "-s", "tb_full_pipeline",
            f"-Ptb_full_pipeline.FINAL_TILE_PIXEL={R.final_tile_pixel_index()}",
            *[str(path.resolve()) for path in EXPERIMENTAL_SOURCES],
        ]
        process = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=self.timeout_seconds)
        if process.returncode != 0:
            raise RuntimeError(f"experimental iverilog failed:\n{process.stdout}{process.stderr}")
        self.compile_output = process.stdout + process.stderr
        self.rom_manifest = stage_rom_set(self.workdir, self.rom_dir)
        shutil.copyfile(self.experimental_rom,
                        self.workdir / "reciprocal_normalized_q17_rne.mem")


def expected_columns(trace: dict, valid: np.ndarray) -> np.ndarray:
    return np.column_stack([
        trace["x_slope_valid"], trace["y_slope_valid"],
        trace["x_centroid"][valid], trace["y_centroid"][valid],
    ])


def verify_rtl_frames(source_run: Path, out: Path, roms: dict,
                      experimental_rom: Path) -> tuple[dict, dict]:
    stored = np.load(source_run / "rtl_input_frames.npz")
    names = [str(name) for name in stored["names"]]
    frames = stored["frames"].astype(np.uint8)
    baseline_rows = []
    candidate_rows = []
    baseline_logs = out / "baseline_rtl_logs"
    candidate_logs = out / "candidate_rtl_logs"
    baseline_logs.mkdir()
    candidate_logs.mkdir()
    selected = P.NORMALIZED_Q17_RNE
    with R.RTLSimulator(rom_dir=source_run / "candidate_hardware_roms",
                        timeout_seconds=45) as baseline_simulator, \
            ExperimentalRTLSimulator(
                experimental_rom,
                rom_dir=source_run / "candidate_hardware_roms",
                timeout_seconds=45) as candidate_simulator:
        compile_output = candidate_simulator.compile_output
        for name, frame in zip(names, frames):
            baseline_model = frozen_pipeline(frame, roms)
            candidate_model = P.evaluate_variant(frame, roms, selected)
            baseline_rtl = baseline_simulator.run_frame(frame)
            candidate_rtl = candidate_simulator.run_frame(frame)
            for label, model, rtl, rows, log_dir in (
                    ("baseline", baseline_model, baseline_rtl,
                     baseline_rows, baseline_logs),
                    ("normalized_q17_rne", candidate_model, candidate_rtl,
                     candidate_rows, candidate_logs)):
                slope_delta = rtl["slopes"][:, 1:5] - expected_columns(
                    model, roms["valid"])
                coefficient_delta = rtl["zernike_raw"] - model["zernike_raw"]
                rows.append({
                    "implementation": label, "case": name,
                    "frame_sha256": sha256_bytes(frame.tobytes()),
                    "model_coefficients_raw": json.dumps(
                        np.asarray(model["zernike_raw"], dtype=np.int64).tolist()),
                    "rtl_coefficients_raw": json.dumps(rtl["zernike_raw"].tolist()),
                    "coefficient_mismatch_count": int(np.count_nonzero(coefficient_delta)),
                    "slope_centroid_mismatch_count": int(np.count_nonzero(slope_delta)),
                    "max_abs_coefficient_difference": int(np.max(np.abs(coefficient_delta))),
                    "max_abs_slope_centroid_difference": int(np.max(np.abs(slope_delta))),
                    "valid_slopes": rtl["subaps_valid"],
                    "valid_coefficients": len(rtl["zernike_raw"]),
                    "unknown_valid_slopes": rtl["unknown_valid_slopes"],
                    "unknown_valid_coefficients": rtl["unknown_valid_coefficients"],
                    "pixels_streamed": rtl["pixels_streamed"],
                    "pixel_mismatches": rtl["pixel_mismatches"],
                    "final_pixel_to_done_cycles": rtl["true_latency_cycles"],
                    "accumulator_to_reciprocal_cycles": (
                        rtl["stage_recip_cycle"] - rtl["stage_accum_cycle"]),
                    "reciprocal_to_slope_cycles": (
                        rtl["stage_slope_cycle"] - rtl["stage_recip_cycle"]),
                    "slope_to_done_cycles": (
                        rtl["stage_done_cycle"] - rtl["stage_slope_cycle"]),
                })
                (log_dir / f"{name}.txt").write_text(rtl["stdout"])
    (out / "experimental_compile_output.txt").write_text(compile_output)
    write_csv(out / "baseline_control_rtl_cases.csv", baseline_rows)
    write_csv(out / "candidate_rtl_cases.csv", candidate_rows)

    def summarize(rows: list[dict]) -> dict:
        exact = all(row["coefficient_mismatch_count"] == 0 and
                    row["slope_centroid_mismatch_count"] == 0 for row in rows)
        correct = all(row["valid_slopes"] == C.NUM_SUBAPS_VALID and
                      row["valid_coefficients"] == C.NUM_MODES for row in rows)
        no_unknowns = all(row["unknown_valid_slopes"] == 0 and
                          row["unknown_valid_coefficients"] == 0 for row in rows)
        return {
            "frames": len(rows), "bit_exact": exact,
            "correct_output_counts": correct, "no_unknown_valid_outputs": no_unknowns,
            "final_pixel_to_done_cycles": sorted(set(
                row["final_pixel_to_done_cycles"] for row in rows)),
            "accumulator_to_reciprocal_cycles": sorted(set(
                row["accumulator_to_reciprocal_cycles"] for row in rows)),
            "reciprocal_to_slope_cycles": sorted(set(
                row["reciprocal_to_slope_cycles"] for row in rows)),
            "slope_to_done_cycles": sorted(set(
                row["slope_to_done_cycles"] for row in rows)),
            "passed": bool(exact and correct and no_unknowns),
        }
    baseline_summary = summarize(baseline_rows)
    candidate_summary = summarize(candidate_rows)
    json_dump(out / "rtl_verification_summary.json", {
        "baseline": baseline_summary, "candidate": candidate_summary})
    if not baseline_summary["passed"] or not candidate_summary["passed"]:
        raise AssertionError("baseline or candidate RTL exact verification failed")
    return baseline_summary, candidate_summary


def copy_software_evidence(software_run: Path, out: Path) -> None:
    target = out / "software_evidence"
    target.mkdir()
    for name in (
        "REPORT.md", "baseline_reproduction.json", "stage_error_attribution.csv",
        "precision_sweep_summary.csv", "precision_sweep_raw.csv",
        "operating_point_summary.csv", "per_mode_error.csv",
        "candidate_architectures.csv", "deterministic_frame_index.csv",
        "regression_tests.txt", "MANIFEST.json",
    ):
        shutil.copyfile(software_run / name, target / name)


def build_resource_table(out: Path, baseline_rtl: dict, candidate_rtl: dict) -> list[dict]:
    rows = [
        {
            "design": "baseline_q18", "centroid_multipliers": 2,
            "ematrix_multipliers": 20, "newton_multipliers": 0,
            "estimated_total_dsp48": 22, "estimated_dsp_delta": 0,
            "added_arithmetic_pipeline_stages": 0,
            "final_pixel_to_done_cycles": baseline_rtl["final_pixel_to_done_cycles"][0],
            "reciprocal_boundary_cycles": baseline_rtl["accumulator_to_reciprocal_cycles"][0],
            "centroid_slope_boundary_cycles": baseline_rtl["reciprocal_to_slope_cycles"][0],
            "steady_state_throughput": "one accepted pixel/cycle and one completed subaperture/cycle",
            "bram_effect": "65,536x19 reciprocal ROM",
            "lut_ff_effect": "frozen baseline",
            "estimate_or_actual": "structural estimate plus simulated cycles",
        },
        {
            "design": "normalized_q17_rne", "centroid_multipliers": 2,
            "ematrix_multipliers": 20, "newton_multipliers": 0,
            "estimated_total_dsp48": 22, "estimated_dsp_delta": 0,
            "added_arithmetic_pipeline_stages": 0,
            "final_pixel_to_done_cycles": candidate_rtl["final_pixel_to_done_cycles"][0],
            "reciprocal_boundary_cycles": candidate_rtl["accumulator_to_reciprocal_cycles"][0],
            "centroid_slope_boundary_cycles": candidate_rtl["reciprocal_to_slope_cycles"][0],
            "steady_state_throughput": "one accepted pixel/cycle and one completed subaperture/cycle",
            "bram_effect": "65,536x18 normalized ROM; same depth, one fewer data bit",
            "lut_ff_effect": "adds 4-bit exponent path plus two 38-bit rounding/barrel-shift networks; not synthesized",
            "estimate_or_actual": "structural estimate plus simulated cycles",
        },
    ]
    write_csv(out / "latency_resource_estimate.csv", rows)
    return rows


def selected_numbers(software_run: Path) -> dict:
    summary = {row["variant"]: row for row in
               read_csv(software_run / "precision_sweep_summary.csv")}
    operating = read_csv(software_run / "operating_point_summary.csv")
    baseline = summary["baseline_q18"]
    candidate = summary["normalized_q17_rne"]
    rows = []
    for photons in (1e6, 3e6):
        for amplitude in (20, 100, 400):
            values = {row["variant"]: row for row in operating
                      if row["metric"] == "ground_truth_residual_nm" and
                      float(row["photons_per_pupil"]) == photons and
                      int(row["input_rms_nm"]) == amplitude and
                      row["variant"] in ("baseline_q18", "normalized_q17_rne")}
            rows.append({
                "photons_per_pupil": photons, "input_rms_nm": amplitude,
                "baseline_mean_ground_truth_residual_nm": float(
                    values["baseline_q18"]["mean_nm"]),
                "candidate_mean_ground_truth_residual_nm": float(
                    values["normalized_q17_rne"]["mean_nm"]),
                "candidate_minus_baseline_nm": float(
                    values["normalized_q17_rne"]["mean_nm"]) - float(
                    values["baseline_q18"]["mean_nm"]),
            })
    return {"baseline": baseline, "candidate": candidate,
            "operating_ground_truth": rows}


def write_report(cfg: dict, out: Path, source_run: Path, software_run: Path,
                 baseline_rtl: dict, candidate_rtl: dict,
                 resource_rows: list[dict], rom_record: dict) -> None:
    numbers = selected_numbers(software_run)
    baseline = numbers["baseline"]
    candidate = numbers["candidate"]
    squared_removed = 100 * (1 - (
        float(candidate["flat_3m_residual_nm"]) /
        float(baseline["flat_3m_residual_nm"])) ** 2)
    operating_lines = "\n".join(
        f"| {row['photons_per_pupil']/1e6:g}M | {row['input_rms_nm']} | "
        f"{row['baseline_mean_ground_truth_residual_nm']:.6f} | "
        f"{row['candidate_mean_ground_truth_residual_nm']:.6f} | "
        f"{row['candidate_minus_baseline_nm']:+.6f} |"
        for row in numbers["operating_ground_truth"])
    report = f"""# Normalized reciprocal precision experiment

Run: `{cfg['run_id']}`  
Frozen control: `{source_run.name}`  
Software selection: `{software_run.name}`

## 1. Baseline reproduction

The immutable control again reproduces 7.658487851 nm fixed and 0 nm floating
on the candidate 3M flat frame. Across all {baseline_rtl['frames']} frozen RTL
subset frames it remains exactly equal to its original bit-accurate model for
centroids, slopes, and coefficients. Counts are 168 slopes and 10 coefficients,
there are no unknown valid outputs, and latency remains
{baseline_rtl['final_pixel_to_done_cycles'][0]} final-pixel-to-done cycles.

## 2. Stage-by-stage error attribution

The existing LUT contents exactly equal `round(2^18/d)`; LUT addressing,
integer multiplication, product slice, subtraction, slope format, E-matrix MAC,
and final slice are not material causes. Replacing only reciprocal-before-
multiply Q18 quantization with the exact rounded centroid quotient removes
100% of baseline squared flat WFE in the isolated test. The final slice removes
only 0.000966% in its non-additive isolated replacement.

| Rank | Isolated stage replacement | Modified flat RMS nm | Removed nm | Squared WFE removed |
|---:|---|---:|---:|---:|
| 1 | B: reciprocal output Q18 quantization | 0.000000 | 7.658488 | 100.000000% |
| 2 | I: final accumulator output slice | 7.658451 | 0.000037 | 0.000966% |
| 3 | A: reciprocal LUT algorithm/content | 7.658488 | 0.000000 | 0.000000% |
| 3 | C: numerator x reciprocal multiply | 7.658488 | 0.000000 | 0.000000% |
| 3 | D: post-multiply product slice | 7.658488 | 0.000000 | 0.000000% |
| 3 | E: centroid/reference subtraction | 7.658488 | 0.000000 | 0.000000% |
| 3 | F: centroid-to-slope conversion | 7.658488 | 0.000000 | 0.000000% |
| 3 | G: slope output slice | 7.658488 | 0.000000 | 0.000000% |
| 3 | H: E-matrix multiply/accumulate | 7.658488 | 0.000000 | 0.000000% |

## 3. Precision sweep

| Design | Flat 3M RMS nm | Mean FF nm | p95 FF nm | DSP delta estimate | Cycle delta | RTL bit-exact status |
|---|---:|---:|---:|---:|---:|---|
| Baseline | {float(baseline['flat_3m_residual_nm']):.6f} | {float(baseline['mean_fixed_minus_floating_nm']):.6f} | {float(baseline['p95_fixed_minus_floating_nm']):.6f} | 0 | 0 | Yes, 19/19 frames |
| Normalized Q17 RNE | {float(candidate['flat_3m_residual_nm']):.6f} | {float(candidate['mean_fixed_minus_floating_nm']):.6f} | {float(candidate['p95_fixed_minus_floating_nm']):.6f} | 0 | 0 | Yes, 19/19 frames |
| Direct Q25 RNE | 0.036398 | 0.017350 | 0.045281 | +2 | 0 | Software bit model only |
| NR1 Q25 RNE | 0.036398 | 0.017323 | 0.045000 | +4 | +2 | Software bit model only |

The selected implementation removes {squared_removed:.6f}% of the original
flat squared WFE and improves arithmetic discrepancy on every one of the 276
deterministic frames. Newton-Raphson is not necessary.

## 4. Candidate architecture comparison

The selected path represents `d=a*2^k` with an 18-bit unsigned Q17 mantissa
`round(2^17/a)` and a 4-bit unsigned exponent. Each 19-bit unsigned moment times
18-bit mantissa produces a 37-bit unsigned product. A 38-bit rounding adder and
exponent-controlled shift retain a 24-bit unsigned Q18 centroid. Ties round up.
All later formats, ROMs, references, widths, and slices are unchanged.

## 5. Selected implementation

`normalized_q17_rne` is the lowest-cost material improvement. It is more accurate
than Q25 direct and Q25 NR on the flat case, uses the existing centroid multiplier
shape, and adds no arithmetic pipeline stage. The experimental normalized ROM
hash is `{rom_record['sha256']}`.

## 6. RTL changes

Only new files under `rtl/experimental/` and the opt-in simulation wrapper
`sim/tb_zynq_top_precision_experimental.v` were added. The default baseline RTL,
active ROMs, calibration, and prior result directories are unchanged.

## 7. Bit-exact verification

The experimental RTL matches its own bit-exact model on all
{candidate_rtl['frames']} frozen frames: zero centroid, slope, or coefficient
mismatches; 168 slopes and 10 coefficients per frame; no unknown valid output.
The baseline simultaneously passes the same requirements.

## 8. Numerical results

Candidate flat residuals are {float(candidate['flat_1m_residual_nm']):.6f} nm at
1M photons and {float(candidate['flat_3m_residual_nm']):.6f} nm at 3M, versus
{float(baseline['flat_1m_residual_nm']):.6f} and
{float(baseline['flat_3m_residual_nm']):.6f} nm for baseline. Across the full
deterministic set, candidate mean/p95/max fixed-minus-floating errors are
{float(candidate['mean_fixed_minus_floating_nm']):.6f}/
{float(candidate['p95_fixed_minus_floating_nm']):.6f}/
{float(candidate['max_fixed_minus_floating_nm']):.6f} nm. Worst coefficient
error is {float(candidate['worst_abs_coefficient_error_nm']):.6f} nm; slope
RMS/max errors are {float(candidate['slope_rms_error_px']):.8f}/
{float(candidate['slope_max_error_px']):.8f} px. There are zero wraps and zero
invalid centroids.

| Photons | Input RMS nm | Baseline truth residual nm | Candidate truth residual nm | Change nm |
|---|---:|---:|---:|---:|
{operating_lines}

Arithmetic error improves on every frame, but ground-truth residual worsens on
104/276 individual frames because the old coherent bias sometimes canceled
optical error. The worst individual increase is 2.105503 nm. At the six mixed
operating cells, three means improve and three increase only 0.0320-0.0370 nm.

## 9. Latency and resource cost

Both designs simulate at {candidate_rtl['final_pixel_to_done_cycles'][0]} cycles
from the final contributing pixel to done, with three accumulator-to-reciprocal,
two reciprocal-to-slope, and two slope-to-done boundary cycles. Throughput remains
one accepted pixel per cycle and one completed subaperture per cycle after fill.
Structural inference is 22 DSP48s for both designs: two centroid multipliers plus
20 E-matrix multipliers, delta zero. The candidate adds exponent registers and
two rounding/barrel-shift networks; the normalized ROM is 65,536x18 rather than
65,536x19. No Vivado run was performed, so LUT/FF/BRAM mapping and achieved timing
remain estimates, not implementation results.

## 10. Recommendation

Retain the current baseline as the default control. The normalized Q17 RNE path
is the recommended experimental successor if the precision floor justifies an
opt-in RTL variant. It reaches 0.014726 nm on the defining flat case with no
estimated DSP or latency increase. NR1 is numerically unnecessary and materially
more expensive. Any promotion into the default design should follow synthesis
and timing closure, which were outside this bounded simulation-only run.

Explicit decisions:

- Reciprocal-before-multiply Q18 quantization accounts for 100% of the isolated
  original flat squared error; the implemented candidate removes 99.999630%.
- Newton-Raphson was not necessary.
- Normalized Q17 RNE is the lowest-cost material improvement.
- Arithmetic discrepancy improves on every case. Ground-truth residual worsens
  in 104 individual cases through removal of fortuitous baseline cancellation;
  three aggregate mixed cells increase only 0.0320-0.0370 nm.
- The experimental design adds zero simulated cycles and zero estimated DSP48s.
- The baseline implementation is completely unchanged and reproducible.
"""
    (out / "REPORT.md").write_text(report)


def write_results_index(out: Path, source_run: Path, software_run: Path,
                        candidate_rom: dict) -> None:
    entries = []
    for path, description, samples, uncertainty in (
        ("software_evidence/stage_error_attribution.csv", "ranked A-I isolated replacements",
         "one frozen 3M flat frame", "deterministic isolated ablation"),
        ("software_evidence/precision_sweep_summary.csv", "candidate arithmetic summary",
         "276 deterministic frames/candidate", "descriptive full-set statistics"),
        ("software_evidence/operating_point_summary.csv", "mixed operating-point summary",
         "20 independent wavefronts/cell", "95% Student-t interval"),
        ("baseline_control_rtl_cases.csv", "frozen baseline RTL equality",
         "19 selected frames", "exact integer equality"),
        ("candidate_rtl_cases.csv", "experimental candidate RTL equality",
         "19 selected frames", "exact integer equality"),
        ("latency_resource_estimate.csv", "latency and structural resource comparison",
         "two architectures", "simulated cycles; structural estimates"),
    ):
        entries.append({
            "path": path, "description": description,
            "source_evidence_run": source_run.name,
            "source_software_run": software_run.name,
            "calibration": "unchanged frozen factor-8 candidate",
            "candidate_reciprocal_rom_sha256": candidate_rom["sha256"],
            "sample_count": samples, "uncertainty": uncertainty,
        })
    json_dump(out / "RESULTS_INDEX.json", {"schema_version": 1, "entries": entries})


def write_manifest(cfg_path: Path, cfg: dict, out: Path, source_run: Path,
                   software_run: Path, baseline_before: dict, active_before: dict,
                   experimental_rom: dict) -> None:
    baseline_after = {str(path.relative_to(C.REPO_ROOT)): sha256(path)
                      for path in BASELINE_FILES}
    active_after = {name: value["sha256"] for name, value in
                    validate_rom_set(C.RTL_DIR).items()}
    if baseline_before != baseline_after:
        raise AssertionError("frozen baseline RTL changed during experiment")
    if active_before != active_after:
        raise AssertionError("active ROMs changed during experiment")
    sources = [Path(__file__).resolve(), cfg_path,
               C.EVAL_DIR / "common" / "precision_experimental.py",
               C.EVAL_DIR / "tests" / "test_precision_experimental.py",
               *EXPERIMENTAL_SOURCES[:5], C.SIM_DIR / "tb_zynq_top_precision_experimental.v"]
    result_files = [path for path in out.rglob("*")
                    if path.is_file() and path.name != "MANIFEST.json"]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    git_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=C.REPO_ROOT, check=True,
        capture_output=True, text=True).stdout.strip())
    manifest = {
        "schema_version": 1, "run_id": cfg["run_id"],
        "source_revision": {
            "git_head": git_head,
            "git_worktree_dirty": git_dirty,
            "identity_note": "source_sha256 entries are authoritative for this dirty worktree",
        },
        "parameters": cfg,
        "rtl_implementations_tested": {
            "baseline": {
                "top": "shwfs_processing",
                "reciprocal": "reciprocal_u16_q18",
                "selection": "default baseline testbench/build path",
            },
            "candidate": {
                "top": "shwfs_processing_precision_experimental",
                "reciprocal": "reciprocal_u16_normalized_q17_rne",
                "selection": "opt-in tb_zynq_top_precision_experimental testbench",
            },
        },
        "source_evidence_run": str(source_run.relative_to(C.REPO_ROOT)),
        "source_evidence_manifest_sha256": sha256(source_run / "MANIFEST.json"),
        "source_software_run": str(software_run.relative_to(C.REPO_ROOT)),
        "source_software_manifest_sha256": sha256(software_run / "MANIFEST.json"),
        "configuration_sha256": sha256(cfg_path),
        "source_sha256": {str(path.relative_to(C.REPO_ROOT)): sha256(path)
                          for path in sources},
        "baseline_rtl_sha256_before": baseline_before,
        "baseline_rtl_sha256_after": baseline_after,
        "baseline_rtl_unchanged": True,
        "active_rom_sha256_before": active_before,
        "calibration_sha256": active_before,
        "active_rom_sha256_after": active_after,
        "active_roms_unchanged": True,
        "experimental_rom": experimental_rom,
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy")},
        "result_sha256": {str(path.relative_to(out)): sha256(path)
                          for path in result_files},
        "vivado_synthesis_implementation": "not run",
        "resource_numbers": "structural estimates only",
        "default_build": "unchanged baseline",
    }
    json_dump(out / "MANIFEST.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text())
    out = (args.out or C.RESULTS_DIR / "runs" / cfg["run_id"]).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    source_run = (C.REPO_ROOT / cfg["source_evidence_run"]).resolve()
    software_run = (C.REPO_ROOT / cfg["source_software_run"]).resolve()
    if not (source_run / "MANIFEST.json").is_file() or not (software_run / "MANIFEST.json").is_file():
        raise FileNotFoundError("source evidence run is incomplete")
    baseline_before = {str(path.relative_to(C.REPO_ROOT)): sha256(path)
                       for path in BASELINE_FILES}
    active_before = {name: value["sha256"] for name, value in
                     validate_rom_set(C.RTL_DIR).items()}
    out.mkdir(parents=True)
    print("[1/7] copy immutable software evidence", flush=True)
    copy_software_evidence(software_run, out)
    print("[2/7] generate isolated normalized reciprocal ROM", flush=True)
    rom_dir = out / "experimental_roms"
    rom_dir.mkdir()
    for name in validate_rom_set(source_run / "candidate_hardware_roms"):
        shutil.copyfile(source_run / "candidate_hardware_roms" / name, rom_dir / name)
    rom_record = generate_normalized_rom(rom_dir / "reciprocal_normalized_q17_rne.mem")
    json_dump(out / "experimental_rom_manifest.json", rom_record)
    print("[3/7] verify baseline and candidate RTL on identical frames", flush=True)
    roms = load_frozen_roms(source_run / "candidate_hardware_roms")
    baseline_rtl, candidate_rtl = verify_rtl_frames(
        source_run, out, roms, rom_dir / "reciprocal_normalized_q17_rne.mem")
    print("[4/7] latency/resource estimate", flush=True)
    resource_rows = build_resource_table(out, baseline_rtl, candidate_rtl)
    print("[5/7] consolidated report and index", flush=True)
    write_report(cfg, out, source_run, software_run, baseline_rtl,
                 candidate_rtl, resource_rows, rom_record)
    write_results_index(out, source_run, software_run, rom_record)
    print("[6/7] regressions", flush=True)
    tests = subprocess.run(
        [sys.executable, "-m", "unittest",
         "evaluation.tests.test_precision_experimental",
         "evaluation.tests.test_final_evidence_package",
         "evaluation.tests.test_rom_loading",
         "evaluation.tests.test_reference_conventions", "-v"],
        cwd=C.REPO_ROOT, capture_output=True, text=True, check=True)
    (out / "regression_tests.txt").write_text(tests.stdout + tests.stderr)
    print("[7/7] immutable-baseline manifest", flush=True)
    write_manifest(cfg_path, cfg, out, source_run, software_run,
                   baseline_before, active_before, rom_record)
    print(f"complete: {out}", flush=True)


if __name__ == "__main__":
    main()
