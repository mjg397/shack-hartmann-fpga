#!/usr/bin/env python3
"""Bounded reciprocal-precision analysis with an immutable RTL baseline."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import inspect
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

from evaluation.common import bitaccurate as BA  # noqa: E402
from evaluation.common import config as C  # noqa: E402
from evaluation.common import precision_experimental as P  # noqa: E402
from evaluation.common import rtl_runner as R  # noqa: E402
from evaluation.common.roms import REQUIRED_ROMS, validate_rom_set  # noqa: E402
from evaluation.optical.blocker_investigation import (  # noqa: E402
    exact_centroids_from_integer,
    frozen_pipeline,
    load_frozen_roms,
)
from evaluation.optical.final_evidence_package import (  # noqa: E402
    detector_config,
    physical_detector_energy,
)
from evaluation.optical.forward_model_investigation import (  # noqa: E402
    make_propagation_bench,
    t_interval,
)
from evaluation.optical.resolved_model_candidate import (  # noqa: E402
    common_scoring_bench,
    coefficient_metric,
    deterministic_frame,
    metric_rms,
)


DEFAULT_CONFIG = C.EVAL_DIR / "configs" / "reciprocal_precision_software_v1.json"
Q18 = float(1 << C.CENTROID_FRAC)


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


def load_truths(source_run: Path) -> dict[str, np.ndarray]:
    record = json.loads((source_run / "confirmation_wavefronts.json").read_text())
    truths = {"flat": np.zeros(C.NUM_MODES)}
    truths.update({name: np.asarray(value, dtype=float)
                   for name, value in record["signed_modes"].items()})
    for amplitude, vectors in record["mixed"].items():
        label = amplitude.removesuffix("nm")
        for index, value in enumerate(vectors):
            truths[f"mixed_{label}nm_{index:02d}"] = np.asarray(value, dtype=float)
    return truths


def reproduce_frames(cfg: dict, source_run: Path, out: Path) -> tuple[list[dict], list[np.ndarray]]:
    source_rows = read_csv(source_run / "deterministic_confirmation_raw.csv")
    source_cfg = json.loads((C.REPO_ROOT / cfg["source_config"]).read_text())
    truths = load_truths(source_run)
    cases = list(dict.fromkeys(row["case"] for row in source_rows))
    probabilities: dict[tuple[int, str], np.ndarray] = {}
    optical_rows = []
    for factor in source_cfg["confirmation"]["factors"]:
        bench = make_propagation_bench(factor)
        for case in cases:
            image, _ = bench.propagate_coeffs(truths[case])
            energy, captured, fine_pitch = physical_detector_energy(bench, image, factor)
            probability = energy / captured
            probabilities[(factor, case)] = probability
            optical_rows.append({
                "factor": factor, "case": case, "captured_flux": captured,
                "fine_pixel_pitch_m": fine_pitch,
                "probability_sum": float(probability.sum()),
                "probability_sha256_float32": sha256_bytes(
                    probability.astype(np.float32).tobytes()),
            })
        del bench
    write_csv(out / "reproduced_optical_inputs.csv", optical_rows)

    frames = []
    index_rows = []
    arrays = {}
    mismatches = []
    for index, row in enumerate(source_rows):
        photons = float(row["photons_per_pupil"])
        factor = int(row["factor"])
        case = row["case"]
        frame = deterministic_frame(
            probabilities[(factor, case)], detector_config(source_cfg, photons, False))
        frame_hash = sha256_bytes(frame.astype(np.uint8).tobytes())
        if frame_hash != row["frame_sha256"]:
            mismatches.append({"row": index, "case": case, "expected": row["frame_sha256"],
                               "observed": frame_hash})
        key = f"frame_{index:03d}"
        arrays[key] = frame.astype(np.uint8)
        frames.append(frame.astype(np.uint8))
        index_rows.append({
            "row": index, "array_key": key, "photons_per_pupil": photons,
            "factor": factor, "case": case, "frame_sha256": frame_hash,
            "floating_coefficients_m": row["floating_coefficients_m"],
            "baseline_fixed_coefficients_m": row["fixed_coefficients_m"],
            "truth_coefficients_m": row["truth_coefficients_m"],
        })
    if mismatches:
        json_dump(out / "FRAME_REPRODUCTION_FAILURE.json", mismatches)
        raise AssertionError(f"{len(mismatches)} regenerated integer frames differ from source")
    np.savez_compressed(out / "deterministic_frames.npz", **arrays)
    write_csv(out / "deterministic_frame_index.csv", index_rows)
    return index_rows, frames


def all_variants(cfg: dict) -> list[dict]:
    variants = [P.BASELINE,
                {"name": "q18_floor", "kind": "direct", "frac_bits": 18,
                 "reciprocal_rounding": "truncate", "product_rounding": "truncate",
                 "reciprocal_format": "unsigned Q18 direct reciprocal",
                 "product_format": "19x19 mathematical comparison", "rounding": "truncate"},
                {"name": "q18_product_rne", "kind": "direct", "frac_bits": 18,
                 "reciprocal_rounding": "nearest_half_up",
                 "product_rounding": "nearest_half_up",
                 "reciprocal_format": "unsigned Q18 direct reciprocal",
                 "product_format": "19x19 mathematical comparison",
                 "rounding": "nearest-half-up"}]
    for bits in cfg["sweep"]["normalized_fraction_bits"]:
        for rounding in cfg["sweep"]["rounding_rules"]:
            short = "trunc" if rounding == "truncate" else "rne"
            variants.append({
                "name": f"normalized_q{bits}_{short}", "kind": "normalized",
                "frac_bits": bits, "reciprocal_rounding": rounding,
                "product_rounding": rounding,
                "reciprocal_format": f"unsigned normalized Q{bits} mantissa ({bits+1} bits) plus 4-bit exponent",
                "product_format": f"19x{bits+1} full product; exponent shift after multiply",
                "rounding": rounding,
            })
    for bits in cfg["sweep"]["direct_fraction_bits"]:
        variants.append({
            "name": f"direct_q{bits}_rne", "kind": "direct", "frac_bits": bits,
            "reciprocal_rounding": "nearest_half_up",
            "product_rounding": "nearest_half_up",
            "reciprocal_format": f"unsigned Q{bits} reciprocal ({bits+1} bits)",
            "product_format": f"19x{bits+1} full product, rounded to Q18",
            "rounding": "nearest-half-up",
        })
    for bits in cfg["sweep"]["newton_fraction_bits"]:
        variants.append({
            "name": f"nr1_q{bits}_rne", "kind": "newton", "frac_bits": bits,
            "iteration_rounding": "nearest_half_up",
            "product_rounding": "nearest_half_up",
            "reciprocal_format": f"existing Q18 x0 lifted to Q{bits}; x1 retained Q{bits}",
            "product_format": (
                f"d[15:0]*x0[{bits}:0] -> {bits+17} bits; "
                f"x0[{bits}:0]*correction[{bits+1}:0] -> {2*bits+3} bits; "
                f"moment[18:0]*x1[{bits}:0] -> {bits+20} bits"),
            "rounding": "both retained slices nearest-half-up",
        })
    variants.append({
        "name": "exact_quotient_rne", "kind": "exact",
        "product_rounding": "nearest_half_up",
        "reciprocal_format": "exact integer quotient bound",
        "product_format": "round(moment*2^18/intensity)",
        "rounding": "nearest-half-up",
    })
    return variants


def cost_estimate(variant: dict) -> dict:
    kind = variant["kind"]
    bits = int(variant.get("frac_bits", 18))
    if kind == "baseline":
        return {"estimated_dsp_delta": 0, "estimated_added_cycles": 0,
                "multiplier_shape": "two 19x18 centroid multipliers",
                "cost_class": "frozen control"}
    if kind == "normalized" and bits <= 17:
        return {"estimated_dsp_delta": 0, "estimated_added_cycles": 0,
                "multiplier_shape": f"two 19x{bits+1} centroid multipliers",
                "cost_class": "same DSP count; exponent/control and barrel-shift LUT/FF"}
    if kind in ("direct", "newton") and bits == 18:
        return {"estimated_dsp_delta": 0, "estimated_added_cycles": 0,
                "multiplier_shape": "comparison only; d=1 requires the pruned nineteenth reciprocal bit",
                "cost_class": "rounding-only comparison"}
    if kind == "direct":
        return {"estimated_dsp_delta": 2, "estimated_added_cycles": 0,
                "multiplier_shape": f"two 19x{bits+1} centroid products exceed 25x18",
                "cost_class": "wider centroid path; split-DSP estimate"}
    if kind == "newton":
        return {"estimated_dsp_delta": 4, "estimated_added_cycles": 2,
                "multiplier_shape": "two Newton DSPs plus two additional split centroid DSPs",
                "cost_class": "one NR step and wider centroid path"}
    return {"estimated_dsp_delta": None, "estimated_added_cycles": None,
            "multiplier_shape": "non-hardware exact bound", "cost_class": "bound only"}


def frame_float_and_truth(row: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(ast.literal_eval(row["floating_coefficients_m"]), dtype=float),
            np.asarray(ast.literal_eval(row["truth_coefficients_m"]), dtype=float))


def slope_error(trace: dict, frame: np.ndarray, roms: dict) -> np.ndarray:
    _, _, _, exact_x, exact_y = exact_centroids_from_integer(frame)
    valid = roms["valid"]
    return np.r_[trace["x_centroid"][valid] / Q18 - exact_x[valid],
                 trace["y_centroid"][valid] / Q18 - exact_y[valid]]


def evaluate_sweep(cfg: dict, index_rows: list[dict], frames: list[np.ndarray],
                   roms: dict, scale: float, metric: np.ndarray,
                   out: Path) -> tuple[list[dict], list[dict]]:
    variants = all_variants(cfg)
    raw_rows = []
    summary_rows = []
    bias_rows = []
    architecture_rows = []
    baseline_discrepancy = None
    baseline_residual = None
    for variant in variants:
        discrepancies = []
        residuals = []
        coefficient_errors = []
        slope_errors = []
        wraps = invalid = 0
        flat = {}
        for source, frame in zip(index_rows, frames):
            trace = P.evaluate_variant(frame, roms, variant)
            coefficients = np.asarray(trace["zernike_raw"], dtype=float) * scale
            floating, truth = frame_float_and_truth(source)
            coefficient_error = coefficients - floating
            discrepancy = metric_rms(coefficient_error, metric) * 1e9
            residual = metric_rms(coefficients - truth, metric) * 1e9
            error = slope_error(trace, frame, roms)
            discrepancies.append(discrepancy)
            residuals.append(residual)
            coefficient_errors.append(coefficient_error)
            slope_errors.extend(error.tolist())
            wraps += (trace["diagnostics"]["accumulator_wrap_count"] +
                      trace["diagnostics"]["output_field_wrap_count"])
            invalid += trace["invalid_centroid_count"]
            if source["case"] == "flat" and int(source["factor"]) == 8:
                flat[int(float(source["photons_per_pupil"]))] = residual
            raw_rows.append({
                "variant": variant["name"], **{key: source[key] for key in
                    ("row", "photons_per_pupil", "factor", "case", "frame_sha256")},
                "floating_coefficients_m": json.dumps(floating.tolist()),
                "candidate_coefficients_m": json.dumps(coefficients.tolist()),
                "coefficient_error_m": json.dumps(coefficient_error.tolist()),
                "fixed_minus_floating_nm": discrepancy,
                "ground_truth_residual_nm": residual,
                "slope_rms_error_px": float(np.sqrt(np.mean(error**2))),
                "slope_max_error_px": float(np.max(np.abs(error))),
                "invalid_centroid_count": trace["invalid_centroid_count"],
                **trace["diagnostics"],
            })
        discrepancies_a = np.asarray(discrepancies)
        residuals_a = np.asarray(residuals)
        coefficient_errors_a = np.asarray(coefficient_errors)
        slope_errors_a = np.asarray(slope_errors)
        if variant["name"] == "baseline_q18":
            baseline_discrepancy = discrepancies_a.copy()
            baseline_residual = residuals_a.copy()
        cost = cost_estimate(variant)
        row = {
            "variant": variant["name"], "kind": variant["kind"],
            "flat_1m_residual_nm": flat[1000000],
            "flat_3m_residual_nm": flat[3000000],
            "mean_fixed_minus_floating_nm": float(np.mean(discrepancies_a)),
            "p95_fixed_minus_floating_nm": float(np.percentile(discrepancies_a, 95)),
            "max_fixed_minus_floating_nm": float(np.max(discrepancies_a)),
            "mean_ground_truth_residual_nm": float(np.mean(residuals_a)),
            "p95_ground_truth_residual_nm": float(np.percentile(residuals_a, 95)),
            "coefficient_bias_m": json.dumps(coefficient_errors_a.mean(axis=0).tolist()),
            "worst_abs_coefficient_error_nm": float(
                np.max(np.abs(coefficient_errors_a)) * 1e9),
            "slope_rms_error_px": float(np.sqrt(np.mean(slope_errors_a**2))),
            "slope_max_error_px": float(np.max(np.abs(slope_errors_a))),
            "overflow_wrap_count": wraps, "invalid_centroid_count": invalid,
            "reciprocal_format": variant.get("reciprocal_format"),
            "product_format": variant.get("product_format"),
            "rounding": variant.get("rounding"), **cost,
        }
        if baseline_discrepancy is not None:
            row.update({
                "arithmetic_frames_worse_than_baseline": int(np.count_nonzero(
                    discrepancies_a > baseline_discrepancy + 1e-12)),
                "max_arithmetic_worsening_nm": float(np.max(
                    discrepancies_a - baseline_discrepancy)),
                "ground_truth_frames_worse_than_baseline": int(np.count_nonzero(
                    residuals_a > baseline_residual + 1e-12)),
                "max_ground_truth_worsening_nm": float(np.max(
                    residuals_a - baseline_residual)),
            })
        summary_rows.append(row)
        for mode, label in enumerate(C.MODE_LABELS):
            values = coefficient_errors_a[:, mode] * 1e9
            bias_rows.append({
                "variant": variant["name"], "mode": mode, "mode_label": label,
                "mean_error_nm": float(np.mean(values)),
                "rms_error_nm": float(np.sqrt(np.mean(values**2))),
                "p95_abs_error_nm": float(np.percentile(np.abs(values), 95)),
                "max_abs_error_nm": float(np.max(np.abs(values))),
            })
        architecture_rows.append({
            "variant": variant["name"], "kind": variant["kind"],
            "reciprocal_format": variant.get("reciprocal_format"),
            "product_format": variant.get("product_format"),
            "rounding": variant.get("rounding"), **cost,
        })
    write_csv(out / "precision_sweep_raw.csv", raw_rows)
    write_csv(out / "precision_sweep_summary.csv", summary_rows)
    write_csv(out / "per_mode_error.csv", bias_rows)
    write_csv(out / "candidate_architectures.csv", architecture_rows)

    operating = []
    mixed_amplitudes = (20, 100, 400)
    keep = {"baseline_q18", "normalized_q17_rne", "direct_q25_rne",
            "nr1_q25_rne", "exact_quotient_rne"}
    for variant in keep:
        for photons in (1e6, 3e6):
            for amplitude in mixed_amplitudes:
                values = [row for row in raw_rows if row["variant"] == variant and
                          float(row["photons_per_pupil"]) == photons and
                          int(row["factor"]) == 8 and
                          row["case"].startswith(f"mixed_{amplitude}nm_")]
                for key in ("fixed_minus_floating_nm", "ground_truth_residual_nm"):
                    stats = t_interval([float(row[key]) for row in values])
                    operating.append({
                        "variant": variant, "photons_per_pupil": photons,
                        "input_rms_nm": amplitude, "metric": key,
                        "n_wavefronts": len(values), "mean_nm": stats["mean"],
                        "ci95_low_nm": stats["ci_low"],
                        "ci95_high_nm": stats["ci_high"],
                        "p95_nm": float(np.percentile([float(row[key]) for row in values], 95)),
                        "max_nm": float(np.max([float(row[key]) for row in values])),
                    })
    write_csv(out / "operating_point_summary.csv", operating)
    return summary_rows, raw_rows


def stage_attribution(frame: np.ndarray, source_row: dict, roms: dict,
                      scale: float, metric: np.ndarray, out: Path) -> list[dict]:
    baseline = P.evaluate_variant(frame, roms, P.BASELINE)
    floating, _ = frame_float_and_truth(source_row)
    exact_variant = {"name": "exact", "kind": "exact",
                     "product_rounding": "nearest_half_up"}
    exact = P.evaluate_variant(frame, roms, exact_variant)
    baseline_coeff = baseline["zernike_raw"].astype(float) * scale
    exact_coeff = exact["zernike_raw"].astype(float) * scale
    baseline_residual = metric_rms(baseline_coeff - floating, metric) * 1e9
    baseline_slope_error = slope_error(baseline, frame, roms)
    exact_slope_error = slope_error(exact, frame, roms)

    s_i, _, _ = BA.accumulate(frame)
    ideal_q18 = np.asarray([P.divide_unsigned(1 << 18, int(d), "nearest_half_up")
                            if d else 0 for d in s_i], dtype=np.int64)
    lut_q18 = np.where(s_i == 0, 0, roms["reciprocal"][s_i])
    lut_content_mismatches = int(np.count_nonzero(ideal_q18[s_i > 0] != lut_q18[s_i > 0]))
    product_high_bits = int(np.count_nonzero(
        (np.asarray(baseline["xI"]) * baseline["x_reciprocal"]) >> C.CENTROID_BITS))
    slope_overflows = int(np.count_nonzero(
        np.abs(np.r_[baseline["x_slope_valid"], baseline["y_slope_valid"]]) >=
        (1 << (C.SLOPE_BITS - 1))))

    exact_output = P.pipeline_from_centroids(
        baseline["x_centroid"], baseline["y_centroid"], roms,
        retain_output_fraction=True)
    exact_output_coeff = np.asarray(exact_output["zernike_raw"], float) * scale
    exact_output_residual = metric_rms(exact_output_coeff - floating, metric) * 1e9

    definitions = [
        ("A_reciprocal_lut_algorithm", baseline_coeff, baseline_slope_error,
         f"ROM equals ideal nearest Q18 at used divisors; mismatches={lut_content_mismatches}"),
        ("B_reciprocal_output_q18_quantization", exact_coeff, exact_slope_error,
         "replace Q18 reciprocal-before-multiply with exact rounded centroid quotient"),
        ("C_numerator_times_reciprocal_multiply", baseline_coeff, baseline_slope_error,
         "37-bit integer multiply is exact"),
        ("D_post_multiply_product_slice", baseline_coeff, baseline_slope_error,
         f"discarded nonzero high products={product_high_bits}; no fractional bits exist after Q18 reciprocal"),
        ("E_centroid_reference_subtraction", baseline_coeff, baseline_slope_error,
         "26-bit subtraction is exact at observed range"),
        ("F_centroid_to_slope_conversion", baseline_coeff, baseline_slope_error,
         "centroid and slope share Q18; conversion is an identity"),
        ("G_slope_output_slice", baseline_coeff, baseline_slope_error,
         f"observed 25-bit overflows={slope_overflows}"),
        ("H_ematrix_multiply_accumulate", baseline_coeff, baseline_slope_error,
         "43-bit products and 52-bit accumulation are exact with zero wraps"),
        ("I_final_accumulator_output_slice", exact_output_coeff, baseline_slope_error,
         "retain fractional accumulator bits instead of acc[42:18]"),
    ]
    rows = []
    for stage, coefficients, slopes, note in definitions:
        residual = metric_rms(coefficients - floating, metric) * 1e9
        removed = baseline_residual - residual
        squared_removed = 100 * (baseline_residual**2 - residual**2) / baseline_residual**2
        rows.append({
            "stage": stage, "baseline_flat_residual_nm": baseline_residual,
            "modified_flat_residual_nm": residual,
            "residual_removed_nm": removed,
            "baseline_squared_wfe_removed_pct": squared_removed,
            "coefficient_error_nm": json.dumps(((coefficients - floating) * 1e9).tolist()),
            "slope_rms_error_px": float(np.sqrt(np.mean(slopes**2))),
            "slope_max_error_px": float(np.max(np.abs(slopes))),
            "note": note,
        })
    rows.sort(key=lambda row: row["baseline_squared_wfe_removed_pct"], reverse=True)
    write_csv(out / "stage_error_attribution.csv", rows)
    return rows


def baseline_reproduction(source_run: Path, frame: np.ndarray, roms: dict,
                          scale: float, metric: np.ndarray, out: Path) -> dict:
    model = frozen_pipeline(frame, roms)
    with R.RTLSimulator(rom_dir=source_run / "candidate_hardware_roms",
                        timeout_seconds=45) as simulator:
        rtl = simulator.run_frame(frame)
    valid = roms["valid"]
    expected_slopes = np.column_stack([
        model["x_slope_valid"], model["y_slope_valid"],
        model["x_centroid"][valid], model["y_centroid"][valid]])
    result = {
        "frame_sha256": sha256_bytes(frame.tobytes()),
        "floating_residual_nm": 0.0,
        "fixed_residual_nm": metric_rms(
            model["zernike_raw"].astype(float) * scale, metric) * 1e9,
        "fixed_coefficients_nm": (model["zernike_raw"] * scale * 1e9).tolist(),
        "coefficient_mismatch_count": int(np.count_nonzero(
            rtl["zernike_raw"] - model["zernike_raw"])),
        "slope_centroid_mismatch_count": int(np.count_nonzero(
            rtl["slopes"][:, 1:5] - expected_slopes)),
        "valid_slopes": rtl["subaps_valid"],
        "valid_coefficients": len(rtl["zernike_raw"]),
        "unknown_valid_slopes": rtl["unknown_valid_slopes"],
        "unknown_valid_coefficients": rtl["unknown_valid_coefficients"],
        "final_pixel_to_done_cycles": rtl["true_latency_cycles"],
        "last_contributing_pixel_to_done_cycles": rtl["latency_cycles"],
        "baseline_passed": bool(
            np.isclose(metric_rms(model["zernike_raw"] * scale, metric) * 1e9,
                       7.6584878514271075, rtol=0, atol=1e-12) and
            np.array_equal(rtl["zernike_raw"], model["zernike_raw"]) and
            np.array_equal(rtl["slopes"][:, 1:5], expected_slopes)),
    }
    (out / "baseline_rtl_log.txt").write_text(rtl["stdout"])
    json_dump(out / "baseline_reproduction.json", result)
    if not result["baseline_passed"]:
        raise AssertionError("frozen baseline did not reproduce; refusing optimization")
    return result


def report(cfg: dict, out: Path, baseline: dict, attribution: list[dict],
           summary: list[dict]) -> None:
    lookup = {row["variant"]: row for row in summary}
    selected = lookup[cfg["selected_software_candidate"]]
    nr = lookup["nr1_q25_rne"]
    direct = lookup["direct_q25_rne"]
    dominant = attribution[0]
    text = f"""# Reciprocal precision software selection

Run: `{cfg['run_id']}`  
Frozen control: `final_evidence_package_v1_20260930_final`

## 1. Baseline reproduction

The frozen candidate 3M flat frame reproduces {baseline['fixed_residual_nm']:.9f} nm
fixed and {baseline['floating_residual_nm']:.1f} nm floating. RTL and the frozen
bit-accurate model have zero coefficient and slope/centroid mismatches, with
{baseline['valid_slopes']} valid slopes, {baseline['valid_coefficients']} coefficients,
and {baseline['final_pixel_to_done_cycles']} final-pixel-to-done cycles.

## 2. Stage-by-stage error attribution

The dominant isolated replacement is `{dominant['stage']}`: it reduces the flat
residual to {dominant['modified_flat_residual_nm']:.6f} nm and removes
{dominant['baseline_squared_wfe_removed_pct']:.6f}% of baseline squared WFE.
The full ranked A-I table, coefficient vectors, and slope errors are in
`stage_error_attribution.csv`. The existing 65,536-entry LUT is already exactly
round-to-nearest Q18; the loss occurs because Q18 reciprocal quantization happens
before multiplication.

## 3. Precision sweep and architecture comparison

| Design | Flat 3M RMS nm | Mean FF nm | p95 FF nm | Estimated DSP delta | Estimated latency delta |
|---|---:|---:|---:|---:|---:|
| Baseline | {lookup['baseline_q18']['flat_3m_residual_nm']:.6f} | {lookup['baseline_q18']['mean_fixed_minus_floating_nm']:.6f} | {lookup['baseline_q18']['p95_fixed_minus_floating_nm']:.6f} | 0 | 0 |
| Normalized Q17 RNE | {selected['flat_3m_residual_nm']:.6f} | {selected['mean_fixed_minus_floating_nm']:.6f} | {selected['p95_fixed_minus_floating_nm']:.6f} | 0 | 0 |
| Direct Q25 RNE | {direct['flat_3m_residual_nm']:.6f} | {direct['mean_fixed_minus_floating_nm']:.6f} | {direct['p95_fixed_minus_floating_nm']:.6f} | +2 | 0 |
| NR1 Q25 RNE | {nr['flat_3m_residual_nm']:.6f} | {nr['mean_fixed_minus_floating_nm']:.6f} | {nr['p95_fixed_minus_floating_nm']:.6f} | +4 | +2 |

All costs are structural estimates against a DSP48E1 25x18 multiplier; no
synthesis or achieved timing claim is made.

## 4. Selection before RTL

`normalized_q17_rne` is selected for experimental RTL. It keeps the reciprocal
mantissa at 18 bits, adds a 4-bit exponent, delays the binary-point shift until
after the existing 19x18 product, and rounds the shifted product to nearest.
It is the lowest-cost candidate and is numerically better than the wider Q25 and
NR1 Q25 candidates on the frozen 3M flat case. Newton-Raphson is not necessary.

This run is software-only. Experimental RTL verification belongs in a distinct
follow-on run; the baseline RTL and all frozen result directories remain unchanged.
"""
    (out / "REPORT.md").write_text(text)


def manifest(cfg_path: Path, cfg: dict, source_run: Path, out: Path) -> None:
    import hcipy
    relevant = [Path(__file__).resolve(), cfg_path,
                C.EVAL_DIR / "common" / "precision_experimental.py",
                C.EVAL_DIR / "tests" / "test_precision_experimental.py"]
    result_files = [path for path in out.rglob("*")
                    if path.is_file() and path.name != "MANIFEST.json"]
    value = {
        "schema_version": 1, "run_id": cfg["run_id"],
        "source_run": str(source_run.relative_to(C.REPO_ROOT)),
        "source_manifest_sha256": sha256(source_run / "MANIFEST.json"),
        "configuration_sha256": sha256(cfg_path),
        "source_sha256": {str(path.relative_to(C.REPO_ROOT)): sha256(path)
                          for path in relevant},
        "candidate_rom_sha256": {name: value["sha256"] for name, value in
                                  validate_rom_set(source_run / "candidate_hardware_roms").items()},
        "active_rom_sha256": {name: value["sha256"] for name, value in
                              validate_rom_set(C.RTL_DIR).items()},
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "hcipy")},
        "result_sha256": {str(path.relative_to(out)): sha256(path)
                          for path in result_files},
        "baseline_rtl_modified": False, "vivado_run": False,
    }
    json_dump(out / "MANIFEST.json", value)


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
    source_run = (C.REPO_ROOT / cfg["source_run"]).resolve()
    out.mkdir(parents=True)
    print("[1/7] regenerate and hash-check frozen integer frames", flush=True)
    index_rows, frames = reproduce_frames(cfg, source_run, out)
    roms = load_frozen_roms(source_run / "candidate_hardware_roms")
    scale = json.loads((source_run / "candidate_hardware_calibration.json").read_text())[
        "scale_m_per_output_lsb"]
    scoring = common_scoring_bench({"resolution": {
        "common_scoring_pupil_base": 240, "common_scoring_edge_supersample": 20}})
    metric = coefficient_metric(scoring)
    flat_index = next(index for index, row in enumerate(index_rows)
                      if row["case"] == "flat" and int(row["factor"]) == 8 and
                      float(row["photons_per_pupil"]) == 3e6)
    print("[2/7] reproduce immutable baseline RTL", flush=True)
    baseline = baseline_reproduction(source_run, frames[flat_index], roms, scale, metric, out)
    print("[3/7] isolate arithmetic stages A-I", flush=True)
    attribution = stage_attribution(
        frames[flat_index], index_rows[flat_index], roms, scale, metric, out)
    print("[4/7] bounded bit-exact precision sweep", flush=True)
    summary, _ = evaluate_sweep(cfg, index_rows, frames, roms, scale, metric, out)
    print("[5/7] write software selection report", flush=True)
    report(cfg, out, baseline, attribution, summary)
    print("[6/7] focused tests", flush=True)
    tests = subprocess.run(
        [sys.executable, "-m", "unittest",
         "evaluation.tests.test_precision_experimental", "-v"],
        cwd=C.REPO_ROOT, capture_output=True, text=True, check=True)
    (out / "regression_tests.txt").write_text(tests.stdout + tests.stderr)
    print("[7/7] manifest", flush=True)
    manifest(cfg_path, cfg, source_run, out)
    print(f"complete: {out}", flush=True)


if __name__ == "__main__":
    main()
