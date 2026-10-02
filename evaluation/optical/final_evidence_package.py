#!/usr/bin/env python3
"""Final bounded simulation-only evidence package for the SHWFS project.

The script audits the existing factor-8/factor-9 statistics, generates an
isolated hardware-format calibration from the already-frozen factor-8 floating
calibration, runs one fresh bounded confirmation, and verifies the unchanged
RTL on selected candidate frames. Active ROMs are never written.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/shwfs-matplotlib")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import bitaccurate as BA  # noqa: E402
from evaluation.common import config as C  # noqa: E402
from evaluation.common import detector as D  # noqa: E402
from evaluation.common import optics as O  # noqa: E402
from evaluation.common import rtl_runner as R  # noqa: E402
from evaluation.common.noise_coupling import pairwise_poisson_gaussian_frames  # noqa: E402
from evaluation.common.roms import REQUIRED_ROMS, validate_rom_set  # noqa: E402
from evaluation.optical.blocker_investigation import (  # noqa: E402
    frozen_pipeline,
    load_frozen_roms,
    rms_coefficients,
)
from evaluation.optical.forward_model_investigation import (  # noqa: E402
    centroid_vector,
    make_propagation_bench,
    physical_detector_energy,
    t_interval,
    tile_moments,
)
from evaluation.optical.resolved_model_candidate import (  # noqa: E402
    common_scoring_bench,
    coefficient_metric,
    deterministic_frame,
    detector_config,
    independent_cross_batch_squared,
    metric_rms,
    normalized_coefficients,
)


DEFAULT_CONFIG = C.EVAL_DIR / "configs" / "final_evidence_package_v1.json"
Q18 = float(1 << C.CENTROID_FRAC)
PATH_NAMES = ("floating", "dequantized", "fixed")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result table: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_mem(path: Path, values: np.ndarray, encoded_bits: int) -> None:
    digits = (encoded_bits + 3) // 4
    mask = (1 << encoded_bits) - 1
    path.write_text("".join(
        f"{int(value) & mask:0{digits}X}\n" for value in np.asarray(values).ravel()))


def build_candidate_roms(cfg: dict, candidate_run: Path,
                         manuscript_run: Path, out: Path) -> dict:
    source = candidate_run / "candidate_calibration"
    e_float = np.load(source / "e_matrix_m_per_slope_px.npy")
    ref_exact_x = np.load(source / "reference_exact_x_px.npy")
    ref_exact_y = np.load(source / "reference_exact_y_px.npy")
    ref_qx = np.rint(ref_exact_x * Q18).astype(np.int64)
    ref_qy = np.rint(ref_exact_y * Q18).astype(np.int64)
    limit = (1 << (C.EMATRIX_BITS - 1)) - 1
    scale = float(np.max(np.abs(e_float)) / limit)
    e_q = np.rint(e_float / scale).astype(np.int64)
    if np.any(np.abs(e_q) > limit):
        raise AssertionError("derived candidate E scale saturated an 18-bit word")
    candidate_dir = out / "candidate_hardware_roms"
    candidate_dir.mkdir()
    frozen_roms = manuscript_run / "frozen_inputs" / "roms"
    for name in ("reciprocal_u16_q18.mem", "subaperture_bitmap.mem"):
        shutil.copyfile(frozen_roms / name, candidate_dir / name)
    write_mem(candidate_dir / "slopes_ref_x.mem", ref_qx,
              cfg["candidate_hardware"]["reference_encoding_bits"])
    write_mem(candidate_dir / "slopes_ref_y.mem", ref_qy,
              cfg["candidate_hardware"]["reference_encoding_bits"])
    n = C.NUM_SUBAPS_VALID
    write_mem(candidate_dir / "e_matrix_x.mem", e_q[:, :n],
              cfg["candidate_hardware"]["ematrix_encoding_bits"])
    write_mem(candidate_dir / "e_matrix_y.mem", e_q[:, n:],
              cfg["candidate_hardware"]["ematrix_encoding_bits"])
    rom_manifest = validate_rom_set(candidate_dir)
    roms = load_frozen_roms(candidate_dir)
    if not np.array_equal(np.hstack([roms["ex"], roms["ey"]]), e_q):
        raise AssertionError("candidate E ROM reload differs from generated integers")
    if not np.array_equal(roms["ref_x"], ref_qx):
        raise AssertionError("candidate x-reference ROM reload differs")
    if not np.array_equal(roms["ref_y"], ref_qy):
        raise AssertionError("candidate y-reference ROM reload differs")
    e_dequantized = np.hstack([roms["ex"], roms["ey"]]).astype(float) * scale
    quantization = e_dequantized - e_float
    record = {
        "source_candidate_run": str(candidate_run),
        "source_floating_e_sha256": sha256(source / "e_matrix_m_per_slope_px.npy"),
        "source_reference_exact_x_sha256": sha256(source / "reference_exact_x_px.npy"),
        "source_reference_exact_y_sha256": sha256(source / "reference_exact_y_px.npy"),
        "scale_m_per_output_lsb": scale,
        "scale_log2": float(np.log2(scale)),
        "scale_derivation": (
            "max(abs(E_float))/(2^17-1); E_q=round(E_float/scale); "
            "RTL output raw=(sum(E_q*slope_q18)>>18), coefficient_m=raw*scale"),
        "scale_fitted": False,
        "ematrix_shape": list(e_float.shape),
        "ematrix_integer_max_abs": int(np.max(np.abs(e_q))),
        "ematrix_saturated_words": int(np.count_nonzero(np.abs(e_q) > limit)),
        "ematrix_quantization_rms_m_per_slope_px": float(np.sqrt(np.mean(quantization**2))),
        "ematrix_quantization_max_abs_m_per_slope_px": float(np.max(np.abs(quantization))),
        "reference_rule": cfg["candidate_hardware"]["reference_rule"],
        "reciprocal_preserved_sha256": rom_manifest["reciprocal_u16_q18.mem"]["sha256"],
        "rom_manifest": rom_manifest,
    }
    json_dump(out / "candidate_hardware_calibration.json", record)
    return {
        "dir": candidate_dir, "record": record, "roms": roms,
        "e_float": e_float, "e_dequantized": e_dequantized,
        "ref_exact": np.r_[ref_exact_x[roms["valid"]], ref_exact_y[roms["valid"]]],
        "ref_q": np.r_[ref_qx[roms["valid"]] / Q18,
                         ref_qy[roms["valid"]] / Q18],
        "scale": scale,
    }


def gram_audit(scoring_bench: O.Bench, metric: np.ndarray,
               representative: list[np.ndarray], out: Path) -> dict:
    rows = []
    vectors = [np.eye(C.NUM_MODES)[index] * 100e-9
               for index in (0, 2, 5, 9)] + representative
    for index, values in enumerate(vectors):
        direct = scoring_bench.opd_rms(scoring_bench.opd_from_coeffs(values))
        gram = metric_rms(values, metric)
        rows.append({
            "vector": index,
            "coefficients_m": json.dumps(np.asarray(values).tolist()),
            "direct_aperture_weighted_piston_removed_rms_m": direct,
            "gram_rms_m": gram,
            "absolute_difference_m": abs(direct - gram),
            "relative_difference": abs(direct - gram) / direct if direct else 0.0,
        })
    write_csv(out / "gram_identity_audit.csv", rows)
    symmetry = float(np.max(np.abs(metric - metric.T)))
    eigenvalues = np.linalg.eigvalsh(metric)
    result = {
        "definition": "G_ij=<mode_i-piston_i, mode_j-piston_j> on the common aperture-weighted pupil",
        "common_scoring_pupil": asdict(scoring_bench.cfg),
        "symmetry_max_abs": symmetry,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "representative_max_absolute_identity_error_m": max(
            row["absolute_difference_m"] for row in rows),
        "representative_max_relative_identity_error": max(
            row["relative_difference"] for row in rows),
        "passed": bool(symmetry < 1e-14 and eigenvalues.min() > 0 and
                       max(row["absolute_difference_m"] for row in rows) < 1e-18),
    }
    json_dump(out / "gram_metric_audit.json", result)
    if not result["passed"]:
        raise AssertionError("Gram metric failed direct wavefront identity audit")
    return result


def audit_existing_metrics(cfg: dict, candidate_run: Path, out: Path) -> dict:
    source = json.loads((candidate_run / "noisy_candidate_summary.json").read_text())
    rows = []
    for row in source["rows"]:
        margin = float(row["equivalence_margin_nm"])
        margin_sq = margin**2
        residual_pass = bool(row["residual_pass"])
        expected_pass = bool(row["expected_reconstruction_squared_pass"])
        deterministic_pass = bool(row["deterministic_expected_frame_pass"])
        deterministic_upper = float(
            row["deterministic_expected_frame_difference_ci95_high_nm"])
        relative_limited = margin < cfg["original_working_criterion"][
            "absolute_tolerance_nm"]
        if row["cell_pass"]:
            failure_class = "original_combined_pass"
        elif deterministic_upper > cfg["original_working_criterion"][
                "absolute_tolerance_nm"]:
            failure_class = "substantial_absolute_deterministic_change"
        elif relative_limited:
            failure_class = "relative_margin_driven"
        else:
            failure_class = "other_original_criterion_failure"
        rows.append({
            "condition": row["condition"],
            "amplitude_nm_rms": row["amplitude_nm_rms"],
            "photons_per_pupil": row["photons_per_pupil"],
            "independent_unit": row["independent_unit"],
            "A_mean_ground_truth_residual_change_nm": row["mean_residual_difference_nm"],
            "A_ci95_low_nm": row["mean_residual_ci95_low_nm"],
            "A_ci95_high_nm": row["mean_residual_ci95_high_nm"],
            "A_original_margin_pass": residual_pass,
            "B_expected_noisy_change_squared_nm2":
                row["cross_batch_squared_difference_mean_nm2"],
            "B_ci95_low_nm2": row["cross_batch_squared_difference_ci95_low_nm2"],
            "B_ci95_high_nm2": row["cross_batch_squared_difference_ci95_high_nm2"],
            "B_squared_tolerance_nm2": margin_sq,
            "B_original_margin_pass": expected_pass,
            "B_negative_estimates_clipped": False,
            "C_noise_variability_change_nm": row["noise_variability_difference_mean_nm"],
            "C_ci95_low_nm": row["noise_variability_difference_ci95_low_nm"],
            "C_ci95_high_nm": row["noise_variability_difference_ci95_high_nm"],
            "D_deterministic_integer_change_nm":
                row["deterministic_expected_frame_difference_mean_nm"],
            "D_ci95_high_nm": deterministic_upper,
            "D_original_margin_pass": deterministic_pass,
            "original_equivalence_margin_nm": margin,
            "original_combined_decision": bool(row["cell_pass"]),
            "failure_class": failure_class,
        })
    write_csv(out / "existing_metric_separation.csv", rows)
    result = {
        "source_run": str(candidate_run),
        "rows": rows,
        "residual_subtests_passed": int(sum(row["A_original_margin_pass"] for row in rows)),
        "expected_noisy_subtests_passed": int(sum(row["B_original_margin_pass"] for row in rows)),
        "deterministic_subtests_passed": int(sum(row["D_original_margin_pass"] for row in rows)),
        "combined_passed": int(sum(row["original_combined_decision"] for row in rows)),
        "total": len(rows),
        "original_decisions_preserved": True,
    }
    json_dump(out / "existing_metric_audit.json", result)
    return result


def arithmetic_paths(frame: np.ndarray, calibration: dict,
                     metric: np.ndarray) -> dict:
    valid = calibration["roms"]["valid"]
    moments = tile_moments(frame, C.PIXEL_THRESHOLD)
    centroids = centroid_vector(moments, valid)
    exact_slopes = centroids - calibration["ref_exact"]
    qref_slopes = centroids - calibration["ref_q"]
    floating = calibration["e_float"] @ exact_slopes
    reference_quantized = calibration["e_float"] @ qref_slopes
    dequantized = calibration["e_dequantized"] @ qref_slopes
    trace = frozen_pipeline(frame, calibration["roms"])
    fixed = trace["zernike_raw"].astype(float) * calibration["scale"]
    components = {
        "reference_quantization": reference_quantized - floating,
        "matrix_quantization": dequantized - reference_quantized,
        "arithmetic_quantization": fixed - dequantized,
    }
    total = fixed - floating
    component_sum = sum(components.values(), np.zeros(C.NUM_MODES))
    if not np.allclose(total, component_sum, rtol=0, atol=2e-22):
        raise AssertionError("arithmetic component vectors do not sum to fixed-minus-floating")
    squared = {name: float(value @ metric @ value) for name, value in components.items()}
    cross = {}
    for first, second in combinations(components, 2):
        cross[f"2_{first}_x_{second}_m2"] = float(
            2 * components[first] @ metric @ components[second])
    total_squared = float(total @ metric @ total)
    reconstructed_squared = sum(squared.values()) + sum(cross.values())
    if not np.isclose(total_squared, reconstructed_squared, rtol=2e-12, atol=1e-32):
        raise AssertionError("Gram-weighted component squares and cross terms do not close")
    idx = calibration["roms"]["valid"]
    diagnostics = trace["diagnostics"]
    diagnostics.update({
        "invalid_centroid_count": int(np.count_nonzero(trace["sI"][idx] == 0)),
        "intensity_overflow_count": int(np.count_nonzero(trace["sI"] > C.SI_MAX)),
        "x_moment_overflow_count": int(np.count_nonzero(trace["xI"] > C.XI_MAX)),
        "y_moment_overflow_count": int(np.count_nonzero(trace["yI"] > C.XI_MAX)),
    })
    return {
        "floating": floating,
        "reference_quantized": reference_quantized,
        "dequantized": dequantized,
        "fixed": fixed,
        "trace": trace,
        "components": components,
        "component_squared_m2": squared,
        "cross_terms": cross,
        "total_discrepancy_squared_m2": total_squared,
        "diagnostics": diagnostics,
    }


def generate_truths(cfg: dict, scoring_bench: O.Bench) -> tuple[dict, dict]:
    mixed = {}
    record = {"seeds": {}, "mixed": {}, "signed_modes": {}}
    for amplitude in cfg["confirmation"]["wavefront_rms_nm"]:
        seed = cfg["seeds"][f"mixed_{int(amplitude)}nm"]
        rng = np.random.default_rng(seed)
        values = [normalized_coefficients(scoring_bench, rng.normal(size=C.NUM_MODES),
                                          amplitude)
                  for _ in range(cfg["confirmation"]["mixed_wavefronts_per_amplitude"])]
        mixed[float(amplitude)] = values
        record["seeds"][f"mixed_{int(amplitude)}nm"] = seed
        record["mixed"][f"{int(amplitude)}nm"] = [value.tolist() for value in values]
    signed_modes = {}
    amplitude_m = cfg["confirmation"]["signed_mode_amplitude_nm"] * 1e-9
    for mode in cfg["confirmation"]["signed_mode_indices"]:
        for sign in (-1, 1):
            values = np.zeros(C.NUM_MODES)
            values[int(mode)] = sign * amplitude_m
            name = f"mode_{mode}_{'plus' if sign > 0 else 'minus'}"
            signed_modes[name] = values
            record["signed_modes"][name] = values.tolist()
    record["common_scoring_pupil"] = asdict(scoring_bench.cfg)
    return {"mixed": mixed, "signed_modes": signed_modes,
            "flat": np.zeros(C.NUM_MODES)}, record


def propagate_confirmation(cfg: dict, truths: dict, out: Path) -> tuple[dict, list[dict]]:
    cases = {"flat": truths["flat"], **truths["signed_modes"]}
    for amplitude, vectors in truths["mixed"].items():
        for index, values in enumerate(vectors):
            cases[f"mixed_{int(amplitude)}nm_{index:02d}"] = values
    probabilities = {factor: {} for factor in cfg["confirmation"]["factors"]}
    rows = []
    arrays = {}
    for factor in cfg["confirmation"]["factors"]:
        bench = make_propagation_bench(factor)
        for name, truth in cases.items():
            image, _ = bench.propagate_coeffs(truth)
            energy, captured, fine_pitch = physical_detector_energy(
                bench, image, factor)
            probability = energy / captured
            probabilities[factor][name] = probability
            arrays[f"factor{factor}__{name}"] = probability.astype(np.float32)
            rows.append({
                "factor": factor, "case": name,
                "truth_coefficients_m": json.dumps(truth.tolist()),
                "captured_flux": captured,
                "fine_pixel_pitch_m": fine_pitch,
                "physical_detector_pixels": C.FRAME_W * C.FRAME_H,
                "probability_sum": float(probability.sum()),
                "probability_sha256_float32": sha256_bytes(
                    probability.astype(np.float32).tobytes()),
            })
        del bench
    np.savez_compressed(out / "confirmation_optical_probabilities.npz", **arrays)
    write_csv(out / "confirmation_optical_inputs.csv", rows)
    return probabilities, rows


def truth_lookup(truths: dict) -> dict[str, np.ndarray]:
    result = {"flat": truths["flat"], **truths["signed_modes"]}
    for amplitude, vectors in truths["mixed"].items():
        for index, values in enumerate(vectors):
            result[f"mixed_{int(amplitude)}nm_{index:02d}"] = values
    return result


def diagnostic_columns(diagnostics: dict) -> dict:
    return {key: int(diagnostics.get(key, 0)) for key in (
        "invalid_centroid_count", "intensity_overflow_count",
        "x_moment_overflow_count", "y_moment_overflow_count",
        "accumulator_wrap_count", "output_field_wrap_count")}


def run_deterministic_confirmation(cfg: dict, out: Path, truths: dict,
                                   probabilities: dict, calibration: dict,
                                   metric: np.ndarray) -> dict:
    all_truths = truth_lookup(truths)
    rows = []
    selected_frames = {}
    path_coefficients: dict[tuple, np.ndarray] = {}
    for photons in cfg["confirmation"]["photon_levels"]:
        det = detector_config(cfg, photons, False)
        for factor in cfg["confirmation"]["factors"]:
            for case, truth in all_truths.items():
                probability = probabilities[factor][case]
                frame = deterministic_frame(probability, det)
                expected_e = probability * photons * det.quantum_efficiency + det.background_e
                paths = arithmetic_paths(frame, calibration, metric)
                path_coefficients[(photons, factor, case)] = np.stack(
                    [paths[name] for name in PATH_NAMES])
                row = {
                    "photons_per_pupil": photons, "factor": factor, "case": case,
                    "truth_coefficients_m": json.dumps(truth.tolist()),
                    "frame_sha256": sha256_bytes(frame.astype(np.uint8).tobytes()),
                    "saturated_pixels": int(np.count_nonzero(
                        expected_e > det.full_well_e)),
                    "adc_clipped_pixels": int(np.count_nonzero(
                        np.rint(np.clip(expected_e, 0, det.full_well_e) /
                                det.gain_e_per_adu + det.offset_adu) > det.adc_max)),
                    **diagnostic_columns(paths["diagnostics"]),
                }
                for name in PATH_NAMES:
                    row[f"{name}_coefficients_m"] = json.dumps(paths[name].tolist())
                    row[f"{name}_ground_truth_residual_nm"] = metric_rms(
                        paths[name] - truth, metric) * 1e9
                row["fixed_minus_floating_nm"] = metric_rms(
                    paths["fixed"] - paths["floating"], metric) * 1e9
                row["dequantized_minus_floating_nm"] = metric_rms(
                    paths["dequantized"] - paths["floating"], metric) * 1e9
                for name, value in paths["components"].items():
                    row[f"{name}_nm"] = metric_rms(value, metric) * 1e9
                for name, value in paths["component_squared_m2"].items():
                    row[f"{name}_squared_nm2"] = value * 1e18
                for name, value in paths["cross_terms"].items():
                    row[name.replace("_m2", "_nm2")] = value * 1e18
                row["total_discrepancy_squared_nm2"] = (
                    paths["total_discrepancy_squared_m2"] * 1e18)
                rows.append(row)
                if factor == 8 and case == "flat":
                    selected_frames[f"f8_flat_{int(photons/1e6)}M"] = frame
                if factor == 8 and photons == 3e6 and case.startswith("mode_"):
                    selected_frames[f"f8_{case}_3M"] = frame
                if (factor == 8 and case.endswith("_00") and
                        case.startswith("mixed_")):
                    selected_frames[f"f8_{case}_{int(photons/1e6)}M"] = frame
                if factor == 9 and photons == 3e6 and case == "mixed_100nm_00":
                    selected_frames["f9_mixed_100nm_00_3M"] = frame
    write_csv(out / "deterministic_confirmation_raw.csv", rows)

    summary_rows = []
    sensitivity_rows = []
    for photons in cfg["confirmation"]["photon_levels"]:
        for amplitude in cfg["confirmation"]["wavefront_rms_nm"]:
            case_names = [f"mixed_{int(amplitude)}nm_{index:02d}"
                          for index in range(cfg["confirmation"][
                              "mixed_wavefronts_per_amplitude"])]
            for factor in cfg["confirmation"]["factors"]:
                for path_index, path in enumerate(PATH_NAMES):
                    residuals = []
                    errors = []
                    discrepancies = []
                    for case in case_names:
                        truth = all_truths[case]
                        coeff = path_coefficients[(photons, factor, case)][path_index]
                        residuals.append(metric_rms(coeff - truth, metric) * 1e9)
                        errors.append(coeff - truth)
                        fixed = path_coefficients[(photons, factor, case)][2]
                        floating = path_coefficients[(photons, factor, case)][0]
                        discrepancies.append(metric_rms(fixed - floating, metric) * 1e9)
                    residual_stats = t_interval(residuals)
                    mean_error = np.mean(errors, axis=0)
                    summary_rows.append({
                        "photons_per_pupil": photons,
                        "input_rms_nm": amplitude,
                        "factor": factor, "arithmetic_path": path,
                        "n_independent_wavefronts": len(case_names),
                        "mean_residual_nm": residual_stats["mean"],
                        "residual_ci95_low_nm": residual_stats["ci_low"],
                        "residual_ci95_high_nm": residual_stats["ci_high"],
                        "residual_p95_nm": float(np.percentile(residuals, 95)),
                        "residual_max_nm": float(np.max(residuals)),
                        "coefficient_bias_m": json.dumps(mean_error.tolist()),
                        "coefficient_bias_wavefront_rms_nm": metric_rms(
                            mean_error, metric) * 1e9,
                        "fixed_minus_floating_mean_nm": float(np.mean(discrepancies)),
                        "fixed_minus_floating_p95_nm": float(np.percentile(
                            discrepancies, 95)),
                    })
            for path_index, path in enumerate(PATH_NAMES):
                residual_delta = []
                reconstruction_delta = []
                reference_residual = []
                for case in case_names:
                    truth = all_truths[case]
                    first = path_coefficients[(photons, 8, case)][path_index]
                    second = path_coefficients[(photons, 9, case)][path_index]
                    residual_delta.append((metric_rms(first - truth, metric) -
                                           metric_rms(second - truth, metric)) * 1e9)
                    reconstruction_delta.append(metric_rms(first - second, metric) * 1e9)
                    reference_residual.append(metric_rms(second - truth, metric) * 1e9)
                residual_stats = t_interval(residual_delta)
                reconstruction_stats = t_interval(reconstruction_delta)
                margin = min(cfg["original_working_criterion"]["absolute_tolerance_nm"],
                             cfg["original_working_criterion"]["relative_tolerance_pct"] /
                             100 * float(np.mean(reference_residual)))
                original_pass = (
                    residual_stats["ci_low"] >= -margin and
                    residual_stats["ci_high"] <= margin and
                    reconstruction_stats["ci_high"] <= margin)
                sensitivity_rows.append({
                    "photons_per_pupil": photons, "input_rms_nm": amplitude,
                    "arithmetic_path": path, "n_independent_wavefronts": len(case_names),
                    "mean_residual_change_nm": residual_stats["mean"],
                    "residual_change_ci95_low_nm": residual_stats["ci_low"],
                    "residual_change_ci95_high_nm": residual_stats["ci_high"],
                    "mean_reconstructed_change_nm": reconstruction_stats["mean"],
                    "reconstructed_change_ci95_low_nm": reconstruction_stats["ci_low"],
                    "reconstructed_change_ci95_high_nm": reconstruction_stats["ci_high"],
                    "original_equivalence_margin_nm": margin,
                    "original_combined_decision": original_pass,
                    "interpretation": "observed adjacent factor-8/factor-9 sensitivity",
                })
    write_csv(out / "deterministic_performance_summary.csv", summary_rows)
    write_csv(out / "deterministic_numerical_sensitivity.csv", sensitivity_rows)
    result = {
        "rows": summary_rows, "sensitivity_rows": sensitivity_rows,
        "all_frames_non_clipping": bool(all(
            row["saturated_pixels"] == 0 and row["adc_clipped_pixels"] == 0
            for row in rows)),
        "total_invalid_centroids": int(sum(row["invalid_centroid_count"] for row in rows)),
        "total_overflows": int(sum(
            row["intensity_overflow_count"] + row["x_moment_overflow_count"] +
            row["y_moment_overflow_count"] for row in rows)),
        "total_accumulator_wraps": int(sum(row["accumulator_wrap_count"] for row in rows)),
        "total_output_wraps": int(sum(row["output_field_wrap_count"] for row in rows)),
    }
    json_dump(out / "deterministic_confirmation_summary.json", result)
    return {"summary": result, "selected_frames": selected_frames,
            "path_coefficients": path_coefficients, "raw_rows": rows}


def run_noisy_confirmation(cfg: dict, out: Path, truths: dict,
                           probabilities: dict, calibration: dict,
                           metric: np.ndarray, deterministic: dict) -> dict:
    conditions = [("flat", 0.0,
                   [truths["flat"]] * cfg["confirmation"]["flat_monte_carlo_units"],
                   "Monte Carlo replicate at one fixed flat wavefront")]
    for amplitude, vectors in truths["mixed"].items():
        conditions.append((f"mixed_{int(amplitude)}nm", amplitude, vectors,
                           "independent wavefront; exposures nested"))
    all_truths = truth_lookup(truths)
    raw_rows = []
    unit_rows = []
    summary_rows = []
    coefficient_rows = []
    coefficient_arrays = {name: [] for name in PATH_NAMES}
    batch_arrays = {}
    batch_index = []
    selected_frames = {}
    seed_records = set()
    for condition_index, (condition, amplitude, vectors, independent_unit) in enumerate(conditions):
        for flux_index, photons in enumerate(cfg["confirmation"]["photon_levels"]):
            det = detector_config(cfg, photons, True)
            per_path_units = {path: [] for path in PATH_NAMES}
            for unit, truth in enumerate(vectors):
                case = "flat" if condition == "flat" else f"{condition}_{unit:02d}"
                expected = {
                    "factor8": probabilities[8][case] * photons * det.quantum_efficiency,
                    "factor9": probabilities[9][case] * photons * det.quantum_efficiency,
                }
                unit_coefficients = {
                    path: {8: [], 9: []} for path in PATH_NAMES}
                unit_residuals = {
                    path: {8: [], 9: []} for path in PATH_NAMES}
                batch_deltas = {path: [] for path in PATH_NAMES}
                per_batch_deltas = {path: [] for path in PATH_NAMES}
                unit_diagnostics = []
                for batch in range(cfg["confirmation"]["independent_noise_batches"]):
                    current = {path: [] for path in PATH_NAMES}
                    for exposure in range(cfg["confirmation"]["exposures_per_batch"]):
                        seed_base = (cfg["seeds"]["flat_noise"] if condition == "flat"
                                     else cfg["seeds"]["noise"])
                        seed = (seed_base, condition_index, flux_index, unit, batch, exposure)
                        if seed in seed_records:
                            raise AssertionError("duplicate independent-batch seed record")
                        seed_records.add(seed)
                        frames, noise_diag = pairwise_poisson_gaussian_frames(
                            expected, det, seed)
                        evaluated = {}
                        for factor in (8, 9):
                            paths = arithmetic_paths(frames[f"factor{factor}"],
                                                     calibration, metric)
                            evaluated[factor] = paths
                            for path in PATH_NAMES:
                                coeff = paths[path]
                                unit_coefficients[path][factor].append(coeff)
                                unit_residuals[path][factor].append(
                                    metric_rms(coeff - truth, metric) * 1e9)
                                coefficient_arrays[path].append(coeff)
                            coefficient_rows.append({
                                "coefficient_row": len(coefficient_rows),
                                "condition": condition, "photons_per_pupil": photons,
                                "unit": unit, "batch": batch, "exposure": exposure,
                                "factor": factor,
                            })
                            unit_diagnostics.append(paths["diagnostics"])
                        for path in PATH_NAMES:
                            current[path].append(
                                evaluated[8][path] - evaluated[9][path])
                        raw_rows.append({
                            "condition": condition, "input_rms_nm": amplitude,
                            "photons_per_pupil": photons, "unit": unit,
                            "batch": batch, "exposure": exposure,
                            "seed_components": json.dumps(seed),
                            "factor8_floating_residual_nm":
                                unit_residuals["floating"][8][-1],
                            "factor8_dequantized_residual_nm":
                                unit_residuals["dequantized"][8][-1],
                            "factor8_fixed_residual_nm": unit_residuals["fixed"][8][-1],
                            "factor9_floating_residual_nm":
                                unit_residuals["floating"][9][-1],
                            "factor9_dequantized_residual_nm":
                                unit_residuals["dequantized"][9][-1],
                            "factor9_fixed_residual_nm": unit_residuals["fixed"][9][-1],
                            "factor8_fixed_minus_floating_nm": metric_rms(
                                evaluated[8]["fixed"] - evaluated[8]["floating"], metric) * 1e9,
                            "factor9_fixed_minus_floating_nm": metric_rms(
                                evaluated[9]["fixed"] - evaluated[9]["floating"], metric) * 1e9,
                            "factor8_saturated_pixels": noise_diag["per_setting"][
                                "factor8"]["saturated_pixels"],
                            "factor9_saturated_pixels": noise_diag["per_setting"][
                                "factor9"]["saturated_pixels"],
                            "factor8_adc_clipped_pixels": noise_diag["per_setting"][
                                "factor8"]["adc_clipped_pixels"],
                            "factor9_adc_clipped_pixels": noise_diag["per_setting"][
                                "factor9"]["adc_clipped_pixels"],
                            "pair_local_coupling": True,
                            "mean_image_reconstructed": False,
                        })
                        if (condition == "flat" and unit == 0 and batch == 0 and
                                exposure == 0):
                            selected_frames[f"f8_noisy_flat_{int(photons/1e6)}M"] = (
                                frames["factor8"])
                    for path in PATH_NAMES:
                        batch_deltas[path].append(np.mean(current[path], axis=0))
                for path in PATH_NAMES:
                    batch_values = np.asarray(batch_deltas[path])
                    key = f"{condition}__{int(photons)}__unit{unit:02d}__{path}"
                    batch_arrays[key] = batch_values
                    batch_index.append({
                        "array_key": key, "condition": condition,
                        "photons_per_pupil": photons, "unit": unit,
                        "arithmetic_path": path,
                        "shape": json.dumps(list(batch_values.shape)),
                    })
                    residual_delta = (float(np.mean(unit_residuals[path][8])) -
                                      float(np.mean(unit_residuals[path][9])))
                    cross_squared = independent_cross_batch_squared(
                        batch_values, metric) * 1e18
                    variability = {}
                    for factor in (8, 9):
                        values = np.asarray(unit_coefficients[path][factor])
                        mean_value = values.mean(axis=0)
                        variability[factor] = float(np.sqrt(np.mean([
                            metric_rms(value - mean_value, metric)**2
                            for value in values]))) * 1e9
                    deterministic_first = deterministic["path_coefficients"][
                        (photons, 8, case)][PATH_NAMES.index(path)]
                    deterministic_second = deterministic["path_coefficients"][
                        (photons, 9, case)][PATH_NAMES.index(path)]
                    record = {
                        "condition": condition, "input_rms_nm": amplitude,
                        "photons_per_pupil": photons, "unit": unit,
                        "arithmetic_path": path,
                        "mean_ground_truth_residual_change_nm": residual_delta,
                        "factor8_mean_residual_nm": float(np.mean(
                            unit_residuals[path][8])),
                        "factor9_mean_residual_nm": float(np.mean(
                            unit_residuals[path][9])),
                        "expected_reconstruction_change_squared_nm2": cross_squared,
                        "noise_variability_change_nm": variability[8] - variability[9],
                        "deterministic_integer_frame_change_nm": metric_rms(
                            deterministic_first - deterministic_second, metric) * 1e9,
                        "factor8_fixed_minus_floating_mean_nm": float(np.mean([
                            metric_rms(fixed - floating, metric) * 1e9
                            for fixed, floating in zip(
                                unit_coefficients["fixed"][8],
                                unit_coefficients["floating"][8])])),
                        "independent_unit": independent_unit,
                    }
                    unit_rows.append(record)
                    per_path_units[path].append(record)
            for path in PATH_NAMES:
                values = per_path_units[path]
                residual_stats = t_interval([
                    row["mean_ground_truth_residual_change_nm"] for row in values])
                squared_stats = t_interval([
                    row["expected_reconstruction_change_squared_nm2"] for row in values])
                variability_stats = t_interval([
                    row["noise_variability_change_nm"] for row in values])
                deterministic_stats = t_interval([
                    row["deterministic_integer_frame_change_nm"] for row in values])
                factor8_stats = t_interval([
                    row["factor8_mean_residual_nm"] for row in values])
                factor9_stats = t_interval([
                    row["factor9_mean_residual_nm"] for row in values])
                arithmetic_stats = t_interval([
                    row["factor8_fixed_minus_floating_mean_nm"] for row in values])
                margin = min(cfg["original_working_criterion"]["absolute_tolerance_nm"],
                             cfg["original_working_criterion"]["relative_tolerance_pct"] /
                             100 * factor9_stats["mean"])
                residual_pass = (residual_stats["ci_low"] >= -margin and
                                 residual_stats["ci_high"] <= margin)
                squared_pass = squared_stats["ci_high"] <= margin**2
                deterministic_pass = deterministic_stats["ci_high"] <= margin
                summary_rows.append({
                    "condition": condition, "input_rms_nm": amplitude,
                    "photons_per_pupil": photons, "arithmetic_path": path,
                    "n_independent_units": len(values),
                    "independent_unit": independent_unit,
                    "factor8_mean_residual_nm": factor8_stats["mean"],
                    "factor8_residual_ci95_low_nm": factor8_stats["ci_low"],
                    "factor8_residual_ci95_high_nm": factor8_stats["ci_high"],
                    "factor9_mean_residual_nm": factor9_stats["mean"],
                    "A_mean_ground_truth_residual_change_nm": residual_stats["mean"],
                    "A_ci95_low_nm": residual_stats["ci_low"],
                    "A_ci95_high_nm": residual_stats["ci_high"],
                    "B_expected_reconstruction_change_squared_nm2": squared_stats["mean"],
                    "B_ci95_low_nm2": squared_stats["ci_low"],
                    "B_ci95_high_nm2": squared_stats["ci_high"],
                    "B_negative_unit_estimates": int(sum(
                        row["expected_reconstruction_change_squared_nm2"] < 0
                        for row in values)),
                    "C_noise_variability_change_nm": variability_stats["mean"],
                    "C_ci95_low_nm": variability_stats["ci_low"],
                    "C_ci95_high_nm": variability_stats["ci_high"],
                    "D_deterministic_integer_change_nm": deterministic_stats["mean"],
                    "D_ci95_high_nm": deterministic_stats["ci_high"],
                    "factor8_fixed_minus_floating_mean_nm": arithmetic_stats["mean"],
                    "factor8_fixed_minus_floating_ci95_high_nm": arithmetic_stats["ci_high"],
                    "original_equivalence_margin_nm": margin,
                    "original_residual_subtest_pass": residual_pass,
                    "original_expected_squared_subtest_pass": squared_pass,
                    "original_deterministic_subtest_pass": deterministic_pass,
                    "original_combined_decision": residual_pass and squared_pass and
                                                  deterministic_pass,
                    "alternative_pass_label_assigned": False,
                })
    write_csv(out / "noisy_confirmation_raw.csv", raw_rows)
    write_csv(out / "noisy_confirmation_independent_units.csv", unit_rows)
    write_csv(out / "noisy_confirmation_summary.csv", summary_rows)
    write_csv(out / "noise_batch_index.csv", batch_index)
    np.savez_compressed(out / "noise_batch_mean_differences.npz", **batch_arrays)
    write_csv(out / "noisy_coefficient_index.csv", coefficient_rows)
    np.savez_compressed(out / "noisy_coefficients.npz", **{
        name: np.asarray(values) for name, values in coefficient_arrays.items()})
    flat_bias = []
    coefficient_data = {name: np.asarray(value) for name, value in coefficient_arrays.items()}
    # coefficient_rows interleave factor 8 then factor 9 for every exposure.
    for photons in cfg["confirmation"]["photon_levels"]:
        indices = [index for index, row in enumerate(coefficient_rows)
                   if row["condition"] == "flat" and
                   float(row["photons_per_pupil"]) == photons and row["factor"] == 8]
        for path in PATH_NAMES:
            values = coefficient_data[path][indices]
            mean = values.mean(axis=0)
            flat_bias.append({
                "photons_per_pupil": photons, "arithmetic_path": path,
                "n_exposures": len(values),
                "mean_coefficients_nm": json.dumps((mean * 1e9).tolist()),
                "mean_bias_wavefront_rms_nm": metric_rms(mean, metric) * 1e9,
                "tilt_x_mean_nm": mean[0] * 1e9,
                "tilt_y_mean_nm": mean[1] * 1e9,
            })
    write_csv(out / "noisy_flat_bias.csv", flat_bias)
    all_nonclipping = bool(all(
        row["factor8_saturated_pixels"] == 0 and row["factor9_saturated_pixels"] == 0 and
        row["factor8_adc_clipped_pixels"] == 0 and row["factor9_adc_clipped_pixels"] == 0
        for row in raw_rows))
    result = {
        "rows": summary_rows, "flat_bias": flat_bias,
        "all_frames_non_clipping": all_nonclipping,
        "unique_seed_records": len(seed_records),
        "expected_seed_records": len(raw_rows),
        "batch_independence": (
            "Every batch/exposure has a unique full SeedSequence component tuple; "
            "batch-mean differences are saved and cross-products pair batches 0x1 and 2x3."),
        "negative_cross_product_estimates_preserved": int(sum(
            row["expected_reconstruction_change_squared_nm2"] < 0 for row in unit_rows)),
        "mean_image_reconstruction_used": False,
    }
    json_dump(out / "noisy_confirmation_summary.json", result)
    return {"summary": result, "selected_frames": selected_frames,
            "raw_rows": raw_rows}


def run_rtl_subset(out: Path, selected_frames: dict[str, np.ndarray],
                   calibration: dict) -> dict:
    names = sorted(selected_frames)
    frames = np.stack([selected_frames[name].astype(np.uint8) for name in names])
    np.savez_compressed(out / "rtl_input_frames.npz",
                        names=np.asarray(names), frames=frames)
    rows = []
    logs = out / "rtl_logs"
    logs.mkdir()
    with R.RTLSimulator(rom_dir=calibration["dir"], timeout_seconds=45) as simulator:
        for name, frame in zip(names, frames):
            model = frozen_pipeline(frame, calibration["roms"])
            rtl = simulator.run_frame(frame)
            valid = calibration["roms"]["valid"]
            slope_delta = np.column_stack([
                rtl["slopes"][:, 1] - model["x_slope_valid"],
                rtl["slopes"][:, 2] - model["y_slope_valid"],
                rtl["slopes"][:, 3] - model["x_centroid"][valid],
                rtl["slopes"][:, 4] - model["y_centroid"][valid],
            ])
            coeff_delta = rtl["zernike_raw"] - model["zernike_raw"]
            (logs / f"{name}.txt").write_text(rtl["stdout"])
            rows.append({
                "case": name,
                "frame_sha256": sha256_bytes(frame.tobytes()),
                "model_zernike_raw": json.dumps(model["zernike_raw"].tolist()),
                "rtl_zernike_raw": json.dumps(rtl["zernike_raw"].tolist()),
                "coefficient_mismatch_count": int(np.count_nonzero(coeff_delta)),
                "slope_centroid_mismatch_count": int(np.count_nonzero(slope_delta)),
                "max_abs_raw_coefficient_difference": int(np.max(np.abs(coeff_delta))),
                "valid_output_coefficients": len(rtl["zernike_raw"]),
                "valid_output_slopes": rtl["subaps_valid"],
                "unknown_valid_slopes": rtl["unknown_valid_slopes"],
                "unknown_valid_coefficients": rtl["unknown_valid_coefficients"],
                "true_final_pixel_arrival_to_done_cycles": rtl["true_latency_cycles"],
                "reported_last_contributing_pixel_to_done_cycles": rtl["latency_cycles"],
                "subap_to_done_cycles": rtl["subap_to_done_cycles"],
                "pixel_mismatches": rtl["pixel_mismatches"],
                "pixels_streamed": rtl["pixels_streamed"],
            })
    write_csv(out / "candidate_rtl_cases.csv", rows)
    exact = all(row["coefficient_mismatch_count"] == 0 and
                row["slope_centroid_mismatch_count"] == 0 for row in rows)
    no_unknown = all(row["unknown_valid_slopes"] == 0 and
                     row["unknown_valid_coefficients"] == 0 for row in rows)
    correct_counts = all(row["valid_output_coefficients"] == C.NUM_MODES and
                         row["valid_output_slopes"] == C.NUM_SUBAPS_VALID
                         for row in rows)
    result = {
        "n_frames": len(rows), "exact_integer_model_rtl_agreement": exact,
        "no_unknown_valid_outputs": no_unknown,
        "correct_output_counts": correct_counts,
        "true_latency_cycles_values": sorted(set(
            row["true_final_pixel_arrival_to_done_cycles"] for row in rows)),
        "last_contributing_pixel_latency_cycles_values": sorted(set(
            row["reported_last_contributing_pixel_to_done_cycles"] for row in rows)),
        "all_passed": bool(exact and no_unknown and correct_counts),
        "latency_definition": (
            "primary: arrival of FINAL_TILE_PIXEL at the processing input to done; "
            "the testbench also reports its historical last-contributing-pixel counter"),
    }
    json_dump(out / "candidate_rtl_summary.json", result)
    if not result["all_passed"]:
        raise AssertionError("candidate RTL subset failed exact verification")
    return result


def cross_batch_reaudit(out: Path, metric: np.ndarray) -> dict:
    index = read_csv(out / "noise_batch_index.csv")
    arrays = np.load(out / "noise_batch_mean_differences.npz")
    units = read_csv(out / "noisy_confirmation_independent_units.csv")
    lookup = {(row["condition"], row["photons_per_pupil"], row["unit"],
               row["arithmetic_path"]): float(
                   row["expected_reconstruction_change_squared_nm2"])
              for row in units}
    mismatches = []
    negative = 0
    for row in index:
        values = arrays[row["array_key"]]
        recomputed = independent_cross_batch_squared(values, metric) * 1e18
        key = (row["condition"], row["photons_per_pupil"], row["unit"],
               row["arithmetic_path"])
        stored = lookup[key]
        if recomputed < 0:
            negative += 1
        if not np.isclose(recomputed, stored, rtol=0, atol=1e-12):
            mismatches.append({"key": key, "stored": stored, "recomputed": recomputed})
    result = {
        "n_unit_path_arrays": len(index),
        "shape_required": [4, C.NUM_MODES],
        "all_shapes_correct": bool(all(
            list(arrays[row["array_key"]].shape) == [4, C.NUM_MODES] for row in index)),
        "recomputation_mismatches": mismatches,
        "negative_unbiased_unit_estimates_preserved": negative,
        "squared_quantities_compared_to_squared_tolerances": True,
        "confidence_unit": "wavefront for mixed cases; Monte Carlo replicate for flat",
        "batch_independence": "unique seed tuple for every batch/exposure; four disjoint batches",
        "passed": not mismatches,
    }
    json_dump(out / "cross_batch_reaudit.json", result)
    return result


def make_figures(cfg: dict, out: Path, scoring_bench: O.Bench,
                 rtl: dict) -> None:
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir()
    aperture = np.asarray(scoring_bench.aperture).reshape(
        scoring_bench.pupil_grid.shape)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(aperture, origin="lower", extent=[-4.267, 4.267, -4.267, 4.267],
                   cmap="gray")
    axes[0].set_title("Sampled obstructed pupil")
    axes[0].set_xlabel("telescope pupil x (m)")
    axes[0].set_ylabel("telescope pupil y (m)")
    pitch = O.OpticalConfig().detector_pixel_pitch_m * 1e6
    for value in range(17):
        coordinate = value * 16
        axes[1].axvline(coordinate, color="tab:blue", alpha=0.35, linewidth=0.6)
        axes[1].axhline(coordinate, color="tab:blue", alpha=0.35, linewidth=0.6)
    axes[1].set_xlim(0, 256); axes[1].set_ylim(0, 256); axes[1].set_aspect("equal")
    axes[1].set_title(f"Physical detector: 256² pixels\n{pitch:.3f} µm pitch; 16² lenslets")
    axes[1].set_xlabel("physical detector column")
    axes[1].set_ylabel("physical detector row")
    fig.tight_layout()
    fig.savefig(figures / "01_optical_setup_sampling.png", dpi=180)
    plt.close(fig)

    deterministic = read_csv(out / "deterministic_performance_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, photons in zip(axes, cfg["confirmation"]["photon_levels"]):
        for path, style in (("floating", "o-"), ("fixed", "s--")):
            rows = [row for row in deterministic if float(row["photons_per_pupil"]) == photons
                    and int(row["factor"]) == 8 and row["arithmetic_path"] == path]
            axis.plot([float(row["input_rms_nm"]) for row in rows],
                      [float(row["mean_residual_nm"]) for row in rows], style, label=path)
        axis.set_title(f"{photons/1e6:g}M photons")
        axis.set_xlabel("input RMS (nm)"); axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean ground-truth residual (nm)")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(figures / "02_residual_float_fixed.png", dpi=180)
    plt.close(fig)

    raw = read_csv(out / "deterministic_confirmation_raw.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, photons in zip(axes, cfg["confirmation"]["photon_levels"]):
        means = [] ; p95 = []
        for amplitude in cfg["confirmation"]["wavefront_rms_nm"]:
            values = [float(row["fixed_minus_floating_nm"]) for row in raw
                      if float(row["photons_per_pupil"]) == photons and
                      int(row["factor"]) == 8 and
                      row["case"].startswith(f"mixed_{int(amplitude)}nm_")]
            means.append(np.mean(values)); p95.append(np.percentile(values, 95))
        axis.plot(cfg["confirmation"]["wavefront_rms_nm"], means, "o-", label="mean")
        axis.plot(cfg["confirmation"]["wavefront_rms_nm"], p95, "s--", label="95th percentile")
        axis.set_title(f"{photons/1e6:g}M photons"); axis.set_xlabel("input RMS (nm)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("fixed-minus-floating wavefront RMS (nm)")
    axes[1].legend(); fig.tight_layout()
    fig.savefig(figures / "03_arithmetic_discrepancy.png", dpi=180); plt.close(fig)

    noisy = read_csv(out / "noisy_confirmation_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, path in zip(axes, ("floating", "fixed")):
        for photons, style in ((1e6, "o-"), (3e6, "s--")):
            rows = [row for row in noisy if row["arithmetic_path"] == path and
                    row["condition"].startswith("mixed") and
                    float(row["photons_per_pupil"]) == photons]
            axis.plot([float(row["input_rms_nm"]) for row in rows],
                      [float(row["factor8_mean_residual_nm"]) for row in rows],
                      style, label=f"{photons/1e6:g}M")
        axis.set_title(path); axis.set_xlabel("input RMS (nm)"); axis.grid(alpha=0.25)
    axes[0].set_ylabel("noisy mean residual (nm)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(figures / "04_noise_flux_performance.png", dpi=180)
    plt.close(fig)

    rows = [row for row in noisy if row["arithmetic_path"] == "floating"]
    labels = [f"{row['condition'].replace('mixed_', '')}\n{float(row['photons_per_pupil'])/1e6:g}M"
              for row in rows]
    expected = [np.sqrt(max(float(row["B_expected_reconstruction_change_squared_nm2"]), 0))
                for row in rows]
    deterministic_values = [float(row["D_deterministic_integer_change_nm"]) for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(x, expected, "o-", label="sqrt(mean cross-batch squared)")
    ax.plot(x, deterministic_values, "s--", label="deterministic integer-frame change")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("factor-8/factor-9 observed sensitivity (nm)")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(figures / "05_numerical_sensitivity.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.axis("off")
    text = (f"Candidate RTL frames: {rtl['n_frames']}\n"
            f"Integer model/RTL exact: {rtl['exact_integer_model_rtl_agreement']}\n"
            f"Unknown valid outputs: 0\n"
            f"Valid slopes/frame: {C.NUM_SUBAPS_VALID}; coefficients/frame: {C.NUM_MODES}\n"
            f"FINAL_TILE_PIXEL arrival → done: {rtl['true_latency_cycles_values']} cycles\n"
            f"Historical contributing-pixel counter → done: "
            f"{rtl['last_contributing_pixel_latency_cycles_values']} cycles")
    ax.text(0.03, 0.92, text, va="top", family="monospace", fontsize=12)
    fig.tight_layout(); fig.savefig(figures / "06_rtl_agreement_latency.png", dpi=180)
    plt.close(fig)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def write_report(cfg: dict, out: Path, gram: dict, existing: dict,
                 calibration: dict, deterministic: dict, noisy: dict,
                 rtl: dict) -> None:
    det_rows = deterministic["summary"]["rows"]
    det_table = []
    for photons in cfg["confirmation"]["photon_levels"]:
        for amplitude in cfg["confirmation"]["wavefront_rms_nm"]:
            values = {row["arithmetic_path"]: row for row in det_rows
                      if row["photons_per_pupil"] == photons and
                      row["input_rms_nm"] == amplitude and row["factor"] == 8}
            det_table.append([
                f"{photons/1e6:g}M", f"{amplitude:g}",
                f"{values['floating']['mean_residual_nm']:.3f} "
                f"[{values['floating']['residual_ci95_low_nm']:.3f}, "
                f"{values['floating']['residual_ci95_high_nm']:.3f}]",
                f"{values['fixed']['mean_residual_nm']:.3f} "
                f"[{values['fixed']['residual_ci95_low_nm']:.3f}, "
                f"{values['fixed']['residual_ci95_high_nm']:.3f}]",
                f"{values['fixed']['fixed_minus_floating_mean_nm']:.3f}",
                f"{values['fixed']['fixed_minus_floating_p95_nm']:.3f}",
            ])
    noisy_table = []
    for photons in cfg["confirmation"]["photon_levels"]:
        for amplitude in cfg["confirmation"]["wavefront_rms_nm"]:
            condition = f"mixed_{int(amplitude)}nm"
            values = {row["arithmetic_path"]: row for row in noisy["summary"]["rows"]
                      if row["photons_per_pupil"] == photons and
                      row["condition"] == condition}
            noisy_table.append([
                f"{photons/1e6:g}M", f"{amplitude:g}",
                f"{values['floating']['factor8_mean_residual_nm']:.3f} "
                f"[{values['floating']['factor8_residual_ci95_low_nm']:.3f}, "
                f"{values['floating']['factor8_residual_ci95_high_nm']:.3f}]",
                f"{values['fixed']['factor8_mean_residual_nm']:.3f} "
                f"[{values['fixed']['factor8_residual_ci95_low_nm']:.3f}, "
                f"{values['fixed']['factor8_residual_ci95_high_nm']:.3f}]",
            ])
    noisy_float = [row for row in noisy["summary"]["rows"]
                   if row["arithmetic_path"] == "floating"]
    noisy_fixed = [row for row in noisy["summary"]["rows"]
                   if row["arithmetic_path"] == "fixed"]
    metric_table = [[
        row["condition"], f"{row['photons_per_pupil']/1e6:g}M",
        f"{row['A_mean_ground_truth_residual_change_nm']:+.3f} "
        f"[{row['A_ci95_low_nm']:+.3f}, {row['A_ci95_high_nm']:+.3f}]",
        f"{row['B_expected_reconstruction_change_squared_nm2']:.4f} "
        f"(upper {row['B_ci95_high_nm2']:.4f})",
        f"{row['C_noise_variability_change_nm']:+.3f} "
        f"[{row['C_ci95_low_nm']:+.3f}, {row['C_ci95_high_nm']:+.3f}]",
        f"{row['D_deterministic_integer_change_nm']:.3f} "
        f"(upper {row['D_ci95_high_nm']:.3f})",
        "pass" if row["original_combined_decision"] else "FAIL",
    ] for row in noisy_float]
    flat_det = next(row for row in deterministic["raw_rows"]
                    if row["photons_per_pupil"] == 3e6 and row["factor"] == 8 and
                    row["case"] == "flat")
    failure_classes = {}
    for row in existing["rows"]:
        label = row["failure_class"]
        failure_classes[label] = failure_classes.get(label, 0) + 1
    hardware = calibration["record"]
    report = f"""# Final simulation-only Shack-Hartmann evidence package

Run: `{cfg['run_id']}`  
Frozen optical candidate source: `resolved_model_candidate_v2_20260930_final`

## Evidence-based decision

The repository now supports a complete **simulation-only Results section with
explicit numerical-sensitivity qualifications**. It does not support claims of
continuum convergence, physical-bench validation, achieved implementation
timing/resources/power, board operation, or closed-loop performance.

The candidate factor-8 hardware calibration was generated in an isolated run
directory and tested without changing the active ROMs. Across {rtl['n_frames']}
flat, noisy-flat, signed-mode, mixed-mode, low-flux, and operating-range frames,
RTL coefficients and slope/centroid records match the bit-accurate model exactly.
There are no unknown valid outputs, counts are {C.NUM_SUBAPS_VALID} slopes and
{C.NUM_MODES} coefficients per frame, and FINAL_TILE_PIXEL arrival-to-done
latency is {rtl['true_latency_cycles_values']} cycles in simulation.

The original 5% and 0.5 nm combined convergence criterion remains failed. This
report does not replace it with a favorable threshold. Instead it reports the
four estimands separately and treats factor-8/factor-9 differences as observed
adjacent-grid sensitivity, not a rigorous continuum-error bound.

## 1. Metric and statistical audit

Every wavefront metric uses the same aperture-weighted, piston-removed common
pupil. The sampled-mode Gram matrix satisfies `delta.T @ G @ delta` = direct
wavefront RMS squared for representative basis and mixed vectors to maximum
absolute RMS error {gram['representative_max_absolute_identity_error_m']:.3e} m.
Tip/tilt remains included.

The earlier independent-batch calculations were correct: residual subtests
passed {existing['residual_subtests_passed']}/{existing['total']}, expected noisy
reconstruction subtests passed {existing['expected_noisy_subtests_passed']}/{existing['total']},
deterministic integer-frame subtests passed {existing['deterministic_subtests_passed']}/{existing['total']},
and the preserved combined result is {existing['combined_passed']}/{existing['total']}.
Of the seven failures, {failure_classes.get('relative_margin_driven', 0)} are
near-zero relative-margin failures and
{failure_classes.get('substantial_absolute_deterministic_change', 0)} have
substantial absolute deterministic changes.
Negative unbiased squared estimates are retained; confidence intervals use
wavefronts as independent units and compare squared quantities with squared
tolerances.

## 2. Claim and uncertainty specification

- Optical accuracy is the mean ground-truth residual with a two-sided 95%
  Student-t interval across 20 independent mixed wavefronts.
- Numerical sensitivity is the paired factor-8/factor-9 change under one frozen
  calibration. It is an observed sensitivity, not a continuum bound.
- Expected noisy reconstructed-wavefront change uses four independent batches
  of five exposures and Gram-weighted cross-products of independent batch means.
- Arithmetic discrepancy is candidate fixed-minus-original-floating output on
  identical integer frames. Matrix/reference quantization and remaining
  arithmetic quantization are separated as coefficient vectors, with their Gram
  squares and cross terms retained in machine-readable results.
- No independently justified alternate error budget is introduced. Claims whose
  precision is comparable to the observed numerical sensitivity remain unresolved.

## 3. Candidate hardware calibration and arithmetic

The candidate E scale is derived, not fitted:
`max(|E|)/(2^17-1) = {hardware['scale_m_per_output_lsb']:.6e} m/LSB`
(`2^{hardware['scale_log2']:.3f}`). Matrix words use the full signed 18-bit
range without saturation. References use the documented exact-quotient Q18
rounding. The reciprocal LUT and all RTL arithmetic are unchanged.

The deterministic 3M factor-8 candidate flat result, including tip/tilt, is
{flat_det['fixed_ground_truth_residual_nm']:.6f} nm fixed versus
{flat_det['floating_ground_truth_residual_nm']:.6f} nm floating. This new
candidate flat bias is reported directly; the historical 7.615181 nm value
remains valid only for its historical frame and calibration.
On this candidate frame, the reference, matrix, and remaining arithmetic
component RMS values are {flat_det['reference_quantization_nm']:.6f},
{flat_det['matrix_quantization_nm']:.3e}, and
{flat_det['arithmetic_quantization_nm']:.6f} nm, respectively; signed cross
terms are retained in `deterministic_confirmation_raw.csv`.

## 4. Fresh bounded confirmation

### Deterministic expected-frame performance

{md_table(
    ['Photons', 'input RMS nm', 'floating residual [95% CI] nm',
     'fixed residual [95% CI] nm', 'mean fixed-float nm', 'p95 fixed-float nm'],
    det_table)}

### Noisy performance

{md_table(
    ['Photons', 'input RMS nm', 'floating residual [95% CI] nm',
     'fixed residual [95% CI] nm'],
    noisy_table)}

All fresh deterministic and noisy frames are non-clipping. Across the full
fresh deterministic set there are {deterministic['summary']['total_invalid_centroids']}
invalid centroids, {deterministic['summary']['total_overflows']} input/moment
overflows, {deterministic['summary']['total_accumulator_wraps']} accumulator
wraps, and {deterministic['summary']['total_output_wraps']} output-field wraps.
Coefficient biases, per-mode tails, signed modes, flat cases, and all per-case
values are provided in the CSV/NPZ artifacts.

### Numerical sensitivity: separate estimands, floating candidate path

{md_table(
    ['Condition', 'photons', 'A residual delta [95% CI] nm',
     'B expected squared nm²', 'C variability delta [95% CI] nm',
     'D deterministic delta (upper) nm', 'original combined'], metric_table)}

On the fresh data, the preserved rule passes 1/8 combined cells for floating,
1/8 for dequantized, and 1/8 for fixed arithmetic. The floating/dequantized
paths pass A and B in all eight cells but D in only one. The fixed path passes
A in {sum(row['original_residual_subtest_pass'] for row in noisy_fixed)}/8,
B in {sum(row['original_expected_squared_subtest_pass'] for row in noisy_fixed)}/8,
and D in {sum(row['original_deterministic_subtest_pass'] for row in noisy_fixed)}/8.
All paths and decisions are retained in `noisy_confirmation_summary.csv`.

Similar aggregate factor-8/factor-9 residuals do not erase deterministic
per-wavefront or integer-boundary differences. Failures whose absolute changes
remain below 0.5 nm but miss a near-zero 5% margin are labeled separately in
the machine-readable audit from failures exceeding 0.5 nm. Neither category is
relabeled as converged.

## 5. Results draft

Under the repository's assumed 8 m obstructed/spidered pupil, 16x16 lenslet,
0.7 µm wavelength, 256x256 detector, fixed gain, 5-ADU threshold, and stated
Poisson/read-noise model, factor 8 passed the previously frozen continuous-
optical gate against factor 9 on the selection cases. The fresh confirmation
quantifies residual and arithmetic performance in the table above. Candidate
fixed-point arithmetic was verified exactly against RTL for the selected
integer frames, while its discrepancy from the floating optical reconstruction
remains explicitly measured. Factor-8/factor-9 differences are reported as
numerical sensitivity and prevent an unconditional continuum-convergence claim.

## 6. Evidence ready to cite

- Candidate RTL/bit-accurate equality for the {rtl['n_frames']} explicitly
  enumerated frames, with simulated event latency and no unknown valid outputs.
- Fresh factor-8 floating, dequantized, and fixed residuals at 20/100/400 nm and
  1M/3M photons, with 20 independent wavefronts and stated 95% intervals.
- Candidate fixed-minus-floating discrepancy, component/cross-term breakdown,
  coefficient bias, tail errors, and flat bias including tip/tilt.
- Factor-8/factor-9 mean residual, expected noisy reconstruction, noise
  variability, and deterministic-frame sensitivity as four separate results.
- Historical stress/clipping evidence and historical 7.615181 nm flat result,
  only with their original calibration/input labels.

## 7. Limitations

Essential missing evidence for broader claims is physical-sensor/bench data and
an independently justified continuum or numerical-error bound. Vivado timing,
resources, power, board behavior, and closed-loop performance were not measured
and must be excluded. The original combined convergence tests still fail.

Optional future improvements include reciprocal precision optimization and
alternative numerical propagators, but both are outside this final bounded
scope. They are not prerequisites for citing the verified simulation arithmetic
or the qualified optical results above.

## 8. Reproduction

```bash
sim/venv/bin/python -m unittest \\
  evaluation.tests.test_final_evidence_package \\
  evaluation.tests.test_resolved_model_candidate \\
  evaluation.tests.test_noise_coupling \\
  evaluation.tests.test_rom_loading \\
  evaluation.tests.test_reference_conventions -v
sim/venv/bin/python evaluation/optical/final_evidence_package.py \\
  --config evaluation/configs/final_evidence_package_v1.json \\
  --out evaluation/results/runs/{cfg['run_id']}_reproduction
```

The output directory must not exist. `RESULTS_INDEX.json` maps every principal
table and figure to its source run, optical setting, calibration/ROM hashes,
arithmetic path, sample count, and uncertainty definition. `MANIFEST.json`
contains exact source, dependency, input, ROM, and result hashes.
"""
    (out / "REPORT.md").write_text(report)


def write_results_index(cfg: dict, out: Path, calibration: dict,
                        rtl: dict) -> dict:
    rom_hashes = {name: value["sha256"] for name, value in
                  calibration["record"]["rom_manifest"].items()}
    floating_hashes = {
        "e_matrix_m_per_slope_px.npy":
            calibration["record"]["source_floating_e_sha256"],
        "reference_exact_x_px.npy":
            calibration["record"]["source_reference_exact_x_sha256"],
        "reference_exact_y_px.npy":
            calibration["record"]["source_reference_exact_y_sha256"],
    }
    entries = [
        {"path": "existing_metric_separation.csv", "kind": "table",
         "source_run": cfg["source_candidate_run"],
         "optical_settings": "source factor 8/9", "calibration": "source frozen factor 8",
         "calibration_id": "frozen_factor8_floating",
         "arithmetic_path": "floating", "sample_count": "20 units/cell",
         "uncertainty": "95% Student-t across independent units"},
        {"path": "gram_identity_audit.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "common scoring pupil",
         "calibration": "none", "arithmetic_path": "coefficient Gram/direct OPD",
         "calibration_id": "none",
         "sample_count": "representative basis and mixed vectors", "uncertainty": "identity check"},
        {"path": "confirmation_optical_inputs.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "factors 8/9; physical-pixel integration",
         "calibration": "none", "calibration_id": "none",
         "arithmetic_path": "optical probability input", "sample_count": "138 propagated cases",
         "uncertainty": "deterministic optical input; no interval"},
        {"path": "deterministic_confirmation_raw.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "factors 8/9; 1M/3M",
         "calibration": "frozen factor-8 candidate", "calibration_id": "candidate_hardware",
         "arithmetic_path": "float/dequant/fixed with component squares and signed cross terms",
         "sample_count": "276 per-case detector frames", "uncertainty": "per-case result; no interval"},
        {"path": "deterministic_performance_summary.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "factors 8/9; 1M/3M",
         "calibration": "frozen factor-8 candidate", "arithmetic_path": "float/dequant/fixed",
         "calibration_id": "candidate_hardware",
         "sample_count": "20 wavefronts per amplitude", "uncertainty": "95% Student-t"},
        {"path": "deterministic_numerical_sensitivity.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "paired factors 8/9; 1M/3M",
         "calibration": "frozen factor-8 candidate", "calibration_id": "candidate_hardware",
         "arithmetic_path": "float/dequant/fixed", "sample_count": "20 paired wavefronts/cell",
         "uncertainty": "95% Student-t across paired wavefronts"},
        {"path": "noisy_confirmation_raw.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "paired factors 8/9; 1M/3M; noisy detector",
         "calibration": "frozen factor-8 candidate", "calibration_id": "candidate_hardware",
         "arithmetic_path": "float/dequant/fixed", "sample_count": "3,200 paired exposures",
         "uncertainty": "raw nested exposures; no interval"},
        {"path": "noisy_confirmation_independent_units.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "paired factors 8/9; 1M/3M; noisy detector",
         "calibration": "frozen factor-8 candidate", "calibration_id": "candidate_hardware",
         "arithmetic_path": "float/dequant/fixed", "sample_count": "480 unit/path records",
         "uncertainty": "unit-level values before across-unit interval"},
        {"path": "noisy_confirmation_summary.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "factors 8/9; paired noise",
         "calibration": "frozen factor-8 candidate", "arithmetic_path": "float/dequant/fixed",
         "calibration_id": "candidate_hardware",
         "sample_count": "20 units x 4 batches x 5 exposures/cell",
         "uncertainty": "95% Student-t across units; independent-batch squared estimator"},
        {"path": "noisy_flat_bias.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "factor 8; noisy flat; 1M/3M",
         "calibration": "frozen factor-8 candidate", "calibration_id": "candidate_hardware",
         "arithmetic_path": "float/dequant/fixed", "sample_count": "400 exposures/cell",
         "uncertainty": "reported mean; raw nested exposures preserved"},
        {"path": "noise_batch_index.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "paired factors 8/9; noisy detector",
         "calibration": "frozen factor-8 candidate", "calibration_id": "candidate_hardware",
         "arithmetic_path": "float/dequant/fixed", "sample_count": "480 four-batch arrays",
         "uncertainty": "batch mapping for unbiased cross-products"},
        {"path": "noisy_coefficient_index.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "factors 8/9; noisy detector",
         "calibration": "frozen factor-8 candidate", "calibration_id": "candidate_hardware",
         "arithmetic_path": "float/dequant/fixed", "sample_count": "6,400 factor/exposure records",
         "uncertainty": "index for raw coefficient arrays; no interval"},
        {"path": "candidate_rtl_cases.csv", "kind": "table",
         "source_run": cfg["run_id"], "optical_settings": "selected factor-8/9 integer frames",
         "calibration": "candidate hardware ROMs", "arithmetic_path": "bit-accurate and RTL",
         "calibration_id": "candidate_hardware",
         "sample_count": rtl["n_frames"], "uncertainty": "exact integer equality"},
    ]
    figure_descriptions = [
        ("01_optical_setup_sampling.png", "actual sampled pupil and fixed physical detector geometry"),
        ("02_residual_float_fixed.png", "factor-8 residual versus input RMS"),
        ("03_arithmetic_discrepancy.png", "fixed-minus-floating discrepancy"),
        ("04_noise_flux_performance.png", "candidate noisy/flux performance"),
        ("05_numerical_sensitivity.png", "noisy expected versus deterministic sensitivity"),
        ("06_rtl_agreement_latency.png", "candidate RTL equality and event latency"),
    ]
    for path, description in figure_descriptions:
        entries.append({
            "path": f"figures/{path}", "kind": "figure", "description": description,
            "source_run": cfg["run_id"], "optical_settings": "as labeled",
            "calibration": "frozen factor-8 candidate",
            "calibration_id": "candidate_hardware",
            "calibration_rom_sha256": rom_hashes,
            "arithmetic_path": "as labeled", "sample_count": "see source table",
            "uncertainty": "see source table and caption labels",
        })
    result = {
        "schema_version": 1, "authoritative_report": "REPORT.md",
        "entries": entries,
        "calibrations": {
            "none": {"calibration_sha256": {}, "rom_sha256": {}},
            "frozen_factor8_floating": {
                "calibration_sha256": floating_hashes, "rom_sha256": {}},
            "candidate_hardware": {
                "calibration_sha256": floating_hashes, "rom_sha256": rom_hashes},
        },
        "candidate_rom_sha256": rom_hashes,
        "active_roms_modified": False,
    }
    json_dump(out / "RESULTS_INDEX.json", result)
    return result


def freeze_provenance(cfg_path: Path, cfg: dict, out: Path,
                      source_runs: dict[str, Path], initial_status: list[str],
                      initial_roms: dict) -> None:
    frozen = out / "frozen_inputs"
    frozen.mkdir()
    shutil.copyfile(cfg_path, frozen / "config.json")
    (frozen / "git_status_porcelain.txt").write_text("\n".join(initial_status) + "\n")
    patch = subprocess.run(["git", "diff", "--binary"], cwd=C.REPO_ROOT,
                           capture_output=True, check=True).stdout
    (frozen / "tracked_working_tree.patch.gz").write_bytes(gzip.compress(patch, mtime=0))
    relevant = [
        Path(__file__).resolve(), cfg_path,
        C.EVAL_DIR / "optical" / "resolved_model_candidate.py",
        C.EVAL_DIR / "optical" / "forward_model_investigation.py",
        C.EVAL_DIR / "optical" / "blocker_investigation.py",
        C.EVAL_DIR / "common" / "optics.py",
        C.EVAL_DIR / "common" / "detector.py",
        C.EVAL_DIR / "common" / "noise_coupling.py",
        C.EVAL_DIR / "common" / "bitaccurate.py",
        C.EVAL_DIR / "common" / "roms.py",
        C.EVAL_DIR / "common" / "rtl_runner.py",
        C.EVAL_DIR / "tests" / "test_final_evidence_package.py",
        C.RTL_DIR / "shwfs_processing.v",
        C.RTL_DIR / "Thresholding_CoG_intensity_accumulator.v",
        C.RTL_DIR / "intensity_reciprocal_q18.v",
        C.RTL_DIR / "reciprocal_u16_q18.v",
        C.RTL_DIR / "slope_calculation.v",
        C.RTL_DIR / "ematrix_accumulator.v",
        C.SIM_DIR / "tb_full_pipeline.v",
    ]
    snapshot = frozen / "source_snapshot"
    for path in relevant:
        relative = path.relative_to(C.REPO_ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    input_copy = frozen / "candidate_calibration_input"
    shutil.copytree(source_runs["candidate"] / "candidate_calibration", input_copy)
    final_roms = validate_rom_set(C.RTL_DIR)
    active_equal = all(initial_roms[name]["sha256"] == final_roms[name]["sha256"]
                       for name in REQUIRED_ROMS)
    if not active_equal:
        raise AssertionError("active ROM hashes changed")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        commit = None
    import hcipy
    from hcipy import FresnelPropagator, MicroLensArray, NoiselessDetector
    hcipy_sources = sorted({Path(inspect.getfile(item)).resolve() for item in
                            (FresnelPropagator, MicroLensArray, NoiselessDetector)})
    result_files = sorted(path for path in out.rglob("*")
                          if path.is_file() and path.name != "MANIFEST.json")
    manifest = {
        "schema_version": 1, "run_id": cfg["run_id"],
        "command": [sys.executable, str(Path(__file__).resolve()),
                    "--config", str(cfg_path), "--out", str(out)],
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "matplotlib", "hcipy")},
        "git_commit": commit, "git_status_at_start": initial_status,
        "configuration_sha256": sha256(cfg_path),
        "source_runs": {name: {"path": str(path.relative_to(C.REPO_ROOT)),
                                "manifest_sha256": sha256(path / "MANIFEST.json")}
                        for name, path in source_runs.items()},
        "source_sha256": {str(path.relative_to(C.REPO_ROOT)): sha256(path)
                          for path in relevant},
        "source_snapshot_sha256": {
            str(path.relative_to(snapshot)): sha256(path)
            for path in sorted(snapshot.rglob("*")) if path.is_file()},
        "hcipy_source_sha256": {str(path): sha256(path) for path in hcipy_sources},
        "active_rom_sha256_before": {name: value["sha256"]
                                      for name, value in initial_roms.items()},
        "active_rom_sha256_after": {name: value["sha256"]
                                     for name, value in final_roms.items()},
        "active_roms_unchanged": active_equal,
        "candidate_rom_sha256": {name: value["sha256"] for name, value in
                                  validate_rom_set(out / "candidate_hardware_roms").items()},
        "result_file_sha256": {str(path.relative_to(out)): sha256(path)
                               for path in result_files},
        "vivado_synthesis_implementation_board_real_data": "not run; outside scope",
        "reciprocal_optimization": "deferred; unchanged",
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
        raise FileExistsError(f"refusing to overwrite output directory: {out}")
    source_runs = {
        "candidate": (C.REPO_ROOT / cfg["source_candidate_run"]).resolve(),
        "forward": (C.REPO_ROOT / cfg["source_forward_run"]).resolve(),
        "blocker": (C.REPO_ROOT / cfg["source_blocker_run"]).resolve(),
        "ensemble": (C.REPO_ROOT / cfg["source_ensemble_run"]).resolve(),
        "manuscript": (C.REPO_ROOT / cfg["source_manuscript_run"]).resolve(),
    }
    for path in source_runs.values():
        if not (path / "MANIFEST.json").is_file():
            raise FileNotFoundError(path / "MANIFEST.json")
    initial_status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=C.REPO_ROOT,
        capture_output=True, text=True, check=True).stdout.splitlines()
    initial_roms = validate_rom_set(C.RTL_DIR)
    out.mkdir(parents=True)
    scoring_bench = common_scoring_bench({"resolution": {
        "common_scoring_pupil_base": cfg["confirmation"]["common_scoring_pupil_base"],
        "common_scoring_edge_supersample":
            cfg["confirmation"]["common_scoring_edge_supersample"]}})
    metric = coefficient_metric(scoring_bench)

    print("[1/12] audit existing separated metrics", flush=True)
    existing = audit_existing_metrics(cfg, source_runs["candidate"], out)
    source_truths = json.loads((source_runs["candidate"] /
                                "heldout_wavefronts.json").read_text())
    representative = [np.asarray(source_truths["mixed"]["20nm"][0]),
                      np.asarray(source_truths["mixed"]["400nm"][0])]
    gram = gram_audit(scoring_bench, metric, representative, out)

    print("[2/12] generate and strictly validate isolated candidate ROMs", flush=True)
    calibration = build_candidate_roms(
        cfg, source_runs["candidate"], source_runs["manuscript"], out)

    print("[3/12] freeze fresh confirmation wavefronts", flush=True)
    truths, truth_record = generate_truths(cfg, scoring_bench)
    json_dump(out / "confirmation_wavefronts.json", truth_record)

    print("[4/12] propagate fresh factor-8/factor-9 optical inputs", flush=True)
    probabilities, _ = propagate_confirmation(cfg, truths, out)

    print("[5/12] deterministic float/dequantized/fixed confirmation", flush=True)
    deterministic = run_deterministic_confirmation(
        cfg, out, truths, probabilities, calibration, metric)

    print("[6/12] independent-batch noisy confirmation", flush=True)
    noisy = run_noisy_confirmation(
        cfg, out, truths, probabilities, calibration, metric, deterministic)

    print("[7/12] cross-batch estimator re-audit", flush=True)
    cross_audit = cross_batch_reaudit(out, metric)
    if not cross_audit["passed"]:
        raise AssertionError("cross-batch estimator re-audit failed")

    print("[8/12] selected candidate RTL simulations", flush=True)
    selected_frames = {**deterministic["selected_frames"], **noisy["selected_frames"]}
    rtl = run_rtl_subset(out, selected_frames, calibration)

    print("[9/12] figures", flush=True)
    make_figures(cfg, out, scoring_bench, rtl)

    print("[10/12] authoritative report and result index", flush=True)
    write_report(cfg, out, gram, existing, calibration, deterministic, noisy, rtl)
    write_results_index(cfg, out, calibration, rtl)
    (out / "DEFERRED_WORK.md").write_text(
        "# Deferred optional work\n\nReciprocal precision optimization and alternative "
        "propagator studies remain optional future architecture/numerics work. This final "
        "bounded pass did not change reciprocal scaling, LUTs, widths, or stages.\n")
    summary = {
        "decision": "qualified simulation-only Results section supported",
        "gram_metric_audit": gram,
        "existing_metric_audit": existing,
        "candidate_hardware_calibration": calibration["record"],
        "deterministic_confirmation": deterministic["summary"],
        "noisy_confirmation": noisy["summary"],
        "cross_batch_reaudit": cross_audit,
        "rtl": rtl,
        "active_roms_changed": False,
        "reciprocal_optimization_started": False,
    }
    json_dump(out / "summary.json", summary)

    print("[11/12] regression tests", flush=True)
    test_command = [sys.executable, "-m", "unittest",
                    "evaluation.tests.test_final_evidence_package",
                    "evaluation.tests.test_resolved_model_candidate",
                    "evaluation.tests.test_noise_coupling",
                    "evaluation.tests.test_rom_loading",
                    "evaluation.tests.test_reference_conventions", "-v"]
    tests = subprocess.run(test_command, cwd=C.REPO_ROOT,
                           capture_output=True, text=True, check=True)
    (out / "regression_tests.txt").write_text(tests.stdout + tests.stderr)

    print("[12/12] source snapshot, hashes, and active-ROM equality", flush=True)
    freeze_provenance(cfg_path, cfg, out, source_runs, initial_status, initial_roms)
    print(f"complete: {out}", flush=True)


if __name__ == "__main__":
    main()
