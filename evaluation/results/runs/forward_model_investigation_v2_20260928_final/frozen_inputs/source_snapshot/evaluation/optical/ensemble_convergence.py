#!/usr/bin/env python3
"""Bounded frozen-reconstructor and noisy-ensemble convergence evaluation.

The active RTL arithmetic and calibration ROMs are inputs, never outputs.  The
script distinguishes the production path (one frozen reference/E matrix across
all forward grids) from the already-completed per-grid recalibration diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/shwfs-matplotlib")

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import bitaccurate as BA  # noqa: E402
from evaluation.common import config as C  # noqa: E402
from evaluation.common import detector as D  # noqa: E402
from evaluation.common import optics as O  # noqa: E402
from evaluation.common import rtl_runner as R  # noqa: E402
from evaluation.optical.blocker_investigation import (  # noqa: E402
    frozen_pipeline,
    integrated_detector_image,
    load_frozen_roms,
    normalized_coefficients,
    rms_coefficients,
)


DEFAULT_CONFIG = C.EVAL_DIR / "configs" / "ensemble_convergence_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result table: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def detector_config(cfg: dict, photons: float, noisy: bool) -> D.DetectorConfig:
    values = {key: value for key, value in cfg["detector"].items()
              if key in D.DetectorConfig.__dataclass_fields__}
    values.update(total_photons=photons, shot_noise=noisy,
                  read_noise_e=cfg["detector"]["read_noise_e"] if noisy else 0.0)
    return D.DetectorConfig(**values)


def setting_specs(cfg: dict) -> dict[str, dict]:
    """Unique forward-model settings; production is shared by both studies."""
    specs = {
        "production": {"edge_supersample": 4, "propagation_factor": 1},
    }
    for edge in cfg["deterministic"]["edge_supersampling"]:
        if edge != 4:
            specs[f"edge_ss{edge}"] = {"edge_supersample": edge,
                                       "propagation_factor": 1}
    for factor in cfg["deterministic"]["propagation_factors"]:
        if factor != 1:
            specs[f"prop_f{factor}"] = {"edge_supersample": 4,
                                        "propagation_factor": factor}
    return specs


def study_setting_key(study: str, setting: int) -> str:
    if study == "pupil_edge_supersampling":
        return "production" if setting == 4 else f"edge_ss{setting}"
    if study == "propagation_grid_with_detector_integration":
        return "production" if setting == 1 else f"prop_f{setting}"
    raise ValueError(study)


def make_bench(spec: dict) -> tuple[O.Bench, int]:
    base = O.OpticalConfig()
    factor = int(spec["propagation_factor"])
    optical = replace(
        base,
        pupil_pixels_base=base.pupil_pixels_base * factor,
        supersample=int(spec["edge_supersample"]),
    )
    return O.Bench(optical), factor


def normalized_detector_probability(image: np.ndarray, factor: int) -> tuple[np.ndarray, float]:
    integrated = integrated_detector_image(image, factor)
    total = float(integrated.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("non-positive/non-finite integrated optical flux")
    return integrated / total, total


def noiseless_adu(probability: np.ndarray, cfg: D.DetectorConfig) -> tuple[np.ndarray, dict]:
    expected_e = probability * cfg.total_photons * cfg.quantum_efficiency + cfg.background_e
    saturated = int(np.count_nonzero(expected_e > cfg.full_well_e))
    electrons = np.clip(expected_e, 0.0, cfg.full_well_e)
    unbounded = np.rint(electrons / cfg.gain_e_per_adu + cfg.offset_adu)
    clipped = int(np.count_nonzero((unbounded < 0) | (unbounded > cfg.adc_max)))
    adu = np.clip(unbounded, 0, cfg.adc_max).astype(np.int64)
    return adu, {"saturated_pixels": saturated, "adc_clipped_pixels": clipped,
                 "expected_electrons": float(expected_e.sum())}


def common_scoring_bench(cfg: dict) -> O.Bench:
    scoring = cfg["common_scoring_pupil"]
    return O.Bench(replace(
        O.OpticalConfig(), pupil_pixels_base=int(scoring["pupil_pixels_base"]),
        supersample=int(scoring["pupil_edge_supersample"])))


def run_deterministic(cfg: dict, out: Path, blocker_run: Path,
                      roms: dict[str, np.ndarray], scale: float,
                      scoring_bench: O.Bench) -> dict:
    case_record = json.loads((blocker_run / "convergence_cases.json").read_text())
    cases = {name: np.asarray(values, dtype=float)
             for name, values in case_record["cases"].items()}
    specs = setting_specs(cfg)
    det = detector_config(cfg, cfg["deterministic"]["photons_per_pupil"], noisy=False)
    valid = roms["valid"]
    captures: dict[str, dict[str, dict]] = {}
    arrays: dict[str, np.ndarray] = {}
    raw_rows: list[dict] = []
    for setting_name, spec in specs.items():
        bench, factor = make_bench(spec)
        captures[setting_name] = {}
        for case_name, truth in cases.items():
            image, _ = bench.propagate_coeffs(truth)
            probability, raw_flux = normalized_detector_probability(image, factor)
            adu, detector_diag = noiseless_adu(probability, det)
            fixed = frozen_pipeline(adu, roms)
            coeff = fixed["zernike_raw"].astype(float) * scale
            residual = rms_coefficients(scoring_bench, coeff - truth)
            centroids = np.r_[fixed["x_centroid"][valid],
                              fixed["y_centroid"][valid]].astype(float) / (1 << C.CENTROID_FRAC)
            expected_adu = (
                probability * det.total_photons * det.quantum_efficiency /
                det.gain_e_per_adu + det.offset_adu
            )
            captures[setting_name][case_name] = {
                "probability": probability,
                "expected_adu": expected_adu,
                "adu": adu,
                "centroids": centroids,
                "coefficients": coeff,
                "residual_m": residual,
            }
            prefix = f"{setting_name}__{case_name}"
            arrays[f"probability__{prefix}"] = probability.astype(np.float32)
            arrays[f"adu__{prefix}"] = adu.astype(np.uint8)
            arrays[f"centroids_px__{prefix}"] = centroids.astype(np.float64)
            arrays[f"coefficients_m__{prefix}"] = coeff.astype(np.float64)
            raw_rows.append({
                "setting_name": setting_name,
                "edge_supersample": spec["edge_supersample"],
                "propagation_factor": spec["propagation_factor"],
                "case": case_name,
                "input_rms_m_common_grid": rms_coefficients(scoring_bench, truth),
                "ground_truth_residual_m": residual,
                "raw_integrated_flux_before_detector_normalization": raw_flux,
                "expected_adu_sum": float(expected_adu.sum()),
                "digitized_adu_sum": int(adu.sum()),
                **detector_diag,
                "frozen_reference_rom": True,
                "frozen_e_matrix_rom": True,
                "coefficients_m": json.dumps(coeff.tolist()),
            })
    np.savez_compressed(out / "deterministic_forward_arrays.npz", **arrays)
    write_csv(out / "deterministic_frozen_raw.csv", raw_rows)

    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    studies = {
        "pupil_edge_supersampling": cfg["deterministic"]["edge_supersampling"],
        "propagation_grid_with_detector_integration": cfg["deterministic"]["propagation_factors"],
    }
    for study, values in studies.items():
        reference_setting = max(values)
        reference_key = study_setting_key(study, reference_setting)
        for setting in values:
            key = study_setting_key(study, setting)
            case_pairs = []
            for case_name in cases:
                current = captures[key][case_name]
                reference = captures[reference_key][case_name]
                probability_delta = current["probability"] - reference["probability"]
                expected_adu_delta = current["expected_adu"] - reference["expected_adu"]
                adu_delta = current["adu"].astype(float) - reference["adu"].astype(float)
                centroid_delta = current["centroids"] - reference["centroids"]
                coeff_delta = current["coefficients"] - reference["coefficients"]
                absolute_nm = abs(current["residual_m"] - reference["residual_m"]) * 1e9
                relative_pct = (absolute_nm / (reference["residual_m"] * 1e9) * 100
                                if reference["residual_m"] > 0 else float("inf"))
                row = {
                    "study": study, "setting": setting,
                    "setting_name": key, "reference_setting": reference_setting,
                    "reference_setting_name": reference_key, "case": case_name,
                    "residual_m": current["residual_m"],
                    "reference_residual_m": reference["residual_m"],
                    "signed_residual_change_m": current["residual_m"] - reference["residual_m"],
                    "absolute_residual_change_nm": absolute_nm,
                    "relative_residual_change_pct": relative_pct,
                    "passes_residual_tolerances": bool(
                        absolute_nm <= cfg["acceptance"]["absolute_tolerance_nm"] and
                        relative_pct <= cfg["acceptance"]["relative_tolerance_pct"]),
                    "normalized_intensity_relative_l2": float(
                        np.linalg.norm(probability_delta) /
                        np.linalg.norm(reference["probability"])),
                    "normalized_intensity_max_abs": float(np.max(np.abs(probability_delta))),
                    "expected_adu_rms_change": float(np.sqrt(np.mean(expected_adu_delta ** 2))),
                    "digitized_adu_rms_change": float(np.sqrt(np.mean(adu_delta ** 2))),
                    "digitized_adu_changed_pixels": int(np.count_nonzero(adu_delta)),
                    "fixed_centroid_rms_change_px": float(np.sqrt(np.mean(centroid_delta ** 2))),
                    "fixed_centroid_max_abs_change_px": float(np.max(np.abs(centroid_delta))),
                    "reconstructed_wavefront_rms_change_m": rms_coefficients(
                        scoring_bench, coeff_delta),
                    "coefficient_rms_change_m": float(np.sqrt(np.mean(coeff_delta ** 2))),
                    "reference_self_comparison_is_evidence": False,
                }
                pair_rows.append(row); case_pairs.append(row)
            summary_rows.append({
                "study": study, "setting": setting,
                "setting_name": key, "reference_setting": reference_setting,
                "n_cases": len(case_pairs),
                "mean_residual_m": float(np.mean([r["residual_m"] for r in case_pairs])),
                "reference_mean_residual_m": float(np.mean([
                    r["reference_residual_m"] for r in case_pairs])),
                "mean_absolute_residual_change_nm": float(np.mean([
                    r["absolute_residual_change_nm"] for r in case_pairs])),
                "max_absolute_residual_change_nm": float(np.max([
                    r["absolute_residual_change_nm"] for r in case_pairs])),
                "mean_reconstructed_wavefront_rms_change_nm": float(np.mean([
                    r["reconstructed_wavefront_rms_change_m"] * 1e9 for r in case_pairs])),
                "max_reconstructed_wavefront_rms_change_nm": float(np.max([
                    r["reconstructed_wavefront_rms_change_m"] * 1e9 for r in case_pairs])),
                "all_cases_pass_residual_tolerances": (None if setting == reference_setting
                    else bool(all(r["passes_residual_tolerances"] for r in case_pairs))),
                "reference_self_comparison_is_evidence": False,
            })
    write_csv(out / "deterministic_frozen_paired.csv", pair_rows)
    write_csv(out / "deterministic_frozen_summary.csv", summary_rows)
    summary = {
        "calibration_policy": "same frozen active reference and E-matrix ROMs at every grid",
        "detector": asdict(det), "case_source": str(blocker_run / "convergence_cases.json"),
        "common_scoring_pupil": asdict(scoring_bench.cfg), "rows": summary_rows,
    }
    json_dump(out / "deterministic_frozen_summary.json", summary)
    return {"summary": summary, "cases": case_record, "specs": specs}


def generate_noisy_truth(cfg: dict, scoring_bench: O.Bench) -> dict[float, list[np.ndarray]]:
    settings = cfg["noisy_ensemble"]
    result: dict[float, list[np.ndarray]] = {}
    for amplitude in settings["wavefront_rms_nm"]:
        seed_key = f"wavefronts_{int(amplitude)}nm"
        rng = np.random.default_rng(cfg["seeds"][seed_key])
        result[float(amplitude)] = [
            normalized_coefficients(scoring_bench, rng.normal(size=C.NUM_MODES), amplitude)
            for _ in range(settings["independent_wavefronts_per_amplitude"])
        ]
    return result


def coupled_noisy_frames(probabilities: dict[str, np.ndarray], det: D.DetectorConfig,
                         seed_components: list[int]) -> tuple[dict[str, np.ndarray], dict]:
    """Couple all grids while preserving every detector-noise marginal exactly."""
    names = sorted(probabilities)
    lambdas = {
        name: (np.asarray(probabilities[name], dtype=float) * det.total_photons *
               det.quantum_efficiency + det.background_e)
        for name in names
    }
    minimum = np.minimum.reduce([lambdas[name] for name in names])
    seed_sequence = np.random.SeedSequence(seed_components)
    children = seed_sequence.spawn(2 + len(names))
    common_shot = np.random.default_rng(children[0]).poisson(
        np.clip(minimum, 0, None)).astype(np.int64)
    common_read = np.random.default_rng(children[1]).normal(
        0.0, det.read_noise_e, minimum.shape) if det.read_noise_e > 0 else 0.0
    frames: dict[str, np.ndarray] = {}
    saturated: dict[str, int] = {}
    clipped: dict[str, int] = {}
    for index, name in enumerate(names):
        increment_mean = np.clip(lambdas[name] - minimum, 0.0, None)
        increment = np.random.default_rng(children[index + 2]).poisson(
            increment_mean).astype(np.int64)
        electrons = common_shot + increment + common_read
        saturated[name] = int(np.count_nonzero(electrons > det.full_well_e))
        electrons = np.clip(electrons, 0.0, det.full_well_e)
        unbounded = np.rint(electrons / det.gain_e_per_adu + det.offset_adu)
        clipped[name] = int(np.count_nonzero(
            (unbounded < 0) | (unbounded > det.adc_max)))
        frames[name] = np.clip(unbounded, 0, det.adc_max).astype(np.int64)
    return frames, {
        "saturated_pixels": saturated, "adc_clipped_pixels": clipped,
        "common_poisson_mean_sum": float(minimum.sum()),
        "coupling_seed_components": seed_components,
        "proof": ("N_setting=N_common+N_increment with independent Poisson means "
                  "min(lambda) and lambda_setting-min(lambda); therefore "
                  "N_setting~Poisson(lambda_setting). Shared read field is N(0,sigma^2)."),
    }


def t_interval(values: list[float] | np.ndarray, confidence: float) -> dict:
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError("Student-t interval requires at least two independent units")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    sem = sd / np.sqrt(x.size)
    critical = float(student_t.ppf((1 + confidence) / 2, x.size - 1))
    half = critical * sem
    return {"n": int(x.size), "mean": mean, "sd": sd, "sem": float(sem),
            "ci_low": float(mean - half), "ci_high": float(mean + half),
            "ci_half_width": float(half), "confidence": confidence,
            "method": "two-sided Student-t interval across independent wavefront means"}


def comparison_definitions(cfg: dict) -> list[dict]:
    result = []
    for study, pairs in cfg["comparisons"].items():
        full_study = ("pupil_edge_supersampling" if study == "edge"
                      else "propagation_grid_with_detector_integration")
        supported = (max(cfg["noisy_ensemble"]["edge_supersampling"])
                     if study == "edge" else
                     max(cfg["noisy_ensemble"]["propagation_factors"]))
        production = 4 if study == "edge" else 1
        for coarse, fine in pairs:
            result.append({
                "study": full_study,
                "comparison": f"{study}_{coarse}_to_{fine}",
                "coarse_setting": coarse, "fine_setting": fine,
                "coarse_key": study_setting_key(full_study, coarse),
                "fine_key": study_setting_key(full_study, fine),
                "comparison_type": ("production_to_supported_reference"
                                    if coarse == production and fine == supported
                                    else "successive_refinement"),
            })
    return result


def run_noisy_ensemble(cfg: dict, out: Path, roms: dict[str, np.ndarray],
                       scale: float, scoring_bench: O.Bench) -> dict:
    ensemble = cfg["noisy_ensemble"]
    specs = setting_specs(cfg)
    truths = generate_noisy_truth(cfg, scoring_bench)
    json_dump(out / "ensemble_wavefronts.json", {
        "seeds": {key: value for key, value in cfg["seeds"].items()
                  if key.startswith("wavefronts_")},
        "normalization_pupil": asdict(scoring_bench.cfg),
        "wavefronts": {f"{amplitude:g}nm": [row.tolist() for row in values]
                       for amplitude, values in truths.items()},
    })

    probabilities: dict[str, dict[tuple[float, int], np.ndarray]] = {}
    optical_rows: list[dict] = []
    for setting_name, spec in specs.items():
        bench, factor = make_bench(spec)
        probabilities[setting_name] = {}
        for amplitude, wavefronts in truths.items():
            for wavefront, truth in enumerate(wavefronts):
                image, _ = bench.propagate_coeffs(truth)
                probability, raw_flux = normalized_detector_probability(image, factor)
                stored = probability.astype(np.float32)
                probabilities[setting_name][(amplitude, wavefront)] = stored
                optical_rows.append({
                    "setting_name": setting_name,
                    "edge_supersample": spec["edge_supersample"],
                    "propagation_factor": spec["propagation_factor"],
                    "amplitude_nm_rms": amplitude, "wavefront": wavefront,
                    "raw_integrated_flux_before_detector_normalization": raw_flux,
                    "normalized_probability_sum": float(stored.sum(dtype=np.float64)),
                    "probability_float32_sha256": sha256_bytes(stored.tobytes()),
                })
    write_csv(out / "ensemble_optical_inputs.csv", optical_rows)

    frame_rows: list[dict] = []
    frame_results: dict[tuple, dict] = {}
    amplitude_values = [float(x) for x in ensemble["wavefront_rms_nm"]]
    photon_values = [float(x) for x in ensemble["photons_per_pupil"]]
    for amplitude_index, amplitude in enumerate(amplitude_values):
        for wavefront, truth in enumerate(truths[amplitude]):
            probability_set = {name: probabilities[name][(amplitude, wavefront)]
                               for name in specs}
            for flux_index, photons in enumerate(photon_values):
                det = detector_config(cfg, photons, noisy=True)
                for repeat in range(ensemble["noise_exposures_per_wavefront"]):
                    seed_components = [cfg["seeds"]["noise_coupling"], amplitude_index,
                                       wavefront, flux_index, repeat]
                    frames, noise_diag = coupled_noisy_frames(
                        probability_set, det, seed_components)
                    for setting_name, frame in frames.items():
                        fixed = frozen_pipeline(frame, roms)
                        coeff = fixed["zernike_raw"].astype(float) * scale
                        residual = rms_coefficients(scoring_bench, coeff - truth)
                        key = (amplitude, photons, wavefront, repeat, setting_name)
                        frame_results[key] = {"coefficients": coeff, "residual_m": residual}
                        frame_rows.append({
                            "amplitude_nm_rms": amplitude,
                            "photons_per_pupil": photons,
                            "wavefront": wavefront, "repeat": repeat,
                            "setting_name": setting_name,
                            "edge_supersample": specs[setting_name]["edge_supersample"],
                            "propagation_factor": specs[setting_name]["propagation_factor"],
                            "ground_truth_residual_m": residual,
                            "coefficients_m": json.dumps(coeff.tolist()),
                            "frame_sha256": sha256_bytes(frame.astype(np.uint8).tobytes()),
                            "adc_clipped_pixels": noise_diag["adc_clipped_pixels"][setting_name],
                            "saturated_pixels": noise_diag["saturated_pixels"][setting_name],
                            "coupling_seed_components": json.dumps(seed_components),
                            "statistical_unit": "wavefront; repeats are nested exposures",
                        })
    write_csv(out / "ensemble_raw.csv", frame_rows)

    confidence = cfg["acceptance"]["confidence_level"]
    setting_summary: list[dict] = []
    for amplitude in amplitude_values:
        for photons in photon_values:
            for setting_name in specs:
                wavefront_means = []
                for wavefront in range(ensemble["independent_wavefronts_per_amplitude"]):
                    wavefront_means.append(float(np.mean([
                        frame_results[(amplitude, photons, wavefront, repeat,
                                       setting_name)]["residual_m"]
                        for repeat in range(ensemble["noise_exposures_per_wavefront"])
                    ])))
                stats = t_interval(wavefront_means, confidence)
                setting_summary.append({
                    "amplitude_nm_rms": amplitude, "photons_per_pupil": photons,
                    "setting_name": setting_name,
                    "edge_supersample": specs[setting_name]["edge_supersample"],
                    "propagation_factor": specs[setting_name]["propagation_factor"],
                    "n_independent_wavefronts": stats["n"],
                    "nested_exposures_per_wavefront": ensemble["noise_exposures_per_wavefront"],
                    "residual_mean_m": stats["mean"], "residual_sd_m": stats["sd"],
                    "residual_sem_m": stats["sem"],
                    "residual_ci95_low_m": stats["ci_low"],
                    "residual_ci95_high_m": stats["ci_high"],
                    "interval_method": stats["method"],
                    "adc_clipped_pixels_total": int(sum(
                        row["adc_clipped_pixels"] for row in frame_rows
                        if row["amplitude_nm_rms"] == amplitude
                        and row["photons_per_pupil"] == photons
                        and row["setting_name"] == setting_name)),
                })
    write_csv(out / "ensemble_setting_summary.csv", setting_summary)

    pair_raw: list[dict] = []
    pair_summary: list[dict] = []
    definitions = comparison_definitions(cfg)
    for definition in definitions:
        coarse = definition["coarse_key"]
        fine = definition["fine_key"]
        for amplitude in amplitude_values:
            for photons in photon_values:
                per_wavefront_residual_delta = []
                per_wavefront_prediction_delta = []
                reference_wavefront_residual = []
                for wavefront in range(ensemble["independent_wavefronts_per_amplitude"]):
                    residual_deltas = []
                    prediction_deltas = []
                    reference_residuals = []
                    for repeat in range(ensemble["noise_exposures_per_wavefront"]):
                        coarse_value = frame_results[(amplitude, photons, wavefront,
                                                      repeat, coarse)]
                        fine_value = frame_results[(amplitude, photons, wavefront,
                                                    repeat, fine)]
                        residual_delta = coarse_value["residual_m"] - fine_value["residual_m"]
                        prediction_delta = rms_coefficients(
                            scoring_bench,
                            coarse_value["coefficients"] - fine_value["coefficients"])
                        residual_deltas.append(residual_delta)
                        prediction_deltas.append(prediction_delta)
                        reference_residuals.append(fine_value["residual_m"])
                        pair_raw.append({
                            **definition, "amplitude_nm_rms": amplitude,
                            "photons_per_pupil": photons, "wavefront": wavefront,
                            "repeat": repeat,
                            "signed_residual_difference_m": residual_delta,
                            "reconstructed_wavefront_rms_difference_m": prediction_delta,
                            "fine_reference_residual_m": fine_value["residual_m"],
                            "pairing": "same physical wavefront and explicitly coupled noise exposure",
                        })
                    per_wavefront_residual_delta.append(float(np.mean(residual_deltas)))
                    per_wavefront_prediction_delta.append(float(np.mean(prediction_deltas)))
                    reference_wavefront_residual.append(float(np.mean(reference_residuals)))
                residual_stats = t_interval(per_wavefront_residual_delta, confidence)
                prediction_stats = t_interval(per_wavefront_prediction_delta, confidence)
                reference_stats = t_interval(reference_wavefront_residual, confidence)
                reference_mean_nm = reference_stats["mean"] * 1e9
                margin_nm = min(cfg["acceptance"]["absolute_tolerance_nm"],
                                cfg["acceptance"]["relative_tolerance_pct"] / 100 *
                                reference_mean_nm)
                residual_low_nm = residual_stats["ci_low"] * 1e9
                residual_high_nm = residual_stats["ci_high"] * 1e9
                prediction_high_nm = prediction_stats["ci_high"] * 1e9
                residual_pass = residual_low_nm >= -margin_nm and residual_high_nm <= margin_nm
                prediction_pass = prediction_high_nm <= margin_nm
                pair_summary.append({
                    **definition, "amplitude_nm_rms": amplitude,
                    "photons_per_pupil": photons,
                    "n_independent_wavefronts": residual_stats["n"],
                    "nested_exposures_per_wavefront": ensemble["noise_exposures_per_wavefront"],
                    "fine_reference_residual_mean_nm": reference_mean_nm,
                    "equivalence_margin_nm": margin_nm,
                    "signed_residual_difference_mean_nm": residual_stats["mean"] * 1e9,
                    "signed_residual_difference_ci95_low_nm": residual_low_nm,
                    "signed_residual_difference_ci95_high_nm": residual_high_nm,
                    "signed_residual_difference_ci95_width_nm": (
                        residual_high_nm - residual_low_nm),
                    "residual_interval_wholly_inside_margin": bool(residual_pass),
                    "prediction_rms_difference_mean_nm": prediction_stats["mean"] * 1e9,
                    "prediction_rms_difference_ci95_low_nm": prediction_stats["ci_low"] * 1e9,
                    "prediction_rms_difference_ci95_high_nm": prediction_high_nm,
                    "prediction_upper_interval_below_margin": bool(prediction_pass),
                    "ensemble_cell_accepted": bool(residual_pass and prediction_pass),
                    "acceptance_rule_frozen_before_run": True,
                    "finest_self_comparison_is_evidence": False,
                })
    write_csv(out / "ensemble_paired_raw.csv", pair_raw)
    write_csv(out / "ensemble_paired_summary.csv", pair_summary)

    comparison_summary = []
    for definition in definitions:
        cells = [row for row in pair_summary
                 if row["comparison"] == definition["comparison"]]
        comparison_summary.append({
            **definition, "n_predeclared_cells": len(cells),
            "accepted_cells": int(sum(row["ensemble_cell_accepted"] for row in cells)),
            "all_cells_accepted": bool(all(row["ensemble_cell_accepted"] for row in cells)),
            "max_abs_residual_difference_mean_nm": float(max(
                abs(row["signed_residual_difference_mean_nm"]) for row in cells)),
            "max_prediction_rms_difference_mean_nm": float(max(
                row["prediction_rms_difference_mean_nm"] for row in cells)),
            "max_prediction_to_margin_ratio": float(max(
                row["prediction_rms_difference_ci95_high_nm"] /
                row["equivalence_margin_nm"] for row in cells)),
        })
    write_csv(out / "ensemble_comparison_summary.csv", comparison_summary)
    summary = {
        "design": ensemble, "acceptance": cfg["acceptance"],
        "coupling": ensemble["coupling"], "setting_rows": setting_summary,
        "paired_rows": pair_summary, "comparison_rows": comparison_summary,
        "all_frames_non_clipping": bool(all(row["adc_clipped_pixels"] == 0
                                             for row in frame_rows)),
        "all_frames_below_full_well": bool(all(row["saturated_pixels"] == 0
                                                for row in frame_rows)),
    }
    json_dump(out / "ensemble_summary.json", summary)
    return summary


def reuse_completed_evidence(blocker_run: Path, out: Path) -> dict:
    reuse_dir = out / "reused_evidence"
    reuse_dir.mkdir()
    names = [
        "MANIFEST.json", "REPORT.md", "flat_summary.json", "flat_rtl_summary.json",
        "convergence_cases.json",
        "convergence_raw.csv", "convergence_summary.csv",
        "convergence_paired_successive.csv", "convergence_physical_checks.csv",
        "clipping_ablation_raw.csv", "clipping_ablation_summary.csv",
        "clipping_ablation_summary.json", "claim_to_evidence.csv",
    ]
    records = {}
    for name in names:
        source = blocker_run / name
        destination = reuse_dir / name
        shutil.copy2(source, destination)
        records[name] = {"source_sha256": sha256(source),
                         "copied_sha256": sha256(destination)}

    with (blocker_run / "convergence_summary.csv").open(newline="") as handle:
        diagnostic_rows = [row for row in csv.DictReader(handle)
                           if row["detector_path"] == "production_adc"]
    for row in diagnostic_rows:
        row["calibration_policy"] = "per-grid recalibration diagnostic; not production"
    write_csv(out / "per_grid_recalibration_diagnostic_summary.csv", diagnostic_rows)
    flat = json.loads((blocker_run / "flat_summary.json").read_text())
    clipping = json.loads((blocker_run / "clipping_ablation_summary.json").read_text())
    summary = {
        "source_blocker_run": str(blocker_run), "files": records,
        "flat_fixed_wavefront_rms_m": flat["fixed_wavefront_rms_m"],
        "clipping": clipping,
        "reuse_policy": "byte-copied completed evidence; no clipping or flat rerun",
    }
    json_dump(out / "reused_evidence_summary.json", summary)
    return summary


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_figures(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)
    frozen = read_csv(out / "deterministic_frozen_summary.csv")
    diagnostic = read_csv(out / "per_grid_recalibration_diagnostic_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    studies = (("pupil_edge_supersampling", "Pupil-edge supersampling"),
               ("propagation_grid_with_detector_integration",
                "Propagation grid + detector integration"))
    for ax, (study, title) in zip(axes, studies):
        fr = [row for row in frozen if row["study"] == study]
        dg = [row for row in diagnostic if row["study"] == study]
        ax.plot([float(row["setting"]) for row in fr],
                [float(row["mean_residual_m"]) * 1e9 for row in fr], "o-",
                label="frozen fixed reconstructor")
        ax.plot([float(row["setting"]) for row in dg],
                [float(row["mean_residual_m"]) * 1e9 for row in dg], "s--",
                label="per-grid recalibration diagnostic")
        ax.set_title(title); ax.set_xlabel("setting"); ax.set_ylabel("mean residual (nm)")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"frozen_vs_recalibrated_convergence.{suffix}", dpi=180)
    plt.close(fig)

    paired = read_csv(out / "deterministic_frozen_paired.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, (study, title) in zip(axes, studies):
        rows = [row for row in paired if row["study"] == study]
        settings = sorted({int(row["setting"]) for row in rows})
        for case in sorted({row["case"] for row in rows}):
            values = [next(float(row["reconstructed_wavefront_rms_change_m"]) * 1e9
                           for row in rows if row["case"] == case
                           and int(row["setting"]) == setting)
                      for setting in settings]
            ax.plot(settings, values, marker="o", linewidth=1, markersize=3, label=case)
        ax.set_title(title); ax.set_xlabel("setting")
        ax.set_ylabel("prediction change vs highest (nm)"); ax.grid(alpha=0.25)
    axes[1].legend(fontsize=6, ncol=2)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"deterministic_paired_prediction_changes.{suffix}", dpi=180)
    plt.close(fig)

    setting_rows = read_csv(out / "ensemble_setting_summary.csv")
    for study, title, settings, filename in (
        ("edge", "Noisy ensemble: pupil-edge sequence", [4, 12, 16, 20],
         "ensemble_edge_residuals"),
        ("prop", "Noisy ensemble: propagation sequence", [1, 2, 3, 4, 5],
         "ensemble_propagation_residuals"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.5), constrained_layout=True,
                                 sharex=True)
        for ax, amplitude, photons in zip(
                axes.ravel(), (20, 20, 100, 100), (1e6, 3e6, 1e6, 3e6)):
            names = [study_setting_key(
                "pupil_edge_supersampling" if study == "edge" else
                "propagation_grid_with_detector_integration", setting)
                     for setting in settings]
            rows = [next(row for row in setting_rows
                         if row["setting_name"] == name
                         and float(row["amplitude_nm_rms"]) == amplitude
                         and float(row["photons_per_pupil"]) == photons)
                    for name in names]
            means = np.asarray([float(row["residual_mean_m"]) * 1e9 for row in rows])
            lows = np.asarray([float(row["residual_ci95_low_m"]) * 1e9 for row in rows])
            highs = np.asarray([float(row["residual_ci95_high_m"]) * 1e9 for row in rows])
            ax.errorbar(settings, means, yerr=np.vstack([means - lows, highs - means]),
                        marker="o", capsize=3)
            ax.set_title(f"{amplitude} nm RMS, {photons/1e6:g}M photons")
            ax.set_ylabel("fixed residual (nm)"); ax.grid(alpha=0.25)
        for ax in axes[-1]: ax.set_xlabel("setting")
        fig.suptitle(title)
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(figure_dir / f"{filename}.{suffix}", dpi=180)
        plt.close(fig)

    comparisons = read_csv(out / "ensemble_paired_summary.csv")
    labels = sorted({row["comparison"] for row in comparisons})
    cells = [(20, 1e6), (20, 3e6), (100, 1e6), (100, 3e6)]
    ratio = np.zeros((len(labels), len(cells)))
    accepted = np.zeros_like(ratio, dtype=bool)
    for i, label in enumerate(labels):
        for j, (amplitude, photons) in enumerate(cells):
            row = next(row for row in comparisons if row["comparison"] == label
                       and float(row["amplitude_nm_rms"]) == amplitude
                       and float(row["photons_per_pupil"]) == photons)
            ratio[i, j] = (float(row["prediction_rms_difference_ci95_high_nm"]) /
                           float(row["equivalence_margin_nm"]))
            accepted[i, j] = row["ensemble_cell_accepted"] == "True"
    fig, ax = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    image = ax.imshow(ratio, aspect="auto", cmap="magma", vmin=0,
                      vmax=max(1.0, float(np.percentile(ratio, 95))))
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(len(cells)),
                  [f"{a} nm\n{p/1e6:g}M ph" for a, p in cells])
    for i in range(len(labels)):
        for j in range(len(cells)):
            ax.text(j, i, ("PASS" if accepted[i, j] else f"{ratio[i,j]:.1f}×"),
                    ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="prediction CI upper bound / equivalence margin")
    ax.set_title("Predeclared noisy-ensemble equivalence decisions")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"ensemble_equivalence.{suffix}", dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def write_report(out: Path, cfg: dict, deterministic: dict,
                 ensemble: dict, reused: dict) -> None:
    det_rows = deterministic["summary"]["rows"]
    det_table = []
    for row in det_rows:
        passes = row["all_cases_pass_residual_tolerances"]
        det_table.append([
            row["study"], str(row["setting"]), f"{row['mean_residual_m']*1e9:.3f}",
            f"{row['mean_reconstructed_wavefront_rms_change_nm']:.3f}",
            f"{row['max_reconstructed_wavefront_rms_change_nm']:.3f}",
            "self—not evidence" if passes is None else ("yes" if passes else "no"),
        ])
    comparison_table = []
    for row in ensemble["comparison_rows"]:
        comparison_table.append([
            row["comparison"], row["comparison_type"],
            f"{row['accepted_cells']}/{row['n_predeclared_cells']}",
            "yes" if row["all_cells_accepted"] else "no",
            f"{row['max_abs_residual_difference_mean_nm']:.3f}",
            f"{row['max_prediction_rms_difference_mean_nm']:.3f}",
            f"{row['max_prediction_to_margin_ratio']:.2f}×",
        ])
    production_cells = [row for row in ensemble["paired_rows"]
                        if row["comparison_type"] == "production_to_supported_reference"]
    production_table = []
    for row in production_cells:
        production_table.append([
            row["comparison"], f"{row['amplitude_nm_rms']:.0f}",
            f"{row['photons_per_pupil']/1e6:g}M",
            f"{row['fine_reference_residual_mean_nm']:.3f}",
            (f"{row['signed_residual_difference_mean_nm']:+.3f} "
             f"[{row['signed_residual_difference_ci95_low_nm']:+.3f}, "
             f"{row['signed_residual_difference_ci95_high_nm']:+.3f}]"),
            (f"{row['prediction_rms_difference_mean_nm']:.3f} "
             f"(upper {row['prediction_rms_difference_ci95_high_nm']:.3f})"),
            f"{row['equivalence_margin_nm']:.3f}",
            "PASS" if row["ensemble_cell_accepted"] else "fail",
        ])
    accepted_cells = [row for row in ensemble["paired_rows"]
                      if row["ensemble_cell_accepted"]]
    residual_only = [row for row in ensemble["paired_rows"]
                     if row["residual_interval_wholly_inside_margin"] and
                     not row["prediction_upper_interval_below_margin"]]
    clipping_rows = reused["clipping"]["rows"]
    clipping_8m = next(row for row in clipping_rows
                       if float(row["photons_per_pupil"]) == 8e6)
    edge_prod = next(row for row in det_rows
                     if row["study"] == "pupil_edge_supersampling" and row["setting"] == 4)
    prop_prod = next(row for row in det_rows
                     if row["study"] == "propagation_grid_with_detector_integration"
                     and row["setting"] == 1)
    deferred = """# Deferred work: reciprocal precision

Reciprocal precision optimization is intentionally deferred. No reciprocal
scaling, LUT contents, operand widths, pipeline stages, active reference ROMs,
or E-matrix ROMs were changed in this run. A future architecture task must be
separate from these frozen optical-convergence results and must repeat exact
bit-accurate/RTL equivalence plus every calibration-dependent campaign it
invalidates.
"""
    (out / "DEFERRED_WORK.md").write_text(deferred)

    text = f"""# Frozen-reconstructor and noisy-ensemble convergence evaluation

Run: `{cfg['run_id']}`  
Frozen source evidence: `blocker_investigation_v1_20260928_final`  
Scope: HCIPy and Python; no RTL rerun was needed because RTL, widths, reciprocal,
references, and E-matrix ROMs are byte-identical to the frozen source run.

## Executive result

The requested remaining tests completed. Forward-grid refinement was evaluated
first with the **same frozen production reference and E matrix** at every grid;
the already-completed per-grid recalibration remains a separately labeled
diagnostic. A new noisy ensemble then used 20 independent wavefronts with five
nested exposures in each of four predeclared non-clipping amplitude/flux cells.

The production settings do not satisfy deterministic convergence. Against the
highest tested settings, the six-case frozen-reconstructor study gives a mean
prediction change of {edge_prod['mean_reconstructed_wavefront_rms_change_nm']:.3f}
nm for edge 4→20 and
{prop_prod['mean_reconstructed_wavefront_rms_change_nm']:.3f} nm for propagation
1→5, with case maxima of {edge_prod['max_reconstructed_wavefront_rms_change_nm']:.3f}
and {prop_prod['max_reconstructed_wavefront_rms_change_nm']:.3f} nm respectively.
No finest-grid self-comparison is counted as evidence.

The new ensemble test accepted {len(accepted_cells)} of
{len(ensemble['paired_rows'])} predeclared comparison/condition cells under the
frozen two-part equivalence rule. {len(residual_only)} cells would have appeared
acceptable from residual means alone but failed the reconstructed-wavefront
difference requirement; they are not called converged.

## 1. Frozen production reconstructor: deterministic forward sensitivity

Every setting uses the same source-run reference ROMs, integer E-matrix ROMs,
reciprocal LUT, fixed-point reconstruction, 3e6-photon noiseless detector, and
common 240-base/20× edge-sampled scoring pupil. Refined optical intensity is block-summed into
the physical 256×256 detector pixels before detector conversion, integer ADC
rounding, and the unchanged 5 ADU threshold.

{markdown_table(['Study', 'Setting', 'Mean residual (nm)',
                 'Mean prediction Δ vs highest (nm)', 'Max prediction Δ (nm)',
                 'All six residual cases pass 5% and 0.5 nm'], det_table)}

`deterministic_frozen_paired.csv` records, per case, normalized detector-intensity
changes, expected and integer ADU changes, fixed-centroid changes, coefficient
changes, reconstructed-wavefront changes, and ground-truth residual changes.
Similar aggregate residuals are therefore not treated as equivalent predictions.

## 2. Per-grid recalibration diagnostic (kept separate)

The per-grid results were not substituted into the production path. They are
byte-copied from the blocker run into `reused_evidence/` and summarized in
`per_grid_recalibration_diagnostic_summary.csv`. They answer how much sensitivity
can be absorbed by recalibration at each numerical grid; the frozen path above
answers what the actual fixed reconstructor predicts when only the forward model
is refined. Their differing curves are shown together only for diagnosis.

## 3. New noisy-ensemble convergence test

The design and acceptance rule were frozen in the configuration before this
run: 20 and 100 nm RMS; 1e6 and 3e6 photons/pupil/frame; 20 independent
wavefronts; five nested exposures. Both fluxes are non-clipping in every tested
frame: **{ensemble['all_frames_non_clipping']}**; all frames remain below full
well: **{ensemble['all_frames_below_full_well']}**.

Noise pairing is explicit, not seed reuse by itself. At each pixel and exposure,
all grids share `Poisson(min λ)` and each receives an independent
`Poisson(λ_grid−min λ)` increment; their sum has exactly the required Poisson
marginal. One shared Gaussian read-noise field has the required N(0,3² e⁻)
marginal on every grid. Wavefront means over the five exposures are the 20
independent units used for Student-t intervals.

Acceptance requires both: (1) the 95% paired interval for signed residual change
lies wholly inside ±`min(0.5 nm, 5% of supported-reference mean residual)`, and
(2) the upper 95% interval for paired reconstructed-wavefront RMS difference is
below that same margin. A nonsignificant difference is not a pass.

{markdown_table(['Comparison', 'Type', 'Accepted cells', 'All cells',
                 'Max |mean residual Δ| (nm)', 'Max mean prediction Δ (nm)',
                 'Worst prediction upper/margin'], comparison_table)}

### Production settings versus supported references

{markdown_table(['Comparison', 'Input RMS (nm)', 'Photons', 'Reference residual (nm)',
                 'Residual Δ mean [95% CI] (nm)', 'Prediction Δ mean (upper 95%)',
                 'Margin (nm)', 'Decision'], production_table)}

This ensemble result is independent of—and does not override—the failed
noiseless per-case test. Full per-frame, per-wavefront, and uncertainty records
are in `ensemble_raw.csv`, `ensemble_paired_raw.csv`, and
`ensemble_paired_summary.csv`.

## 4. Completed evidence reused without rerun

- The 7.615181 nm noiseless fixed-flat result, including tip/tilt, is preserved.
  It was not re-investigated or corrected away.
- The clipping ablation is reused. At 8e6 photons, production float residual was
  {clipping_8m['production_float_residual_m_mean']*1e9:.2f} nm versus
  {clipping_8m['no_adc_clip_diagnostic_float_residual_m_mean']*1e9:.2f} nm with
  only the upper ADC clamp removed. Expected valid-tile electrons remain
  distinct from nominal QE×photons/168.
- The prior 35-case RTL suite and exact flat replay remain valid. No RTL
  simulation was rerun because no RTL or calibration input changed.

## 5. Manuscript use and limitations

Still usable, with the prior qualifications: the frozen-grid 300-wavefront
mixed residuals (4.03/4.76/6.49/12.63/24.77 nm primary float and
4.27/5.01/6.75/12.65/24.83 nm fixed at 20/50/100/200/400 nm RMS), finite-case
RTL equality, simulated nine-cycle latency, the fixed-flat trace, and the paired
clipping result. These are simulation results at the stated discretization and
detector assumptions—not numerically grid-independent optical predictions.

Whether any refinement comparison is stable is reported condition by condition
in the ensemble tables. A comparison is not promoted to a general convergence
claim unless all four cells pass the predeclared rule. Practical convergence of
the production optical model remains unresolved wherever that rule or a
successive deterministic comparison fails.

No physical-bench equivalence, FPGA timing closure, resource, power, synthesis,
or implementation claim is supported.

## 6. Claim-to-evidence

| Claim | Status | Evidence |
|---|---|---|
| Frozen production predictions are insensitive to tested refinements | {'ESTABLISHED' if all(r['all_cells_accepted'] for r in ensemble['comparison_rows']) else 'NOT ESTABLISHED'} | deterministic and ensemble paired tables |
| Per-grid recalibration is production calibration | NOT CLAIMED | diagnostic is explicitly separated |
| Noisy ensemble intervals establish agreement | CONDITION-SPECIFIC | `ensemble_paired_summary.csv`; frozen equivalence rule |
| 7.615181 nm fixed flat and active calibration are preserved | ESTABLISHED | ROM hashes, source flat summary, manifest |
| Clipping is a major but not sole high-flux effect in the assumed detector | COMPLETED PRIOR EVIDENCE | byte-copied clipping ablation |
| Physical bench or implemented FPGA performance | UNTESTED | outside scope |

## 7. Reproduction

```bash
sim/venv/bin/python -m unittest \\
  evaluation.tests.test_rom_loading \\
  evaluation.tests.test_reference_conventions -v
sim/venv/bin/python evaluation/optical/ensemble_convergence.py \\
  --config evaluation/configs/ensemble_convergence_v1.json \\
  --out evaluation/results/runs/{cfg['run_id']}_reproduction
```

The output directory must not exist. `MANIFEST.json` freezes the exact source,
configuration, calibration inputs, reused evidence hashes, dependency versions,
working-tree patch, result hashes, and commands. Reciprocal optimization is
listed only in `DEFERRED_WORK.md` and was not started.
"""
    (out / "REPORT.md").write_text(text)

    claims = [
        {"claim": "Frozen production predictions are insensitive to tested refinements",
         "status": ("ESTABLISHED" if all(row["all_cells_accepted"]
                                          for row in ensemble["comparison_rows"])
                    else "NOT ESTABLISHED"),
         "evidence": "deterministic_frozen_paired.csv; ensemble_paired_summary.csv"},
        {"claim": "Per-grid recalibration is production calibration",
         "status": "NOT CLAIMED",
         "evidence": "per_grid_recalibration_diagnostic_summary.csv is diagnostic only"},
        {"claim": "Noisy ensemble intervals establish agreement",
         "status": "CONDITION-SPECIFIC",
         "evidence": "ensemble_paired_summary.csv; ensemble_comparison_summary.csv"},
        {"claim": "7.615181 nm fixed flat and active calibration are preserved",
         "status": "ESTABLISHED",
         "evidence": "reused_evidence/flat_summary.json; MANIFEST.json ROM hashes"},
        {"claim": "Physical bench or implemented FPGA performance",
         "status": "UNTESTED", "evidence": "outside scope"},
    ]
    write_csv(out / "claim_to_evidence.csv", claims)


def freeze_provenance(cfg_path: Path, blocker_run: Path, calibration_run: Path,
                      out: Path, command: list[str], initial_status: list[str],
                      tracked_patch: bytes) -> dict:
    frozen = out / "frozen_inputs"
    source_snapshot = frozen / "source_snapshot"
    source_snapshot.mkdir(parents=True)
    shutil.copy2(cfg_path, frozen / "config.json")
    shutil.copytree(calibration_run / "frozen_inputs", frozen / "calibration")

    relevant = [
        Path(__file__).resolve(), cfg_path.resolve(),
        C.EVAL_DIR / "optical" / "blocker_investigation.py",
        C.EVAL_DIR / "optical" / "manuscript_evaluation.py",
        C.EVAL_DIR / "common" / "bitaccurate.py",
        C.EVAL_DIR / "common" / "config.py",
        C.EVAL_DIR / "common" / "detector.py",
        C.EVAL_DIR / "common" / "optics.py",
        C.EVAL_DIR / "common" / "roms.py",
        C.EVAL_DIR / "common" / "rtl_runner.py",
        C.EVAL_DIR / "tests" / "test_rom_loading.py",
        C.EVAL_DIR / "tests" / "test_reference_conventions.py",
    ]
    relevant += [C.RTL_DIR / name for name in R.RTL_SOURCES]
    relevant += [C.SIM_DIR / name for name in R.TB_SOURCES]
    for source in relevant:
        destination = source_snapshot / source.relative_to(C.REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    with gzip.open(frozen / "tracked_working_tree.patch.gz", "wb", compresslevel=9) as handle:
        handle.write(tracked_patch)
    (frozen / "git_status_porcelain.txt").write_text("\n".join(initial_status) + "\n")

    rom_names = ("slopes_ref_x.mem", "slopes_ref_y.mem", "e_matrix_x.mem",
                 "e_matrix_y.mem", "reciprocal_u16_q18.mem",
                 "subaperture_bitmap.mem")
    active_roms = {name: sha256(C.RTL_DIR / name) for name in rom_names}
    source_rom_dir = calibration_run / "frozen_inputs" / "roms"
    source_roms = {name: sha256(source_rom_dir / name) for name in rom_names}
    if active_roms != source_roms:
        raise RuntimeError("active RTL calibration/reciprocal ROMs changed during frozen run")

    frozen_files = sorted(path for path in frozen.rglob("*") if path.is_file())
    result_files = sorted(path for path in out.rglob("*")
                          if path.is_file() and frozen not in path.parents
                          and path.name != "MANIFEST.json")
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT, capture_output=True,
        text=True, check=True).stdout.strip()
    packages = {name: importlib.metadata.version(name)
                for name in ("hcipy", "numpy", "scipy", "matplotlib")}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "git_revision": git_revision,
        "git_dirty_before_run": bool(initial_status),
        "git_status_porcelain_before_run": initial_status,
        "tracked_patch": {
            "path": "frozen_inputs/tracked_working_tree.patch.gz",
            "uncompressed_bytes": len(tracked_patch),
            "sha256_uncompressed": sha256_bytes(tracked_patch),
        },
        "python": sys.version, "platform": platform.platform(),
        "dependencies": packages,
        "source_blocker_run": str(blocker_run.relative_to(C.REPO_ROOT)),
        "source_blocker_manifest_sha256": sha256(blocker_run / "MANIFEST.json"),
        "source_calibration_run": str(calibration_run.relative_to(C.REPO_ROOT)),
        "source_calibration_manifest_sha256": sha256(calibration_run / "MANIFEST.json"),
        "active_rom_sha256": active_roms,
        "frozen_source_rom_sha256": source_roms,
        "active_roms_equal_frozen_source": True,
        "rtl_simulation": ("not rerun: no RTL, arithmetic, reciprocal, reference, "
                           "or E-matrix change; prior finite simulation evidence reused"),
        "source_snapshot_paths": [str(path.relative_to(C.REPO_ROOT)) for path in relevant],
        "frozen_file_sha256": {str(path.relative_to(out)): sha256(path)
                               for path in frozen_files},
        "result_file_sha256": {str(path.relative_to(out)): sha256(path)
                               for path in result_files},
        "acceptance_rule_frozen_before_run": True,
        "configuration_sha256": sha256(cfg_path),
    }
    json_dump(out / "MANIFEST.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text())
    blocker_run = (C.REPO_ROOT / cfg["source_blocker_run"]).resolve()
    calibration_run = (C.REPO_ROOT / cfg["source_calibration_run"]).resolve()
    out = (args.out or C.RESULTS_DIR / "runs" / cfg["run_id"]).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")
    for required in (blocker_run / "MANIFEST.json", calibration_run / "MANIFEST.json"):
        if not required.is_file():
            raise FileNotFoundError(required)

    initial_status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=C.REPO_ROOT,
        capture_output=True, text=True, check=True).stdout.splitlines()
    tracked_patch = subprocess.run(
        ["git", "diff", "--binary"], cwd=C.REPO_ROOT,
        capture_output=True, check=True).stdout
    out.mkdir(parents=True)

    rom_dir = calibration_run / "frozen_inputs" / "roms"
    roms = load_frozen_roms(rom_dir)
    scale = float(json.loads(
        (calibration_run / "frozen_inputs" / "derived_scale.json").read_text()
    )["scale_m_per_lsb"])
    source_flat = json.loads((blocker_run / "flat_summary.json").read_text())
    expected_flat = cfg["freeze_policy"]["fixed_flat_result_nm"] * 1e-9
    if not np.isclose(source_flat["fixed_wavefront_rms_m"], expected_flat,
                      rtol=0.0, atol=1e-21):
        raise RuntimeError("source fixed-flat value does not match frozen configuration")
    scoring_bench = common_scoring_bench(cfg)

    print("[1/5] copy and hash completed flat/clipping/recalibration evidence", flush=True)
    reused = reuse_completed_evidence(blocker_run, out)
    print("[2/5] deterministic forward refinement with frozen production reconstructor",
          flush=True)
    deterministic = run_deterministic(
        cfg, out, blocker_run, roms, scale, scoring_bench)
    print("[3/5] paired noisy ensemble with exact marginal-preserving coupling", flush=True)
    ensemble = run_noisy_ensemble(cfg, out, roms, scale, scoring_bench)
    summary = {"deterministic": deterministic["summary"], "ensemble": ensemble,
               "reused_evidence": reused,
               "fixed_flat_result_m": source_flat["fixed_wavefront_rms_m"],
               "active_calibration_changed": False,
               "rtl_rerun": False}
    json_dump(out / "summary.json", summary)
    print("[4/5] figures, consolidated report, claims, and deferred-work entry", flush=True)
    make_figures(out)
    write_report(out, cfg, deterministic, ensemble, reused)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(cfg_path),
               "--out", str(out)]
    print("[5/5] exact source/configuration/calibration/result manifest", flush=True)
    freeze_provenance(cfg_path, blocker_run, calibration_run, out, command,
                      initial_status, tracked_patch)
    print(f"complete: {out}", flush=True)


if __name__ == "__main__":
    main()
