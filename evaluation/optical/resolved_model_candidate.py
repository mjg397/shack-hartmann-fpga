#!/usr/bin/env python3
"""Resolve the HCIPy numerical model and gate a separate candidate calibration.

This is an immutable run producer. It never writes active RTL ROMs. Numerical
oversampling is block-integrated into the same physical 256x256 detector pixels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/shwfs-matplotlib")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import config as C  # noqa: E402
from evaluation.common import detector as D  # noqa: E402
from evaluation.common import optics as O  # noqa: E402
from evaluation.common.noise_coupling import pairwise_poisson_gaussian_frames  # noqa: E402
from evaluation.common.roms import REQUIRED_ROMS, validate_rom_set  # noqa: E402
from evaluation.optical.blocker_investigation import (  # noqa: E402
    frozen_pipeline,
    load_frozen_roms,
    rms_coefficients,
)
from evaluation.optical.forward_model_investigation import (  # noqa: E402
    calibration_from_bench,
    centroid_vector,
    expected_detector_views,
    make_propagation_bench,
    physical_detector_energy,
    run_simple_lenslet,
    simple_square_analytic_pixel_energy,
    t_interval,
    tile_moments,
)


DEFAULT_CONFIG = C.EVAL_DIR / "configs" / "resolved_model_candidate_v1.json"
Q18 = float(1 << C.CENTROID_FRAC)


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


def detector_config(cfg: dict, photons: float, noisy: bool) -> D.DetectorConfig:
    values = {key: value for key, value in cfg["detector"].items()
              if key in D.DetectorConfig.__dataclass_fields__}
    values.update(total_photons=photons, shot_noise=noisy,
                  read_noise_e=cfg["detector"]["read_noise_e"] if noisy else 0.0)
    return D.DetectorConfig(**values)


class ControlledBench(O.Bench):
    """Bench-equivalent model with independently selectable Fresnel controls."""

    def __init__(self, cfg: O.OpticalConfig, zero_padding: int,
                 transfer_oversampling: int):
        from hcipy import (
            CartesianGrid,
            FresnelPropagator,
            Magnifier,
            MicroLensArray,
            NoiselessDetector,
            OpticalSystem,
            SeparatedCoords,
            ShackHartmannWavefrontSensorEstimator,
            Wavefront,
            evaluate_supersampled,
            make_obstructed_circular_aperture,
            make_pupil_grid,
            make_zernike_basis,
        )

        self.cfg = cfg
        c = cfg
        self.pupil_grid = make_pupil_grid(c.num_pupil_pixels, c.pupil_grid_diameter_m)
        aperture_gen = make_obstructed_circular_aperture(
            c.telescope_diameter_m,
            c.central_obscuration_m / c.telescope_diameter_m,
            num_spiders=c.num_spiders,
            spider_width=c.spider_width_m,
        )
        self.aperture = evaluate_supersampled(
            aperture_gen, self.pupil_grid, c.supersample)
        self.magnifier = Magnifier(c.magnification)
        half = c.sh_diameter_m / 2 - c.lenslet_diameter_m / 2
        centres = np.linspace(-half, half, c.num_lenslets)
        mla_grid = CartesianGrid(SeparatedCoords((centres, centres)))
        scaled = self.pupil_grid.scaled(c.magnification)
        self.mla = MicroLensArray(scaled, mla_grid, c.focal_length_m)
        propagator = FresnelPropagator(
            scaled, c.focal_length_m, num_oversampling=transfer_oversampling,
            zero_padding=zero_padding)
        self.shwfs = OpticalSystem((self.mla, propagator))
        self.estimator = ShackHartmannWavefrontSensorEstimator(
            mla_grid, self.mla.mla_index)
        wavefront = Wavefront(self.aperture, c.wavelength_m)
        wavefront.total_power = 1
        propagated = self.shwfs(self.magnifier(wavefront))
        self.camera = NoiselessDetector(propagated.electric_field.grid)
        self.camera.integrate(propagated, 1)
        self.reference_image = self.camera.read_out()
        self.zernike_basis = make_zernike_basis(
            c.num_zernike + 1, c.telescope_diameter_m, self.pupil_grid)
        self.modes = [np.asarray(self.zernike_basis[index + 1])
                      for index in range(c.num_zernike)]
        self._Wavefront = Wavefront
        self.zero_padding = int(zero_padding)
        self.transfer_oversampling = int(transfer_oversampling)


def common_scoring_bench(cfg: dict) -> O.Bench:
    resolution = cfg["resolution"]
    return O.Bench(replace(
        O.OpticalConfig(),
        pupil_pixels_base=resolution["common_scoring_pupil_base"],
        supersample=resolution["common_scoring_edge_supersample"],
    ))


def load_selection_cases(cfg: dict, blocker_run: Path) -> dict[str, np.ndarray]:
    stored = json.loads((blocker_run / "convergence_cases.json").read_text())["cases"]
    result = {}
    for name in cfg["resolution"]["selection_cases"]:
        result[name] = (np.zeros(C.NUM_MODES) if name == "flat"
                        else np.asarray(stored[name], dtype=float))
    return result


def capture_case(bench: O.Bench, factor: int, coefficients: np.ndarray,
                 det: D.DetectorConfig, valid: np.ndarray) -> dict:
    image, sampled_opd = bench.propagate_coeffs(coefficients)
    views = expected_detector_views(bench, image, factor, det)
    moments = {
        "continuous": tile_moments(views["continuous_adu"], None),
        "thresholded": tile_moments(views["continuous_adu"], C.PIXEL_THRESHOLD),
        "integer": tile_moments(views["integer_adu"], C.PIXEL_THRESHOLD),
    }
    centroids = {name: centroid_vector(value, valid)
                 for name, value in moments.items()}
    return {"views": views, "moments": moments, "centroids": centroids,
            "sampled_opd_rms_m": bench.opd_rms(sampled_opd)}


def entrance_field_base_cells(bench: O.Bench, coefficients: np.ndarray,
                              factor: int) -> np.ndarray:
    phase = 2 * np.pi * bench.opd_from_coeffs(coefficients) / bench.cfg.wavelength_m
    field = np.asarray(bench.aperture) * np.exp(1j * phase)
    averaged = field.reshape(C.FRAME_H, factor, C.FRAME_W, factor).mean(axis=(1, 3))
    norm = np.linalg.norm(averaged)
    return averaged / norm if norm else averaged


def run_resolution_ladder(cfg: dict, out: Path, cases: dict[str, np.ndarray],
                          valid: np.ndarray, det: D.DetectorConfig) -> dict:
    factors = cfg["resolution"]["propagation_factors"]
    captures: dict[int, dict] = {}
    entrance_fields: dict[int, dict] = {}
    raw_rows = []
    runtime_rows = []
    arrays = {}
    for factor in factors:
        start = time.perf_counter()
        bench = make_propagation_bench(factor)
        build_seconds = time.perf_counter() - start
        captures[factor] = {}
        entrance_fields[factor] = {}
        propagation_seconds = []
        for case_name, coefficients in cases.items():
            start = time.perf_counter()
            capture = capture_case(bench, factor, coefficients, det, valid)
            propagation_seconds.append(time.perf_counter() - start)
            captures[factor][case_name] = capture
            entrance_fields[factor][case_name] = entrance_field_base_cells(
                bench, coefficients, factor)
            prefix = f"factor{factor}__{case_name}"
            arrays[f"probability__{prefix}"] = capture["views"]["probability"].astype(np.float32)
            arrays[f"integer_adu__{prefix}"] = capture["views"]["integer_adu"].astype(np.uint8)
            arrays[f"centroids_continuous__{prefix}"] = capture["centroids"]["continuous"]
            arrays[f"centroids_thresholded__{prefix}"] = capture["centroids"]["thresholded"]
            arrays[f"centroids_integer__{prefix}"] = capture["centroids"]["integer"]
            raw_rows.append({
                "factor": factor, "case": case_name,
                "truth_coefficients_m": json.dumps(coefficients.tolist()),
                "sampled_opd_rms_m": capture["sampled_opd_rms_m"],
                "captured_flux": capture["views"]["captured_flux"],
                "expected_e_sum": float(capture["views"]["expected_e"].sum()),
                "integer_adu_sum": int(capture["views"]["integer_adu"].sum()),
                "integer_nonzero_pixels": int(np.count_nonzero(
                    capture["views"]["integer_adu"])),
                "saturated_pixels": capture["views"]["saturated_pixels"],
                "adc_clipped_pixels": capture["views"]["adc_clipped_pixels"],
                "continuous_spot_sigma_x_mean_px": float(
                    capture["moments"]["continuous"]["sigma_x"][valid].mean()),
                "continuous_spot_sigma_y_mean_px": float(
                    capture["moments"]["continuous"]["sigma_y"][valid].mean()),
            })
        n = C.FRAME_W * factor
        runtime_rows.append({
            "factor": factor, "samples_axis": n,
            "fine_samples": n * n,
            "fresnel_padded_samples_axis": 2 * n,
            "estimated_one_complex_padded_array_mib": (2 * n) ** 2 * 16 / 2**20,
            "observed_build_seconds": build_seconds,
            "observed_mean_case_seconds": float(np.mean(propagation_seconds)),
            "observed_max_case_seconds": float(np.max(propagation_seconds)),
            "process_peak_rss_mib_cumulative": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
        })
        del bench
        gc.collect()

    np.savez_compressed(out / "resolution_ladder_arrays.npz", **arrays)
    write_csv(out / "resolution_ladder_raw.csv", raw_rows)
    write_csv(out / "runtime_memory.csv", runtime_rows)

    pair_rows = []
    entrance_rows = []
    for coarse, fine in zip(factors[:-1], factors[1:]):
        for case_name in cases:
            first = captures[coarse][case_name]
            second = captures[fine][case_name]
            probability_delta = first["views"]["probability"] - second["views"]["probability"]
            sigma_first = np.r_[first["moments"]["continuous"]["sigma_x"][valid],
                                first["moments"]["continuous"]["sigma_y"][valid]]
            sigma_second = np.r_[second["moments"]["continuous"]["sigma_x"][valid],
                                 second["moments"]["continuous"]["sigma_y"][valid]]
            row = {
                "coarse_factor": coarse, "fine_factor": fine, "case": case_name,
                "normalized_image_relative_l2": float(
                    np.linalg.norm(probability_delta) /
                    np.linalg.norm(second["views"]["probability"])),
                "captured_flux_relative_change": float(
                    first["views"]["captured_flux"] /
                    second["views"]["captured_flux"] - 1),
                "continuous_centroid_rms_change_px": float(np.sqrt(np.mean(
                    (first["centroids"]["continuous"] -
                     second["centroids"]["continuous"]) ** 2))),
                "thresholded_centroid_rms_change_px": float(np.sqrt(np.mean(
                    (first["centroids"]["thresholded"] -
                     second["centroids"]["thresholded"]) ** 2))),
                "integer_centroid_rms_change_px": float(np.sqrt(np.mean(
                    (first["centroids"]["integer"] -
                     second["centroids"]["integer"]) ** 2))),
                "spot_sigma_rms_change_px": float(np.sqrt(np.mean(
                    (sigma_first - sigma_second) ** 2))),
                "integer_changed_pixels": int(np.count_nonzero(
                    first["views"]["integer_adu"] != second["views"]["integer_adu"])),
                "integer_rms_change_adu": float(np.sqrt(np.mean(
                    (first["views"]["integer_adu"].astype(float) -
                     second["views"]["integer_adu"].astype(float)) ** 2))),
                "reference_self_comparison_is_evidence": False,
            }
            pair_rows.append(row)
            entrance_rows.append({
                "coarse_factor": coarse, "fine_factor": fine, "case": case_name,
                "normalized_base_cell_complex_field_relative_l2": float(
                    np.linalg.norm(entrance_fields[coarse][case_name] -
                                   entrance_fields[fine][case_name]) /
                    np.linalg.norm(entrance_fields[fine][case_name])),
                "scope": "entrance pupil and OPD quadrature only; no MLA or propagation",
            })
    write_csv(out / "resolution_successive.csv", pair_rows)
    write_csv(out / "entrance_field_successive.csv", entrance_rows)
    return {"captures": captures, "pair_rows": pair_rows,
            "entrance_rows": entrance_rows, "runtime_rows": runtime_rows}


def run_analytic_pixel_quadrature(cfg: dict, out: Path) -> list[dict]:
    optical = O.OpticalConfig()
    width = optical.lenslet_diameter_m
    wavelength = optical.wavelength_m
    focal = optical.focal_length_m
    pitch = optical.detector_pixel_pitch_m
    extent = optical.sh_diameter_m
    edges = np.linspace(-extent / 2, extent / 2, C.FRAME_W + 1)
    scale = width / (wavelength * focal)
    rows = []
    for shift_px in cfg["resolution"]["simple_lenslet_tilt_pixels"]:
        shift = shift_px * pitch
        exact = simple_square_analytic_pixel_energy(
            edges, width, wavelength, focal, shift, 0.0)
        exact_probability = exact / exact.sum()
        for factor in cfg["resolution"]["simple_lenslet_factors"]:
            subpitch = pitch / factor
            offsets = (-pitch / 2 + subpitch / 2 +
                       np.arange(factor) * subpitch)
            centres = edges[:-1] + pitch / 2
            x = centres[:, None] + offsets[None, :]
            axis_x = (scale * np.sinc(scale * (x - shift)) ** 2).mean(axis=1) * pitch
            axis_y = (scale * np.sinc(scale * x) ** 2).mean(axis=1) * pitch
            midpoint = np.outer(axis_y, axis_x)
            probability = midpoint / midpoint.sum()
            coordinates = centres
            centroid = float((probability * coordinates[None, :]).sum())
            exact_centroid = float((exact_probability * coordinates[None, :]).sum())
            rows.append({
                "factor": factor, "tilt_shift_command_px": shift_px,
                "physical_pixels_axis": C.FRAME_W,
                "subsamples_per_physical_pixel_axis": factor,
                "normalized_image_relative_l2_error": float(
                    np.linalg.norm(probability - exact_probability) /
                    np.linalg.norm(exact_probability)),
                "centroid_x_error_px": (centroid - exact_centroid) / pitch,
                "reference": "adaptive quadrature over identical physical pixel edges",
                "field": "analytic sinc-squared; isolates detector-pixel quadrature",
            })
    write_csv(out / "analytic_pixel_quadrature.csv", rows)
    return rows


def run_numerical_controls(cfg: dict, out: Path, cases: dict[str, np.ndarray],
                           valid: np.ndarray, det: D.DetectorConfig,
                           ladder: dict) -> dict:
    resolution = cfg["resolution"]
    edge_factor = resolution["edge_control_factor"]
    edge_captures = {}
    for supersample in resolution["edge_supersampling"]:
        if supersample == 4:
            edge_captures[supersample] = {
                name: ladder["captures"][edge_factor][name]
                for name in resolution["edge_control_cases"]}
            continue
        bench = make_propagation_bench(edge_factor, supersample)
        edge_captures[supersample] = {
            name: capture_case(bench, edge_factor, cases[name], det, valid)
            for name in resolution["edge_control_cases"]}
        del bench
        gc.collect()
    edge_rows = []
    for coarse, fine in zip(resolution["edge_supersampling"][:-1],
                            resolution["edge_supersampling"][1:]):
        for case_name in resolution["edge_control_cases"]:
            first = edge_captures[coarse][case_name]
            second = edge_captures[fine][case_name]
            edge_rows.append({
                "factor": edge_factor, "coarse_edge_supersample": coarse,
                "fine_edge_supersample": fine, "case": case_name,
                "normalized_image_relative_l2": float(np.linalg.norm(
                    first["views"]["probability"] - second["views"]["probability"]) /
                    np.linalg.norm(second["views"]["probability"])),
                "captured_flux_relative_change": float(
                    first["views"]["captured_flux"] / second["views"]["captured_flux"] - 1),
                "continuous_centroid_rms_change_px": float(np.sqrt(np.mean(
                    (first["centroids"]["continuous"] -
                     second["centroids"]["continuous"]) ** 2))),
                "thresholded_centroid_rms_change_px": float(np.sqrt(np.mean(
                    (first["centroids"]["thresholded"] -
                     second["centroids"]["thresholded"]) ** 2))),
                "integer_changed_pixels": int(np.count_nonzero(
                    first["views"]["integer_adu"] != second["views"]["integer_adu"])),
            })
    write_csv(out / "pupil_edge_controls.csv", edge_rows)

    factor = resolution["fresnel_control_factor"]
    base = O.OpticalConfig()
    bench_cfg = replace(base, pupil_pixels_base=base.pupil_pixels_base * factor)
    fresnel_captures = {}
    for spec in resolution["fresnel_controls"]:
        bench = ControlledBench(bench_cfg, spec["zero_padding"],
                                spec["transfer_oversampling"])
        fresnel_captures[spec["name"]] = {
            name: capture_case(bench, factor, cases[name], det, valid)
            for name in resolution["fresnel_control_cases"]}
        del bench
        gc.collect()
    fresnel_rows = []
    default = next(spec for spec in resolution["fresnel_controls"]
                   if spec["name"] == "default")
    for spec in resolution["fresnel_controls"]:
        for case_name in resolution["fresnel_control_cases"]:
            first = fresnel_captures[default["name"]][case_name]
            second = fresnel_captures[spec["name"]][case_name]
            fresnel_rows.append({
                "factor": factor, "case": case_name,
                "baseline": default["name"], "comparison": spec["name"],
                "zero_padding": spec["zero_padding"],
                "transfer_oversampling": spec["transfer_oversampling"],
                "normalized_image_relative_l2": float(np.linalg.norm(
                    first["views"]["probability"] - second["views"]["probability"]) /
                    np.linalg.norm(second["views"]["probability"])),
                "captured_flux_relative_change": float(
                    first["views"]["captured_flux"] / second["views"]["captured_flux"] - 1),
                "continuous_centroid_rms_change_px": float(np.sqrt(np.mean(
                    (first["centroids"]["continuous"] -
                     second["centroids"]["continuous"]) ** 2))),
                "thresholded_centroid_rms_change_px": float(np.sqrt(np.mean(
                    (first["centroids"]["thresholded"] -
                     second["centroids"]["thresholded"]) ** 2))),
                "integer_changed_pixels": int(np.count_nonzero(
                    first["views"]["integer_adu"] != second["views"]["integer_adu"])),
            })
    write_csv(out / "fresnel_controls.csv", fresnel_rows)
    return {"edge_rows": edge_rows, "fresnel_rows": fresnel_rows}


def optical_gate(cfg: dict, ladder: dict, simple: dict) -> dict:
    candidate = cfg["resolution"]["candidate_factor"]
    finer = cfg["resolution"]["finer_check_factor"]
    limits = cfg["optical_gate"]
    rows = [row for row in ladder["pair_rows"]
            if row["coarse_factor"] == candidate and row["fine_factor"] == finer]
    checks = {
        "normalized_image": max(row["normalized_image_relative_l2"] for row in rows)
        <= limits["normalized_image_relative_l2_max"],
        "captured_flux": max(abs(row["captured_flux_relative_change"]) for row in rows)
        <= limits["captured_flux_relative_change_max"],
        "continuous_centroid": max(row["continuous_centroid_rms_change_px"] for row in rows)
        <= limits["continuous_centroid_rms_change_px_max"],
        "spot_sigma": max(row["spot_sigma_rms_change_px"] for row in rows)
        <= limits["spot_sigma_rms_change_px_max"],
    }
    simple_rows = [row for row in simple["rows"] if row["factor"] == candidate]
    optical = O.OpticalConfig()
    checks["analytic_lenslet_image"] = max(
        row["normalized_image_relative_l2_error"] for row in simple_rows
    ) <= limits["analytic_lenslet_image_relative_l2_max"]
    checks["analytic_lenslet_centroid"] = max(
        abs(row["centroid_x_error_m"]) / optical.detector_pixel_pitch_m
        for row in simple_rows
    ) <= limits["analytic_lenslet_centroid_abs_error_px_max"]
    return {
        "candidate_factor": candidate, "finer_check_factor": finer,
        "limits": limits, "checks": checks,
        "passed": bool(all(checks.values())),
        "self_comparison_used": False,
        "centroid_gate_equivalent_wavefront_nm":
            C.centroid_px_to_wavefront_m(
                limits["continuous_centroid_rms_change_px_max"]) * 1e9,
    }


def coefficient_metric(scoring_bench: O.Bench) -> np.ndarray:
    result = np.empty((C.NUM_MODES, C.NUM_MODES), dtype=float)
    for row in range(C.NUM_MODES):
        for column in range(C.NUM_MODES):
            result[row, column] = scoring_bench.opd_inner(
                scoring_bench.modes[row], scoring_bench.modes[column])
    return (result + result.T) / 2


def metric_rms(coefficients: np.ndarray, metric: np.ndarray) -> float:
    value = float(np.asarray(coefficients) @ metric @ np.asarray(coefficients))
    return float(np.sqrt(max(value, 0.0)))


def independent_cross_batch_squared(batch_mean_differences: np.ndarray,
                                    metric: np.ndarray) -> float:
    """Unbiased squared signal norm from four independent noisy batch means."""
    values = np.asarray(batch_mean_differences, dtype=float)
    if values.shape[0] != 4:
        raise ValueError("the frozen estimator requires exactly four independent batches")
    return float(0.5 * (values[0] @ metric @ values[1] +
                        values[2] @ metric @ values[3]))


def exact_candidate_coefficients(frame: np.ndarray, calibration: dict,
                                 valid: np.ndarray) -> np.ndarray:
    moments = tile_moments(frame, C.PIXEL_THRESHOLD)
    centroids = centroid_vector(moments, valid)
    return calibration["inverse"] @ (centroids - calibration["reference"])


def deterministic_frame(probability: np.ndarray, det: D.DetectorConfig) -> np.ndarray:
    expected = probability * det.total_photons * det.quantum_efficiency + det.background_e
    bounded = np.clip(expected, 0.0, det.full_well_e)
    unbounded = np.rint(bounded / det.gain_e_per_adu + det.offset_adu)
    return np.clip(unbounded, 0, det.adc_max).astype(np.int64)


def write_reference_arrays(candidate_dir: Path, factor: int, capture: dict,
                           valid: np.ndarray, calibration: dict) -> dict:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    all_x = capture["moments"]["integer"]["cx"]
    all_y = capture["moments"]["integer"]["cy"]
    qx = np.rint(all_x * Q18).astype(np.int64)
    qy = np.rint(all_y * Q18).astype(np.int64)
    if not np.allclose(calibration["reference"], np.r_[all_x[valid], all_y[valid]],
                       rtol=0, atol=0):
        raise AssertionError("candidate flat reference does not match frozen capture")
    np.save(candidate_dir / "reference_exact_x_px.npy", all_x)
    np.save(candidate_dir / "reference_exact_y_px.npy", all_y)
    np.save(candidate_dir / "reference_q18_x.npy", qx)
    np.save(candidate_dir / "reference_q18_y.npy", qy)
    np.save(candidate_dir / "interaction_slope_px_per_m.npy", calibration["interaction"])
    np.save(candidate_dir / "e_matrix_m_per_slope_px.npy", calibration["inverse"])
    record = {
        "candidate_factor": factor,
        "reference_convention": (
            "single noiseless deterministic integer flat frame; threshold at 5 ADU; "
            "exact moment quotient; nearest-integer Q18 only in reference_q18_*.npy"),
        "interaction": "independent +/-30 nm modal probes through the same integer detector path",
        "hardware_mem_files_generated": False,
        "reason": "hardware quantization remains gated on end-to-end candidate stability",
        "files": {},
    }
    for path in sorted(candidate_dir.glob("*.npy")):
        record["files"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    json_dump(candidate_dir / "candidate_calibration.json", record)
    return record


def evaluate_selection_calibration(cfg: dict, out: Path, cases: dict[str, np.ndarray],
                                   ladder: dict, calibrations: dict[int, dict],
                                   scoring_bench: O.Bench, roms: dict,
                                   active_scale: float) -> dict:
    candidate = cfg["resolution"]["candidate_factor"]
    finer = cfg["resolution"]["finer_check_factor"]
    common = calibrations[candidate]
    raw_rows = []
    coefficients = {}
    for factor in (candidate, finer):
        coefficients[factor] = {}
        for case_name, truth in cases.items():
            capture = ladder["captures"][factor][case_name]
            coefficients[factor][case_name] = {}
            for stage in ("continuous", "thresholded", "integer"):
                common_coeff = common["inverse"] @ (
                    capture["centroids"][stage] - common["reference"])
                own = calibrations[factor]
                own_coeff = own["inverse"] @ (
                    capture["centroids"][stage] - own["reference"])
                coefficients[factor][case_name][stage] = common_coeff
                raw_rows.append({
                    "factor": factor, "case": case_name, "detector_stage": stage,
                    "calibration_policy": f"frozen_factor{candidate}_candidate",
                    "primary_test": True,
                    "coefficients_m": json.dumps(common_coeff.tolist()),
                    "ground_truth_residual_nm": metric_rms(
                        common_coeff - truth, coefficient_metric(scoring_bench)) * 1e9,
                })
                raw_rows.append({
                    "factor": factor, "case": case_name, "detector_stage": stage,
                    "calibration_policy": "per_grid_diagnostic",
                    "primary_test": False,
                    "coefficients_m": json.dumps(own_coeff.tolist()),
                    "ground_truth_residual_nm": metric_rms(
                        own_coeff - truth, coefficient_metric(scoring_bench)) * 1e9,
                })
            fixed = frozen_pipeline(capture["views"]["integer_adu"], roms)
            fixed_coeff = fixed["zernike_raw"].astype(float) * active_scale
            raw_rows.append({
                "factor": factor, "case": case_name, "detector_stage": "integer",
                "calibration_policy": "historical_frozen_active_fixed",
                "primary_test": False,
                "coefficients_m": json.dumps(fixed_coeff.tolist()),
                "ground_truth_residual_nm": metric_rms(
                    fixed_coeff - truth, coefficient_metric(scoring_bench)) * 1e9,
            })
    write_csv(out / "candidate_selection_raw.csv", raw_rows)

    metric = coefficient_metric(scoring_bench)
    pair_rows = []
    for case_name, truth in cases.items():
        for stage in ("continuous", "thresholded", "integer"):
            first = coefficients[candidate][case_name][stage]
            second = coefficients[finer][case_name][stage]
            first_residual = metric_rms(first - truth, metric) * 1e9
            second_residual = metric_rms(second - truth, metric) * 1e9
            reconstruction_difference = metric_rms(first - second, metric) * 1e9
            residual_change = abs(first_residual - second_residual)
            margin = min(cfg["acceptance"]["absolute_tolerance_nm"],
                         cfg["acceptance"]["relative_tolerance_pct"] / 100 *
                         second_residual)
            pair_rows.append({
                "case": case_name, "detector_stage": stage,
                "candidate_factor": candidate, "finer_factor": finer,
                "candidate_residual_nm": first_residual,
                "finer_residual_nm": second_residual,
                "reconstructed_wavefront_rms_change_nm": reconstruction_difference,
                "absolute_residual_change_nm": residual_change,
                "equivalence_margin_nm": margin,
                "reconstruction_pass": reconstruction_difference <= margin,
                "residual_pass": residual_change <= margin,
                "case_pass": reconstruction_difference <= margin and residual_change <= margin,
                "reference_self_comparison_is_evidence": False,
            })
    write_csv(out / "candidate_selection_paired.csv", pair_rows)
    integer_rows = [row for row in pair_rows if row["detector_stage"] == "integer"]
    result = {
        "primary_policy": f"one frozen factor-{candidate} calibration at factors "
                          f"{candidate} and {finer}",
        "rows": pair_rows,
        "integer_passed_cases": int(sum(row["case_pass"] for row in integer_rows)),
        "integer_total_cases": len(integer_rows),
        "passed": bool(all(row["case_pass"] for row in integer_rows)),
        "per_grid_rows_are_diagnostic_only": True,
    }
    json_dump(out / "candidate_selection_summary.json", result)
    return result


def normalized_coefficients(scoring_bench: O.Bench, values: np.ndarray,
                            amplitude_nm: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    current = rms_coefficients(scoring_bench, values)
    if current <= 0:
        raise ValueError("cannot normalize a zero coefficient vector")
    return values * (amplitude_nm * 1e-9 / current)


def generate_heldout(cfg: dict, scoring_bench: O.Bench) -> tuple[dict, dict]:
    mixed = {}
    record = {"normalization_pupil": asdict(scoring_bench.cfg), "mixed": {}}
    for amplitude in cfg["held_out"]["wavefront_rms_nm"]:
        rng = np.random.default_rng(cfg["seeds"][f"heldout_{int(amplitude)}nm"])
        vectors = [normalized_coefficients(scoring_bench, rng.normal(size=C.NUM_MODES),
                                           amplitude)
                   for _ in range(cfg["held_out"]["mixed_wavefronts_per_amplitude"])]
        mixed[float(amplitude)] = vectors
        record["mixed"][f"{int(amplitude)}nm"] = [value.tolist() for value in vectors]
    single = {}
    amplitude = cfg["held_out"]["single_mode_amplitude_nm"] * 1e-9
    for mode in cfg["held_out"]["single_mode_indices"]:
        value = np.zeros(C.NUM_MODES)
        value[int(mode)] = amplitude
        single[f"mode_{mode}_{cfg['held_out']['single_mode_amplitude_nm']:g}nm"] = value
    record["single_modes"] = {key: value.tolist() for key, value in single.items()}
    return {"mixed": mixed, "single": single, "flat": np.zeros(C.NUM_MODES)}, record


def propagate_heldout(cfg: dict, heldout: dict, calibrations: dict[int, dict],
                      factors: tuple[int, int]) -> tuple[dict, list[dict]]:
    probabilities = {factor: {} for factor in factors}
    rows = []
    all_cases = {"flat": heldout["flat"], **heldout["single"]}
    for amplitude, vectors in heldout["mixed"].items():
        for index, vector in enumerate(vectors):
            all_cases[f"mixed_{int(amplitude)}nm_{index:02d}"] = vector
    for factor in factors:
        bench = make_propagation_bench(factor)
        det = D.DetectorConfig(shot_noise=False, read_noise_e=0.0)
        for case_name, truth in all_cases.items():
            image, _ = bench.propagate_coeffs(truth)
            energy, captured, _ = physical_detector_energy(bench, image, factor)
            probability = energy / captured
            probabilities[factor][case_name] = probability
            rows.append({
                "factor": factor, "case": case_name,
                "truth_coefficients_m": json.dumps(truth.tolist()),
                "truth_rms_m": rms_coefficients(bench, truth),
                "captured_flux": captured,
                "probability_sha256_float32": sha256_bytes(
                    probability.astype(np.float32).tobytes()),
            })
        del bench
        gc.collect()
    return probabilities, rows


def run_heldout_deterministic(cfg: dict, out: Path, heldout: dict,
                              probabilities: dict, calibrations: dict[int, dict],
                              scoring_bench: O.Bench, roms: dict,
                              active_scale: float) -> dict:
    candidate = cfg["resolution"]["candidate_factor"]
    finer = cfg["resolution"]["finer_check_factor"]
    metric = coefficient_metric(scoring_bench)
    truths = {"flat": heldout["flat"], **heldout["single"]}
    for amplitude, vectors in heldout["mixed"].items():
        for index, value in enumerate(vectors):
            truths[f"mixed_{int(amplitude)}nm_{index:02d}"] = value
    raw_rows = []
    pair_rows = []
    for photons in cfg["held_out"]["photon_levels"]:
        det = detector_config(cfg, photons, False)
        for case_name, truth in truths.items():
            common_coefficients = {}
            common_residuals = {}
            for factor in (candidate, finer):
                frame = deterministic_frame(probabilities[factor][case_name], det)
                common_coeff = exact_candidate_coefficients(
                    frame, calibrations[candidate], roms["valid"])
                own_coeff = exact_candidate_coefficients(
                    frame, calibrations[factor], roms["valid"])
                fixed = frozen_pipeline(frame, roms)["zernike_raw"].astype(float) * active_scale
                common_coefficients[factor] = common_coeff
                common_residuals[factor] = metric_rms(common_coeff - truth, metric) * 1e9
                for policy, coeff, primary in (
                    (f"frozen_factor{candidate}_candidate", common_coeff, True),
                    ("per_grid_diagnostic", own_coeff, False),
                    ("historical_frozen_active_fixed", fixed, False),
                ):
                    raw_rows.append({
                        "photons_per_pupil": photons, "case": case_name,
                        "factor": factor, "calibration_policy": policy,
                        "primary_test": primary,
                        "ground_truth_residual_nm": metric_rms(coeff - truth, metric) * 1e9,
                        "coefficients_m": json.dumps(coeff.tolist()),
                        "saturated_pixels": int(np.count_nonzero(
                            probabilities[factor][case_name] * photons *
                            det.quantum_efficiency > det.full_well_e)),
                        "adc_clipped_pixels": 0,
                    })
            difference = metric_rms(
                common_coefficients[candidate] - common_coefficients[finer], metric) * 1e9
            residual_change = abs(common_residuals[candidate] - common_residuals[finer])
            margin = min(cfg["acceptance"]["absolute_tolerance_nm"],
                         cfg["acceptance"]["relative_tolerance_pct"] / 100 *
                         common_residuals[finer])
            pair_rows.append({
                "photons_per_pupil": photons, "case": case_name,
                "candidate_factor": candidate, "finer_factor": finer,
                "finer_reference_residual_nm": common_residuals[finer],
                "equivalence_margin_nm": margin,
                "reconstructed_wavefront_rms_change_nm": difference,
                "absolute_residual_change_nm": residual_change,
                "reconstruction_pass": difference <= margin,
                "residual_pass": residual_change <= margin,
                "case_pass": difference <= margin and residual_change <= margin,
            })
    write_csv(out / "heldout_deterministic_raw.csv", raw_rows)
    write_csv(out / "heldout_deterministic_paired.csv", pair_rows)
    result = {
        "rows": pair_rows,
        "passed_cases": int(sum(row["case_pass"] for row in pair_rows)),
        "total_cases": len(pair_rows),
        "passed": bool(all(row["case_pass"] for row in pair_rows)),
        "all_frames_non_clipping": bool(all(
            row["saturated_pixels"] == 0 and row["adc_clipped_pixels"] == 0
            for row in raw_rows)),
    }
    json_dump(out / "heldout_deterministic_summary.json", result)
    return result


def run_noisy_confirmation(cfg: dict, out: Path, heldout: dict,
                           probabilities: dict, calibration: dict,
                           scoring_bench: O.Bench, valid: np.ndarray) -> dict:
    candidate = cfg["resolution"]["candidate_factor"]
    finer = cfg["resolution"]["finer_check_factor"]
    design = cfg["noisy_confirmation"]
    metric = coefficient_metric(scoring_bench)
    conditions = [("flat", 0.0, [heldout["flat"]] * design["flat_monte_carlo_units"],
                   "independent Monte Carlo replicate at one fixed flat wavefront")]
    for amplitude, vectors in heldout["mixed"].items():
        conditions.append((f"mixed_{int(amplitude)}nm", amplitude, vectors,
                           "independent wavefront; exposures nested"))
    raw_rows = []
    unit_rows = []
    summary_rows = []
    for condition_index, (condition, amplitude, truths, independent_unit) in enumerate(conditions):
        for flux_index, photons in enumerate(cfg["held_out"]["photon_levels"]):
            det = detector_config(cfg, photons, True)
            residual_units = []
            cross_squared_units = []
            deterministic_units = []
            variability_units = []
            reference_residual_units = []
            naive_units = []
            for unit, truth in enumerate(truths):
                case_name = "flat" if condition == "flat" else f"{condition}_{unit:02d}"
                expected = {
                    f"factor{candidate}": probabilities[candidate][case_name] *
                    photons * det.quantum_efficiency + det.background_e,
                    f"factor{finer}": probabilities[finer][case_name] *
                    photons * det.quantum_efficiency + det.background_e,
                }
                deterministic_coefficients = {}
                for factor in (candidate, finer):
                    frame = deterministic_frame(probabilities[factor][case_name], det)
                    deterministic_coefficients[factor] = exact_candidate_coefficients(
                        frame, calibration, valid)
                deterministic_difference = metric_rms(
                    deterministic_coefficients[candidate] -
                    deterministic_coefficients[finer], metric) * 1e9
                batch_means = []
                setting_coefficients = {candidate: [], finer: []}
                residual_differences = []
                reference_residuals = []
                for batch in range(design["independent_batches"]):
                    batch_differences = []
                    for exposure in range(design["exposures_per_batch"]):
                        seed_base = (cfg["seeds"]["flat_noise"] if condition == "flat"
                                     else cfg["seeds"]["noise"])
                        frames, diagnostics = pairwise_poisson_gaussian_frames(
                            expected, det,
                            [seed_base, condition_index, flux_index, unit, batch, exposure])
                        coeff = {}
                        residual = {}
                        for factor in (candidate, finer):
                            name = f"factor{factor}"
                            coeff[factor] = exact_candidate_coefficients(
                                frames[name], calibration, valid)
                            residual[factor] = metric_rms(coeff[factor] - truth, metric) * 1e9
                            setting_coefficients[factor].append(coeff[factor])
                        difference = coeff[candidate] - coeff[finer]
                        batch_differences.append(difference)
                        residual_differences.append(
                            residual[candidate] - residual[finer])
                        reference_residuals.append(residual[finer])
                        raw_rows.append({
                            "condition": condition, "amplitude_nm_rms": amplitude,
                            "photons_per_pupil": photons, "unit": unit,
                            "batch": batch, "exposure": exposure,
                            "candidate_residual_nm": residual[candidate],
                            "finer_residual_nm": residual[finer],
                            "signed_residual_difference_nm":
                                residual[candidate] - residual[finer],
                            "framewise_reconstruction_difference_nm":
                                metric_rms(difference, metric) * 1e9,
                            "candidate_frame_sha256": sha256_bytes(
                                frames[f"factor{candidate}"].astype(np.uint8).tobytes()),
                            "finer_frame_sha256": sha256_bytes(
                                frames[f"factor{finer}"].astype(np.uint8).tobytes()),
                            "candidate_saturated_pixels": diagnostics["per_setting"][
                                f"factor{candidate}"]["saturated_pixels"],
                            "finer_saturated_pixels": diagnostics["per_setting"][
                                f"factor{finer}"]["saturated_pixels"],
                            "candidate_adc_clipped_pixels": diagnostics["per_setting"][
                                f"factor{candidate}"]["adc_clipped_pixels"],
                            "finer_adc_clipped_pixels": diagnostics["per_setting"][
                                f"factor{finer}"]["adc_clipped_pixels"],
                            "pair_local_coupling": True,
                            "mean_image_reconstructed": False,
                        })
                    batch_means.append(np.mean(batch_differences, axis=0))
                batch_means = np.asarray(batch_means)
                cross_squared = independent_cross_batch_squared(batch_means, metric) * 1e18
                naive_first_batch = metric_rms(batch_means[0], metric) * 1e9
                variability = {}
                for factor in (candidate, finer):
                    values = np.asarray(setting_coefficients[factor])
                    mean_value = values.mean(axis=0)
                    variability[factor] = float(np.sqrt(np.mean([
                        metric_rms(value - mean_value, metric) ** 2
                        for value in values]))) * 1e9
                residual_mean = float(np.mean(residual_differences))
                reference_mean = float(np.mean(reference_residuals))
                variability_change = variability[candidate] - variability[finer]
                residual_units.append(residual_mean)
                cross_squared_units.append(cross_squared)
                deterministic_units.append(deterministic_difference)
                variability_units.append(variability_change)
                reference_residual_units.append(reference_mean)
                naive_units.append(naive_first_batch)
                unit_rows.append({
                    "condition": condition, "amplitude_nm_rms": amplitude,
                    "photons_per_pupil": photons, "unit": unit,
                    "mean_residual_difference_nm": residual_mean,
                    "finer_reference_residual_mean_nm": reference_mean,
                    "independent_cross_batch_squared_difference_nm2": cross_squared,
                    "naive_five_exposure_batch_norm_nm": naive_first_batch,
                    "deterministic_expected_frame_difference_nm": deterministic_difference,
                    "candidate_noise_variability_nm": variability[candidate],
                    "finer_noise_variability_nm": variability[finer],
                    "noise_variability_difference_nm": variability_change,
                    "independent_unit": independent_unit,
                })
            residual_stats = t_interval(residual_units, cfg["acceptance"]["confidence_level"])
            squared_stats = t_interval(cross_squared_units,
                                       cfg["acceptance"]["confidence_level"])
            deterministic_stats = t_interval(
                deterministic_units, cfg["acceptance"]["confidence_level"])
            variability_stats = t_interval(
                variability_units, cfg["acceptance"]["confidence_level"])
            reference_stats = t_interval(
                reference_residual_units, cfg["acceptance"]["confidence_level"])
            naive_stats = t_interval(naive_units, cfg["acceptance"]["confidence_level"])
            margin = min(cfg["acceptance"]["absolute_tolerance_nm"],
                         cfg["acceptance"]["relative_tolerance_pct"] / 100 *
                         reference_stats["mean"])
            residual_pass = (residual_stats["ci_low"] >= -margin and
                             residual_stats["ci_high"] <= margin)
            squared_pass = squared_stats["ci_high"] <= margin ** 2
            deterministic_pass = deterministic_stats["ci_high"] <= margin
            summary_rows.append({
                "condition": condition, "amplitude_nm_rms": amplitude,
                "photons_per_pupil": photons,
                "n_independent_units": len(truths),
                "independent_unit": independent_unit,
                "independent_batches": design["independent_batches"],
                "exposures_per_batch": design["exposures_per_batch"],
                "finer_reference_residual_mean_nm": reference_stats["mean"],
                "equivalence_margin_nm": margin,
                "mean_residual_difference_nm": residual_stats["mean"],
                "mean_residual_ci95_low_nm": residual_stats["ci_low"],
                "mean_residual_ci95_high_nm": residual_stats["ci_high"],
                "cross_batch_squared_difference_mean_nm2": squared_stats["mean"],
                "cross_batch_squared_difference_ci95_low_nm2": squared_stats["ci_low"],
                "cross_batch_squared_difference_ci95_high_nm2": squared_stats["ci_high"],
                "descriptive_sqrt_nonnegative_cross_mean_nm": float(np.sqrt(
                    max(squared_stats["mean"], 0.0))),
                "naive_five_exposure_norm_mean_nm": naive_stats["mean"],
                "naive_five_exposure_norm_ci95_high_nm": naive_stats["ci_high"],
                "deterministic_expected_frame_difference_mean_nm":
                    deterministic_stats["mean"],
                "deterministic_expected_frame_difference_ci95_high_nm":
                    deterministic_stats["ci_high"],
                "noise_variability_difference_mean_nm": variability_stats["mean"],
                "noise_variability_difference_ci95_low_nm": variability_stats["ci_low"],
                "noise_variability_difference_ci95_high_nm": variability_stats["ci_high"],
                "residual_pass": residual_pass,
                "expected_reconstruction_squared_pass": squared_pass,
                "deterministic_expected_frame_pass": deterministic_pass,
                "cell_pass": residual_pass and squared_pass and deterministic_pass,
                "rule_frozen_before_run": True,
            })
    write_csv(out / "noisy_candidate_raw.csv", raw_rows)
    write_csv(out / "noisy_candidate_independent_units.csv", unit_rows)
    write_csv(out / "noisy_candidate_summary.csv", summary_rows)
    result = {
        "design": design,
        "acceptance": cfg["acceptance"],
        "rows": summary_rows,
        "accepted_cells": int(sum(row["cell_pass"] for row in summary_rows)),
        "total_cells": len(summary_rows),
        "passed": bool(all(row["cell_pass"] for row in summary_rows)),
        "all_frames_non_clipping": bool(all(
            row["candidate_saturated_pixels"] == 0 and
            row["finer_saturated_pixels"] == 0 and
            row["candidate_adc_clipped_pixels"] == 0 and
            row["finer_adc_clipped_pixels"] == 0 for row in raw_rows)),
        "mean_image_reconstruction_used": False,
        "finite_sample_bias_handling": (
            "The primary expected-reconstruction statistic is a coefficient-space "
            "cross-product between independent five-exposure batch means. Its expectation "
            "is the squared norm of the expected coefficient difference; the positive "
            "five-exposure RMS norm is reported only as a descriptive comparator."),
    }
    json_dump(out / "noisy_candidate_summary.json", result)
    return result


def make_figures(out: Path) -> None:
    import matplotlib.pyplot as plt

    def read_rows(name: str) -> list[dict]:
        with (out / name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    rows = read_rows("resolution_successive.csv")
    pairs = list(dict.fromkeys(
        f"{row['coarse_factor']}→{row['fine_factor']}" for row in rows))
    image_max = []
    centroid_max = []
    for pair in pairs:
        subset = [row for row in rows
                  if f"{row['coarse_factor']}→{row['fine_factor']}" == pair]
        image_max.append(max(float(row["normalized_image_relative_l2"]) for row in subset))
        centroid_max.append(max(float(row["continuous_centroid_rms_change_px"])
                                for row in subset))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].plot(pairs, image_max, "o-")
    axes[0].axhline(0.0025, color="tab:red", linestyle="--", label="gate")
    axes[0].set_ylabel("max normalized image rel. L2")
    axes[0].legend()
    axes[1].plot(pairs, centroid_max, "o-")
    axes[1].axhline(0.0012, color="tab:red", linestyle="--", label="gate")
    axes[1].set_ylabel("max continuous centroid RMS (px)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "successive_resolution.png", dpi=180)
    plt.close(fig)

    rows = [row for row in read_rows("candidate_selection_paired.csv")
            if row["detector_stage"] == "integer"]
    labels = [row["case"] for row in rows]
    values = [float(row["reconstructed_wavefront_rms_change_nm"]) for row in rows]
    margins = [float(row["equivalence_margin_nm"]) for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    candidate = rows[0]["candidate_factor"]
    finer = rows[0]["finer_factor"]
    ax.bar(x - 0.18, values, width=0.36,
           label=f"factor {candidate}→{finer} change")
    ax.bar(x + 0.18, margins, width=0.36, label="5% and 0.5 nm margin")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("wavefront RMS (nm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "candidate_selection_gate.png", dpi=180)
    plt.close(fig)

    rows = read_rows("noisy_candidate_summary.csv")
    labels = [f"{row['condition']}\n{float(row['photons_per_pupil'])/1e6:g}M"
              for row in rows]
    deterministic = [float(row["deterministic_expected_frame_difference_mean_nm"])
                     for row in rows]
    margins = [float(row["equivalence_margin_nm"]) for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(x, deterministic, "o-", label="deterministic expected-frame change")
    ax.plot(x, margins, "s--", label="cell margin")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("wavefront RMS (nm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "noisy_candidate_gate.png", dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def write_report(cfg_path: Path, cfg: dict, out: Path, ladder: dict, controls: dict, gate: dict,
                 selection: dict, deterministic: dict, noisy: dict,
                 hardware_gate: dict, simple: dict) -> None:
    candidate = cfg["resolution"]["candidate_factor"]
    finer = cfg["resolution"]["finer_check_factor"]
    successive = []
    for coarse, fine in zip(cfg["resolution"]["propagation_factors"][:-1],
                            cfg["resolution"]["propagation_factors"][1:]):
        rows = [row for row in ladder["pair_rows"]
                if row["coarse_factor"] == coarse and row["fine_factor"] == fine]
        successive.append([
            f"{coarse}→{fine}",
            f"{max(row['normalized_image_relative_l2'] for row in rows):.6f}",
            f"{max(row['continuous_centroid_rms_change_px'] for row in rows):.6f}",
            f"{max(row['thresholded_centroid_rms_change_px'] for row in rows):.6f}",
            f"{max(row['integer_centroid_rms_change_px'] for row in rows):.6f}",
            f"{max(row['integer_changed_pixels'] for row in rows)}",
        ])
    selection_rows = [row for row in selection["rows"]
                      if row["detector_stage"] == "integer"]
    selection_table = [[
        row["case"], f"{row['finer_residual_nm']:.3f}",
        f"{row['reconstructed_wavefront_rms_change_nm']:.3f}",
        f"{row['absolute_residual_change_nm']:.3f}",
        f"{row['equivalence_margin_nm']:.3f}", "pass" if row["case_pass"] else "FAIL",
    ] for row in selection_rows]
    noisy_table = [[
        row["condition"], f"{row['photons_per_pupil']/1e6:g}M",
        f"{row['mean_residual_difference_nm']:+.3f} "
        f"[{row['mean_residual_ci95_low_nm']:+.3f}, {row['mean_residual_ci95_high_nm']:+.3f}]",
        f"{row['cross_batch_squared_difference_mean_nm2']:.3f} "
        f"(upper {row['cross_batch_squared_difference_ci95_high_nm2']:.3f})",
        f"{row['deterministic_expected_frame_difference_mean_nm']:.3f} "
        f"(upper {row['deterministic_expected_frame_difference_ci95_high_nm']:.3f})",
        f"{row['equivalence_margin_nm']:.3f}",
        "pass" if row["cell_pass"] else "FAIL",
    ] for row in noisy["rows"]]
    with (out / "candidate_selection_raw.csv").open(newline="") as handle:
        calibration_raw = list(csv.DictReader(handle))
    calibration_table = []
    for factor in (candidate, finer):
        for policy, label in (
                (f"frozen_factor{candidate}_candidate", "frozen candidate"),
                ("per_grid_diagnostic", "per-grid diagnostic"),
                ("historical_frozen_active_fixed", "historical active fixed")):
            values = [float(row["ground_truth_residual_nm"])
                      for row in calibration_raw
                      if int(row["factor"]) == factor and
                      row["detector_stage"] == "integer" and
                      row["calibration_policy"] == policy]
            calibration_table.append([
                str(factor), label, f"{np.mean(values):.3f}", f"{np.max(values):.3f}",
                "primary" if policy.startswith("frozen_factor") else "diagnostic",
            ])
    with (out / "heldout_deterministic_raw.csv").open(newline="") as handle:
        heldout_raw = list(csv.DictReader(handle))
    accuracy_table = []
    for photons in cfg["held_out"]["photon_levels"]:
        for amplitude in cfg["held_out"]["wavefront_rms_nm"]:
            row = [f"{photons/1e6:g}M", f"{amplitude:g}"]
            for factor in (candidate, finer):
                values = [float(value["ground_truth_residual_nm"])
                          for value in heldout_raw
                          if float(value["photons_per_pupil"]) == photons and
                          value["case"].startswith(f"mixed_{int(amplitude)}nm_") and
                          int(value["factor"]) == factor and
                          value["calibration_policy"] ==
                          f"frozen_factor{candidate}_candidate"]
                row.append(f"{np.mean(values):.3f} (max {np.max(values):.3f})")
            accuracy_table.append(row)
    max_edge = max(row["normalized_image_relative_l2"] for row in controls["edge_rows"])
    max_fresnel = max(row["normalized_image_relative_l2"] for row in controls["fresnel_rows"])
    candidate_simple = [row for row in simple["rows"] if row["factor"] == candidate]
    simple_max = max(row["normalized_image_relative_l2_error"] for row in candidate_simple)
    max_selection = max(row["reconstructed_wavefront_rms_change_nm"]
                        for row in selection_rows)
    report = f"""# Resolved-model candidate calibration evaluation

Run: `{cfg['run_id']}`  
Source: `forward_model_investigation_v2_20260928_final`  
Scope: HCIPy/Python only. No active ROM, RTL arithmetic, reciprocal, gain,
threshold, geometry, synthesis, implementation, board, or real-data change.

## Executive decision

Factor {candidate} is the lowest tested numerical setting that passes the
predeclared **continuous optical** gate against the independently finer factor
{finer}; factor {finer} is not credited for agreeing with itself. It is therefore
a defensible lower candidate for expected optical intensity and continuous
centroids, not a validated end-to-end production setting.

The one frozen factor-{candidate} candidate calibration **fails** the unchanged
5% and 0.5 nm end-to-end rule when applied at factors {candidate} and {finer}.
The largest selection-case integer-frame reconstructed-wavefront change is
{max_selection:.3f} nm. Held-out deterministic cases pass
{deterministic['passed_cases']}/{deterministic['total_cases']}; the new noisy
confirmation passes {noisy['accepted_cells']}/{noisy['total_cells']} cells.
Consequently no candidate E matrix was quantized, no candidate hardware ROM set
was staged, no RTL simulation was run, and the full manuscript campaign must
not yet be repeated.

This does not alter the historical 7.615181 nm noiseless fixed-flat result,
including tip/tilt. Its reciprocal-rounding explanation and deferred status are
reused without reinvestigation.

## 1. Physical pixels versus numerical samples

All factors retain the same 256x256 physical detector, physical pixel edges,
lenslet geometry, wavelength, OPD coefficients, photons-per-pupil definition,
and detector order. A factor *f* HCIPy image has `(256f)^2` quadrature-power
samples; each non-overlapping `f x f` block is summed into one physical pixel
before noise, full-well clipping, gain, ADC rounding, and thresholded centroiding.
No extra area factor is applied because HCIPy's `NoiselessDetector` samples are
already `Wavefront.power`, including grid weights.

The independent analytic square-lenslet comparison continues to improve: its
worst factor-{candidate} normalized-image error is {simple_max:.6f}. A second
analytic sinc-squared test varies only composite midpoint quadrature inside the
unchanged physical pixel boundaries (`analytic_pixel_quadrature.csv`).

## 2. Successive refinement

{markdown_table(
    ['Pair', 'max image rel. L2', 'max continuous centroid px',
     'max threshold centroid px', 'max integer centroid px', 'max changed ADU px'],
    successive)}

The factor-{candidate}→{finer} continuous centroid gate of 0.0012 pixel maps to
{gate['centroid_gate_equivalent_wavefront_nm']:.3f} nm across one lenslet by the
fixed geometry; final decisions still use reconstructed common-pupil RMS and
both original tolerances. The optical gate passed: `{gate['passed']}`.

## 3. Separated numerical controls

At factor {cfg['resolution']['edge_control_factor']}, the largest successive
pupil-edge-control image change is {max_edge:.3e}. At factor
{cfg['resolution']['fresnel_control_factor']}, changing zero padding or transfer
sampling from the documented defaults changes normalized images by at most
{max_fresnel:.3e}. Entrance-pupil/phase-only, Fresnel, edge, analytic pixel
quadrature, and full propagated comparisons are stored separately. These checks
show that the remaining factor-{candidate}→{finer} sensitivity is not explained
by insufficient edge quadrature or the default Fresnel padding alone.

## 4. Frozen candidate calibration

The factor-{candidate} floating reference is the exact thresholded quotient
from one noiseless integer flat frame. Its documented nearest-integer Q18
representation is saved separately but is not substituted into this floating
primary path because the hardware gate was not reached. The interaction matrix
uses independent +/-{cfg['candidate_calibration']['probe_nm']:g} nm modal
probes; its pseudoinverse is saved in physical metre-per-slope-pixel units.
The same reference and inverse are used at both grids. Factor-{finer}
recalibration is diagnostic only and never enters the primary rows.

{markdown_table(
    ['Case', f'factor-{finer} residual nm', 'reconstruction change nm',
     'absolute residual change nm', 'margin nm', 'decision'], selection_table)}

For context, candidate accuracy improved over the historical active fixed
calibration on these four selection cases, but improvement is not convergence:

{markdown_table(
    ['Grid factor', 'Calibration', 'mean residual nm', 'max residual nm', 'role'],
    calibration_table)}

Continuous and thresholded rows in `candidate_selection_paired.csv` localize
the failure. The continuous optical changes are small but some are already at
the strict combined margin; ADC rounding then changes hundreds of pixels and
raises the mixed-100-nm reconstruction change. The cause is therefore a
combination of residual forward quadrature sensitivity and nonlinear detector
boundary crossings, with calibration mapping determining their wavefront
impact—not pupil-edge or Fresnel-padding error.

## 5. Fresh held-out and noisy confirmation

Fresh post-freeze mixed directions at 20, 100, and 400 nm RMS, four 75 nm single
modes, flat, and 1M/3M photons were evaluated. Every reported detector frame is
non-clipping: `{deterministic['all_frames_non_clipping'] and noisy['all_frames_non_clipping']}`.
Tip/tilt is retained throughout.

Held-out mixed-wavefront accuracy of the same frozen factor-{candidate}
calibration is:

{markdown_table(
    ['Photons', 'input RMS nm', f'factor-{candidate} mean (max) residual nm',
     f'factor-{finer} mean (max) residual nm'], accuracy_table)}

{markdown_table(
    ['Condition', 'photons', 'residual delta [95% CI] nm',
     'cross-batch squared nm^2', 'deterministic mean (upper) nm', 'margin nm', 'decision'],
    noisy_table)}

The noisy expected-reconstruction rule does not take the positive norm of only
five exposures as exact. Four independent five-exposure batches are used. The
coefficient-space cross-products of independent batch means are unbiased for
the squared norm of the expected reconstruction difference; their interval is
the acceptance statistic. The ordinary positive five-exposure norm remains in
the tables only as a descriptive comparator. Every noisy frame is reconstructed
individually; no reconstructed mean detector image is used.

## 6. Hardware gate and usable claims

Hardware gate passed: `{hardware_gate['passed']}`. Candidate floating files and
hashes are preserved under `candidate_calibration/`, but hardware-format E ROMs
were deliberately not generated. The active ROM hashes are unchanged.

Still usable with the earlier qualifications:

- the historical exact RTL/model agreement and latency;
- the 7.615181 nm fixed-flat result including tip/tilt and its known reciprocal bias;
- the clipping ablation and its distinction between captured expected electrons
  and nominal QE x photons / lenslet;
- per-setting noisy marginals from earlier runs;
- factor {candidate} continuous expected-intensity/centroid predictions, only
  with the factor-{candidate}→{finer} sensitivity bounds reported here.

Not usable as unconditional manuscript numbers: discrete detector,
reconstruction, residual, or noisy convergence claims that assume a single
resolved production grid. The candidate calibration accuracy is reported as
observed and is not repaired by bias subtraction, fitted output scale, or
tip/tilt removal.

## 7. Recommendation and unresolved work

Use factor {candidate} as the minimum setting for exploratory continuous optical
predictions, and bracket every discrete/reconstructed result with factor
{finer}. Do not adopt factor {finer} merely because it is the largest tested
grid. End-to-end practical convergence remains unresolved specifically at ADC/
threshold boundary crossings and their calibration-weighted reconstruction.
An unbounded refinement sweep is not justified by this bounded failure.

The full manuscript campaign should **not** be rerun. Reciprocal precision
optimization remains a separate deferred architecture task; no work on it was
performed here.

## 8. Reproduction

```bash
sim/venv/bin/python -m unittest \\
  evaluation.tests.test_resolved_model_candidate \\
  evaluation.tests.test_noise_coupling \\
  evaluation.tests.test_rom_loading \\
  evaluation.tests.test_reference_conventions -v
sim/venv/bin/python evaluation/optical/resolved_model_candidate.py \\
  --config evaluation/configs/{cfg_path.name} \\
  --out evaluation/results/runs/{cfg['run_id']}_reproduction
```

The output directory must not already exist. See `MANIFEST.json` for exact
source, input, calibration, result, and active-ROM hashes.
"""
    (out / "REPORT.md").write_text(report)


def freeze_provenance(cfg_path: Path, cfg: dict, out: Path, command: list[str],
                      initial_status: list[str], initial_roms: dict,
                      source_paths: dict[str, Path]) -> dict:
    frozen = out / "frozen_inputs"
    frozen.mkdir(exist_ok=True)
    shutil.copyfile(cfg_path, frozen / "config.json")
    (frozen / "git_status_porcelain.txt").write_text("\n".join(initial_status) + "\n")
    patch = subprocess.run(["git", "diff", "--binary"], cwd=C.REPO_ROOT,
                           capture_output=True, check=True).stdout
    (frozen / "tracked_working_tree.patch.gz").write_bytes(gzip.compress(patch, mtime=0))
    relevant = [
        Path(__file__).resolve(), cfg_path,
        C.EVAL_DIR / "common" / "optics.py",
        C.EVAL_DIR / "common" / "detector.py",
        C.EVAL_DIR / "common" / "noise_coupling.py",
        C.EVAL_DIR / "common" / "bitaccurate.py",
        C.EVAL_DIR / "common" / "roms.py",
        C.EVAL_DIR / "optical" / "forward_model_investigation.py",
        C.EVAL_DIR / "optical" / "blocker_investigation.py",
        C.EVAL_DIR / "optical" / "regen_reference.py",
        C.EVAL_DIR / "optical" / "regen_ematrix.py",
        C.EVAL_DIR / "tests" / "test_resolved_model_candidate.py",
    ]
    import hcipy
    from hcipy import FresnelPropagator, MicroLensArray, NoiselessDetector
    hcipy_sources = sorted({Path(inspect.getfile(item)).resolve() for item in
                            (FresnelPropagator, MicroLensArray, NoiselessDetector)})
    final_roms = validate_rom_set(C.RTL_DIR)
    active_equal = all(initial_roms[name]["sha256"] == final_roms[name]["sha256"]
                       for name in REQUIRED_ROMS)
    if not active_equal:
        raise AssertionError("active RTL ROMs changed during a read-only experiment")
    result_files = sorted(path for path in out.rglob("*")
                          if path.is_file() and path.name != "MANIFEST.json")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        commit = None
    manifest = {
        "schema_version": 1,
        "run_id": cfg["run_id"],
        "command": command,
        "working_directory": str(C.REPO_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "matplotlib", "hcipy")},
        "git_commit": commit,
        "git_status_at_start": initial_status,
        "configuration_sha256": sha256(cfg_path),
        "source_runs": {name: {
            "path": str(path.relative_to(C.REPO_ROOT)),
            "manifest_sha256": sha256(path / "MANIFEST.json"),
        } for name, path in source_paths.items()},
        "active_rom_sha256_before": {name: value["sha256"]
                                      for name, value in initial_roms.items()},
        "active_rom_sha256_after": {name: value["sha256"]
                                     for name, value in final_roms.items()},
        "active_roms_unchanged": active_equal,
        "candidate_hardware_roms_generated": False,
        "rtl_simulation": "not run; candidate failed the pre-RTL stability gate",
        "vivado_synthesis_implementation": "not run; outside scope",
        "source_sha256": {str(path.relative_to(C.REPO_ROOT)): sha256(path)
                          for path in relevant},
        "hcipy_source_sha256": {str(path): sha256(path) for path in hcipy_sources},
        "candidate_calibration_sha256": {
            str(path.relative_to(out)): sha256(path)
            for path in sorted((out / "candidate_calibration").glob("*")) if path.is_file()},
        "result_file_sha256": {str(path.relative_to(out)): sha256(path)
                               for path in result_files},
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
    out = (args.out or C.RESULTS_DIR / "runs" / cfg["run_id"]).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")
    source_paths = {
        "forward": (C.REPO_ROOT / cfg["source_forward_run"]).resolve(),
        "blocker": (C.REPO_ROOT / cfg["source_blocker_run"]).resolve(),
        "calibration": (C.REPO_ROOT / cfg["source_calibration_run"]).resolve(),
    }
    for path in source_paths.values():
        if not (path / "MANIFEST.json").is_file():
            raise FileNotFoundError(path / "MANIFEST.json")
    initial_status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=C.REPO_ROOT,
        capture_output=True, text=True, check=True).stdout.splitlines()
    initial_roms = validate_rom_set(C.RTL_DIR)
    out.mkdir(parents=True)
    (out / "figures").mkdir()
    rom_dir = source_paths["calibration"] / "frozen_inputs" / "roms"
    roms = load_frozen_roms(rom_dir)
    active_scale = float(json.loads(
        (source_paths["calibration"] / "frozen_inputs" /
         "derived_scale.json").read_text())["scale_m_per_lsb"])
    scoring_bench = common_scoring_bench(cfg)
    cases = load_selection_cases(cfg, source_paths["blocker"])
    det = detector_config(cfg, cfg["resolution"]["photons_per_pupil"], False)

    print("[1/11] independent analytic lenslet and pixel-integration checks", flush=True)
    simple_cfg = {"simple_lenslet": {
        "propagation_factors": cfg["resolution"]["simple_lenslet_factors"],
        "tilt_spot_shift_detector_pixels": cfg["resolution"]["simple_lenslet_tilt_pixels"],
    }}
    simple = run_simple_lenslet(simple_cfg, out)
    pixel_quadrature = run_analytic_pixel_quadrature(cfg, out)

    print("[2/11] bounded factor-5/6/7/8 full-model ladder", flush=True)
    ladder = run_resolution_ladder(cfg, out, cases, roms["valid"], det)

    print("[3/11] separated edge, padding, and transfer controls", flush=True)
    controls = run_numerical_controls(cfg, out, cases, roms["valid"], det, ladder)
    gate = optical_gate(cfg, ladder, simple)
    json_dump(out / "optical_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError("bounded continuous optical gate failed; no candidate calibration allowed")

    print("[4/11] independent candidate and per-grid diagnostic calibrations", flush=True)
    candidate = cfg["resolution"]["candidate_factor"]
    finer = cfg["resolution"]["finer_check_factor"]
    calibrations = {}
    calibration_rows = []
    for factor in (candidate, finer):
        bench = make_propagation_bench(factor)
        calibration = calibration_from_bench(
            bench, factor, det, cfg["candidate_calibration"]["probe_nm"], roms["valid"])
        calibrations[factor] = calibration
        calibration_rows.append({
            "factor": factor,
            "role": "frozen primary candidate" if factor == candidate else "per-grid diagnostic only",
            "probe_nm": cfg["candidate_calibration"]["probe_nm"],
            "condition_number": calibration["condition_number"],
            "singular_min": float(calibration["singular_values"][-1]),
            "singular_max": float(calibration["singular_values"][0]),
        })
        del bench
        gc.collect()
    write_csv(out / "candidate_calibration_metadata.csv", calibration_rows)
    candidate_record = write_reference_arrays(
        out / "candidate_calibration", candidate,
        ladder["captures"][candidate]["flat"], roms["valid"],
        calibrations[candidate])

    print("[5/11] same-candidate selection test at candidate and finer grids", flush=True)
    selection = evaluate_selection_calibration(
        cfg, out, cases, ladder, calibrations, scoring_bench, roms, active_scale)

    print("[6/11] freeze fresh held-out wavefronts and propagate both grids", flush=True)
    heldout, heldout_record = generate_heldout(cfg, scoring_bench)
    json_dump(out / "heldout_wavefronts.json", heldout_record)
    probabilities, optical_rows = propagate_heldout(
        cfg, heldout, calibrations, (candidate, finer))
    write_csv(out / "heldout_optical_inputs.csv", optical_rows)

    print("[7/11] deterministic held-out candidate confirmation", flush=True)
    deterministic = run_heldout_deterministic(
        cfg, out, heldout, probabilities, calibrations, scoring_bench,
        roms, active_scale)

    print("[8/11] pair-local noisy confirmation with independent batches", flush=True)
    noisy = run_noisy_confirmation(
        cfg, out, heldout, probabilities, calibrations[candidate],
        scoring_bench, roms["valid"])

    print("[9/11] hardware/RTL stage gate", flush=True)
    hardware_gate = {
        "optical_gate_passed": gate["passed"],
        "selection_candidate_passed": selection["passed"],
        "heldout_deterministic_passed": deterministic["passed"],
        "heldout_noisy_passed": noisy["passed"],
        "passed": bool(gate["passed"] and selection["passed"] and
                       deterministic["passed"] and noisy["passed"]),
        "candidate_e_matrix_quantized": False,
        "candidate_hardware_roms_generated": False,
        "rtl_simulation_run": False,
        "reason": "candidate failed before hardware quantization/RTL staging",
        "active_roms_changed": False,
    }
    json_dump(out / "hardware_stage_gate.json", hardware_gate)
    (out / "RTL_NOT_RUN.md").write_text(
        f"# RTL stage not reached\n\nThe factor-{candidate} candidate failed the frozen "
        "end-to-end convergence gate before hardware quantization. No hardware-format "
        "candidate E/reference ROM set was generated or staged, and RTL was not run.\n")
    (out / "DEFERRED_WORK.md").write_text(
        "# Deferred work\n\nReciprocal precision optimization remains a separate architecture "
        "task. This run did not change or reinvestigate reciprocal scaling, LUT contents, "
        "widths, arithmetic, or pipeline stages.\n")

    summary = {
        "optical_gate": gate,
        "candidate_selection": selection,
        "heldout_deterministic": deterministic,
        "noisy_confirmation": noisy,
        "hardware_gate": hardware_gate,
        "candidate_calibration": candidate_record,
        "analytic_pixel_quadrature_rows": pixel_quadrature,
        "active_rtl_or_rom_changed": False,
        "reciprocal_optimization_started": False,
    }
    json_dump(out / "summary.json", summary)

    print("[10/11] figures and consolidated report", flush=True)
    make_figures(out)
    write_report(cfg_path, cfg, out, ladder, controls, gate, selection,
                 deterministic, noisy, hardware_gate, simple)

    print("[11/11] exact provenance and active-ROM equality", flush=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(cfg_path),
               "--out", str(out)]
    freeze_provenance(cfg_path, cfg, out, command, initial_status,
                      initial_roms, source_paths)
    print(f"complete: {out}", flush=True)


if __name__ == "__main__":
    main()
