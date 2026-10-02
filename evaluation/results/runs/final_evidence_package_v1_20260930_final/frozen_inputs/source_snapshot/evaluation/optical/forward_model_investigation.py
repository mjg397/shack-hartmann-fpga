#!/usr/bin/env python3
"""Focused audit of SHWFS numerical propagation and convergence methodology.

This run producer preserves the active RTL/calibration inputs. It diagnoses the
all-grid noise coupling, validates one square lenslet against the independent
analytic focal-plane sinc-squared solution, traces full-model differences stage
by stage, and performs only the affected pairwise noisy confirmations.
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
from scipy.integrate import quad
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import config as C  # noqa: E402
from evaluation.common import detector as D  # noqa: E402
from evaluation.common import optics as O  # noqa: E402
from evaluation.common.noise_coupling import pairwise_poisson_gaussian_frames  # noqa: E402
from evaluation.optical.blocker_investigation import (  # noqa: E402
    frozen_pipeline,
    load_frozen_roms,
    rms_coefficients,
)
from evaluation.optical.ensemble_convergence import coupled_noisy_frames  # noqa: E402


DEFAULT_CONFIG = C.EVAL_DIR / "configs" / "forward_model_investigation_v1.json"
Q18 = float(1 << C.CENTROID_FRAC)


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


def common_scoring_bench(cfg: dict) -> O.Bench:
    full = cfg["full_forward_model"]
    return O.Bench(replace(
        O.OpticalConfig(), pupil_pixels_base=full["common_scoring_pupil_base"],
        supersample=full["common_scoring_edge_supersample"]))


def make_propagation_bench(factor: int, edge_supersample: int = 4) -> O.Bench:
    base = O.OpticalConfig()
    return O.Bench(replace(
        base, pupil_pixels_base=base.pupil_pixels_base * int(factor),
        supersample=int(edge_supersample)))


def physical_detector_energy(bench: O.Bench, image: np.ndarray,
                             factor: int) -> tuple[np.ndarray, float, float]:
    """Block-integrate HCIPy quadrature power into fixed physical pixels."""
    fine = np.asarray(image, dtype=float).reshape(C.FRAME_H * factor,
                                                   C.FRAME_W * factor)
    dx = float(np.asarray(bench.reference_image.grid.delta)[0])
    # Bench.propagate() reads HCIPy's NoiselessDetector. That detector
    # accumulates Wavefront.power, and Wavefront.power already multiplies
    # intensity by grid.weights. These samples are therefore quadrature power,
    # not an unweighted intensity density; multiplying by dx**2 again would
    # spuriously make captured power fall as factor**-2.
    energy = fine.reshape(C.FRAME_H, factor, C.FRAME_W, factor).sum(axis=(1, 3))
    return energy, float(energy.sum()), dx


def expected_detector_views(bench: O.Bench, image: np.ndarray, factor: int,
                            det: D.DetectorConfig) -> dict:
    energy, captured, dx = physical_detector_energy(bench, image, factor)
    probability = energy / captured
    expected_e = probability * det.total_photons * det.quantum_efficiency + det.background_e
    absolute_e = energy * det.total_photons * det.quantum_efficiency + det.background_e
    bounded = np.clip(expected_e, 0.0, det.full_well_e)
    continuous_adu = bounded / det.gain_e_per_adu + det.offset_adu
    unbounded = np.rint(continuous_adu)
    integer_adu = np.clip(unbounded, 0, det.adc_max).astype(np.int64)
    return {
        "energy": energy, "captured_flux": captured, "fine_pixel_pitch_m": dx,
        "probability": probability, "expected_e": expected_e,
        "absolute_expected_e": absolute_e, "continuous_adu": continuous_adu,
        "integer_adu": integer_adu,
        "saturated_pixels": int(np.count_nonzero(expected_e > det.full_well_e)),
        "adc_clipped_pixels": int(np.count_nonzero(
            (unbounded < 0) | (unbounded > det.adc_max))),
    }


def tile_moments(image: np.ndarray, threshold: float | None = None) -> dict:
    values = np.asarray(image, dtype=float).reshape(C.FRAME_H, C.FRAME_W)
    tiles = values.reshape(C.NUM_LENSLETS, C.SUBAP_PIXELS,
                           C.NUM_LENSLETS, C.SUBAP_PIXELS).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(C.NUM_SUBAPS_GRID, C.SUBAP_PIXELS, C.SUBAP_PIXELS)
    if threshold is not None:
        tiles = np.where(tiles < threshold, 0.0, tiles)
    x = np.arange(C.SUBAP_PIXELS, dtype=float)[None, None, :]
    y = np.arange(C.SUBAP_PIXELS, dtype=float)[None, :, None]
    flux = tiles.sum(axis=(1, 2))
    x1 = (tiles * x).sum(axis=(1, 2))
    y1 = (tiles * y).sum(axis=(1, 2))
    cx = np.divide(x1, flux, out=np.zeros_like(flux), where=flux > 0)
    cy = np.divide(y1, flux, out=np.zeros_like(flux), where=flux > 0)
    var_x = np.divide((tiles * (x - cx[:, None, None])**2).sum(axis=(1, 2)), flux,
                      out=np.zeros_like(flux), where=flux > 0)
    var_y = np.divide((tiles * (y - cy[:, None, None])**2).sum(axis=(1, 2)), flux,
                      out=np.zeros_like(flux), where=flux > 0)
    return {"flux": flux, "cx": cx, "cy": cy,
            "sigma_x": np.sqrt(var_x), "sigma_y": np.sqrt(var_y)}


def centroid_vector(moments: dict, valid: np.ndarray) -> np.ndarray:
    return np.r_[moments["cx"][valid], moments["cy"][valid]]


def reconstruct_dequantized(centroids: np.ndarray, roms: dict,
                            scale_m_per_lsb: float) -> np.ndarray:
    valid = roms["valid"]
    reference = np.r_[roms["ref_x"][valid] / Q18,
                      roms["ref_y"][valid] / Q18]
    matrix = np.hstack([roms["ex"], roms["ey"]]).astype(float) * scale_m_per_lsb
    return matrix @ (centroids - reference)


def t_interval(values: list[float] | np.ndarray, confidence: float = 0.95) -> dict:
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError("Student-t interval needs at least two independent units")
    mean = float(x.mean()); sd = float(x.std(ddof=1)); sem = sd / np.sqrt(x.size)
    critical = float(student_t.ppf((1 + confidence) / 2, x.size - 1))
    half = critical * sem
    return {"n": int(x.size), "mean": mean, "sd": sd, "sem": float(sem),
            "ci_low": float(mean - half), "ci_high": float(mean + half),
            "ci_half_width": float(half),
            "method": "two-sided Student-t interval across independent wavefront units"}


def simple_square_analytic_pixel_energy(edges: np.ndarray, aperture_width: float,
                                        wavelength: float, focal_length: float,
                                        shift_x: float, shift_y: float) -> np.ndarray:
    scale = aperture_width / (wavelength * focal_length)

    def axis_integrals(shift: float) -> np.ndarray:
        values = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            integral = quad(lambda coordinate: aperture_width / (wavelength * focal_length) *
                            np.sinc(scale * (coordinate - shift))**2,
                            lower, upper, epsabs=2e-13, epsrel=2e-11, limit=100)[0]
            values.append(integral)
        return np.asarray(values)

    return np.outer(axis_integrals(shift_y), axis_integrals(shift_x))


def full_image_moments(energy: np.ndarray, pitch: float) -> dict:
    n = energy.shape[0]
    coordinates = -n * pitch / 2 + pitch / 2 + np.arange(n) * pitch
    total = float(energy.sum())
    cx = float((energy * coordinates[None, :]).sum() / total)
    cy = float((energy * coordinates[:, None]).sum() / total)
    sx = float(np.sqrt((energy * (coordinates[None, :] - cx)**2).sum() / total))
    sy = float(np.sqrt((energy * (coordinates[:, None] - cy)**2).sum() / total))
    return {"centroid_x_m": cx, "centroid_y_m": cy,
            "sigma_x_m_finite_field": sx, "sigma_y_m_finite_field": sy}


def run_simple_lenslet(cfg: dict, out: Path) -> dict:
    from hcipy import (CartesianGrid, Field, MicroLensArray, SeparatedCoords,
                       ShackHartmannWavefrontSensorOptics, Wavefront,
                       make_pupil_grid, make_rectangular_aperture)

    optical = O.OpticalConfig()
    pitch = optical.lenslet_diameter_m
    focal = optical.focal_length_m
    wavelength = optical.wavelength_m
    detector_pitch = optical.detector_pixel_pitch_m
    extent = optical.sh_diameter_m
    edges = np.linspace(-extent / 2, extent / 2, C.FRAME_W + 1)
    shifts_px = cfg["simple_lenslet"]["tilt_spot_shift_detector_pixels"]
    rows = []
    arrays = {}
    for factor in cfg["simple_lenslet"]["propagation_factors"]:
        grid = make_pupil_grid(C.FRAME_W * factor, extent)
        centre_grid = CartesianGrid(SeparatedCoords((np.asarray([0.0]), np.asarray([0.0]))))
        square = make_rectangular_aperture(pitch)
        aperture = square(grid)
        mla = MicroLensArray(grid, centre_grid, focal, lenslet_shape=square)
        optics = ShackHartmannWavefrontSensorOptics(grid, mla)
        dx = float(np.asarray(grid.delta)[0])
        for shift_px in shifts_px:
            shift_m = float(shift_px) * detector_pitch
            angle = shift_m / focal
            field = aperture * np.exp(1j * 2 * np.pi / wavelength * angle * grid.x)
            wavefront = Wavefront(Field(field, grid), wavelength)
            input_power_before = float(wavefront.total_power)
            wavefront.total_power = 1.0
            propagated = optics(wavefront)
            fine_intensity = np.asarray(propagated.intensity)
            energy = fine_intensity.reshape(C.FRAME_H, factor,
                                             C.FRAME_W, factor).sum(axis=(1, 3)) * dx**2
            analytic = simple_square_analytic_pixel_energy(
                edges, pitch, wavelength, focal, shift_m, 0.0)
            numeric_moments = full_image_moments(energy, detector_pitch)
            analytic_moments = full_image_moments(analytic, detector_pitch)
            numeric_probability = energy / energy.sum()
            analytic_probability = analytic / analytic.sum()
            first_zero = wavelength * focal / pitch
            xx, yy = np.meshgrid(
                -extent / 2 + detector_pitch / 2 + np.arange(C.FRAME_W) * detector_pitch,
                -extent / 2 + detector_pitch / 2 + np.arange(C.FRAME_H) * detector_pitch)
            central = (np.abs(xx - shift_m) <= first_zero) & (np.abs(yy) <= first_zero)
            row = {
                "factor": factor, "tilt_shift_command_px": shift_px,
                "tilt_angle_rad": angle, "expected_spot_shift_m": shift_m,
                "pupil_samples_across_lenslet": int(round(pitch / dx)),
                "fine_pixel_pitch_m": dx, "physical_detector_pitch_m": detector_pitch,
                "input_power_before_normalization": input_power_before,
                "captured_output_power": float(energy.sum()),
                "analytic_captured_power": float(analytic.sum()),
                "captured_flux_error": float(energy.sum() - analytic.sum()),
                "normalized_image_relative_l2_error": float(
                    np.linalg.norm(numeric_probability - analytic_probability) /
                    np.linalg.norm(analytic_probability)),
                "normalized_image_l1_error": float(np.abs(
                    numeric_probability - analytic_probability).sum()),
                "centroid_x_m": numeric_moments["centroid_x_m"],
                "analytic_centroid_x_m": analytic_moments["centroid_x_m"],
                "centroid_x_error_m": (numeric_moments["centroid_x_m"] -
                                         analytic_moments["centroid_x_m"]),
                "centroid_y_m": numeric_moments["centroid_y_m"],
                "finite_field_sigma_x_m": numeric_moments["sigma_x_m_finite_field"],
                "analytic_finite_field_sigma_x_m": analytic_moments["sigma_x_m_finite_field"],
                "central_first_zero_box_energy": float(energy[central].sum() / energy.sum()),
                "analytic_central_first_zero_box_energy": float(
                    analytic[central].sum() / analytic.sum()),
                "analytic_axis_first_zero_m": first_zero,
                "analytic_fwhm_m": 0.885892941 * first_zero,
            }
            rows.append(row)
            key = f"factor{factor}_shift{shift_px:+.1f}px".replace(".", "p")
            arrays[f"numeric_probability__{key}"] = numeric_probability.astype(np.float32)
            arrays[f"analytic_probability__{key}"] = analytic_probability.astype(np.float32)
    write_csv(out / "simple_lenslet_validation.csv", rows)
    np.savez_compressed(out / "simple_lenslet_arrays.npz", **arrays)
    summary = {
        "aperture": "one hard-edged square lenslet",
        "analytic_reference": (
            "thin-lens focal-plane square-aperture sinc-squared intensity, integrated "
            "independently with scipy.quad over the fixed physical detector-pixel edges"
        ),
        "spot_shift_law": "x=f*theta",
        "rows": rows,
    }
    json_dump(out / "simple_lenslet_summary.json", summary)
    return summary


def setting_specs(cfg: dict) -> dict[str, dict]:
    full = cfg["full_forward_model"]
    specs = {"production": {"factor": 1, "edge_supersample": 4,
                             "study": "both", "setting": 1}}
    for factor in full["propagation_factors"]:
        if factor != 1:
            specs[f"prop_f{factor}"] = {
                "factor": factor, "edge_supersample": 4,
                "study": "propagation", "setting": factor,
            }
    for supersample in full["pupil_edge_supersampling"]:
        if supersample != 4:
            specs[f"edge_ss{supersample}"] = {
                "factor": 1, "edge_supersample": supersample,
                "study": "pupil_edge", "setting": supersample,
            }
    return specs


def build_benches(specs: dict[str, dict]) -> dict[str, O.Bench]:
    return {name: make_propagation_bench(spec["factor"], spec["edge_supersample"])
            for name, spec in specs.items()}


def run_geometry_audit(cfg: dict, out: Path, specs: dict,
                       benches: dict[str, O.Bench]) -> dict:
    rows = []
    base = O.OpticalConfig()
    for name, spec in specs.items():
        bench = benches[name]
        pupil_grid = bench.pupil_grid
        detector_grid = bench.reference_image.grid
        pupil_x = np.unique(np.asarray(pupil_grid.x, dtype=float))
        detector_x = np.unique(np.asarray(detector_grid.x, dtype=float))
        pupil_dx = float(np.asarray(pupil_grid.delta)[0])
        detector_dx = float(np.asarray(detector_grid.delta)[0])
        index = np.asarray(bench.mla.mla_index, dtype=int)
        labels, counts = np.unique(index, return_counts=True)
        expected_samples = (C.SUBAP_PIXELS * spec["factor"])**2
        grouped_centres = detector_x.reshape(C.FRAME_W, spec["factor"]).mean(axis=1)
        physical_centres = (-base.sh_diameter_m / 2 + base.detector_pixel_pitch_m / 2 +
                            np.arange(C.FRAME_W) * base.detector_pixel_pitch_m)
        aperture = np.asarray(bench.aperture, dtype=float)
        lmax = float(np.max(np.asarray(detector_grid.dims) *
                           np.asarray(detector_grid.delta)))
        fresnel_limit = base.wavelength_m * base.focal_length_m / lmax
        rows.append({
            "setting_name": name, **spec,
            "wavelength_m": bench.cfg.wavelength_m,
            "telescope_diameter_m": bench.cfg.telescope_diameter_m,
            "central_obscuration_m": bench.cfg.central_obscuration_m,
            "spider_width_m": bench.cfg.spider_width_m,
            "num_spiders": bench.cfg.num_spiders,
            "magnification": bench.cfg.magnification,
            "sh_diameter_m": bench.cfg.sh_diameter_m,
            "lenslet_pitch_m": bench.cfg.lenslet_diameter_m,
            "lenslet_focal_length_m": bench.cfg.focal_length_m,
            "lenslet_centres_min_m": float(np.min(bench.mla.mla_grid.x)),
            "lenslet_centres_max_m": float(np.max(bench.mla.mla_grid.x)),
            "lenslet_count": int(bench.mla.mla_grid.size),
            "pupil_samples_axis": int(pupil_grid.shape[0]),
            "pupil_sample_pitch_m": pupil_dx,
            "pupil_lower_edge_m": float(pupil_x[0] - pupil_dx / 2),
            "pupil_upper_edge_m": float(pupil_x[-1] + pupil_dx / 2),
            "pupil_even_grid": bool(pupil_grid.shape[0] % 2 == 0),
            "pupil_contains_zero_sample": bool(np.any(pupil_x == 0)),
            "detector_fine_samples_axis": int(detector_grid.shape[0]),
            "detector_fine_pitch_m": detector_dx,
            "detector_lower_edge_m": float(detector_x[0] - detector_dx / 2),
            "detector_upper_edge_m": float(detector_x[-1] + detector_dx / 2),
            "detector_contains_zero_sample": bool(np.any(detector_x == 0)),
            "integrated_pixel_center_alignment_max_abs_m": float(
                np.max(np.abs(grouped_centres - physical_centres))),
            "mla_label_min": int(labels.min()), "mla_label_max": int(labels.max()),
            "mla_samples_per_lenslet_min": int(counts.min()),
            "mla_samples_per_lenslet_max": int(counts.max()),
            "mla_expected_samples_per_lenslet": expected_samples,
            "mla_all_counts_expected": bool(np.all(counts == expected_samples)),
            "aperture_coherent_area_quadrature_m2": float(aperture.sum() * pupil_dx**2),
            "aperture_energy_area_quadrature_m2": float((aperture**2).sum() * pupil_dx**2),
            "aperture_partial_cells": int(np.count_nonzero(
                (aperture > 0) & (aperture < 1))),
            "fresnel_zero_padding": 2,
            "fresnel_transfer_oversampling": 2,
            "fresnel_native_transfer_branch": bool(detector_dx >= fresnel_limit),
            "fresnel_branch_pitch_limit_m": fresnel_limit,
            "orientation": "array columns increase physical x; rows increase physical y",
        })
    write_csv(out / "physical_geometry_audit.csv", rows)

    # Validate fractional amplitude as a coherent cell-average quadrature, not
    # an intensity fill fraction. A factor-5 binary point quadrature is block-
    # averaged onto the production cells as a separately sampled reference.
    from hcipy import make_obstructed_circular_aperture
    aperture_gen = make_obstructed_circular_aperture(
        base.telescope_diameter_m,
        base.central_obscuration_m / base.telescope_diameter_m,
        num_spiders=base.num_spiders, spider_width=base.spider_width_m)
    resolved_bench = benches["prop_f5"]
    resolved_binary = np.asarray(aperture_gen(resolved_bench.pupil_grid), dtype=float)
    resolved_cell_mean = resolved_binary.reshape(
        C.FRAME_H, 5, C.FRAME_W, 5).mean(axis=(1, 3))
    aperture_rows = []
    arrays = {"factor5_binary_block_mean": resolved_cell_mean.astype(np.float32)}
    for supersample in cfg["full_forward_model"]["pupil_edge_supersampling"]:
        key = "production" if supersample == 4 else f"edge_ss{supersample}"
        aperture = np.asarray(benches[key].aperture, dtype=float).reshape(
            C.FRAME_H, C.FRAME_W)
        delta = aperture - resolved_cell_mean
        dx = float(np.asarray(benches[key].pupil_grid.delta)[0])
        arrays[f"edge_ss{supersample}"] = aperture.astype(np.float32)
        aperture_rows.append({
            "edge_supersample": supersample,
            "reference": "factor-5 binary point quadrature block-averaged to production cells",
            "amplitude_relative_l2_error": float(
                np.linalg.norm(delta) / np.linalg.norm(resolved_cell_mean)),
            "amplitude_max_abs_error": float(np.max(np.abs(delta))),
            "coherent_area_m2": float(aperture.sum() * dx**2),
            "energy_area_m2": float((aperture**2).sum() * dx**2),
            "sqrt_area_fraction_not_used": True,
            "reason": ("cell mean is the coherent field-amplitude quadrature; square-root "
                       "would preserve cell energy but alter coherent diffraction"),
        })
    write_csv(out / "aperture_discretization_audit.csv", aperture_rows)
    np.savez_compressed(out / "aperture_discretization_arrays.npz", **arrays)
    summary = {"physical_rows": rows, "aperture_rows": aperture_rows,
               "all_physical_constants_identical": bool(all(
                   np.isclose(row["wavelength_m"], rows[0]["wavelength_m"],
                              rtol=0, atol=1e-18) and
                   np.isclose(row["sh_diameter_m"], rows[0]["sh_diameter_m"],
                              rtol=0, atol=1e-18) and
                   np.isclose(row["lenslet_pitch_m"], rows[0]["lenslet_pitch_m"],
                              rtol=0, atol=1e-18) and
                   np.isclose(row["lenslet_focal_length_m"],
                              rows[0]["lenslet_focal_length_m"], rtol=0, atol=1e-18) and
                   np.isclose(row["detector_lower_edge_m"],
                              rows[0]["detector_lower_edge_m"], rtol=0, atol=1e-18) and
                   np.isclose(row["detector_upper_edge_m"],
                              rows[0]["detector_upper_edge_m"], rtol=0, atol=1e-18)
                   for row in rows)),
               "all_mla_counts_exact": bool(all(row["mla_all_counts_expected"] for row in rows)),
               "all_pixel_centres_aligned": bool(all(
                   row["integrated_pixel_center_alignment_max_abs_m"] < 1e-15
                   for row in rows))}
    json_dump(out / "physical_geometry_summary.json", summary)
    return summary


def load_deterministic_cases(cfg: dict, blocker_run: Path) -> dict[str, np.ndarray]:
    source = json.loads((blocker_run / "convergence_cases.json").read_text())["cases"]
    cases = {"flat": np.zeros(C.NUM_MODES)}
    for name in cfg["full_forward_model"]["cases"]:
        if name == "flat":
            continue
        cases[name] = np.asarray(source[name], dtype=float)
    return cases


def run_stage_audit(cfg: dict, out: Path, specs: dict, benches: dict,
                    cases: dict[str, np.ndarray], roms: dict,
                    scale: float, scoring_bench: O.Bench) -> dict:
    det = detector_config(cfg, cfg["full_forward_model"]["photons_per_pupil"], False)
    valid = roms["valid"]
    captures = {}
    raw_rows = []
    arrays = {}
    lenslet_rows = []
    for setting_name, spec in specs.items():
        bench = benches[setting_name]
        captures[setting_name] = {}
        for case_name, truth in cases.items():
            image, sampled_opd = bench.propagate_coeffs(truth)
            views = expected_detector_views(bench, image, spec["factor"], det)
            no_threshold = tile_moments(views["continuous_adu"], None)
            continuous_threshold = tile_moments(
                views["continuous_adu"], cfg["detector"]["pixel_threshold_adu"])
            integer = tile_moments(
                views["integer_adu"], cfg["detector"]["pixel_threshold_adu"])
            centroids_b = centroid_vector(no_threshold, valid)
            centroids_c = centroid_vector(continuous_threshold, valid)
            centroids_d = centroid_vector(integer, valid)
            coeff_b = reconstruct_dequantized(centroids_b, roms, scale)
            coeff_c = reconstruct_dequantized(centroids_c, roms, scale)
            coeff_d = reconstruct_dequantized(centroids_d, roms, scale)
            fixed_trace = frozen_pipeline(views["integer_adu"], roms)
            coeff_e = fixed_trace["zernike_raw"].astype(float) * scale
            sampled_opd_rms = bench.opd_rms(sampled_opd)
            capture = {
                "views": views, "moments_b": no_threshold,
                "moments_c": continuous_threshold, "moments_d": integer,
                "centroids_b": centroids_b, "centroids_c": centroids_c,
                "centroids_d": centroids_d, "coeff_b": coeff_b,
                "coeff_c": coeff_c, "coeff_d": coeff_d, "coeff_e": coeff_e,
            }
            captures[setting_name][case_name] = capture
            prefix = f"{setting_name}__{case_name}"
            arrays[f"physical_energy__{prefix}"] = views["energy"].astype(np.float32)
            arrays[f"expected_e__{prefix}"] = views["expected_e"].astype(np.float32)
            arrays[f"integer_adu__{prefix}"] = views["integer_adu"].astype(np.uint8)
            arrays[f"centroids_no_threshold__{prefix}"] = centroids_b
            arrays[f"centroids_continuous_threshold__{prefix}"] = centroids_c
            arrays[f"centroids_integer__{prefix}"] = centroids_d
            arrays[f"coefficients_fixed_m__{prefix}"] = coeff_e
            raw_rows.append({
                "setting_name": setting_name, **spec, "case": case_name,
                "truth_coefficients_m": json.dumps(truth.tolist()),
                "truth_rms_common_m": rms_coefficients(scoring_bench, truth),
                "sampled_opd_rms_on_setting_m": sampled_opd_rms,
                "captured_flux_before_normalization": views["captured_flux"],
                "normalized_expected_e_sum": float(views["expected_e"].sum()),
                "absolute_expected_e_sum_without_renormalization": float(
                    views["absolute_expected_e"].sum()),
                "saturated_pixels": views["saturated_pixels"],
                "adc_clipped_pixels": views["adc_clipped_pixels"],
                "stage_b_no_threshold_centroid_mean_px": float(centroids_b.mean()),
                "stage_c_thresholded_continuous_centroid_mean_px": float(centroids_c.mean()),
                "stage_d_integer_centroid_mean_px": float(centroids_d.mean()),
                "stage_b_residual_m": rms_coefficients(scoring_bench, coeff_b - truth),
                "stage_c_residual_m": rms_coefficients(scoring_bench, coeff_c - truth),
                "stage_d_residual_m": rms_coefficients(scoring_bench, coeff_d - truth),
                "stage_e_fixed_residual_m": rms_coefficients(scoring_bench, coeff_e - truth),
                "stage_e_fixed_coefficients_m": json.dumps(coeff_e.tolist()),
                "valid_spot_sigma_x_mean_px": float(no_threshold["sigma_x"][valid].mean()),
                "valid_spot_sigma_y_mean_px": float(no_threshold["sigma_y"][valid].mean()),
            })
    np.savez_compressed(out / "forward_stage_arrays.npz", **arrays)
    write_csv(out / "forward_stage_raw.csv", raw_rows)

    studies = {
        "propagation": ([1, 2, 3, 4, 5], 5,
                        lambda value: "production" if value == 1 else f"prop_f{value}"),
        "pupil_edge": ([4, 12, 20], 20,
                       lambda value: "production" if value == 4 else f"edge_ss{value}"),
    }
    pair_rows = []
    for study, (settings, reference_setting, key_for) in studies.items():
        reference_key = key_for(reference_setting)
        for setting in settings:
            setting_key = key_for(setting)
            for case_name, truth in cases.items():
                current = captures[setting_key][case_name]
                reference = captures[reference_key][case_name]
                image_delta = current["views"]["energy"] - reference["views"]["energy"]
                expected_delta = current["views"]["expected_e"] - reference["views"]["expected_e"]
                row = {
                    "study": study, "setting": setting,
                    "setting_name": setting_key, "reference_setting": reference_setting,
                    "reference_setting_name": reference_key, "case": case_name,
                    "stage_a_captured_flux": current["views"]["captured_flux"],
                    "stage_a_reference_captured_flux": reference["views"]["captured_flux"],
                    "stage_a_captured_flux_relative_change": (
                        current["views"]["captured_flux"] /
                        reference["views"]["captured_flux"] - 1),
                    "stage_a_absolute_energy_relative_l2": float(
                        np.linalg.norm(image_delta) / np.linalg.norm(reference["views"]["energy"])),
                    "stage_a_normalized_expected_e_relative_l2": float(
                        np.linalg.norm(expected_delta) /
                        np.linalg.norm(reference["views"]["expected_e"])),
                    "stage_a_normalized_expected_e_max_abs": float(np.max(np.abs(expected_delta))),
                }
                for label in ("b", "c", "d"):
                    centroid_delta = current[f"centroids_{label}"] - reference[f"centroids_{label}"]
                    coeff_delta = current[f"coeff_{label}"] - reference[f"coeff_{label}"]
                    row[f"stage_{label}_centroid_rms_change_px"] = float(
                        np.sqrt(np.mean(centroid_delta**2)))
                    row[f"stage_{label}_centroid_max_abs_change_px"] = float(
                        np.max(np.abs(centroid_delta)))
                    row[f"stage_{label}_reconstructed_wavefront_rms_change_m"] = (
                        rms_coefficients(scoring_bench, coeff_delta))
                    row[f"stage_{label}_ground_truth_residual_m"] = rms_coefficients(
                        scoring_bench, current[f"coeff_{label}"] - truth)
                fixed_delta = current["coeff_e"] - reference["coeff_e"]
                row.update({
                    "stage_e_fixed_reconstructed_wavefront_rms_change_m": rms_coefficients(
                        scoring_bench, fixed_delta),
                    "stage_e_fixed_ground_truth_residual_m": rms_coefficients(
                        scoring_bench, current["coeff_e"] - truth),
                    "stage_e_fixed_reference_ground_truth_residual_m": rms_coefficients(
                        scoring_bench, reference["coeff_e"] - truth),
                    "stage_e_fixed_signed_residual_change_m": (
                        rms_coefficients(scoring_bench, current["coeff_e"] - truth) -
                        rms_coefficients(scoring_bench, reference["coeff_e"] - truth)),
                    "reference_self_comparison_is_evidence": False,
                })
                pair_rows.append(row)
                for tile_position, tile in enumerate(valid):
                    lenslet_rows.append({
                        "study": study, "setting": setting,
                        "reference_setting": reference_setting, "case": case_name,
                        "tile": int(tile), "lenslet_row": int(tile // C.NUM_LENSLETS),
                        "lenslet_column": int(tile % C.NUM_LENSLETS),
                        "expected_e_flux": float(current["moments_b"]["flux"][tile]),
                        "reference_expected_e_flux": float(reference["moments_b"]["flux"][tile]),
                        "stage_b_centroid_x_change_px": float(
                            current["moments_b"]["cx"][tile] - reference["moments_b"]["cx"][tile]),
                        "stage_b_centroid_y_change_px": float(
                            current["moments_b"]["cy"][tile] - reference["moments_b"]["cy"][tile]),
                        "stage_c_centroid_x_change_px": float(
                            current["moments_c"]["cx"][tile] - reference["moments_c"]["cx"][tile]),
                        "stage_c_centroid_y_change_px": float(
                            current["moments_c"]["cy"][tile] - reference["moments_c"]["cy"][tile]),
                        "stage_d_centroid_x_change_px": float(
                            current["moments_d"]["cx"][tile] - reference["moments_d"]["cx"][tile]),
                        "stage_d_centroid_y_change_px": float(
                            current["moments_d"]["cy"][tile] - reference["moments_d"]["cy"][tile]),
                    })
    write_csv(out / "forward_stage_paired.csv", pair_rows)
    write_csv(out / "forward_stage_lenslets.csv", lenslet_rows)
    summary = {"detector": asdict(det), "rows": pair_rows,
               "calibration_policy": "frozen active reference and E matrix at stages B-E",
               "normalization_note": (
                   "absolute HCIPy output power is recorded before the existing detector model "
                   "renormalizes captured energy to its photons-per-pupil definition")}
    json_dump(out / "forward_stage_summary.json", summary)
    return {"summary": summary, "captures": captures, "raw_rows": raw_rows}


def calibration_from_bench(bench: O.Bench, factor: int, det: D.DetectorConfig,
                           probe_nm: float, valid: np.ndarray) -> dict:
    flat_image, _ = bench.propagate_coeffs(np.zeros(C.NUM_MODES))
    flat_views = expected_detector_views(bench, flat_image, factor, det)
    flat_moments = tile_moments(flat_views["integer_adu"], C.PIXEL_THRESHOLD)
    reference = centroid_vector(flat_moments, valid)
    interaction = np.zeros((2 * len(valid), C.NUM_MODES))
    probe = probe_nm * 1e-9
    for mode in range(C.NUM_MODES):
        plus = np.zeros(C.NUM_MODES); minus = np.zeros(C.NUM_MODES)
        plus[mode] = probe; minus[mode] = -probe
        plus_image, _ = bench.propagate_coeffs(plus)
        minus_image, _ = bench.propagate_coeffs(minus)
        plus_frame = expected_detector_views(bench, plus_image, factor, det)["integer_adu"]
        minus_frame = expected_detector_views(bench, minus_image, factor, det)["integer_adu"]
        plus_centroids = centroid_vector(
            tile_moments(plus_frame, C.PIXEL_THRESHOLD), valid)
        minus_centroids = centroid_vector(
            tile_moments(minus_frame, C.PIXEL_THRESHOLD), valid)
        interaction[:, mode] = (plus_centroids - minus_centroids) / (2 * probe)
    singular = np.linalg.svd(interaction, compute_uv=False)
    return {"reference": reference, "interaction": interaction,
            "inverse": np.linalg.pinv(interaction),
            "condition_number": float(singular[0] / singular[-1]),
            "singular_values": singular}


def run_calibration_diagnostic(cfg: dict, out: Path, benches: dict, cases: dict,
                               stage: dict, roms: dict, scale: float,
                               scoring_bench: O.Bench) -> dict:
    full = cfg["full_forward_model"]
    det = detector_config(cfg, full["photons_per_pupil"], False)
    factors = full["propagation_factors"]
    valid = roms["valid"]
    calibrations = {}
    arrays = {}
    metadata = []
    for factor in factors:
        key = "production" if factor == 1 else f"prop_f{factor}"
        calibration = calibration_from_bench(
            benches[key], factor, det, full["calibration_probe_nm"], valid)
        calibrations[factor] = calibration
        arrays[f"reference_factor{factor}"] = calibration["reference"]
        arrays[f"interaction_factor{factor}"] = calibration["interaction"]
        arrays[f"inverse_factor{factor}"] = calibration["inverse"]
        metadata.append({
            "factor": factor, "probe_nm": full["calibration_probe_nm"],
            "condition_number": calibration["condition_number"],
            "singular_min": float(calibration["singular_values"][-1]),
            "singular_max": float(calibration["singular_values"][0]),
        })
    np.savez_compressed(out / "diagnostic_calibration_matrices.npz", **arrays)
    write_csv(out / "diagnostic_calibration_metadata.csv", metadata)

    common = calibrations[max(factors)]
    rows = []
    captures = stage["captures"]
    for factor in factors:
        key = "production" if factor == 1 else f"prop_f{factor}"
        own = calibrations[factor]
        for case_name, truth in cases.items():
            centroids = captures[key][case_name]["centroids_d"]
            fixed_coeff = captures[key][case_name]["coeff_e"]
            frozen_dequant = reconstruct_dequantized(centroids, roms, scale)
            common_coeff = common["inverse"] @ (centroids - common["reference"])
            per_grid_coeff = own["inverse"] @ (centroids - own["reference"])
            for policy, coefficients in (
                    ("frozen_fixed_production", fixed_coeff),
                    ("frozen_dequantized_rom", frozen_dequant),
                    ("common_factor5_diagnostic", common_coeff),
                    ("per_grid_diagnostic", per_grid_coeff)):
                rows.append({
                    "factor": factor, "case": case_name,
                    "calibration_policy": policy,
                    "active_production_path": policy == "frozen_fixed_production",
                    "coefficients_m": json.dumps(coefficients.tolist()),
                    "ground_truth_residual_m": rms_coefficients(
                        scoring_bench, coefficients - truth),
                })
    write_csv(out / "calibration_separation_raw.csv", rows)
    summary_rows = []
    for factor in factors:
        for policy in sorted({row["calibration_policy"] for row in rows}):
            subset = [row for row in rows if row["factor"] == factor and
                      row["calibration_policy"] == policy]
            summary_rows.append({
                "factor": factor, "calibration_policy": policy,
                "n_cases": len(subset),
                "mean_residual_nm": float(np.mean([
                    row["ground_truth_residual_m"] for row in subset]) * 1e9),
                "max_residual_nm": float(np.max([
                    row["ground_truth_residual_m"] for row in subset]) * 1e9),
                "active_production_path": policy == "frozen_fixed_production",
            })
    write_csv(out / "calibration_separation_summary.csv", summary_rows)
    summary = {"rows": summary_rows, "metadata": metadata,
               "diagnostic_only": True,
               "active_roms_changed": False,
               "common_calibration_source_factor": max(factors)}
    json_dump(out / "calibration_separation_summary.json", summary)
    return summary


def frame_difference_metrics(first: np.ndarray, second: np.ndarray, roms: dict,
                             scale: float, scoring_bench: O.Bench) -> dict:
    delta = first.astype(float) - second.astype(float)
    first_coeff = frozen_pipeline(first, roms)["zernike_raw"].astype(float) * scale
    second_coeff = frozen_pipeline(second, roms)["zernike_raw"].astype(float) * scale
    return {
        "detector_rms_adu": float(np.sqrt(np.mean(delta**2))),
        "detector_changed_pixels": int(np.count_nonzero(delta)),
        "detector_max_abs_adu": float(np.max(np.abs(delta))),
        "reconstructed_wavefront_rms_difference_nm": rms_coefficients(
            scoring_bench, first_coeff - second_coeff) * 1e9,
    }


def run_noise_coupling_audit(cfg: dict, out: Path, stage: dict, roms: dict,
                             scale: float, scoring_bench: O.Bench) -> dict:
    audit = cfg["noise_audit"]
    det = detector_config(cfg, audit["photons_per_pupil"], True)
    case = audit["control_case"]
    captures = stage["captures"]
    probabilities = {
        "production": captures["production"][case]["views"]["probability"],
        "prop_f5": captures["prop_f5"][case]["views"]["probability"],
        "edge_ss20": captures["edge_ss20"][case]["views"]["probability"],
    }
    expected = {name: probability * det.total_photons * det.quantum_efficiency +
                det.background_e for name, probability in probabilities.items()}
    rows = []
    permutation_failures = 0
    pair_repeat_failures = 0
    for exposure in range(audit["control_exposures"]):
        seed = [cfg["seeds"]["noise_controls"], exposure]

        legacy_only, _ = coupled_noisy_frames(
            {"clone_a": probabilities["production"],
             "clone_b": probabilities["production"]}, det, seed)
        rows.append({"exposure": exposure, "control": "identical_pair_legacy_only",
                     **frame_difference_metrics(legacy_only["clone_a"],
                                                legacy_only["clone_b"], roms,
                                                scale, scoring_bench)})

        legacy_embedded, _ = coupled_noisy_frames(
            {"clone_a": probabilities["production"],
             "clone_b": probabilities["production"],
             "unrelated": probabilities["prop_f5"]}, det, seed)
        rows.append({"exposure": exposure, "control": "identical_pair_legacy_embedded",
                     **frame_difference_metrics(legacy_embedded["clone_a"],
                                                legacy_embedded["clone_b"], roms,
                                                scale, scoring_bench)})

        # The legacy implementation sorts keys, so insertion-order permutation
        # should be stable even though adding/removing a setting is not.
        legacy_permuted, _ = coupled_noisy_frames(
            {"unrelated": probabilities["prop_f5"],
             "clone_b": probabilities["production"],
             "clone_a": probabilities["production"]}, det, seed)
        if (not np.array_equal(legacy_embedded["clone_a"], legacy_permuted["clone_a"])
                or not np.array_equal(legacy_embedded["clone_b"],
                                      legacy_permuted["clone_b"])):
            permutation_failures += 1

        pair_identical, _ = pairwise_poisson_gaussian_frames(
            {"clone_a": expected["production"], "clone_b": expected["production"]},
            det, seed)
        rows.append({"exposure": exposure, "control": "identical_pair_corrected",
                     **frame_difference_metrics(pair_identical["clone_a"],
                                                pair_identical["clone_b"], roms,
                                                scale, scoring_bench)})

        legacy_pair, _ = coupled_noisy_frames(
            {"production": probabilities["production"],
             "prop_f5": probabilities["prop_f5"]}, det, seed)
        rows.append({"exposure": exposure, "control": "nonidentical_legacy_pair_only",
                     **frame_difference_metrics(legacy_pair["production"],
                                                legacy_pair["prop_f5"], roms,
                                                scale, scoring_bench)})
        legacy_three, _ = coupled_noisy_frames(
            {"production": probabilities["production"],
             "prop_f5": probabilities["prop_f5"],
             "edge_ss20": probabilities["edge_ss20"]}, det, seed)
        rows.append({"exposure": exposure, "control": "nonidentical_legacy_with_unrelated",
                     **frame_difference_metrics(legacy_three["production"],
                                                legacy_three["prop_f5"], roms,
                                                scale, scoring_bench)})
        corrected_pair, _ = pairwise_poisson_gaussian_frames(
            {"production": expected["production"], "prop_f5": expected["prop_f5"]},
            det, seed)
        # Draw an unrelated pair and then repeat the target pair; pair-local
        # construction must make this a bit-exact no-op for the target.
        pairwise_poisson_gaussian_frames(
            {"production": expected["production"], "edge_ss20": expected["edge_ss20"]},
            det, [cfg["seeds"]["noise_controls"], exposure, 999])
        corrected_repeat, _ = pairwise_poisson_gaussian_frames(
            {"production": expected["production"], "prop_f5": expected["prop_f5"]},
            det, seed)
        if any(not np.array_equal(corrected_pair[name], corrected_repeat[name])
               for name in corrected_pair):
            pair_repeat_failures += 1
        rows.append({"exposure": exposure, "control": "nonidentical_corrected_pair",
                     **frame_difference_metrics(corrected_pair["production"],
                                                corrected_pair["prop_f5"], roms,
                                                scale, scoring_bench)})
    write_csv(out / "noise_coupling_controls_raw.csv", rows)

    summary_rows = []
    for control in sorted({row["control"] for row in rows}):
        subset = [row for row in rows if row["control"] == control]
        for metric in ("detector_rms_adu", "detector_changed_pixels",
                       "reconstructed_wavefront_rms_difference_nm"):
            stats = t_interval([row[metric] for row in subset])
            summary_rows.append({
                "control": control, "metric": metric,
                "n_exposures": stats["n"], "mean": stats["mean"],
                "sd": stats["sd"], "ci95_low": stats["ci_low"],
                "ci95_high": stats["ci_high"], "interval_method": stats["method"],
            })
    write_csv(out / "noise_coupling_controls_summary.csv", summary_rows)
    embedded_floor = next(row for row in summary_rows
                          if row["control"] == "identical_pair_legacy_embedded" and
                          row["metric"] == "reconstructed_wavefront_rms_difference_nm")
    corrected_floor = next(row for row in summary_rows
                           if row["control"] == "identical_pair_corrected" and
                           row["metric"] == "reconstructed_wavefront_rms_difference_nm")
    summary = {
        "control_case": case, "n_exposures": audit["control_exposures"],
        "legacy_insertion_order_permutation_failures": permutation_failures,
        "corrected_pair_repeat_failures_after_unrelated_draw": pair_repeat_failures,
        "legacy_embedded_identical_reconstruction_floor_nm": embedded_floor,
        "corrected_identical_reconstruction_floor_nm": corrected_floor,
        "interpretation": {
            "per_setting_marginals": "valid in the prior run",
            "per_setting_residual_summaries": "interpretable",
            "paired_mean_residual_difference": (
                "mean estimand remains marginally valid, but its uncertainty depends on coupling"
            ),
            "framewise_prediction_rms_difference": (
                "confounded and superseded because it can be nonzero for identical expectations "
                "and changes when an unrelated grid is added"
            ),
        },
        "rows": summary_rows,
    }
    json_dump(out / "noise_coupling_audit.json", summary)
    return summary


def corrected_comparison_specs() -> list[dict]:
    return [
        {"comparison": "edge_4_to_20", "coarse": "production", "fine": "edge_ss20",
         "study": "pupil_edge_supersampling"},
        {"comparison": "propagation_1_to_5", "coarse": "production", "fine": "prop_f5",
         "study": "propagation_grid_with_detector_integration"},
    ]


def run_corrected_noisy_confirmation(cfg: dict, out: Path, benches: dict,
                                     roms: dict, scale: float,
                                     scoring_bench: O.Bench,
                                     ensemble_run: Path) -> dict:
    design = cfg["corrected_noisy_confirmation"]
    acceptance = cfg["acceptance"]
    det = detector_config(cfg, design["photons_per_pupil"], True)
    wavefront_record = json.loads((ensemble_run / "ensemble_wavefronts.json").read_text())
    truths = {float(amplitude): [np.asarray(values, dtype=float) for values in
              wavefront_record["wavefronts"][f"{int(amplitude)}nm"]]
              for amplitude in design["wavefront_rms_nm"]}
    comparisons = corrected_comparison_specs()
    needed_settings = sorted({item[key] for item in comparisons for key in ("coarse", "fine")})
    probabilities = {}
    optical_rows = []
    spec_lookup = setting_specs(cfg)
    for setting_name in needed_settings:
        factor = spec_lookup[setting_name]["factor"]
        bench = benches[setting_name]
        probabilities[setting_name] = {}
        for amplitude, vectors in truths.items():
            for wavefront, truth in enumerate(vectors):
                image, _ = bench.propagate_coeffs(truth)
                views = expected_detector_views(bench, image, factor, det)
                probabilities[setting_name][(amplitude, wavefront)] = views["probability"]
                optical_rows.append({
                    "setting_name": setting_name, "amplitude_nm_rms": amplitude,
                    "wavefront": wavefront, "captured_flux_before_normalization":
                    views["captured_flux"],
                    "probability_sha256": sha256_bytes(
                        views["probability"].astype(np.float32).tobytes()),
                })
    write_csv(out / "corrected_noisy_optical_inputs.csv", optical_rows)

    raw_rows = []
    wavefront_rows = []
    summary_rows = []
    previous_rows = []
    with (ensemble_run / "ensemble_paired_summary.csv").open(newline="") as handle:
        old_all = list(csv.DictReader(handle))
    for comparison_index, comparison in enumerate(comparisons):
        coarse_name = comparison["coarse"]; fine_name = comparison["fine"]
        for amplitude_index, amplitude in enumerate(design["wavefront_rms_nm"]):
            per_wavefront_residual = []
            per_wavefront_mean_reconstruction = []
            per_wavefront_deterministic = []
            per_wavefront_variability_change = []
            per_wavefront_reference_residual = []
            for wavefront, truth in enumerate(truths[float(amplitude)]):
                coarse_probability = probabilities[coarse_name][(float(amplitude), wavefront)]
                fine_probability = probabilities[fine_name][(float(amplitude), wavefront)]
                expected = {
                    coarse_name: coarse_probability * det.total_photons * det.quantum_efficiency,
                    fine_name: fine_probability * det.total_photons * det.quantum_efficiency,
                }
                coarse_deterministic = frozen_pipeline(
                    np.clip(np.rint(expected[coarse_name] / det.gain_e_per_adu),
                            0, det.adc_max).astype(np.int64), roms)
                fine_deterministic = frozen_pipeline(
                    np.clip(np.rint(expected[fine_name] / det.gain_e_per_adu),
                            0, det.adc_max).astype(np.int64), roms)
                coarse_det_coeff = coarse_deterministic["zernike_raw"].astype(float) * scale
                fine_det_coeff = fine_deterministic["zernike_raw"].astype(float) * scale
                deterministic_difference = rms_coefficients(
                    scoring_bench, coarse_det_coeff - fine_det_coeff) * 1e9
                coarse_coefficients = []
                fine_coefficients = []
                residual_differences = []
                fine_residuals = []
                for repeat in range(design["nested_exposures_per_wavefront"]):
                    seed = [cfg["seeds"]["corrected_confirmation"], comparison_index,
                            amplitude_index, wavefront, repeat]
                    frames, diagnostics = pairwise_poisson_gaussian_frames(
                        expected, det, seed)
                    coeffs = {}
                    residuals = {}
                    for name in (coarse_name, fine_name):
                        coeffs[name] = frozen_pipeline(
                            frames[name], roms)["zernike_raw"].astype(float) * scale
                        residuals[name] = rms_coefficients(
                            scoring_bench, coeffs[name] - truth)
                    coarse_coefficients.append(coeffs[coarse_name])
                    fine_coefficients.append(coeffs[fine_name])
                    residual_differences.append(
                        residuals[coarse_name] - residuals[fine_name])
                    fine_residuals.append(residuals[fine_name])
                    raw_rows.append({
                        **comparison, "amplitude_nm_rms": amplitude,
                        "photons_per_pupil": design["photons_per_pupil"],
                        "wavefront": wavefront, "repeat": repeat,
                        "signed_residual_difference_m": residual_differences[-1],
                        "coarse_residual_m": residuals[coarse_name],
                        "fine_residual_m": residuals[fine_name],
                        "framewise_reconstructed_wavefront_rms_difference_m":
                            rms_coefficients(scoring_bench,
                                             coeffs[coarse_name] - coeffs[fine_name]),
                        "coarse_frame_sha256": sha256_bytes(
                            frames[coarse_name].astype(np.uint8).tobytes()),
                        "fine_frame_sha256": sha256_bytes(
                            frames[fine_name].astype(np.uint8).tobytes()),
                        "pair_local_coupling": True,
                        "coarse_adc_clipped_pixels": diagnostics["per_setting"][coarse_name][
                            "adc_clipped_pixels"],
                        "fine_adc_clipped_pixels": diagnostics["per_setting"][fine_name][
                            "adc_clipped_pixels"],
                    })
                coarse_coefficients = np.asarray(coarse_coefficients)
                fine_coefficients = np.asarray(fine_coefficients)
                coarse_mean = coarse_coefficients.mean(axis=0)
                fine_mean = fine_coefficients.mean(axis=0)
                mean_reconstruction_difference = rms_coefficients(
                    scoring_bench, coarse_mean - fine_mean) * 1e9
                coarse_variability = float(np.sqrt(np.mean([
                    rms_coefficients(scoring_bench, value - coarse_mean)**2
                    for value in coarse_coefficients]))) * 1e9
                fine_variability = float(np.sqrt(np.mean([
                    rms_coefficients(scoring_bench, value - fine_mean)**2
                    for value in fine_coefficients]))) * 1e9
                residual_mean = float(np.mean(residual_differences)) * 1e9
                reference_residual_mean = float(np.mean(fine_residuals)) * 1e9
                per_wavefront_residual.append(residual_mean)
                per_wavefront_mean_reconstruction.append(mean_reconstruction_difference)
                per_wavefront_deterministic.append(deterministic_difference)
                per_wavefront_variability_change.append(coarse_variability - fine_variability)
                per_wavefront_reference_residual.append(reference_residual_mean)
                wavefront_rows.append({
                    **comparison, "amplitude_nm_rms": amplitude,
                    "photons_per_pupil": design["photons_per_pupil"],
                    "wavefront": wavefront,
                    "mean_residual_difference_nm": residual_mean,
                    "mean_reconstruction_difference_nm": mean_reconstruction_difference,
                    "deterministic_expected_frame_reconstruction_difference_nm":
                        deterministic_difference,
                    "coarse_noise_variability_nm": coarse_variability,
                    "fine_noise_variability_nm": fine_variability,
                    "noise_variability_difference_nm": coarse_variability - fine_variability,
                    "fine_reference_residual_mean_nm": reference_residual_mean,
                    "independent_unit": "wavefront; exposures nested",
                })
            residual_stats = t_interval(per_wavefront_residual)
            reconstruction_stats = t_interval(per_wavefront_mean_reconstruction)
            deterministic_stats = t_interval(per_wavefront_deterministic)
            variability_stats = t_interval(per_wavefront_variability_change)
            reference_stats = t_interval(per_wavefront_reference_residual)
            margin = min(acceptance["absolute_tolerance_nm"],
                         acceptance["relative_tolerance_pct"] / 100 * reference_stats["mean"])
            residual_pass = (residual_stats["ci_low"] >= -margin and
                             residual_stats["ci_high"] <= margin)
            reconstruction_pass = reconstruction_stats["ci_high"] <= margin
            old = next(row for row in old_all if row["comparison"] == comparison["comparison"]
                       and float(row["amplitude_nm_rms"]) == float(amplitude)
                       and float(row["photons_per_pupil"]) ==
                       float(design["photons_per_pupil"]))
            previous_rows.append({
                "comparison": comparison["comparison"], "amplitude_nm_rms": amplitude,
                "old_framewise_prediction_mean_nm": float(
                    old["prediction_rms_difference_mean_nm"]),
                "old_framewise_prediction_ci95_high_nm": float(
                    old["prediction_rms_difference_ci95_high_nm"]),
                "new_pair_local_mean_reconstruction_difference_nm":
                    reconstruction_stats["mean"],
                "new_pair_local_mean_reconstruction_ci95_high_nm":
                    reconstruction_stats["ci_high"],
                "old_decision": old["ensemble_cell_accepted"],
                "old_prediction_metric_status": "SUPERSEDED: comparison-set-dependent coupling",
            })
            summary_rows.append({
                **comparison, "amplitude_nm_rms": amplitude,
                "photons_per_pupil": design["photons_per_pupil"],
                "n_independent_wavefronts": residual_stats["n"],
                "nested_exposures_per_wavefront": design["nested_exposures_per_wavefront"],
                "fine_reference_residual_mean_nm": reference_stats["mean"],
                "equivalence_margin_nm": margin,
                "mean_residual_difference_nm": residual_stats["mean"],
                "mean_residual_difference_ci95_low_nm": residual_stats["ci_low"],
                "mean_residual_difference_ci95_high_nm": residual_stats["ci_high"],
                "residual_interval_wholly_inside_margin": bool(residual_pass),
                "mean_reconstruction_difference_nm": reconstruction_stats["mean"],
                "mean_reconstruction_difference_ci95_low_nm": reconstruction_stats["ci_low"],
                "mean_reconstruction_difference_ci95_high_nm": reconstruction_stats["ci_high"],
                "mean_reconstruction_upper_below_margin": bool(reconstruction_pass),
                "deterministic_expected_frame_difference_mean_nm": deterministic_stats["mean"],
                "deterministic_expected_frame_difference_ci95_low_nm":
                    deterministic_stats["ci_low"],
                "deterministic_expected_frame_difference_ci95_high_nm":
                    deterministic_stats["ci_high"],
                "noise_variability_difference_mean_nm": variability_stats["mean"],
                "noise_variability_difference_ci95_low_nm": variability_stats["ci_low"],
                "noise_variability_difference_ci95_high_nm": variability_stats["ci_high"],
                "corrected_cell_accepted": bool(residual_pass and reconstruction_pass),
                "rule_frozen_before_run": True,
            })
    write_csv(out / "corrected_noisy_raw.csv", raw_rows)
    write_csv(out / "corrected_noisy_wavefront_units.csv", wavefront_rows)
    write_csv(out / "corrected_noisy_summary.csv", summary_rows)
    write_csv(out / "noisy_before_after.csv", previous_rows)
    summary = {
        "design": design, "acceptance": acceptance, "rows": summary_rows,
        "previous_rows": previous_rows,
        "all_frames_non_clipping": bool(all(
            row["coarse_adc_clipped_pixels"] == 0 and
            row["fine_adc_clipped_pixels"] == 0 for row in raw_rows)),
        "accepted_cells": int(sum(row["corrected_cell_accepted"] for row in summary_rows)),
        "total_cells": len(summary_rows),
        "old_prediction_metric_status": "superseded; historical files preserved",
    }
    json_dump(out / "corrected_noisy_summary.json", summary)
    return summary


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_figures(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir()

    simple = read_csv(out / "simple_lenslet_validation.csv")
    shifts = sorted({float(row["tilt_shift_command_px"]) for row in simple})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for shift in shifts:
        rows = sorted((row for row in simple
                       if float(row["tilt_shift_command_px"]) == shift),
                      key=lambda row: int(row["factor"]))
        factor = [int(row["factor"]) for row in rows]
        label = f"{shift:+.1f} pixel tilt"
        axes[0].plot(factor, [float(row["normalized_image_relative_l2_error"])
                              for row in rows], "o-", label=label)
        axes[1].plot(factor, [float(row["centroid_x_error_m"]) * 1e9
                              for row in rows], "o-", label=label)
        axes[2].plot(factor, [float(row["captured_flux_error"]) for row in rows],
                     "o-", label=label)
    axes[0].set_ylabel("relative L2 error"); axes[0].set_title("Pixel-integrated image")
    axes[1].set_ylabel("centroid error (nm)"); axes[1].set_title("Tilt displacement")
    axes[2].set_ylabel("captured-power error"); axes[2].set_title("Finite detector field")
    for ax in axes:
        ax.set_xlabel("propagation factor"); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures / f"simple_lenslet_validation.{suffix}", dpi=180)
    plt.close(fig)

    paired = read_csv(out / "forward_stage_paired.csv")
    prop = [row for row in paired if row["study"] == "propagation"]
    cases = sorted({row["case"] for row in prop})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for case in cases:
        rows = sorted((row for row in prop if row["case"] == case),
                      key=lambda row: int(row["setting"]))
        settings = [int(row["setting"]) for row in rows]
        axes[0, 0].plot(settings, [float(row["stage_a_normalized_expected_e_relative_l2"])
                                   for row in rows], "o-", label=case)
        axes[0, 1].plot(settings, [float(row["stage_b_centroid_rms_change_px"])
                                   for row in rows], "o-", label=case)
        axes[1, 0].plot(settings, [float(row["stage_c_centroid_rms_change_px"])
                                   for row in rows], "o-", label=case)
        axes[1, 1].plot(settings, [float(row[
            "stage_e_fixed_reconstructed_wavefront_rms_change_m"]) * 1e9
                                   for row in rows], "o-", label=case)
    titles = (("A: expected-electron image", "relative L2 vs factor 5"),
              ("B: continuous centroid", "RMS change (pixel)"),
              ("C: after continuous threshold", "RMS change (pixel)"),
              ("E: fixed reconstruction", "wavefront change (nm)"))
    for ax, (title, ylabel) in zip(axes.ravel(), titles):
        ax.set_title(title); ax.set_xlabel("propagation factor")
        ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures / f"forward_stage_localization.{suffix}", dpi=180)
    plt.close(fig)

    arrays = np.load(out / "forward_stage_arrays.npz")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for row, case in enumerate(("flat", "single_tilt_x_100nm")):
        production = arrays[f"expected_e__production__{case}"]
        reference = arrays[f"expected_e__prop_f5__{case}"]
        difference = production - reference
        im0 = axes[row, 0].imshow(difference, origin="lower", cmap="coolwarm")
        axes[row, 0].set_title(f"{case}: production − factor 5 (e⁻)")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046)
        crop = difference[96:160, 96:160]
        im1 = axes[row, 1].imshow(crop, origin="lower", cmap="coolwarm")
        axes[row, 1].set_title(f"{case}: central 4×4 lenslets")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures / f"expected_electron_differences.{suffix}", dpi=180)
    plt.close(fig)

    controls = read_csv(out / "noise_coupling_controls_summary.csv")
    controls = [row for row in controls
                if row["metric"] == "reconstructed_wavefront_rms_difference_nm"]
    labels = [row["control"].replace("_", "\n") for row in controls]
    means = np.asarray([float(row["mean"]) for row in controls])
    low = np.asarray([float(row["ci95_low"]) for row in controls])
    high = np.asarray([float(row["ci95_high"]) for row in controls])
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    ax.bar(range(len(labels)), means, color="#4477AA")
    ax.errorbar(range(len(labels)), means,
                yerr=np.vstack([means - low, high - means]), fmt="none",
                color="black", capsize=3)
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set_ylabel("reconstructed difference (nm)")
    ax.set_title("Noise-coupling controls (mean and 95% CI)")
    ax.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures / f"noise_coupling_controls.{suffix}", dpi=180)
    plt.close(fig)

    calibration = read_csv(out / "calibration_separation_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    policies = sorted({row["calibration_policy"] for row in calibration})
    for policy in policies:
        rows = sorted((row for row in calibration if row["calibration_policy"] == policy),
                      key=lambda row: int(row["factor"]))
        ax.plot([int(row["factor"]) for row in rows],
                [float(row["mean_residual_nm"]) for row in rows], "o-",
                label=policy.replace("_", " "))
    ax.set_xlabel("propagation factor"); ax.set_ylabel("mean residual (nm)")
    ax.set_title("Calibration policy remains separate from the forward model")
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures / f"calibration_separation.{suffix}", dpi=180)
    plt.close(fig)

    before_after = read_csv(out / "noisy_before_after.csv")
    labels = [f"{row['comparison']}\n{float(row['amplitude_nm_rms']):.0f} nm"
              for row in before_after]
    old = np.asarray([float(row["old_framewise_prediction_mean_nm"])
                      for row in before_after])
    new = np.asarray([float(row["new_pair_local_mean_reconstruction_difference_nm"])
                      for row in before_after])
    x = np.arange(len(labels)); width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, old, width, label="old framewise metric (superseded)")
    ax.bar(x + width / 2, new, width, label="pair-local mean-reconstruction metric")
    ax.set_xticks(x, labels); ax.set_ylabel("reconstructed difference (nm)")
    ax.set_title("Bounded noisy before/after comparison")
    ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures / f"corrected_noisy_before_after.{suffix}", dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def write_report(out: Path, cfg: dict, simple: dict, geometry: dict,
                 stage: dict, calibration: dict, noise: dict,
                 corrected: dict) -> None:
    simple_rows = simple["rows"]
    simple_table = []
    for factor in cfg["simple_lenslet"]["propagation_factors"]:
        row = next(item for item in simple_rows if item["factor"] == factor and
                   item["tilt_shift_command_px"] == 0.4)
        simple_table.append([
            str(factor), str(row["pupil_samples_across_lenslet"]),
            f"{row['normalized_image_relative_l2_error']:.5f}",
            f"{row['centroid_x_error_m']*1e9:+.3f}",
            f"{row['captured_output_power']:.9f}",
            f"{row['central_first_zero_box_energy']:.5f}",
        ])

    paired = stage["summary"]["rows"]
    propagation_production = [row for row in paired if row["study"] == "propagation"
                              and row["setting"] == 1]
    stage_table = []
    for row in propagation_production:
        stage_table.append([
            row["case"], f"{row['stage_a_captured_flux_relative_change']*100:+.4f}%",
            f"{row['stage_a_normalized_expected_e_relative_l2']:.5f}",
            f"{row['stage_b_centroid_rms_change_px']:.5f}",
            f"{row['stage_c_centroid_rms_change_px']:.5f}",
            f"{row['stage_d_centroid_rms_change_px']:.5f}",
            f"{row['stage_e_fixed_reconstructed_wavefront_rms_change_m']*1e9:.3f}",
        ])
    nonflat = [row for row in propagation_production if row["case"] != "flat"]
    mean_a = float(np.mean([row["stage_a_normalized_expected_e_relative_l2"]
                            for row in nonflat]))
    mean_b = float(np.mean([row["stage_b_centroid_rms_change_px"] for row in nonflat]))
    mean_c = float(np.mean([row["stage_c_centroid_rms_change_px"] for row in nonflat]))
    mean_d = float(np.mean([row["stage_d_centroid_rms_change_px"] for row in nonflat]))
    mean_e = float(np.mean([row["stage_e_fixed_reconstructed_wavefront_rms_change_m"]
                            for row in nonflat])) * 1e9

    calibration_table = []
    for row in calibration["rows"]:
        if row["factor"] in (1, 5):
            calibration_table.append([
                str(row["factor"]), row["calibration_policy"],
                f"{row['mean_residual_nm']:.3f}", f"{row['max_residual_nm']:.3f}",
                "yes" if row["active_production_path"] else "diagnostic",
            ])

    control_table = []
    for control in ("identical_pair_legacy_only", "identical_pair_legacy_embedded",
                    "identical_pair_corrected"):
        detector = next(row for row in noise["rows"] if row["control"] == control and
                        row["metric"] == "detector_rms_adu")
        reconstruction = next(row for row in noise["rows"] if row["control"] == control and
                              row["metric"] ==
                              "reconstructed_wavefront_rms_difference_nm")
        control_table.append([
            control, f"{detector['mean']:.5f}",
            f"{reconstruction['mean']:.3f} "
            f"[{reconstruction['ci95_low']:.3f}, {reconstruction['ci95_high']:.3f}]",
        ])

    corrected_table = []
    for row in corrected["rows"]:
        corrected_table.append([
            row["comparison"], f"{row['amplitude_nm_rms']:.0f}",
            f"{row['fine_reference_residual_mean_nm']:.3f}",
            (f"{row['mean_residual_difference_nm']:+.3f} "
             f"[{row['mean_residual_difference_ci95_low_nm']:+.3f}, "
             f"{row['mean_residual_difference_ci95_high_nm']:+.3f}]"),
            (f"{row['mean_reconstruction_difference_nm']:.3f} "
             f"(upper {row['mean_reconstruction_difference_ci95_high_nm']:.3f})"),
            f"{row['deterministic_expected_frame_difference_mean_nm']:.3f}",
            f"{row['noise_variability_difference_mean_nm']:+.3f}",
            f"{row['equivalence_margin_nm']:.3f}",
            "PASS" if row["corrected_cell_accepted"] else "fail",
        ])

    factor4_rows = [row for row in paired if row["study"] == "propagation" and
                    row["setting"] == 4 and row["case"] != "flat"]
    factor4_max_residual_delta = max(abs(row["stage_e_fixed_signed_residual_change_m"])
                                     for row in factor4_rows) * 1e9
    factor4_max_prediction = max(row[
        "stage_e_fixed_reconstructed_wavefront_rms_change_m"] for row in factor4_rows) * 1e9
    claims = [
        {"claim": "Prior all-grid-coupled framewise prediction difference is valid",
         "status": "SUPERSEDED", "evidence": "noise_coupling_controls_summary.csv"},
        {"claim": "Refinement preserves physical geometry and detector boundaries",
         "status": "ESTABLISHED" if geometry["all_physical_constants_identical"] and
         geometry["all_mla_counts_exact"] and geometry["all_pixel_centres_aligned"]
         else "NOT ESTABLISHED", "evidence": "physical_geometry_audit.csv"},
        {"claim": "Square-lenslet propagation follows independent analytic diffraction",
         "status": "QUANTIFIED", "evidence": "simple_lenslet_validation.csv"},
        {"claim": "Full frozen production optical predictions are converged",
         "status": "NOT ESTABLISHED", "evidence": "forward_stage_paired.csv"},
        {"claim": "Corrected bounded noisy comparisons establish stability",
         "status": ("ESTABLISHED" if corrected["accepted_cells"] == corrected["total_cells"]
                    else "NOT ESTABLISHED"),
         "evidence": "corrected_noisy_summary.csv"},
        {"claim": "Active RTL/calibration was changed", "status": "NO",
         "evidence": "MANIFEST.json"},
    ]
    write_csv(out / "claim_to_evidence.csv", claims)
    deferred = """# Deferred work: reciprocal precision

Reciprocal precision optimization remains deferred. This investigation did not
change RTL arithmetic, widths, reciprocal scaling or LUT contents, pipeline
stages, active reference ROMs, or active E-matrix ROMs. Any future reciprocal
work remains a separate architecture task requiring renewed RTL equivalence and
calibration-dependent evaluation.
"""
    (out / "DEFERRED_WORK.md").write_text(deferred)

    text = f"""# Focused optical-forward-model and convergence-method investigation

Run: `{cfg['run_id']}`  
Source: `ensemble_convergence_v1_20260928_final` (preserved)  
Scope: HCIPy/Python diagnostics; no RTL, ROM, gain, threshold, geometry, or
acceptance-tolerance change.

## Executive conclusions

The previous noisy **framewise prediction-difference** test was confounded.
Its all-grid shared-minimum construction preserved every setting's marginal
detector distribution, but an unrelated third grid changed the common Poisson
component and caused independent excess draws even for identical expectations.
That metric and its 0/36 decision are therefore **superseded**, not deleted.
Per-setting ensemble residuals remain interpretable; paired mean differences
retain their mean estimand but their uncertainty was coupling-dependent and the
affected production comparisons were repeated pair-locally.

The physical optical setup is preserved across refinement. Wavelength, OPD
coefficients, aperture dimensions, magnification, lenslet centres/pitch/focal
length, detector edges, propagation distance, and orientation are invariant;
every lenslet owns exactly the expected `(16×factor)^2` samples and block-
integrated detector centres align within 1e-15 m. All grids are even and omit a
sample exactly at zero in the same symmetric convention.

The large deterministic propagation sensitivity is not a geometry change. It
is already present before thresholding: for the three nonflat production→factor-5
cases, normalized expected-electron images differ by mean relative L2
{mean_a:.5f} and continuous centroids by {mean_b:.5f} pixel RMS. The unchanged
threshold changes that to {mean_c:.5f} pixel and integer ADC sampling to
{mean_d:.5f} pixel; the frozen reconstructor maps the result to {mean_e:.3f} nm
mean wavefront change. The simple-lenslet result identifies coarse sampling of
an approximately 31 µm FWHM diffraction spot by 20.833 µm production pixels,
followed by discrete physical-pixel integration, as the initiating numerical
sensitivity. Threshold/ADC discontinuities and calibration then amplify or
reshape it; they do not create the pre-threshold difference.

No production optical source, RTL, or calibration was changed. The justified
source fix is confined to the **test methodology**: the corrected pairwise
coupling uses only the two expectations, assigns streams in stable name order,
has exact Poisson/Gaussian marginals, and produces identical frames for
identical expectations.

## 1. Noise-coupling controls

{markdown_table(['Control', 'Mean detector RMS (ADU)',
                 'Mean reconstructed difference [95% CI] (nm)'], control_table)}

Insertion-order permutation produced {noise['legacy_insertion_order_permutation_failures']}
failures in {noise['n_exposures']} controls because the old implementation
sorted keys. Adding/removing the unrelated grid did change the joint draw.
The corrected pair repeated bit-exactly after an unrelated pair was drawn
({noise['corrected_pair_repeat_failures_after_unrelated_draw']} failures).
No empirical floor was subtracted.

## 2. Independent simple-lenslet validation

One hard-edged square lenslet uses the production HCIPy `MicroLensArray` and
`FresnelPropagator`, without noise, threshold, clipping, or ADC rounding. The
independent reference is the thin-lens focal-plane
`sinc²(a x/(λf)) sinc²(a y/(λf))` solution integrated with adaptive quadrature
over the exact detector-pixel edges. Tilt is checked against `x=fθ`.

{markdown_table(['Factor', 'Samples/lenslet', 'Image rel. L2',
                 'Tilt centroid error (nm)', 'Captured power',
                 'First-zero-box energy'], simple_table)}

The analytic first zero is 35 µm and FWHM is 31.006 µm. Captured flux and the
finite-field spot width are reported separately from normalized image shape.

## 3. Full-model physical audit

`physical_geometry_audit.csv` records every physical invariant, grid centre and
edge, MLA assignment count, Fresnel padding/branch condition, and orientation.
The Fresnel propagator uses its documented 2× zero padding and 2× transfer-
function sampling; no wraparound setting changes with factor 1–5.

The pupil aperture values are coherent **cell-average amplitudes**. This is the
appropriate quadrature for a binary coherent field. Replacing them by square
roots of area fractions would preserve per-cell energy but change coherent
diffraction, so it was not done. `aperture_discretization_audit.csv` compares
edge supersampling to a factor-5 binary point quadrature block-averaged onto the
production cells.

## 4. Stage-by-stage propagation localization

All cases use the same coefficient vectors and common scoring pupil. Absolute
HCIPy output power is recorded before the existing detector model normalizes to
its stated photons-per-pupil definition, exposing rather than hiding finite-
field differences.

{markdown_table(['Case', 'Captured flux Δ', 'A expected-e rel. L2',
                 'B centroid RMS (px)', 'C threshold RMS (px)',
                 'D integer RMS (px)', 'E fixed wavefront Δ (nm)'], stage_table)}

Stage definitions: A integrated expected photoelectrons; B continuous centroid
without threshold; C continuous centroid after the unchanged 5-ADU threshold;
D deterministic integer detector plus floating centroid; E frozen fixed-point
reconstruction. Per-lenslet spatial differences are in
`forward_stage_lenslets.csv`; image and coefficient arrays are in
`forward_stage_arrays.npz`.

## 5. Calibration sensitivity (diagnostic only)

{markdown_table(['Factor', 'Calibration policy', 'Mean residual (nm)',
                 'Max residual (nm)', 'Production?'], calibration_table)}

The factor-5 common calibration and every per-grid calibration are diagnostics.
They were not written to active ROMs. Their contrast with the frozen rows shows
how calibration can absorb a portion of the forward-model change without
making the underlying detector images equal.

## 6. Bounded corrected noisy confirmation

Only the affected production→supported-reference comparisons at 3M photons
were repeated: 20 and 100 nm RMS, 20 independent wavefronts, and five nested
exposures. The new estimands were frozen before running: paired mean residual,
the wavefront RMS of the five-exposure **mean coefficient difference** as an
estimate of expected-reconstruction change, the deterministic expected-frame
difference, and separately the within-wavefront noise-variability change.

{markdown_table(['Comparison', 'Input (nm)', 'Reference residual (nm)',
                 'Residual Δ [95% CI] (nm)', 'Mean reconstruction Δ (upper) (nm)',
                 'Expected-frame Δ (nm)', 'Noise variability Δ (nm)',
                 'Margin (nm)', 'Decision'], corrected_table)}

Accepted cells: **{corrected['accepted_cells']}/{corrected['total_cells']}**.
All corrected frames were non-clipping: **{corrected['all_frames_non_clipping']}**.
The old framewise numbers are retained beside the replacement estimand in
`noisy_before_after.csv`; coupled framewise RMS is not claimed to be a
coupling-independent physical quantity.

## 7. Stability and remaining limits

No full-model production setting is promoted as converged unless both original
tolerances hold without a self-comparison. For propagation factor 4→5, the
nonflat-case maximum residual change is {factor4_max_residual_delta:.3f} nm and
maximum fixed reconstructed-wavefront change is {factor4_max_prediction:.3f} nm.
The corrected noisy decisions do not override deterministic failures.

Historical RTL/model equality, nine-cycle simulated latency, the 7.615181 nm
fixed-flat result including tip/tilt, broad frozen-grid residuals, and clipping
ablation do not depend on the flawed coupling and need no rerun. All historical
noisy per-setting marginals remain usable. Claims based on the old 0/36
framewise prediction-difference acceptance table must instead cite this run;
other grid-dependent optical predictions remain conditional on discretization.

Practical full-model propagation and pupil-edge convergence remain unresolved.
The bounded study does not establish a highest grid as truth, physical-bench
validity, or implemented-FPGA performance.

## 8. Claim-to-evidence and reproduction

See `claim_to_evidence.csv` and `MANIFEST.json`.

```bash
sim/venv/bin/python -m unittest \\
  evaluation.tests.test_noise_coupling \\
  evaluation.tests.test_rom_loading \\
  evaluation.tests.test_reference_conventions -v
sim/venv/bin/python evaluation/optical/forward_model_investigation.py \\
  --config evaluation/configs/forward_model_investigation_v1.json \\
  --out evaluation/results/runs/{cfg['run_id']}_reproduction
```

The output directory must not exist. Reciprocal optimization remains deferred.
"""
    (out / "REPORT.md").write_text(text)


def freeze_provenance(cfg_path: Path, cfg: dict, out: Path, command: list[str],
                      initial_status: list[str], tracked_patch: bytes) -> dict:
    calibration_run = (C.REPO_ROOT / cfg["source_calibration_run"]).resolve()
    blocker_run = (C.REPO_ROOT / cfg["source_blocker_run"]).resolve()
    ensemble_run = (C.REPO_ROOT / cfg["source_ensemble_run"]).resolve()
    frozen = out / "frozen_inputs"
    snapshot = frozen / "source_snapshot"
    snapshot.mkdir(parents=True)
    shutil.copy2(cfg_path, frozen / "config.json")
    shutil.copytree(calibration_run / "frozen_inputs", frozen / "calibration")
    relevant = [
        Path(__file__).resolve(), cfg_path.resolve(),
        C.EVAL_DIR / "common" / "noise_coupling.py",
        C.EVAL_DIR / "common" / "optics.py",
        C.EVAL_DIR / "common" / "detector.py",
        C.EVAL_DIR / "common" / "bitaccurate.py",
        C.EVAL_DIR / "tests" / "test_noise_coupling.py",
        C.EVAL_DIR / "optical" / "ensemble_convergence.py",
        C.EVAL_DIR / "optical" / "blocker_investigation.py",
    ]
    for source in relevant:
        destination = snapshot / source.relative_to(C.REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    hcipy_root = Path(importlib.import_module("hcipy").__file__).resolve().parent
    hcipy_sources = [
        hcipy_root / "wavefront_sensing" / "shack_hartmann.py",
        hcipy_root / "optics" / "micro_lens_array.py",
        hcipy_root / "optics" / "magnifier.py",
        hcipy_root / "propagation" / "fresnel.py",
        hcipy_root / "field" / "util.py",
    ]
    for source in hcipy_sources:
        destination = snapshot / "third_party" / "hcipy" / source.relative_to(hcipy_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    with gzip.open(frozen / "tracked_working_tree.patch.gz", "wb", compresslevel=9) as handle:
        handle.write(tracked_patch)
    (frozen / "git_status_porcelain.txt").write_text("\n".join(initial_status) + "\n")

    rom_names = ("slopes_ref_x.mem", "slopes_ref_y.mem", "e_matrix_x.mem",
                 "e_matrix_y.mem", "reciprocal_u16_q18.mem", "subaperture_bitmap.mem")
    active_roms = {name: sha256(C.RTL_DIR / name) for name in rom_names}
    source_roms = {name: sha256(calibration_run / "frozen_inputs" / "roms" / name)
                   for name in rom_names}
    if active_roms != source_roms:
        raise RuntimeError("active RTL ROMs differ from the frozen calibration source")
    frozen_files = sorted(path for path in frozen.rglob("*") if path.is_file())
    result_files = sorted(path for path in out.rglob("*") if path.is_file() and
                          frozen not in path.parents and path.name != "MANIFEST.json")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "git_revision": revision,
        "git_dirty_before_run": bool(initial_status),
        "git_status_porcelain_before_run": initial_status,
        "tracked_patch": {"path": "frozen_inputs/tracked_working_tree.patch.gz",
                          "uncompressed_bytes": len(tracked_patch),
                          "sha256_uncompressed": sha256_bytes(tracked_patch)},
        "python": sys.version, "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("hcipy", "numpy", "scipy", "matplotlib")},
        "source_runs": {
            "ensemble": {"path": cfg["source_ensemble_run"],
                         "manifest_sha256": sha256(ensemble_run / "MANIFEST.json")},
            "blocker": {"path": cfg["source_blocker_run"],
                        "manifest_sha256": sha256(blocker_run / "MANIFEST.json")},
            "calibration": {"path": cfg["source_calibration_run"],
                            "manifest_sha256": sha256(calibration_run / "MANIFEST.json")},
        },
        "active_rom_sha256": active_roms,
        "frozen_source_rom_sha256": source_roms,
        "active_roms_equal_frozen_source": True,
        "rtl_simulation": "not rerun; no RTL or active calibration input changed",
        "vivado_synthesis_implementation": "not run; outside scope",
        "configuration_sha256": sha256(cfg_path),
        "source_snapshot_paths": [str(path.relative_to(C.REPO_ROOT)) for path in relevant],
        "hcipy_source_snapshot_paths": [str(path.relative_to(hcipy_root))
                                        for path in hcipy_sources],
        "frozen_file_sha256": {str(path.relative_to(out)): sha256(path)
                               for path in frozen_files},
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
    source_paths = {name: (C.REPO_ROOT / cfg[key]).resolve() for name, key in (
        ("ensemble", "source_ensemble_run"), ("blocker", "source_blocker_run"),
        ("calibration", "source_calibration_run"))}
    for path in source_paths.values():
        if not (path / "MANIFEST.json").is_file():
            raise FileNotFoundError(path / "MANIFEST.json")

    initial_status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=C.REPO_ROOT,
        capture_output=True, text=True, check=True).stdout.splitlines()
    tracked_patch = subprocess.run(
        ["git", "diff", "--binary"], cwd=C.REPO_ROOT,
        capture_output=True, check=True).stdout
    out.mkdir(parents=True)
    rom_dir = source_paths["calibration"] / "frozen_inputs" / "roms"
    roms = load_frozen_roms(rom_dir)
    scale = float(json.loads((source_paths["calibration"] / "frozen_inputs" /
                              "derived_scale.json").read_text())["scale_m_per_lsb"])
    scoring_bench = common_scoring_bench(cfg)
    specs = setting_specs(cfg)

    print("[1/8] independent simple-lenslet diffraction validation", flush=True)
    simple = run_simple_lenslet(cfg, out)
    print("[2/8] build fixed-geometry full benches", flush=True)
    benches = build_benches(specs)
    print("[3/8] physical grid, registration, Fresnel, and aperture audit", flush=True)
    geometry = run_geometry_audit(cfg, out, specs, benches)
    print("[4/8] full-model stage-by-stage deterministic localization", flush=True)
    cases = load_deterministic_cases(cfg, source_paths["blocker"])
    stage = run_stage_audit(cfg, out, specs, benches, cases, roms, scale, scoring_bench)
    print("[5/8] identical-grid and comparison-set noise controls", flush=True)
    noise = run_noise_coupling_audit(cfg, out, stage, roms, scale, scoring_bench)
    print("[6/8] frozen/common/per-grid calibration separation", flush=True)
    calibration = run_calibration_diagnostic(
        cfg, out, benches, cases, stage, roms, scale, scoring_bench)
    print("[7/8] bounded corrected pairwise noisy confirmation", flush=True)
    corrected = run_corrected_noisy_confirmation(
        cfg, out, benches, roms, scale, scoring_bench, source_paths["ensemble"])
    summary = {
        "noise_coupling": noise, "simple_lenslet": simple,
        "geometry": geometry, "forward_stage": stage["summary"],
        "calibration": calibration, "corrected_noisy": corrected,
        "active_rtl_or_calibration_changed": False,
        "reciprocal_optimization_started": False,
    }
    json_dump(out / "summary.json", summary)
    print("[8/8] figures, report, claims, and exact provenance", flush=True)
    make_figures(out)
    write_report(out, cfg, simple, geometry, stage, calibration, noise, corrected)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(cfg_path),
               "--out", str(out)]
    freeze_provenance(cfg_path, cfg, out, command, initial_status, tracked_patch)
    print(f"complete: {out}", flush=True)


if __name__ == "__main__":
    main()
