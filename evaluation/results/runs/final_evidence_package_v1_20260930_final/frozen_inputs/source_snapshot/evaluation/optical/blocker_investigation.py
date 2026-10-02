#!/usr/bin/env python3
"""Investigate the frozen manuscript run's flat, convergence, and clipping blockers.

This script is intentionally a new, immutable run producer.  It reads the
calibration and ROMs frozen by ``manuscript_v1_20260928_final`` and never
overwrites that run or the active RTL ROMs.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import bitaccurate as BA  # noqa: E402
from evaluation.common import config as C  # noqa: E402
from evaluation.common import detector as D  # noqa: E402
from evaluation.common import optics as O  # noqa: E402
from evaluation.common import rtl_runner as R  # noqa: E402
from evaluation.common.roms import REQUIRED_ROMS, read_hex_words  # noqa: E402
from evaluation.optical.manuscript_evaluation import bin_detector  # noqa: E402


DEFAULT_CONFIG = C.EVAL_DIR / "configs" / "blocker_investigation_v1.json"
Q18 = float(1 << C.CENTROID_FRAC)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def load_words(rom_dir: Path, name: str, signed_bits: int | None = None) -> np.ndarray:
    values = np.asarray(read_hex_words(rom_dir / name, REQUIRED_ROMS[name]), dtype=np.int64)
    return BA.to_signed(values, signed_bits) if signed_bits is not None else values


def load_frozen_roms(rom_dir: Path) -> dict[str, np.ndarray]:
    # The bitmap is a single 256-bit word and intentionally does not fit int64.
    bitmap = int(read_hex_words(
        rom_dir / "subaperture_bitmap.mem", REQUIRED_ROMS["subaperture_bitmap.mem"])[0])
    valid = np.asarray([i for i in range(C.NUM_SUBAPS_GRID) if (bitmap >> i) & 1], dtype=np.int64)
    if valid.size != C.NUM_SUBAPS_VALID:
        raise ValueError(f"frozen bitmap has {valid.size} valid subapertures")
    return {
        "reciprocal": load_words(rom_dir, "reciprocal_u16_q18.mem"),
        "ref_x": load_words(rom_dir, "slopes_ref_x.mem", C.SLOPE_BITS),
        "ref_y": load_words(rom_dir, "slopes_ref_y.mem", C.SLOPE_BITS),
        "ex": load_words(rom_dir, "e_matrix_x.mem", C.EMATRIX_BITS).reshape(
            C.NUM_MODES, C.NUM_SUBAPS_VALID),
        "ey": load_words(rom_dir, "e_matrix_y.mem", C.EMATRIX_BITS).reshape(
            C.NUM_MODES, C.NUM_SUBAPS_VALID),
        "valid": valid,
    }


def exact_centroids_from_integer(adu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_i, x_i, y_i = BA.accumulate(adu)
    cx = np.divide(x_i, s_i, out=np.zeros_like(x_i, dtype=float), where=s_i > 0)
    cy = np.divide(y_i, s_i, out=np.zeros_like(y_i, dtype=float), where=s_i > 0)
    return s_i, x_i, y_i, cx, cy


def thresholded_centroids_float(adu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thresholded CoG for diagnostic ADU values wider than the RTL input."""
    image = np.asarray(adu, dtype=float).reshape(C.FRAME_H, C.FRAME_W)
    tiles = image.reshape(C.NUM_LENSLETS, C.SUBAP_PIXELS,
                          C.NUM_LENSLETS, C.SUBAP_PIXELS).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(-1, C.SUBAP_PIXELS, C.SUBAP_PIXELS)
    tiles = np.where(tiles < C.PIXEL_THRESHOLD, 0.0, tiles)
    x_coord = np.arange(C.SUBAP_PIXELS, dtype=float)[None, None, :]
    y_coord = np.arange(C.SUBAP_PIXELS, dtype=float)[None, :, None]
    total = tiles.sum(axis=(1, 2))
    x_moment = (tiles * x_coord).sum(axis=(1, 2))
    y_moment = (tiles * y_coord).sum(axis=(1, 2))
    cx = np.divide(x_moment, total, out=np.zeros_like(total), where=total > 0)
    cy = np.divide(y_moment, total, out=np.zeros_like(total), where=total > 0)
    return cx, cy


def reconstruct_integer(x_slope: np.ndarray, y_slope: np.ndarray,
                        ex: np.ndarray, ey: np.ndarray) -> tuple[np.ndarray, dict]:
    """Frozen-ROM version of the RTL MAC, independent of active repository ROMs."""
    xs = np.asarray(x_slope, dtype=object)
    ys = np.asarray(y_slope, dtype=object)
    out = np.zeros(C.NUM_MODES, dtype=np.int64)
    accumulator_wraps = output_wraps = 0
    for mode in range(C.NUM_MODES):
        acc = 0
        for subap in range(C.NUM_SUBAPS_VALID):
            px = BA.to_signed(BA.truncate(int(ex[mode, subap]) * int(xs[subap]),
                                          C.PRODUCT_BITS), C.PRODUCT_BITS)
            py = BA.to_signed(BA.truncate(int(ey[mode, subap]) * int(ys[subap]),
                                          C.PRODUCT_BITS), C.PRODUCT_BITS)
            unwrapped = acc + int(px) + int(py)
            if not -(1 << (C.ACC_BITS - 1)) <= unwrapped < (1 << (C.ACC_BITS - 1)):
                accumulator_wraps += 1
            acc = int(BA.to_signed(BA.truncate(unwrapped, C.ACC_BITS), C.ACC_BITS))
        sliced = acc >> C.ZERNIKE_SLICE_LOW
        if not -(1 << (C.ZERNIKE_BITS - 1)) <= sliced < (1 << (C.ZERNIKE_BITS - 1)):
            output_wraps += 1
        out[mode] = BA.to_signed(BA.truncate(sliced, C.ZERNIKE_BITS), C.ZERNIKE_BITS)
    return out, {"accumulator_wrap_count": accumulator_wraps,
                 "output_field_wrap_count": output_wraps}


def frozen_pipeline(adu: np.ndarray, roms: dict[str, np.ndarray]) -> dict:
    s_i, x_i, y_i = BA.accumulate(adu)
    if np.any(s_i > 0xFFFF):
        raise OverflowError("intensity outside reciprocal ROM address range")
    r_i = np.where(s_i == 0, 0x7FFFF, roms["reciprocal"][s_i])
    rec_port = BA.truncate(r_i, C.RECIP_PORT_BITS)
    cx = BA.centroid(x_i, rec_port)
    cy = BA.centroid(y_i, rec_port)
    sx = BA.slope(cx, roms["ref_x"])
    sy = BA.slope(cy, roms["ref_y"])
    idx = roms["valid"]
    zernike, diagnostics = reconstruct_integer(sx[idx], sy[idx], roms["ex"], roms["ey"])
    return {"sI": s_i, "xI": x_i, "yI": y_i, "rI": r_i,
            "rec_port": rec_port, "x_centroid": cx, "y_centroid": cy,
            "x_slope": sx, "y_slope": sy, "x_slope_valid": sx[idx],
            "y_slope_valid": sy[idx], "zernike_raw": zernike,
            "diagnostics": diagnostics}


def coefficients_from_frame(adu: np.ndarray, roms: dict[str, np.ndarray],
                            ideal_e: np.ndarray, scale_m_per_lsb: float) -> dict:
    s_i, _, _, cx, cy = exact_centroids_from_integer(adu)
    idx = roms["valid"]
    exact_slopes = np.r_[cx[idx] - roms["ref_x"][idx] / Q18,
                         cy[idx] - roms["ref_y"][idx] / Q18]
    fixed = frozen_pipeline(adu, roms)
    dequantized_e = np.hstack([roms["ex"], roms["ey"]]) * scale_m_per_lsb
    return {
        "ideal_float": ideal_e @ exact_slopes,
        "dequantized_float": dequantized_e @ exact_slopes,
        "fixed": fixed["zernike_raw"].astype(float) * scale_m_per_lsb,
        "fixed_trace": fixed,
        "invalid_centroids": int(np.count_nonzero(s_i[idx] == 0)),
    }


def rms_coefficients(bench: O.Bench, coefficients: np.ndarray) -> float:
    return bench.opd_rms(bench.opd_from_coeffs(coefficients))


def summarize_reconstructed_ensemble(bench: O.Bench, coeffs: np.ndarray) -> dict:
    coeffs = np.asarray(coeffs, dtype=float)
    mean_coeff = coeffs.mean(axis=0)
    per_frame = np.asarray([rms_coefficients(bench, row) for row in coeffs])
    deviations = np.asarray([rms_coefficients(bench, row - mean_coeff) for row in coeffs])
    return {
        "n_frames": int(len(coeffs)),
        "mean_per_frame_wavefront_rms_m": float(per_frame.mean()),
        "root_mean_square_per_frame_wavefront_rms_m": float(np.sqrt(np.mean(per_frame ** 2))),
        "sd_per_frame_wavefront_rms_m": float(per_frame.std(ddof=1)),
        "rms_of_mean_reconstructed_wavefront_m": rms_coefficients(bench, mean_coeff),
        "rms_frame_to_frame_wavefront_variability_m": float(np.sqrt(np.mean(deviations ** 2))),
        "mean_coefficients_m": mean_coeff.tolist(),
    }


def run_flat_investigation(cfg: dict, source_run: Path, out: Path,
                           roms: dict[str, np.ndarray], ideal_e: np.ndarray,
                           scale: float, bench: O.Bench) -> dict:
    frames = np.load(source_run / "rtl_input_frames.npz")
    names = [str(x) for x in frames["names"]]
    frame = frames["frames"][names.index("optical_flat_noiseless")].astype(np.int64).ravel()
    frame_hash = sha256_bytes(frame.astype(np.uint8).tobytes())
    s_i, x_i, y_i, exact_x, exact_y = exact_centroids_from_integer(frame)
    trace = frozen_pipeline(frame, roms)
    idx = roms["valid"]
    exact_qx = np.rint(exact_x * Q18).astype(np.int64)
    exact_qy = np.rint(exact_y * Q18).astype(np.int64)
    error_x = trace["x_centroid"] - exact_qx
    error_y = trace["y_centroid"] - exact_qy
    reciprocal_exact = np.divide(Q18, s_i, out=np.zeros_like(s_i, dtype=float), where=s_i > 0)
    reciprocal_error = trace["rec_port"].astype(float) - reciprocal_exact

    raw_image = frame.reshape(C.FRAME_H, C.FRAME_W)
    thresholded_image = np.where(raw_image < C.PIXEL_THRESHOLD, 0, raw_image)
    np.savez_compressed(
        out / "flat_stage_arrays.npz", raw_adu=raw_image.astype(np.uint8),
        thresholded_adu=thresholded_image.astype(np.uint8), valid_idx=idx,
        sI=s_i, xI=x_i, yI=y_i, reciprocal_q18=trace["rec_port"],
        exact_reciprocal_q18=reciprocal_exact, exact_x_q18=exact_qx,
        exact_y_q18=exact_qy, fixed_x_q18=trace["x_centroid"],
        fixed_y_q18=trace["y_centroid"], reference_x_q18=roms["ref_x"],
        reference_y_q18=roms["ref_y"], x_slope_q18=trace["x_slope"],
        y_slope_q18=trace["y_slope"], x_error_q18=error_x, y_error_q18=error_y)

    stage_rows = []
    for tile in idx:
        stage_rows.append({
            "tile": int(tile), "lenslet_row": int(tile // C.NUM_LENSLETS),
            "lenslet_col": int(tile % C.NUM_LENSLETS),
            "thresholded_sum_sI": int(s_i[tile]), "x_moment_xI": int(x_i[tile]),
            "y_moment_yI": int(y_i[tile]),
            "exact_reciprocal_q18": float(reciprocal_exact[tile]),
            "rtl_reciprocal_q18": int(trace["rec_port"][tile]),
            "reciprocal_error_q18": float(reciprocal_error[tile]),
            "exact_x_centroid_px": float(exact_x[tile]),
            "exact_y_centroid_px": float(exact_y[tile]),
            "rounded_exact_x_q18": int(exact_qx[tile]),
            "rounded_exact_y_q18": int(exact_qy[tile]),
            "rtl_x_centroid_q18": int(trace["x_centroid"][tile]),
            "rtl_y_centroid_q18": int(trace["y_centroid"][tile]),
            "stored_reference_x_q18": int(roms["ref_x"][tile]),
            "stored_reference_y_q18": int(roms["ref_y"][tile]),
            "fixed_minus_exact_x_q18": int(error_x[tile]),
            "fixed_minus_exact_y_q18": int(error_y[tile]),
            "reference_subtracted_x_slope_q18": int(trace["x_slope"][tile]),
            "reference_subtracted_y_slope_q18": int(trace["y_slope"][tile]),
        })
    write_csv(out / "flat_stage_trace.csv", stage_rows)

    dequantized_e = np.hstack([roms["ex"], roms["ey"]]) * scale
    exact_slopes = np.r_[exact_x[idx] - roms["ref_x"][idx] / Q18,
                         exact_y[idx] - roms["ref_y"][idx] / Q18]
    exact_q_slopes = np.r_[(exact_qx[idx] - roms["ref_x"][idx]) / Q18,
                           (exact_qy[idx] - roms["ref_y"][idx]) / Q18]
    fixed_slopes = np.r_[trace["x_slope_valid"] / Q18, trace["y_slope_valid"] / Q18]
    ablation_values = [
        ("original_float_matrix_exact_centroid", ideal_e @ exact_slopes),
        ("dequantized_matrix_exact_centroid", dequantized_e @ exact_slopes),
        ("original_float_matrix_rounded_exact_q18_centroid", ideal_e @ exact_q_slopes),
        ("dequantized_matrix_rounded_exact_q18_centroid", dequantized_e @ exact_q_slopes),
        ("original_float_matrix_rtl_centroid", ideal_e @ fixed_slopes),
        ("dequantized_matrix_rtl_centroid", dequantized_e @ fixed_slopes),
        ("rtl_integer_mac_rtl_centroid", trace["zernike_raw"].astype(float) * scale),
    ]
    ablation_rows = []
    for label, coeff in ablation_values:
        row = {"stage": label, "wavefront_rms_m": rms_coefficients(bench, coeff)}
        row.update({f"mode_{i + 1}_m": float(value) for i, value in enumerate(coeff)})
        ablation_rows.append(row)
    write_csv(out / "flat_coefficient_ablations.csv", ablation_rows)

    with R.RTLSimulator(rom_dir=source_run / "frozen_inputs" / "roms") as simulator:
        rtl = simulator.run_frame(frame)
    rtl_slope_delta = np.column_stack([
        rtl["slopes"][:, 1] - trace["x_slope_valid"],
        rtl["slopes"][:, 2] - trace["y_slope_valid"],
        rtl["slopes"][:, 3] - trace["x_centroid"][idx],
        rtl["slopes"][:, 4] - trace["y_centroid"][idx],
    ])
    rtl_coeff = rtl["zernike_raw"].astype(float) * scale
    rtl_summary = {
        "frame_sha256": frame_hash,
        "model_zernike_raw": trace["zernike_raw"].tolist(),
        "rtl_zernike_raw": rtl["zernike_raw"].tolist(),
        "rtl_coefficients_m": rtl_coeff.tolist(),
        "rtl_wavefront_rms_m": rms_coefficients(bench, rtl_coeff),
        "coefficient_mismatch_count": int(np.count_nonzero(
            rtl["zernike_raw"] - trace["zernike_raw"])),
        "slope_centroid_mismatch_count": int(np.count_nonzero(rtl_slope_delta)),
        "unknown_valid_values": int(rtl["unknown_valid_slopes"] +
                                    rtl["unknown_valid_coefficients"]),
        "latency_cycles": int(rtl["true_latency_cycles"]),
    }
    json_dump(out / "flat_rtl_summary.json", rtl_summary)

    optical_image = bench.propagate(np.zeros_like(np.asarray(bench.aperture)))
    detector_cfg = D.DetectorConfig(**{
        key: value for key, value in cfg["detector"].items()
        if key in D.DetectorConfig.__dataclass_fields__
    })
    rng = np.random.default_rng(cfg["seeds"]["flat_noise"])
    ensemble = {"ideal_float": [], "dequantized_float": [], "fixed": []}
    for _ in range(cfg["flat"]["fresh_noisy_exposures"]):
        exposure = D.expose(optical_image, detector_cfg, rng)
        values = coefficients_from_frame(exposure.adu, roms, ideal_e, scale)
        for key in ensemble:
            ensemble[key].append(values[key])
    noisy_summary = {key: summarize_reconstructed_ensemble(bench, np.asarray(values))
                     for key, values in ensemble.items()}
    noisy_summary["detector"] = asdict(detector_cfg)
    noisy_summary["seed"] = cfg["seeds"]["flat_noise"]
    json_dump(out / "flat_noisy_metrics.json", noisy_summary)

    vx, vy = error_x[idx].astype(float), error_y[idx].astype(float)
    summary = {
        "frame_sha256": frame_hash,
        "detector_settings_identical_to_historical_noisy_flat_except_noise_enable": {
            "photons": detector_cfg.total_photons, "gain_e_per_adu": detector_cfg.gain_e_per_adu,
            "threshold_adu": C.PIXEL_THRESHOLD, "adc_bits": detector_cfg.adc_bits,
            "pupil_and_calibration": "same frozen source run",
        },
        "thresholded_pixel_count": int(np.count_nonzero(raw_image < C.PIXEL_THRESHOLD)),
        "nonzero_pixels_after_threshold": int(np.count_nonzero(thresholded_image)),
        "stored_reference_equals_rounded_exact_q18": {
            "x": bool(np.array_equal(roms["ref_x"][idx], exact_qx[idx])),
            "y": bool(np.array_equal(roms["ref_y"][idx], exact_qy[idx])),
        },
        "fixed_minus_exact_centroid_q18": {
            "x_mean_lsb": float(vx.mean()), "y_mean_lsb": float(vy.mean()),
            "x_rms_lsb": float(np.sqrt(np.mean(vx ** 2))),
            "y_rms_lsb": float(np.sqrt(np.mean(vy ** 2))),
            "x_min_lsb": int(vx.min()), "x_max_lsb": int(vx.max()),
            "y_min_lsb": int(vy.min()), "y_max_lsb": int(vy.max()),
            "x_correlation_with_reciprocal_error": float(np.corrcoef(vx, reciprocal_error[idx])[0, 1]),
            "y_correlation_with_reciprocal_error": float(np.corrcoef(vy, reciprocal_error[idx])[0, 1]),
        },
        "fixed_mac_coefficients_m": (trace["zernike_raw"].astype(float) * scale).tolist(),
        "fixed_wavefront_rms_m": rms_coefficients(
            bench, trace["zernike_raw"].astype(float) * scale),
        "rtl": rtl_summary,
        "noisy_flat": noisy_summary,
        "cause": ("the stored reference is round(moment/sum * 2^18), while RTL forms "
                  "moment * round(2^18/sum); reciprocal rounding is multiplied by the "
                  "moment before reference subtraction"),
    }
    json_dump(out / "flat_summary.json", summary)
    return summary


def normalized_coefficients(bench: O.Bench, values: np.ndarray, rms_nm: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    current = rms_coefficients(bench, values)
    if current <= 0 or not np.isfinite(current):
        raise ValueError("cannot normalize zero/non-finite coefficient vector")
    return values * (rms_nm * 1e-9 / current)


def integrated_detector_image(image: np.ndarray, factor: int) -> np.ndarray:
    return np.asarray(bin_detector(image, factor), dtype=float).reshape(C.FRAME_H, C.FRAME_W)


def convergence_detector_views(image: np.ndarray, factor: int,
                               detector_cfg: D.DetectorConfig) -> dict[str, dict]:
    integrated = integrated_detector_image(image, factor)
    total = float(integrated.sum())
    expected_e = (integrated / total * detector_cfg.total_photons *
                  detector_cfg.quantum_efficiency + detector_cfg.background_e)
    expected_e = np.clip(expected_e, 0.0, detector_cfg.full_well_e)
    continuous_adu = expected_e / detector_cfg.gain_e_per_adu + detector_cfg.offset_adu
    production = D.expose(integrated, detector_cfg).adu.reshape(C.FRAME_H, C.FRAME_W)
    views = {
        "production_adc": production,
        "continuous_diagnostic": continuous_adu,
    }
    result = {}
    for label, adu in views.items():
        cx, cy = thresholded_centroids_float(adu)
        result[label] = {"adu": adu, "centroids": np.r_[cx, cy]}
    result["physical"] = {
        "integrated": integrated, "expected_e": expected_e,
        "raw_flux": total,
    }
    return result


def detector_grid_record(bench: O.Bench, integration_factor: int) -> dict:
    grid = bench.reference_image.grid
    x = np.unique(np.asarray(grid.x, dtype=float))
    dx = float(np.mean(np.diff(x)))
    grouped = x.reshape(C.FRAME_W, integration_factor).mean(axis=1)
    base_cfg = O.OpticalConfig()
    base_dx = base_cfg.detector_pixel_pitch_m
    base_centres = (-base_cfg.sh_diameter_m / 2 + base_dx / 2 +
                    np.arange(C.FRAME_W) * base_dx)
    mla_index = np.asarray(bench.mla.mla_index)
    _, lenslet_counts = np.unique(mla_index, return_counts=True)
    return {
        "fine_detector_samples_per_axis": int(x.size),
        "fine_pixel_pitch_m": dx,
        "fine_first_center_m": float(x[0]),
        "fine_last_center_m": float(x[-1]),
        "fine_lower_edge_m": float(x[0] - dx / 2),
        "fine_upper_edge_m": float(x[-1] + dx / 2),
        "integrated_pixel_center_alignment_max_abs_m": float(
            np.max(np.abs(grouped - base_centres))),
        "lenslet_sample_count_min": int(lenslet_counts.min()),
        "lenslet_sample_count_max": int(lenslet_counts.max()),
        "lenslet_index_count": int(lenslet_counts.size),
    }


def run_convergence_investigation(cfg: dict, out: Path) -> dict:
    settings = cfg["convergence"]
    detector_kwargs = {
        key: value for key, value in cfg["detector"].items()
        if key in D.DetectorConfig.__dataclass_fields__
    }
    detector_kwargs.update(shot_noise=False, read_noise_e=0.0)
    detector_cfg = D.DetectorConfig(**detector_kwargs)
    common_cfg = replace(O.OpticalConfig(),
                         supersample=settings["common_scoring_supersample"])
    common_bench = O.Bench(common_cfg)
    rng = np.random.default_rng(cfg["seeds"]["convergence"])
    basis = np.eye(C.NUM_MODES)
    cases: list[tuple[str, np.ndarray]] = [
        ("single_tilt_x_100nm", normalized_coefficients(common_bench, basis[0], 100.0)),
        ("single_defocus_100nm", normalized_coefficients(common_bench, basis[2], 100.0)),
    ]
    for level, count in ((settings["mixed_low_nm_rms"], 2),
                         (settings["moderate_nm_rms"], 2)):
        for case in range(count):
            direction = rng.normal(size=C.NUM_MODES)
            cases.append((f"mixed_{level:g}nm_{case}",
                          normalized_coefficients(common_bench, direction, level)))
    case_truth = {name: coeff.tolist() for name, coeff in cases}
    json_dump(out / "convergence_cases.json", {
        "seed": cfg["seeds"]["convergence"],
        "normalization_grid": asdict(common_cfg),
        "cases": case_truth,
    })

    raw_rows: list[dict] = []
    check_rows: list[dict] = []
    matrices: dict[str, np.ndarray] = {}
    flat_spots: dict[str, np.ndarray] = {}

    def evaluate_setting(study: str, setting: int, bench: O.Bench,
                         integration_factor: int) -> None:
        zero = np.zeros(C.NUM_MODES)
        flat_image = bench.propagate(np.zeros_like(np.asarray(bench.aperture)))
        flat_views = convergence_detector_views(flat_image, integration_factor, detector_cfg)
        references = {path: flat_views[path]["centroids"]
                      for path in settings["detector_paths"]}
        interactions = {path: np.zeros((2 * C.NUM_SUBAPS_VALID, C.NUM_MODES))
                        for path in settings["detector_paths"]}
        idx = BA.valid_subaps()
        probe = settings["probe_nm"] * 1e-9
        for mode in range(C.NUM_MODES):
            plus = zero.copy(); minus = zero.copy()
            plus[mode] = probe; minus[mode] = -probe
            plus_views = convergence_detector_views(
                bench.propagate_coeffs(plus)[0], integration_factor, detector_cfg)
            minus_views = convergence_detector_views(
                bench.propagate_coeffs(minus)[0], integration_factor, detector_cfg)
            for path in settings["detector_paths"]:
                p = plus_views[path]["centroids"]
                m = minus_views[path]["centroids"]
                interactions[path][:, mode] = np.r_[
                    (p[:C.NUM_SUBAPS_GRID][idx] - m[:C.NUM_SUBAPS_GRID][idx]) / (2 * probe),
                    (p[C.NUM_SUBAPS_GRID:][idx] - m[C.NUM_SUBAPS_GRID:][idx]) / (2 * probe),
                ]
        inverses = {path: np.linalg.pinv(interactions[path])
                    for path in settings["detector_paths"]}
        for path in settings["detector_paths"]:
            key = f"{study}_setting_{setting}_{path}"
            matrices[f"interaction_{key}"] = interactions[path]
            matrices[f"inverse_{key}"] = inverses[path]
            flat_spots[f"flat_adu_{key}"] = flat_views[path]["adu"]

        grid = detector_grid_record(bench, integration_factor)
        aperture = np.asarray(bench.aperture, dtype=float)
        physical = flat_views["physical"]
        for path in settings["detector_paths"]:
            singular = np.linalg.svd(interactions[path], compute_uv=False)
            flat_adu = flat_views[path]["adu"]
            ref = references[path]
            check_rows.append({
                "study": study, "setting": setting, "detector_path": path,
                **grid,
                "pupil_grid_samples_per_axis": bench.cfg.num_pupil_pixels,
                "pupil_pixel_pitch_m": bench.cfg.pupil_grid_diameter_m / bench.cfg.num_pupil_pixels,
                "pupil_supersample": bench.cfg.supersample,
                "pupil_partial_transmission_samples": int(np.count_nonzero(
                    (aperture > 0) & (aperture < 1))),
                "pupil_transmission_sum": float(aperture.sum()),
                "pupil_transmission_squared_sum": float(np.sum(aperture ** 2)),
                "pupil_semantics": ("mean of binary aperture sub-samples, used directly as "
                                    "electric-field amplitude"),
                "raw_flat_flux_before_detector_normalization": physical["raw_flux"],
                "integrated_flat_flux": float(physical["integrated"].sum()),
                "expected_electrons_sum_after_normalization": float(physical["expected_e"].sum()),
                "flat_adu_sum": float(flat_adu.sum()),
                "flat_adc_clipped_pixels": int(np.count_nonzero(
                    flat_adu >= detector_cfg.adc_max)) if path == "production_adc" else 0,
                "flat_reference_mean_px": float(ref.mean()),
                "flat_reference_sd_px": float(ref.std()),
                "interaction_singular_min": float(singular[-1]),
                "interaction_singular_max": float(singular[0]),
                "interaction_condition_number": float(singular[0] / singular[-1]),
                "interaction_singular_values": json.dumps(singular.tolist()),
                "calibration_probe_nm": settings["probe_nm"],
            })

        for case_name, truth in cases:
            image, _ = bench.propagate_coeffs(truth)
            views = convergence_detector_views(image, integration_factor, detector_cfg)
            for path in settings["detector_paths"]:
                centroid = views[path]["centroids"]
                slopes = np.r_[
                    centroid[:C.NUM_SUBAPS_GRID][idx] - references[path][:C.NUM_SUBAPS_GRID][idx],
                    centroid[C.NUM_SUBAPS_GRID:][idx] - references[path][C.NUM_SUBAPS_GRID:][idx],
                ]
                recovered = inverses[path] @ slopes
                residual = rms_coefficients(common_bench, recovered - truth)
                raw_rows.append({
                    "study": study, "setting": setting, "detector_path": path,
                    "case": case_name,
                    "input_rms_m_on_common_grid": rms_coefficients(common_bench, truth),
                    "residual_m_on_common_grid": residual,
                    "truth_coefficients_m": json.dumps(truth.tolist()),
                    "recovered_coefficients_m": json.dumps(recovered.tolist()),
                    "scoring_grid": (f"base 256x256, edge supersample "
                                     f"{settings['common_scoring_supersample']}"),
                })

    for supersample in settings["pupil_edge_supersampling"]:
        evaluate_setting("pupil_edge_supersampling", supersample,
                         O.Bench(O.OpticalConfig(supersample=supersample)), 1)
    base = O.OpticalConfig()
    for factor in settings["propagation_grid_factors"]:
        evaluate_setting(
            "propagation_grid_with_detector_integration", factor,
            O.Bench(replace(base, pupil_pixels_base=base.pupil_pixels_base * factor)), factor)

    # Compare flat spots and references against the highest tested setting.
    for study in ("pupil_edge_supersampling", "propagation_grid_with_detector_integration"):
        tested = sorted({row["setting"] for row in check_rows if row["study"] == study})
        highest = tested[-1]
        for path in settings["detector_paths"]:
            ref_key = f"flat_adu_{study}_setting_{highest}_{path}"
            reference_flat = flat_spots[ref_key]
            for row in check_rows:
                if row["study"] != study or row["detector_path"] != path:
                    continue
                key = f"flat_adu_{study}_setting_{row['setting']}_{path}"
                delta = np.asarray(flat_spots[key], dtype=float) - reference_flat
                row["flat_adu_rms_difference_vs_highest"] = float(np.sqrt(np.mean(delta ** 2)))
                row["flat_adu_max_abs_difference_vs_highest"] = float(np.max(np.abs(delta)))

    write_csv(out / "convergence_raw.csv", raw_rows)
    write_csv(out / "convergence_physical_checks.csv", check_rows)
    np.savez_compressed(out / "convergence_matrices.npz", **matrices)
    np.savez_compressed(out / "convergence_flat_spots.npz", **flat_spots)

    tolerance_rel = settings["acceptance_relative_pct"]
    tolerance_abs = settings["acceptance_absolute_nm"]
    paired_rows: list[dict] = []
    summary_rows: list[dict] = []
    for study in ("pupil_edge_supersampling", "propagation_grid_with_detector_integration"):
        for path in settings["detector_paths"]:
            subset = [row for row in raw_rows
                      if row["study"] == study and row["detector_path"] == path]
            tested = sorted({row["setting"] for row in subset})
            highest = tested[-1]
            cases_here = sorted({row["case"] for row in subset})
            values = {(row["setting"], row["case"]): row["residual_m_on_common_grid"]
                      for row in subset}
            for pos, setting in enumerate(tested):
                mean = float(np.mean([values[(setting, case)] for case in cases_here]))
                high_mean = float(np.mean([values[(highest, case)] for case in cases_here]))
                abs_high = abs(mean - high_mean) * 1e9
                rel_high = abs(mean / high_mean - 1) * 100 if high_mean else float("inf")
                previous = tested[pos - 1] if pos else None
                if previous is None:
                    abs_prev = rel_prev = None
                    all_case_prev = None
                else:
                    previous_mean = float(np.mean([values[(previous, case)] for case in cases_here]))
                    abs_prev = abs(mean - previous_mean) * 1e9
                    rel_prev = abs(mean / previous_mean - 1) * 100 if previous_mean else float("inf")
                    all_case_prev = True
                    for case in cases_here:
                        old = values[(previous, case)]; new = values[(setting, case)]
                        absolute_nm = abs(new - old) * 1e9
                        relative_pct = abs(new / old - 1) * 100 if old else float("inf")
                        passed = relative_pct <= tolerance_rel and absolute_nm <= tolerance_abs
                        all_case_prev = all_case_prev and passed
                        paired_rows.append({
                            "study": study, "detector_path": path,
                            "previous_setting": previous, "setting": setting, "case": case,
                            "previous_residual_m": old, "residual_m": new,
                            "absolute_change_nm": absolute_nm,
                            "relative_change_pct": relative_pct,
                            "passes_both_tolerances": bool(passed),
                        })
                summary_rows.append({
                    "study": study, "detector_path": path, "setting": setting,
                    "n_cases": len(cases_here), "mean_residual_m": mean,
                    "highest_tested_setting": highest,
                    "absolute_delta_vs_highest_nm": abs_high,
                    "relative_delta_vs_highest_pct": rel_high,
                    "passes_vs_highest": (None if setting == highest else bool(
                        rel_high <= tolerance_rel and abs_high <= tolerance_abs)),
                    "highest_self_comparison_is_evidence": False,
                    "previous_setting": previous,
                    "absolute_change_vs_previous_nm": abs_prev,
                    "relative_change_vs_previous_pct": rel_prev,
                    "all_paired_cases_pass_vs_previous": all_case_prev,
                })
    write_csv(out / "convergence_paired_successive.csv", paired_rows)
    write_csv(out / "convergence_summary.csv", summary_rows)

    def selected_result(study: str, selected: int, path: str = "production_adc") -> dict:
        row = next(item for item in summary_rows if item["study"] == study
                   and item["setting"] == selected and item["detector_path"] == path)
        tail = [item for item in summary_rows if item["study"] == study
                and item["detector_path"] == path][-1]
        return {"selected_setting": selected, "selected_vs_highest": row,
                "last_successive_pair": tail,
                "demonstrably_converged": bool(row["passes_vs_highest"] and
                                                tail["all_paired_cases_pass_vs_previous"])}

    summary = {
        "criterion": {"relative_pct": tolerance_rel, "absolute_nm": tolerance_abs,
                      "rule": "both thresholds must pass; finest self-comparison is not evidence"},
        "common_scoring_grid": asdict(common_cfg),
        "same_physical_inputs": case_truth,
        "production_pupil_edge": selected_result("pupil_edge_supersampling", 4),
        "production_propagation_grid": selected_result(
            "propagation_grid_with_detector_integration", 1),
        "rows": summary_rows,
        "audit_corrections": [
            "one coefficient vector per case is propagated at every setting",
            "all residuals are evaluated on the same 256x256, 20x-edge-scored pupil",
            "refined detector intensities are block-summed before detector quantization",
            "each setting retains its own flat reference and push-pull interaction inverse",
        ],
    }
    json_dump(out / "convergence_summary.json", summary)
    return summary


def valid_tile_values(image: np.ndarray, valid_idx: np.ndarray) -> np.ndarray:
    data = np.asarray(image).reshape(C.FRAME_H, C.FRAME_W)
    tiles = data.reshape(C.NUM_LENSLETS, C.SUBAP_PIXELS,
                         C.NUM_LENSLETS, C.SUBAP_PIXELS).transpose(0, 2, 1, 3)
    return tiles.reshape(-1, C.SUBAP_PIXELS, C.SUBAP_PIXELS)[valid_idx]


def paired_detector_exposure(intensity: np.ndarray, cfg: D.DetectorConfig,
                             seed: int) -> dict:
    """One electron realization, digitized with and without upper ADC clipping."""
    optical = np.asarray(intensity, dtype=float)
    normalized = optical / optical.sum() if optical.sum() > 0 else optical
    expected_e = normalized * cfg.total_photons * cfg.quantum_efficiency + cfg.background_e
    rng = np.random.default_rng(seed)
    electrons = rng.poisson(np.clip(expected_e, 0, None)).astype(float)
    if cfg.read_noise_e > 0:
        electrons += rng.normal(0.0, cfg.read_noise_e, electrons.shape)
    saturated_mask = electrons > cfg.full_well_e
    electrons = np.clip(electrons, 0.0, cfg.full_well_e)
    unbounded = np.rint(electrons / cfg.gain_e_per_adu + cfg.offset_adu).astype(np.int64)
    clipped_mask = (unbounded > cfg.adc_max) | (unbounded < 0)
    production = np.clip(unbounded, 0, cfg.adc_max).astype(np.int64)
    return {
        "production_adu": production,
        "no_adc_clip_adu": unbounded,
        "expected_e": expected_e,
        "realized_e_after_full_well": electrons,
        "saturated_mask": saturated_mask,
        "clipped_mask": clipped_mask,
    }


def float_coefficients_wide(adu: np.ndarray, roms: dict[str, np.ndarray],
                            ideal_e: np.ndarray) -> np.ndarray:
    cx, cy = thresholded_centroids_float(adu)
    idx = roms["valid"]
    slopes = np.r_[cx[idx] - roms["ref_x"][idx] / Q18,
                   cy[idx] - roms["ref_y"][idx] / Q18]
    return ideal_e @ slopes


def simple_independent_stats(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    sem = sd / np.sqrt(len(x))
    return {"n_independent_wavefronts": int(len(x)), "mean": float(x.mean()),
            "sd": sd, "sem": float(sem),
            "mean_ci95_low": float(x.mean() - 1.96 * sem),
            "mean_ci95_high": float(x.mean() + 1.96 * sem),
            "interval_method": "normal 1.96*SEM across wavefront means"}


def run_clipping_ablation(cfg: dict, out: Path, roms: dict[str, np.ndarray],
                          ideal_e: np.ndarray, scale: float) -> dict:
    settings = cfg["clipping_ablation"]
    bench = O.Bench(O.OpticalConfig())
    rng = np.random.default_rng(cfg["seeds"]["clipping_wavefronts"])
    wavefronts = []
    for wavefront in range(settings["wavefronts"]):
        coeff = normalized_coefficients(
            bench, rng.normal(size=C.NUM_MODES), settings["input_rms_nm"])
        image, _ = bench.propagate_coeffs(coeff)
        wavefronts.append((coeff, image))
    idx = roms["valid"]
    rows: list[dict] = []
    for flux_index, photons in enumerate(settings["photons_per_pupil"]):
        detector_kwargs = {key: value for key, value in cfg["detector"].items()
                           if key in D.DetectorConfig.__dataclass_fields__}
        detector_kwargs.update(total_photons=photons, shot_noise=True)
        detector_cfg = D.DetectorConfig(**detector_kwargs)
        for wavefront, (truth, image) in enumerate(wavefronts):
            for repeat in range(settings["noise_repeats_per_wavefront"]):
                seed = (cfg["seeds"]["clipping_exposures"] + 10000 * flux_index +
                        100 * wavefront + repeat)
                paired = paired_detector_exposure(image, detector_cfg, seed)
                production = paired["production_adu"]
                diagnostic = paired["no_adc_clip_adu"]
                prod_float = float_coefficients_wide(production, roms, ideal_e)
                diag_float = float_coefficients_wide(diagnostic, roms, ideal_e)
                prod_fixed = frozen_pipeline(production, roms)["zernike_raw"] * scale
                clipped_tiles = valid_tile_values(paired["clipped_mask"], idx)
                expected_tiles = valid_tile_values(paired["expected_e"], idx)
                realized_tiles = valid_tile_values(
                    paired["realized_e_after_full_well"], idx)
                rows.append({
                    "photons_per_pupil": photons, "wavefront": wavefront,
                    "repeat": repeat, "exposure_seed": seed,
                    "production_float_residual_m": rms_coefficients(
                        bench, prod_float - truth),
                    "production_fixed_residual_m": rms_coefficients(
                        bench, prod_fixed - truth),
                    "no_adc_clip_diagnostic_float_residual_m": rms_coefficients(
                        bench, diag_float - truth),
                    "paired_no_clip_minus_production_wavefront_m": rms_coefficients(
                        bench, diag_float - prod_float),
                    "adc_clipped_pixels": int(np.count_nonzero(paired["clipped_mask"])),
                    "adc_clipped_valid_tile_pixels": int(np.count_nonzero(clipped_tiles)),
                    "all_pixel_denominator": C.FRAME_H * C.FRAME_W,
                    "valid_tile_pixel_denominator": C.NUM_SUBAPS_VALID * C.SUBAP_PIXELS ** 2,
                    "adc_clipped_fraction_all_pixels": float(np.mean(paired["clipped_mask"])),
                    "adc_clipped_fraction_valid_tile_pixels": float(np.mean(clipped_tiles)),
                    "thresholded_to_zero_production_pixels": int(np.count_nonzero(
                        production < C.PIXEL_THRESHOLD)),
                    "thresholded_to_zero_no_clip_pixels": int(np.count_nonzero(
                        diagnostic < C.PIXEL_THRESHOLD)),
                    "expected_photoelectrons_per_valid_lenslet": float(
                        expected_tiles.sum(axis=(1, 2)).mean()),
                    "realized_electrons_after_full_well_per_valid_lenslet": float(
                        realized_tiles.sum(axis=(1, 2)).mean()),
                    "nominal_qe_times_incident_photons_per_168": float(
                        photons * detector_cfg.quantum_efficiency / C.NUM_SUBAPS_VALID),
                    "saturated_pixels": int(np.count_nonzero(paired["saturated_mask"])),
                })
    write_csv(out / "clipping_ablation_raw.csv", rows)

    summary_rows: list[dict] = []
    metrics = ("production_float_residual_m", "production_fixed_residual_m",
               "no_adc_clip_diagnostic_float_residual_m",
               "paired_no_clip_minus_production_wavefront_m")
    for photons in settings["photons_per_pupil"]:
        subset = [row for row in rows if row["photons_per_pupil"] == photons]
        entry = {
            "photons_per_pupil": photons,
            "n_wavefronts": settings["wavefronts"],
            "noise_repeats_per_wavefront": settings["noise_repeats_per_wavefront"],
            "n_frames": len(subset),
            "statistical_unit": "wavefront mean over paired repeated noise exposures",
            "adc_clipped_pixels_total": int(sum(row["adc_clipped_pixels"] for row in subset)),
            "adc_clipped_pixel_denominator": int(len(subset) * C.FRAME_H * C.FRAME_W),
            "adc_clipped_fraction_all_pixels": float(
                sum(row["adc_clipped_pixels"] for row in subset) /
                (len(subset) * C.FRAME_H * C.FRAME_W)),
            "adc_clipped_valid_tile_pixels_total": int(sum(
                row["adc_clipped_valid_tile_pixels"] for row in subset)),
            "adc_clipped_valid_tile_pixel_denominator": int(
                len(subset) * C.NUM_SUBAPS_VALID * C.SUBAP_PIXELS ** 2),
            "expected_photoelectrons_per_valid_lenslet_mean": float(np.mean([
                row["expected_photoelectrons_per_valid_lenslet"] for row in subset])),
            "realized_electrons_after_full_well_per_valid_lenslet_mean": float(np.mean([
                row["realized_electrons_after_full_well_per_valid_lenslet"] for row in subset])),
            "nominal_qe_times_incident_photons_per_168": float(
                subset[0]["nominal_qe_times_incident_photons_per_168"]),
            "electron_metric_definition": (
                "expected metric sums normalized expected detected photoelectrons only over "
                "the 168 valid 16x16 tiles; nominal divides full-pupil QE*photons by 168"),
        }
        for metric in metrics:
            wavefront_means = []
            for wavefront in range(settings["wavefronts"]):
                wavefront_means.append(float(np.mean([
                    row[metric] for row in subset if row["wavefront"] == wavefront])))
            stats = simple_independent_stats(wavefront_means)
            entry.update({f"{metric}_{key}": value for key, value in stats.items()})
        summary_rows.append(entry)
    write_csv(out / "clipping_ablation_summary.csv", summary_rows)
    summary = {
        "pairing": ("production and no-ADC-clipping diagnostic use identical optical input, "
                    "Poisson draw, read-noise draw, full-well clipping, gain, integer rounding, "
                    "and threshold; only the upper ADC clamp differs"),
        "fixed_path_scope": "production 8-bit path only; wide diagnostic is software floating-point",
        "rows": summary_rows,
    }
    json_dump(out / "clipping_ablation_summary.json", summary)
    return summary


def make_figures(out: Path, flat: dict, convergence: dict, clipping: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)
    arrays = np.load(out / "flat_stage_arrays.npz")
    valid = arrays["valid_idx"].astype(int)
    valid_mask = np.zeros(C.NUM_SUBAPS_GRID, dtype=bool)
    valid_mask[valid] = True
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    for ax, axis in zip(axes, ("x", "y")):
        values = arrays[f"{axis}_error_q18"].astype(float).copy()
        values[~valid_mask] = np.nan
        image = ax.imshow(values.reshape(16, 16), origin="lower", cmap="coolwarm",
                          vmin=-2100, vmax=2100)
        ax.set_title(f"{axis.upper()} fixed − exact centroid")
        ax.set_xlabel("lenslet column"); ax.set_ylabel("lenslet row")
        fig.colorbar(image, ax=ax, label="Q18 LSB")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"flat_centroid_error_map.{suffix}", dpi=180)
    plt.close(fig)

    coeff = np.asarray(flat["fixed_mac_coefficients_m"]) * 1e9
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    ax.bar(np.arange(1, C.NUM_MODES + 1), coeff, color="#3465a4")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(np.arange(1, C.NUM_MODES + 1), C.MODE_LABELS, rotation=35, ha="right")
    ax.set_ylabel("fixed flat coefficient (nm)")
    ax.set_title("All modes retained in the reported flat residual")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"flat_coefficients.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for ax, study, title in zip(
            axes,
            ("pupil_edge_supersampling", "propagation_grid_with_detector_integration"),
            ("Pupil-edge supersampling", "Propagation grid + detector integration")):
        for path, marker, label in (("production_adc", "o", "production ADC"),
                                    ("continuous_diagnostic", "s", "continuous diagnostic")):
            rows = [r for r in convergence["rows"] if r["study"] == study
                    and r["detector_path"] == path]
            ax.plot([r["setting"] for r in rows],
                    [r["mean_residual_m"] * 1e9 for r in rows], marker=marker,
                    label=label)
        ax.set_xlabel("setting"); ax.set_ylabel("paired mean residual (nm)")
        ax.set_title(title); ax.grid(alpha=0.25); ax.legend()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"convergence_audit.{suffix}", dpi=180)
    plt.close(fig)

    rows = clipping["rows"]
    photons = np.asarray([r["photons_per_pupil"] for r in rows])
    prod = np.asarray([r["production_float_residual_m_mean"] for r in rows]) * 1e9
    wide = np.asarray([r["no_adc_clip_diagnostic_float_residual_m_mean"] for r in rows]) * 1e9
    clipped = np.asarray([r["adc_clipped_fraction_all_pixels"] for r in rows]) * 100
    fig, left = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    left.plot(photons, prod, "o-", label="production ADC")
    left.plot(photons, wide, "s-", label="no-ADC-clip diagnostic")
    left.set_xscale("log"); left.set_xlabel("photons / pupil / frame")
    left.set_ylabel("float residual (nm)"); left.grid(alpha=0.25); left.legend(loc="upper left")
    right = left.twinx(); right.plot(photons, clipped, "^-", color="#a40000",
                                    label="clipped pixels")
    right.set_ylabel("ADC-clipped detector pixels (%)", color="#a40000")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"clipping_ablation.{suffix}", dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def write_report(out: Path, cfg: dict, flat: dict, convergence: dict,
                 clipping: dict) -> None:
    fixed_coeff_nm = np.asarray(flat["fixed_mac_coefficients_m"]) * 1e9
    noisy = flat["noisy_flat"]
    conv_rows = []
    for row in convergence["rows"]:
        if row["detector_path"] != "production_adc":
            continue
        prev = "—" if row["previous_setting"] is None else (
            f"{row['absolute_change_vs_previous_nm']:.3f} nm / "
            f"{row['relative_change_vs_previous_pct']:.1f}%")
        paired = "—" if row["all_paired_cases_pass_vs_previous"] is None else (
            "yes" if row["all_paired_cases_pass_vs_previous"] else "no")
        conv_rows.append([
            row["study"], str(row["setting"]), f"{row['mean_residual_m']*1e9:.3f}",
            prev, paired,
        ])
    clip_rows = []
    for row in clipping["rows"]:
        clip_rows.append([
            f"{row['photons_per_pupil']:.3g}",
            f"{row['expected_photoelectrons_per_valid_lenslet_mean']:.0f}",
            f"{row['nominal_qe_times_incident_photons_per_168']:.0f}",
            f"{row['production_float_residual_m_mean']*1e9:.2f}",
            f"{row['no_adc_clip_diagnostic_float_residual_m_mean']*1e9:.2f}",
            f"{100*row['adc_clipped_fraction_all_pixels']:.3f}% "
            f"({row['adc_clipped_pixels_total']}/{row['adc_clipped_pixel_denominator']})",
        ])
    selected_edge = convergence["production_pupil_edge"]
    selected_prop = convergence["production_propagation_grid"]
    text = f"""# Flat-bias, convergence, and clipping investigation

Run: `{cfg['run_id']}`  
Source run: `manuscript_v1_20260928_final` (read-only)  
Scope: HCIPy, Python fixed-point model, and Icarus RTL simulation only.

## Executive findings

The 7.615181 nm flat result is real for the frozen ROMs and includes all ten
reported modes.  It is created at the reciprocal/centroid multiply, before
reference subtraction: the reference stores the rounded exact quotient, while
RTL multiplies the moment by a rounded reciprocal.  The exact frozen flat frame
produces Tilt X = {fixed_coeff_nm[0]:+.7f} nm, Tilt Y = {fixed_coeff_nm[1]:+.7f}
nm, Coma X = {fixed_coeff_nm[5]:+.7f} nm, and Coma Y =
{fixed_coeff_nm[6]:+.7f} nm.  No tip/tilt or fitted bias was removed.

The exact frame agrees with RTL at every checked centroid, slope, and output
coefficient: {flat['rtl']['coefficient_mismatch_count']} coefficient mismatches,
{flat['rtl']['slope_centroid_mismatch_count']} slope/centroid mismatches, and
{flat['rtl']['unknown_valid_values']} unknown valid values.  RTL residual:
{flat['rtl']['rtl_wavefront_rms_m']*1e9:.6f} nm.

The corrected convergence comparison uses identical coefficient vectors and a
single common scoring pupil.  The production edge setting is demonstrably
converged: **{'yes' if selected_edge['demonstrably_converged'] else 'no'}**.  The production
propagation setting is demonstrably converged: **{'yes' if selected_prop['demonstrably_converged'] else 'no'}**.
The highest setting is never credited for agreeing with itself.

## 1. Flat-wavefront stage trace

The stored Q18 reference equals `round(moment / sum × 2^18)` in every valid tile.
The RTL result is `moment × round(2^18 / sum)`, truncated to the existing
centroid width.  Fixed-minus-exact centroid error is
{flat['fixed_minus_exact_centroid_q18']['x_mean_lsb']:.2f} LSB mean and
{flat['fixed_minus_exact_centroid_q18']['x_rms_lsb']:.2f} LSB RMS in x (the y
values are identical to numerical precision).  Its correlation with reciprocal
rounding error is {flat['fixed_minus_exact_centroid_q18']['x_correlation_with_reciprocal_error']:.5f}.

Spatially, most fully illuminated interior lenslets have a +2040 LSB error,
while the obscuration, spiders, and pupil rim create coherent negative and
intermediate bands.  The y map is the transpose-symmetric counterpart of the x
map.  This is why the reconstruction projects strongly and equally onto x/y
tilt and, more weakly, x/y coma; it is not an independent constant added to the
ten reported coefficients.

The integer trace also confirms the actual widths and signs: unsigned 16-bit
intensity sums, 19-bit moments, a 19-bit Q18 reciprocal narrowed to the 18-bit
multiplier port, unsigned 24-bit Q18 centroids, signed 25-bit Q18 slopes, signed
18-bit Q1.16 matrix words, a signed 52-bit accumulator, and signed output slice
`[42:18]`.  The flat frame exercises no moment, accumulator, or output wrap.

`flat_stage_trace.csv` contains all 168 valid tiles: thresholded intensity sums,
moments, exact and LUT reciprocals, both centroid conventions, stored references,
and signed reference-subtracted slopes.  `flat_stage_arrays.npz` also preserves
the raw and thresholded 65,536-pixel frames.

The original floating matrix and dequantized matrix both reproduce the same
coherent tilt/coma pattern when fed the RTL centroids.  The final integer MAC
changes the dominant tilt coefficients by less than 0.00002 nm.  This rules out
matrix quantization and the reconstruction MAC as the source.

### Noise metrics (fresh seed, same 3e6-photon operating point)

| Path | Mean per-frame RMS | RMS of mean wavefront | RMS frame variability |
|---|---:|---:|---:|
| Primary float | {noisy['ideal_float']['mean_per_frame_wavefront_rms_m']*1e9:.3f} nm | {noisy['ideal_float']['rms_of_mean_reconstructed_wavefront_m']*1e9:.3f} nm | {noisy['ideal_float']['rms_frame_to_frame_wavefront_variability_m']*1e9:.3f} nm |
| Fixed | {noisy['fixed']['mean_per_frame_wavefront_rms_m']*1e9:.3f} nm | {noisy['fixed']['rms_of_mean_reconstructed_wavefront_m']*1e9:.3f} nm | {noisy['fixed']['rms_frame_to_frame_wavefront_variability_m']*1e9:.3f} nm |

Noise changes tile sums and therefore changes the reciprocal rounding pattern;
it does not constitute subtraction of the deterministic flat.  Mean per-frame
RMS, RMS of the mean reconstruction, and frame variability are reported
separately.

### Why the earlier report said approximately zero

The historical `docs/OPTICAL_EVALUATION.md` headline used the exact-floating
flat value (0.000033 nm) but called it generic “flat-wavefront bias.”  Its own
archived `evaluation/results/optical/campaign_summary.json` records the fixed
value as 7.615181 nm.  This investigation resolves a labeling ambiguity; it does
not revise the archived arithmetic result.

## 2. Numerical convergence audit

The prior comparison had two confounders: it normalized each common random
direction separately on each setting's pupil and scored on that changing pupil.
Both are removed here.  Refined detector intensity is block-summed over the
physical production pixels before normalization, threshold, and ADC rounding.
Detector outer edges, block centers, and lenslet registration are recorded in
`convergence_physical_checks.csv`; each setting retains its own flat reference
and ±{cfg['convergence']['probe_nm']:g} nm interaction calibration.

{markdown_table(['Study', 'Setting', 'Mean residual (nm)', 'Change vs previous',
                 'Every paired case passes 5% and 0.5 nm'], conv_rows)}

The production pupil uses HCIPy's mean of binary aperture sub-samples directly
as electric-field amplitude.  Fractional edge pixels therefore represent an
amplitude average (their propagated intensity contribution is its square), not
an area/intensity fill fraction.  Thin spiders and curved boundaries move among
these fractional samples as supersampling changes; their transmission sums and
squared sums are preserved per setting in the physical-check table.

The continuous-detector diagnostic in `convergence_summary.csv` separates
optical/grid dependence from threshold/ADC stair-steps.  Interaction matrices,
singular values, condition numbers, flat spots, and integrated flux are all
preserved.  Claims remain conditional whenever the selected production row or
the final successive pair fails the frozen criterion.

The mechanism is identifiable rather than merely inferred from an irregular
aggregate.  In the continuous diagnostic, propagation residual decreases at
every factor from 1 through 5, and every representative edge case passes the
frozen criterion from 16× to 20×.  The production-integer results keep large,
case-dependent jumps even though these frames have no ADC clipping.  Therefore
the propagation parity-looking sequence is an ADC-rounding/5-ADU-threshold
staircase in the per-setting calibration and test centroids.  The edge study
also has a genuine aperture-boundary/spider sampling dependence below 12×;
integer detector quantization prevents that continuous tail stability from
carrying over to production.  Matrix condition numbers remain of order six, so
ill-conditioning is not the explanation.

## 3. ADC-clipping ablation

Production and diagnostic paths use the same optical input, Poisson sample,
read-noise sample, full-well clipping, gain, integer rounding, and 5 ADU
threshold.  The diagnostic removes only the upper ADC clamp and is not an RTL
input path.

{markdown_table(['Photons/pupil', 'Expected e⁻/valid lenslet', 'Nominal QE·photons/168',
                 'Production float (nm)', 'No-ADC-clip diagnostic (nm)',
                 'Clipped pixels'], clip_rows)}

“Expected e⁻/valid lenslet” is measured by summing normalized expected detected
photoelectrons inside the 168 valid tiles.  The nominal allocation is shown
separately and is not described as collected valid-lenslet flux.  Clipping
fractions use all detector pixels as the stated denominator; valid-tile counts
and denominators are also in the raw and summary tables.

At 8e6 photons, removing only the ADC clamp reduces the paired floating residual
from 21.77 nm to 6.51 nm, directly establishing clipping as the dominant tested
cause there.  At 12e6 photons the diagnostic still leaves 15.53 nm, despite no
full-well saturation.  ADC clipping is therefore a major contributor, not the
sole high-flux mechanism.  The remainder includes non-clipping signal-level
effects (the fixed threshold/reference and optical/calibration nonlinearity);
this one-factor ablation does not partition those remaining causes further.

## 4. Corrections and historical validity

- No arithmetic width, threshold, gain, active ROM, or RTL architecture was
  changed.  The 7.615181 nm result is retained with tilt included.
- `regen_reference.py` now explicitly distinguishes exact-quotient and RTL
  reciprocal reference conventions.  Its historical/default exact behavior is
  unchanged, and no new reference was applied.
- Historical optical residual tables and the 35-case RTL equivalence counts
  remain numerically valid because their code, frames, and ROMs were not changed.
- The old generic “zero flat bias” wording is superseded: zero applied only to
  the exact-floating path.
- The old convergence table is superseded by this paired/common-grid audit.
- The old noise-table electron column was a nominal allocation.  Its residuals
  remain valid, but the column must not be cited as measured valid-tile flux.

## 5. Claim-to-evidence

| Claim | Status | Evidence |
|---|---|---|
| Frozen RTL creates the 7.615181 nm flat residual at reciprocal/centroid arithmetic | Established for exact frozen frame | stage CSV/NPZ, ablations, exact RTL trace |
| RTL equals the bit-accurate model | Established for finite tested cases | old 35 cases plus this exact flat; not formal proof |
| Production optical grid is numerically converged | {'Established' if selected_edge['demonstrably_converged'] and selected_prop['demonstrably_converged'] else 'Not established'} | paired common-grid convergence tables |
| ADC clipping is a major, but not sole, high-flux contributor in this detector model | Established for paired simulated cases | clipping raw/summary tables; not a camera measurement |
| Physical-bench equivalence, FPGA timing, power, or resources | Untested | outside this simulation-only scope |

## 6. Reproduction

From the repository root:

```bash
sim/venv/bin/python -m unittest evaluation.tests.test_rom_loading evaluation.tests.test_reference_conventions -v
sim/venv/bin/python evaluation/optical/blocker_investigation.py \\
  --config evaluation/configs/blocker_investigation_v1.json \\
  --out evaluation/results/runs/{cfg['run_id']}_reproduction
```

The output directory must not already exist.  `MANIFEST.json`, the compressed
tracked-working-tree patch, relevant untracked source copies, frozen source-run
calibration/ROMs, dependency versions, commands, and SHA-256 hashes provide the
reproduction snapshot.
"""
    (out / "REPORT.md").write_text(text)
    write_csv(out / "claim_to_evidence.csv", [
        {
            "claim": "Frozen RTL creates the 7.615181 nm flat residual at reciprocal/centroid arithmetic",
            "status": "ESTABLISHED FOR EXACT FROZEN FRAME",
            "evidence": "flat_stage_trace.csv; flat_coefficient_ablations.csv; flat_rtl_summary.json",
        },
        {
            "claim": "RTL equals the bit-accurate model",
            "status": "ESTABLISHED FOR FINITE TEST SET",
            "evidence": "35 historical cases plus exact flat replay; not formal proof",
        },
        {
            "claim": "Production optical grid is numerically converged",
            "status": ("ESTABLISHED" if selected_edge["demonstrably_converged"] and
                       selected_prop["demonstrably_converged"] else "NOT ESTABLISHED"),
            "evidence": "convergence_raw.csv; convergence_paired_successive.csv; convergence_physical_checks.csv",
        },
        {
            "claim": "ADC clipping is a major but not sole high-flux contributor in this detector model",
            "status": "ESTABLISHED FOR PAIRED SIMULATED CASES",
            "evidence": "clipping_ablation_raw.csv; clipping_ablation_summary.csv",
        },
        {
            "claim": "Physical-bench equivalence, FPGA timing, power, or resources",
            "status": "UNTESTED",
            "evidence": "outside simulation-only scope",
        },
    ])


def freeze_provenance(cfg_path: Path, source_run: Path, out: Path,
                      command: list[str], initial_status: list[str],
                      tracked_patch: bytes) -> dict:
    frozen = out / "frozen_inputs"
    source_snapshot = frozen / "source_snapshot"
    source_snapshot.mkdir(parents=True)
    shutil.copy2(cfg_path, frozen / "config.json")
    shutil.copytree(source_run / "frozen_inputs", frozen / "source_run_calibration")
    relevant = [
        Path(__file__).resolve(),
        C.EVAL_DIR / "common" / "bitaccurate.py",
        C.EVAL_DIR / "common" / "config.py",
        C.EVAL_DIR / "common" / "detector.py",
        C.EVAL_DIR / "common" / "optics.py",
        C.EVAL_DIR / "common" / "roms.py",
        C.EVAL_DIR / "common" / "rtl_runner.py",
        C.EVAL_DIR / "optical" / "manuscript_evaluation.py",
        C.EVAL_DIR / "optical" / "regen_reference.py",
        C.EVAL_DIR / "tests" / "test_reference_conventions.py",
        C.EVAL_DIR / "tests" / "test_rom_loading.py",
        cfg_path.resolve(),
    ]
    relevant += [C.RTL_DIR / name for name in R.RTL_SOURCES]
    relevant += [C.SIM_DIR / name for name in R.TB_SOURCES]
    for source in relevant:
        relative = source.relative_to(C.REPO_ROOT)
        destination = source_snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    with gzip.open(frozen / "tracked_working_tree.patch.gz", "wb", compresslevel=9) as handle:
        handle.write(tracked_patch)
    (frozen / "git_status_porcelain.txt").write_text("\n".join(initial_status) + "\n")

    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT, capture_output=True,
        text=True, check=True).stdout.strip()
    packages = {name: importlib.metadata.version(name)
                for name in ("hcipy", "numpy", "scipy", "matplotlib")}
    iverilog = subprocess.run(["iverilog", "-V"], capture_output=True, text=True)
    frozen_files = sorted(path for path in frozen.rglob("*") if path.is_file())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git_revision": git_revision,
        "git_dirty_before_run": bool(initial_status),
        "git_status_porcelain_before_run": initial_status,
        "tracked_patch": {
            "path": "frozen_inputs/tracked_working_tree.patch.gz",
            "uncompressed_bytes": len(tracked_patch),
            "sha256_uncompressed": sha256_bytes(tracked_patch),
            "note": "complete `git diff --binary`; untracked executed sources are copied separately",
        },
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": packages,
        "iverilog": (iverilog.stdout + iverilog.stderr).splitlines()[0],
        "source_run": str(source_run.relative_to(C.REPO_ROOT)),
        "source_run_manifest_sha256": sha256(source_run / "MANIFEST.json"),
        "source_snapshot_paths": [str(path.relative_to(C.REPO_ROOT)) for path in relevant],
        "frozen_file_sha256": {
            str(path.relative_to(out)): sha256(path) for path in frozen_files
        },
        "rom_policy": "all flat/RTL results use the source run's frozen ROM directory",
        "rtl_invocation": {
            "compiler": "iverilog -g2012",
            "testbench_top": "tb_full_pipeline",
            "sources": R.RTL_SOURCES,
            "testbench_sources": R.TB_SOURCES,
            "parameter_final_tile_pixel": R.final_tile_pixel_index(),
        },
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
    source_run = (C.REPO_ROOT / cfg["source_run"]).resolve()
    if not (source_run / "MANIFEST.json").is_file():
        raise FileNotFoundError(f"source run is incomplete: {source_run}")
    out = (args.out or C.RESULTS_DIR / "runs" / cfg["run_id"]).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out}")

    initial_status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=C.REPO_ROOT,
        capture_output=True, text=True, check=True).stdout.splitlines()
    tracked_patch = subprocess.run(
        ["git", "diff", "--binary"], cwd=C.REPO_ROOT,
        capture_output=True, check=True).stdout
    out.mkdir(parents=True)

    rom_dir = source_run / "frozen_inputs" / "roms"
    roms = load_frozen_roms(rom_dir)
    ideal_e = np.load(source_run / "frozen_inputs" / "E_measured_m_per_slopepx.npy")
    scale = float(json.loads(
        (source_run / "frozen_inputs" / "derived_scale.json").read_text()
    )["scale_m_per_lsb"])
    bench = O.Bench(O.OpticalConfig())

    print("[1/5] flat stage trace and exact RTL replay", flush=True)
    flat = run_flat_investigation(cfg, source_run, out, roms, ideal_e, scale, bench)
    print("[2/5] paired/common-grid convergence audit", flush=True)
    convergence = run_convergence_investigation(cfg, out)
    print("[3/5] paired ADC-clipping ablation", flush=True)
    clipping = run_clipping_ablation(cfg, out, roms, ideal_e, scale)
    combined = {"flat": flat, "convergence": convergence, "clipping": clipping,
                "historical_results_changed": False,
                "historical_results_superseded": [
                    "generic zero-flat wording", "old convergence qualification",
                    "nominal electron allocation label"],
                "active_roms_changed": False}
    json_dump(out / "summary.json", combined)
    print("[4/5] figures and report", flush=True)
    make_figures(out, flat, convergence, clipping)
    write_report(out, cfg, flat, convergence, clipping)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(cfg_path),
               "--out", str(out)]
    print("[5/5] exact source/configuration/calibration snapshot", flush=True)
    freeze_provenance(cfg_path, source_run, out, command, initial_status, tracked_patch)
    print(f"complete: {out}", flush=True)


if __name__ == "__main__":
    main()
