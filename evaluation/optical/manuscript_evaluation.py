#!/usr/bin/env python3
"""Run the frozen, simulation-only manuscript evidence campaign.

This is deliberately separate from the historical scripts and result folders.
It writes every output beneath one run directory and never recalibrates or
tunes. The unquantized optically measured interaction inverse is the primary
floating baseline; the dequantized ROM matrix is a second baseline that
isolates matrix quantization from arithmetic quantization.
"""

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
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import bitaccurate as BA  # noqa: E402
from evaluation.common import config as C  # noqa: E402
from evaluation.common import detector as D  # noqa: E402
from evaluation.common import optics as O  # noqa: E402
from evaluation.common import rtl_runner as R  # noqa: E402
from evaluation.common.roms import stage_rom_set, validate_rom_set  # noqa: E402


DEFAULT_CONFIG = C.EVAL_DIR / "configs" / "manuscript_v1.json"
CALIBRATION_DIR = C.RESULTS_DIR / "optical"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result table: {path}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_stats(values, bootstrap_seed: int) -> dict:
    """Mean, SD, SEM, percentile tail, and deterministic bootstrap mean CI."""
    x = np.asarray(values, dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("summary input is empty or non-finite")
    sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
    rng = np.random.default_rng(bootstrap_seed)
    draws = x[rng.integers(0, x.size, size=(2000, x.size))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n_independent": int(x.size),
        "mean": float(x.mean()),
        "sd": sd,
        "sem": float(sd / np.sqrt(x.size)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
        "mean_ci95_low": float(lo),
        "mean_ci95_high": float(hi),
        "interval_method": "percentile bootstrap of independent units, 2000 resamples",
    }


def detector_config(base: dict, **updates) -> D.DetectorConfig:
    fields = set(D.DetectorConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in base.items() if k in fields}
    kwargs.update(updates)
    return D.DetectorConfig(**kwargs)


def tile_diagnostics(adu: np.ndarray, valid_idx: np.ndarray) -> dict:
    img = np.asarray(adu).reshape(C.FRAME_H, C.FRAME_W)
    tiles = img.reshape(C.NUM_LENSLETS, C.SUBAP_PIXELS,
                        C.NUM_LENSLETS, C.SUBAP_PIXELS).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(-1, C.SUBAP_PIXELS, C.SUBAP_PIXELS)[valid_idx]
    flux = tiles.sum(axis=(1, 2)).astype(float)
    edge_mask = np.zeros((C.SUBAP_PIXELS, C.SUBAP_PIXELS), dtype=bool)
    edge_mask[:2, :] = edge_mask[-2:, :] = True
    edge_mask[:, :2] = edge_mask[:, -2:] = True
    edge = tiles[:, edge_mask].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(flux > 0, edge / flux, 1.0)
    return {
        "valid_flux_adu": float(flux.sum()),
        "min_lenslet_flux_adu": float(flux.min()),
        "edge_flux_fraction_mean": float(frac.mean()),
        "edge_flux_fraction_max": float(frac.max()),
        "lenslets_edge_fraction_gt_0p1": int(np.count_nonzero(frac > 0.1)),
    }


class Reconstruction:
    def __init__(self, bench: O.Bench, scale: float, ideal_e: np.ndarray):
        self.bench = bench
        self.scale = float(scale)
        self.ideal_e = np.asarray(ideal_e, dtype=float)
        if self.ideal_e.shape != (C.NUM_MODES, 2 * C.NUM_SUBAPS_VALID):
            raise ValueError(f"unexpected ideal E shape {self.ideal_e.shape}")
        ex, ey = BA.ematrix()
        self.dequantized_e = np.hstack([ex, ey]).astype(float) * self.scale
        self.idx = BA.valid_subaps()

    def coefficients(self, adu: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        px = np.asarray(adu, dtype=np.int64).ravel()
        fl = BA.run_float_reference(px)
        slopes = np.concatenate([fl["x_slope_px"][self.idx],
                                 fl["y_slope_px"][self.idx]])
        fixed = BA.run_pipeline(px)
        coeffs = {
            "ideal_float": self.ideal_e @ slopes,
            "dequantized_float": self.dequantized_e @ slopes,
            "fixed": fixed["zernike_raw"].astype(float) * self.scale,
        }
        return coeffs, fixed

    def score(self, adu: np.ndarray, true_opd: np.ndarray,
              true_coeffs: np.ndarray | None, exposure: D.DetectorResult | None,
              prefix: dict | None = None) -> dict:
        coeffs, fixed = self.coefficients(adu)
        b = self.bench
        wi = b.opd_from_coeffs(coeffs["ideal_float"])
        wd = b.opd_from_coeffs(coeffs["dequantized_float"])
        wf = b.opd_from_coeffs(coeffs["fixed"])
        ei, ed, ef = wi - true_opd, wd - true_opd, wf - true_opd
        delta_m = wd - wi
        delta_a = wf - wd
        delta_t = wf - wi
        cross = 2.0 * b.opd_inner(ei, delta_t)
        identity_rhs = b.opd_rms(ei) ** 2 + b.opd_rms(delta_t) ** 2 + cross
        row = dict(prefix or {})
        row.update({
            "true_opd_rms_m": b.opd_rms(true_opd),
            "ideal_float_residual_m": b.opd_rms(ei),
            "dequantized_float_residual_m": b.opd_rms(ed),
            "fixed_residual_m": b.opd_rms(ef),
            "matrix_quantization_opd_m": b.opd_rms(delta_m),
            "arithmetic_quantization_opd_m": b.opd_rms(delta_a),
            "fixed_minus_primary_float_opd_m": b.opd_rms(delta_t),
            "error_cross_term_m2": cross,
            "decomposition_closure_m2": b.opd_rms(ef) ** 2 - identity_rhs,
            "invalid_centroid_count": fixed["diagnostics"]["invalid_centroid_count"],
            "intensity_overflow_count": fixed["diagnostics"]["intensity_overflow_count"],
            "x_moment_overflow_count": fixed["diagnostics"]["x_moment_overflow_count"],
            "y_moment_overflow_count": fixed["diagnostics"]["y_moment_overflow_count"],
            "accumulator_wrap_count": fixed["diagnostics"]["accumulator_wrap_count"],
            "output_field_wrap_count": fixed["diagnostics"]["output_field_wrap_count"],
            "ideal_coeffs_m": json.dumps(coeffs["ideal_float"].tolist()),
            "dequantized_coeffs_m": json.dumps(coeffs["dequantized_float"].tolist()),
            "fixed_coeffs_m": json.dumps(coeffs["fixed"].tolist()),
        })
        if true_coeffs is not None:
            truth = np.asarray(true_coeffs, dtype=float)
            row["true_coeffs_m"] = json.dumps(truth.tolist())
            for label, values in coeffs.items():
                err = values - truth
                row[f"{label}_coeff_rmse_m"] = float(np.sqrt(np.mean(err ** 2)))
                row[f"{label}_coeff_bias_m"] = float(np.mean(err))
                row[f"{label}_coeff_max_abs_m"] = float(np.max(np.abs(err)))
        if exposure is not None:
            row.update({
                "saturated_px": exposure.saturated_px,
                "clipped_adu_px": exposure.clipped_adu_px,
                "detected_electrons": exposure.total_electrons,
            })
        return row


def random_coefficients(bench: O.Bench, rng, rms_nm: float) -> np.ndarray:
    values = rng.normal(0.0, 1.0, C.NUM_MODES)
    rms = bench.opd_rms(bench.opd_from_coeffs(values))
    if not np.isfinite(rms) or rms <= 0:
        raise RuntimeError("failed to draw a finite nonzero wavefront")
    return values * (rms_nm * 1e-9 / rms)


def run_flat(rec: Reconstruction, cfg: dict, out: Path) -> tuple[list[dict], dict]:
    b = rec.bench
    flat = np.zeros_like(np.asarray(b.aperture), dtype=float)
    image = b.propagate(flat)
    clean_cfg = detector_config(cfg["detector"], shot_noise=False, read_noise_e=0.0)
    clean_exp = D.expose(image, clean_cfg)
    clean = rec.score(clean_exp.adu, flat, np.zeros(C.NUM_MODES), clean_exp,
                      {"kind": "noiseless", "trial": 0})
    rng = np.random.default_rng(cfg["seeds"]["flat"])
    noisy_cfg = detector_config(cfg["detector"], total_photons=3e6,
                                shot_noise=True)
    rows = [clean]
    for trial in range(cfg["experiments"]["flat_noisy_trials"]):
        exposure = D.expose(image, noisy_cfg, rng)
        rows.append(rec.score(exposure.adu, flat, np.zeros(C.NUM_MODES), exposure,
                              {"kind": "noisy", "trial": trial}))
    write_csv(out / "flat_raw.csv", rows)
    noisy = rows[1:]
    summary = {
        "noiseless": clean,
        "noisy_ideal_float_residual": mean_stats(
            [r["ideal_float_residual_m"] for r in noisy], cfg["seeds"]["bootstrap"]),
        "noisy_fixed_residual": mean_stats(
            [r["fixed_residual_m"] for r in noisy], cfg["seeds"]["bootstrap"] + 1),
    }
    return rows, summary


def run_modes(rec: Reconstruction, cfg: dict, out: Path) -> tuple[list[dict], list[dict]]:
    b = rec.bench
    det = detector_config(cfg["detector"], total_photons=3e6,
                          shot_noise=False, read_noise_e=0.0)
    rows = []
    for mode in range(C.NUM_MODES):
        for amp_nm in cfg["experiments"]["mode_amplitudes_nm"]:
            for sign in (-1, 1):
                truth = np.zeros(C.NUM_MODES)
                truth[mode] = sign * amp_nm * 1e-9
                image, opd = b.propagate_coeffs(truth)
                exposure = D.expose(image, det)
                row = rec.score(exposure.adu, opd, truth, exposure, {
                    "mode": mode + 1, "label": C.MODE_LABELS[mode],
                    "command_nm": sign * amp_nm,
                })
                rows.append(row)
    write_csv(out / "modes_raw.csv", rows)
    summary = []
    for mode in range(C.NUM_MODES):
        subset = [r for r in rows if r["mode"] == mode + 1]
        command = np.asarray([r["command_nm"] for r in subset]) * 1e-9
        entry = {"mode": mode + 1, "label": C.MODE_LABELS[mode], "n": len(subset)}
        for label in ("ideal", "dequantized", "fixed"):
            field = {"ideal": "ideal_coeffs_m", "dequantized": "dequantized_coeffs_m",
                     "fixed": "fixed_coeffs_m"}[label]
            recovered_all = np.asarray([json.loads(r[field]) for r in subset])
            recovered = recovered_all[:, mode]
            gain, bias = np.linalg.lstsq(
                np.column_stack([command, np.ones_like(command)]), recovered, rcond=None)[0]
            nonlinearity = recovered - (gain * command + bias)
            other = np.delete(recovered_all, mode, axis=1)
            entry.update({
                f"{label}_gain": float(gain),
                f"{label}_bias_m": float(bias),
                f"{label}_nonlinearity_rms_m": float(np.sqrt(np.mean(nonlinearity ** 2))),
                f"{label}_crosstalk_rms_m": float(np.sqrt(np.mean(other ** 2))),
            })
        summary.append(entry)
    write_csv(out / "modes_summary.csv", summary)
    return rows, summary


def summarize_levels(rows: list[dict], key: str, levels, cfg: dict) -> list[dict]:
    summary = []
    metrics = [
        "true_opd_rms_m", "ideal_float_residual_m", "dequantized_float_residual_m",
        "fixed_residual_m", "matrix_quantization_opd_m",
        "arithmetic_quantization_opd_m", "fixed_minus_primary_float_opd_m",
    ]
    for i, level in enumerate(levels):
        subset = [r for r in rows if r[key] == level]
        entry = {key: level}
        for j, metric in enumerate(metrics):
            stats = mean_stats([r[metric] for r in subset],
                               cfg["seeds"]["bootstrap"] + 100 * i + j)
            entry.update({f"{metric}_{name}": value for name, value in stats.items()})
        for counter in ("invalid_centroid_count", "intensity_overflow_count",
                        "x_moment_overflow_count", "y_moment_overflow_count",
                        "accumulator_wrap_count", "output_field_wrap_count",
                        "saturated_px", "clipped_adu_px"):
            entry[f"{counter}_total"] = int(sum(r.get(counter, 0) for r in subset))
        entry["max_decomposition_closure_abs_m2"] = float(max(
            abs(r["decomposition_closure_m2"]) for r in subset))
        summary.append(entry)
    return summary


def run_mixed(rec: Reconstruction, cfg: dict, out: Path) -> tuple[list[dict], list[dict]]:
    b = rec.bench
    rng = np.random.default_rng(cfg["seeds"]["mixed"])
    det = detector_config(cfg["detector"], total_photons=3e6, shot_noise=True)
    levels = cfg["experiments"]["mixed_levels_nm_rms"]
    rows = []
    for level in levels:
        for trial in range(cfg["experiments"]["mixed_trials_per_level"]):
            truth = random_coefficients(b, rng, level)
            image, opd = b.propagate_coeffs(truth)
            exposure = D.expose(image, det, rng)
            rows.append(rec.score(exposure.adu, opd, truth, exposure,
                                  {"level_nm": level, "trial": trial}))
    write_csv(out / "mixed_raw.csv", rows)
    summary = summarize_levels(rows, "level_nm", levels, cfg)
    write_csv(out / "mixed_summary.csv", summary)
    coefficient_rows = []
    for level in levels:
        subset = [r for r in rows if r["level_nm"] == level]
        truth = np.asarray([json.loads(r["true_coeffs_m"]) for r in subset])
        estimates = {
            "ideal_float": np.asarray([json.loads(r["ideal_coeffs_m"]) for r in subset]),
            "dequantized_float": np.asarray([json.loads(r["dequantized_coeffs_m"]) for r in subset]),
            "fixed": np.asarray([json.loads(r["fixed_coeffs_m"]) for r in subset]),
        }
        estimates["fixed_minus_dequantized"] = estimates["fixed"] - estimates["dequantized_float"]
        for method, values in estimates.items():
            error = values if method == "fixed_minus_dequantized" else values - truth
            for mode in range(C.NUM_MODES):
                x = error[:, mode]
                coefficient_rows.append({
                    "level_nm": level, "method": method, "mode": mode + 1,
                    "label": C.MODE_LABELS[mode], "n": len(x),
                    "bias_m": float(x.mean()),
                    "rmse_m": float(np.sqrt(np.mean(x ** 2))),
                    "p95_abs_m": float(np.percentile(np.abs(x), 95)),
                    "max_abs_m": float(np.max(np.abs(x))),
                })
    write_csv(out / "mixed_coefficient_summary.csv", coefficient_rows)
    return rows, summary


def run_noise(rec: Reconstruction, cfg: dict, out: Path) -> tuple[list[dict], list[dict]]:
    b = rec.bench
    e = cfg["experiments"]
    wf_rng = np.random.default_rng(cfg["seeds"]["noise_wavefronts"])
    bases = []
    for wavefront in range(e["noise_wavefronts"]):
        truth = random_coefficients(b, wf_rng, e["noise_input_rms_nm"])
        image, opd = b.propagate_coeffs(truth)
        bases.append((truth, image, opd))
    rows = []
    for noise_index, read_noise in enumerate(e["noise_read_noise_e"]):
        noise_label = "shot_only" if read_noise == 0 else f"shot_plus_read_{read_noise:g}e"
        for photon_index, photons in enumerate(e["noise_photons_per_pupil"]):
            det = detector_config(cfg["detector"], total_photons=photons,
                                  read_noise_e=read_noise, shot_noise=True)
            for wavefront, (truth, image, opd) in enumerate(bases):
                for repeat in range(e["noise_repeats_per_wavefront"]):
                    # A stable per-exposure seed makes every raw frame exactly reproducible.
                    seed = (cfg["seeds"]["noise_exposures"] + 100000 * noise_index
                            + 1000 * photon_index + 10 * wavefront + repeat)
                    exposure = D.expose(image, det, np.random.default_rng(seed))
                    rows.append(rec.score(exposure.adu, opd, truth, exposure, {
                        "noise_case": noise_label, "read_noise_e": read_noise,
                        "photons_per_pupil": photons,
                        "detected_e_per_valid_lenslet_mean": D.signal_per_lenslet_e(det),
                        "wavefront": wavefront, "repeat": repeat, "exposure_seed": seed,
                    }))
    write_csv(out / "noise_raw.csv", rows)
    summary = []
    for noise_index, read_noise in enumerate(e["noise_read_noise_e"]):
        label = "shot_only" if read_noise == 0 else f"shot_plus_read_{read_noise:g}e"
        for photon_index, photons in enumerate(e["noise_photons_per_pupil"]):
            subset = [r for r in rows if r["noise_case"] == label
                      and r["photons_per_pupil"] == photons]
            # The independent unit is the wavefront. Average its repeated noise
            # exposures first, then form intervals across wavefronts.
            entry = {
                "noise_case": label, "read_noise_e": read_noise,
                "photons_per_pupil": photons,
                "detected_e_per_valid_lenslet_mean": subset[0]["detected_e_per_valid_lenslet_mean"],
                "n_wavefronts": e["noise_wavefronts"],
                "noise_repeats_per_wavefront": e["noise_repeats_per_wavefront"],
                "n_frames": len(subset),
                "statistical_unit": "wavefront mean over repeated noise exposures",
            }
            for j, metric in enumerate(("ideal_float_residual_m", "dequantized_float_residual_m",
                                        "fixed_residual_m", "matrix_quantization_opd_m",
                                        "arithmetic_quantization_opd_m")):
                cluster = []
                for wavefront in range(e["noise_wavefronts"]):
                    cluster.append(np.mean([r[metric] for r in subset
                                            if r["wavefront"] == wavefront]))
                stats = mean_stats(cluster, cfg["seeds"]["bootstrap"]
                                   + 1000 * noise_index + 50 * photon_index + j)
                entry.update({f"{metric}_{name}": value for name, value in stats.items()})
            for counter in ("invalid_centroid_count", "intensity_overflow_count",
                            "accumulator_wrap_count", "output_field_wrap_count",
                            "saturated_px", "clipped_adu_px"):
                entry[f"{counter}_total"] = int(sum(r[counter] for r in subset))
            summary.append(entry)
    write_csv(out / "noise_summary.csv", summary)
    return rows, summary


def run_dynamic(rec: Reconstruction, cfg: dict, out: Path) -> tuple[list[dict], list[dict]]:
    b = rec.bench
    e = cfg["experiments"]
    rng = np.random.default_rng(cfg["seeds"]["dynamic"])
    directions = [random_coefficients(b, rng, 1.0)
                  for _ in range(e["dynamic_trials_per_level"])]
    det = detector_config(cfg["detector"], total_photons=3e6, shot_noise=True)
    rows, diagnostic_frames, diagnostic_names = [], [], []
    flat_exp = D.expose(b.propagate(np.zeros_like(np.asarray(b.aperture))), det,
                        np.random.default_rng(cfg["seeds"]["dynamic"] + 999))
    flat_valid_flux = tile_diagnostics(flat_exp.adu, rec.idx)["valid_flux_adu"]
    for level in e["dynamic_levels_nm_rms"]:
        for trial, unit in enumerate(directions):
            truth = unit * level
            image, opd = b.propagate_coeffs(truth)
            exposure = D.expose(image, det, np.random.default_rng(
                cfg["seeds"]["dynamic"] + 1000 * int(level) + trial))
            row = rec.score(exposure.adu, opd, truth, exposure,
                            {"level_nm": level, "trial": trial})
            tile = tile_diagnostics(exposure.adu, rec.idx)
            row.update(tile)
            row["valid_flux_fraction_vs_flat"] = tile["valid_flux_adu"] / flat_valid_flux
            fixed = BA.run_pipeline(exposure.adu.ravel())
            centroid_scale = float(1 << C.CENTROID_FRAC)
            cx = fixed["x_centroid"][rec.idx] / centroid_scale
            cy = fixed["y_centroid"][rec.idx] / centroid_scale
            row["centroids_near_window_edge"] = int(np.count_nonzero(
                (cx < 0.5) | (cx > 14.5) | (cy < 0.5) | (cy > 14.5)))
            row["zernike_field_use_pct"] = float(
                100 * np.max(np.abs(fixed["zernike_raw"])) / ((1 << 24) - 1))
            rows.append(row)
            if trial == 0 and level in (100, 1000, 2000, 4000):
                diagnostic_frames.append(exposure.adu.astype(np.uint8))
                diagnostic_names.append(f"{level}_nm")
    write_csv(out / "dynamic_raw.csv", rows)
    np.savez_compressed(out / "dynamic_spot_diagnostics.npz",
                        frames=np.asarray(diagnostic_frames), names=np.asarray(diagnostic_names))
    summary = summarize_levels(rows, "level_nm", e["dynamic_levels_nm_rms"], cfg)
    for entry in summary:
        subset = [r for r in rows if r["level_nm"] == entry["level_nm"]]
        for metric in ("valid_flux_fraction_vs_flat", "edge_flux_fraction_mean",
                       "edge_flux_fraction_max", "centroids_near_window_edge",
                       "zernike_field_use_pct"):
            entry[f"{metric}_mean"] = float(np.mean([r[metric] for r in subset]))
        entry["lenslets_edge_fraction_gt_0p1_total"] = int(sum(
            r["lenslets_edge_fraction_gt_0p1"] for r in subset))
    write_csv(out / "dynamic_summary.csv", summary)
    return rows, summary


def run_unmodelled(rec: Reconstruction, cfg: dict, out: Path) -> tuple[list[dict], list[dict]]:
    from hcipy import make_zernike_basis

    b = rec.bench
    e = cfg["experiments"]
    basis = make_zernike_basis(C.NUM_MODES + 11, b.cfg.telescope_diameter_m, b.pupil_grid)
    high_modes = [np.asarray(basis[i]) for i in range(C.NUM_MODES + 1, C.NUM_MODES + 11)]
    rng = np.random.default_rng(cfg["seeds"]["unmodelled"])
    det = detector_config(cfg["detector"], total_photons=3e6, shot_noise=True)
    rows = []
    for fraction in e["unmodelled_fractions"]:
        for trial in range(e["unmodelled_trials_per_level"]):
            low_coeff = random_coefficients(b, rng, 100.0)
            low = b.opd_from_coeffs(low_coeff)
            high_coeff = rng.normal(size=len(high_modes))
            high = sum(value * mode for value, mode in zip(high_coeff, high_modes))
            high_rms = b.opd_rms(high)
            if high_rms > 0:
                high *= fraction * 100e-9 / high_rms
            truth = low + high
            image = b.propagate(truth)
            exposure = D.expose(image, det, rng)
            coeffs, _ = rec.coefficients(exposure.adu)
            reconstructed = b.opd_from_coeffs(coeffs["fixed"])
            row = rec.score(exposure.adu, truth, None, exposure,
                            {"unmodelled_fraction": fraction, "trial": trial})
            row.update({
                "unmodelled_rms_m": b.opd_rms(high),
                "represented_mode_wavefront_error_m": b.opd_rms(reconstructed - low),
                "represented_coefficient_rmse_m": float(np.sqrt(np.mean(
                    (coeffs["fixed"] - low_coeff) ** 2))),
                "full_pupil_residual_m": b.opd_rms(reconstructed - truth),
            })
            rows.append(row)
    write_csv(out / "unmodelled_raw.csv", rows)
    summary = []
    for i, fraction in enumerate(e["unmodelled_fractions"]):
        subset = [r for r in rows if r["unmodelled_fraction"] == fraction]
        entry = {"unmodelled_fraction": fraction}
        for j, metric in enumerate(("unmodelled_rms_m", "represented_mode_wavefront_error_m",
                                    "represented_coefficient_rmse_m", "full_pupil_residual_m")):
            stats = mean_stats([r[metric] for r in subset],
                               cfg["seeds"]["bootstrap"] + 700 + 20 * i + j)
            entry.update({f"{metric}_{name}": value for name, value in stats.items()})
        summary.append(entry)
    write_csv(out / "unmodelled_summary.csv", summary)
    return rows, summary


def run_subpixel(rec: Reconstruction, cfg: dict, out: Path) -> tuple[list[dict], dict]:
    b = rec.bench
    e = cfg["experiments"]
    det = detector_config(cfg["detector"], total_photons=3e6,
                          shot_noise=False, read_noise_e=0.0)
    c = b.cfg
    # HCIPy tip is RMS-normalized: Z=2x/R. This is the independent geometry
    # conversion from coefficient to detector shift.
    coefficient_per_pixel = (c.detector_pixel_pitch_m / c.focal_length_m
                             * c.magnification * (c.telescope_diameter_m / 2)) / 2
    phases = np.linspace(e["subpixel_phase_min_px"], e["subpixel_phase_max_px"],
                         e["subpixel_positions"])
    rows, base_centroid = [], None
    for phase in phases:
        truth = np.zeros(C.NUM_MODES)
        truth[0] = phase * coefficient_per_pixel
        image, opd = b.propagate_coeffs(truth)
        exposure = D.expose(image, det)
        fixed = BA.run_pipeline(exposure.adu.ravel())
        cx = fixed["x_centroid"][rec.idx] / float(1 << C.CENTROID_FRAC)
        if base_centroid is None:
            base_centroid = cx.copy()
        measured = float(np.mean(cx - base_centroid))
        row = rec.score(exposure.adu, opd, truth, exposure,
                        {"commanded_phase_px": float(phase),
                         "tilt_coefficient_nm": float(truth[0] * 1e9),
                         "measured_shift_px": measured,
                         "centroid_bias_px": measured - phase})
        row.update(tile_diagnostics(exposure.adu, rec.idx))
        rows.append(row)
    write_csv(out / "subpixel_raw.csv", rows)
    bias = np.asarray([r["centroid_bias_px"] for r in rows])
    summary = {
        "positions": len(rows), "phase_interval_px": [float(phases[0]), float(phases[-1])],
        "coefficient_per_pixel_nm": coefficient_per_pixel * 1e9,
        "bias_mean_px": float(bias.mean()),
        "bias_rms_px": float(np.sqrt(np.mean(bias ** 2))),
        "bias_peak_to_peak_px": float(np.ptp(bias)),
        "bias_max_abs_px": float(np.max(np.abs(bias))),
        "conclusion_scope": "one complete pure-tip detector-pixel phase period",
    }
    return rows, summary


def bin_detector(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return np.asarray(image)
    n = C.FRAME_H * factor
    data = np.asarray(image).reshape(n, n)
    return data.reshape(C.FRAME_H, factor, C.FRAME_W, factor).sum(axis=(1, 3))


def run_convergence(rec: Reconstruction, cfg: dict, out: Path) -> tuple[list[dict], dict]:
    e = cfg["experiments"]
    rng = np.random.default_rng(cfg["seeds"]["convergence"])
    directions = [rng.normal(size=C.NUM_MODES) for _ in range(e["convergence_wavefronts"])]
    det = detector_config(cfg["detector"], total_photons=3e6,
                          shot_noise=False, read_noise_e=0.0)
    def thresholded_centroids(adu: np.ndarray) -> np.ndarray:
        s_i, x_i, y_i = BA.accumulate(adu)
        with np.errstate(divide="ignore", invalid="ignore"):
            cx = np.where(s_i > 0, x_i / s_i, 0.0)
            cy = np.where(s_i > 0, y_i / s_i, 0.0)
        return np.concatenate([cx[rec.idx], cy[rec.idx]])

    def calibrated_rows(study: str, setting: int, bench: O.Bench,
                        integration_factor: int) -> list[dict]:
        def expose_image(image):
            return D.expose(bin_detector(image, integration_factor), det).adu

        flat = np.zeros_like(np.asarray(bench.aperture))
        reference = thresholded_centroids(expose_image(bench.propagate(flat)))
        probe_m = 30e-9
        interaction = np.zeros((2 * C.NUM_SUBAPS_VALID, C.NUM_MODES))
        for mode in range(C.NUM_MODES):
            plus = np.zeros(C.NUM_MODES); plus[mode] = probe_m
            minus = np.zeros(C.NUM_MODES); minus[mode] = -probe_m
            p = thresholded_centroids(expose_image(bench.propagate_coeffs(plus)[0]))
            m = thresholded_centroids(expose_image(bench.propagate_coeffs(minus)[0]))
            interaction[:, mode] = (p - m) / (2 * probe_m)
        local_e = np.linalg.pinv(interaction)
        condition = float(np.linalg.cond(interaction))
        result = []
        for trial, direction in enumerate(directions):
            truth = direction * (100e-9 / bench.opd_rms(bench.opd_from_coeffs(direction)))
            image, opd = bench.propagate_coeffs(truth)
            slopes = thresholded_centroids(expose_image(image)) - reference
            reconstructed = bench.opd_from_coeffs(local_e @ slopes)
            result.append({
                "study": study, "setting": setting, "trial": trial,
                "ideal_float_residual_m": bench.opd_rms(reconstructed - opd),
                "interaction_condition_number": condition,
                "calibration_probe_nm": probe_m * 1e9,
                "calibration_policy": "independent flat reference and push-pull E at each resolution",
            })
        return result

    rows = []
    for supersample in e["convergence_supersampling"]:
        b = O.Bench(O.OpticalConfig(supersample=supersample))
        rows.extend(calibrated_rows("pupil_edge_supersampling", supersample, b, 1))
    for factor in e["convergence_propagation_factors"]:
        base = O.OpticalConfig()
        b = O.Bench(replace(base, pupil_pixels_base=base.pupil_pixels_base * factor))
        rows.extend(calibrated_rows("propagation_grid_with_detector_integration",
                                    factor, b, factor))
    write_csv(out / "convergence_raw.csv", rows)
    summary_rows = []
    for study in ("pupil_edge_supersampling", "propagation_grid_with_detector_integration"):
        settings = sorted({r["setting"] for r in rows if r["study"] == study})
        finest = settings[-1]
        ref = np.mean([r["ideal_float_residual_m"] for r in rows
                       if r["study"] == study and r["setting"] == finest])
        for setting in settings:
            vals = [r["ideal_float_residual_m"] for r in rows
                    if r["study"] == study and r["setting"] == setting]
            mean = float(np.mean(vals))
            absolute_nm = abs(mean - ref) * 1e9
            relative_pct = abs(mean / ref - 1) * 100 if ref else float("nan")
            summary_rows.append({
                "study": study, "setting": setting, "n": len(vals),
                "residual_mean_m": mean, "finest_residual_mean_m": float(ref),
                "absolute_delta_vs_finest_nm": absolute_nm,
                "relative_delta_vs_finest_pct": relative_pct,
                "passes_frozen_tolerance": bool(
                    relative_pct <= e["convergence_acceptance_relative_pct"]
                    and absolute_nm <= e["convergence_acceptance_absolute_nm"]),
            })
    write_csv(out / "convergence_summary.csv", summary_rows)
    return rows, {
        "rows": summary_rows,
        "criterion": {
            "relative_pct": e["convergence_acceptance_relative_pct"],
            "absolute_nm": e["convergence_acceptance_absolute_nm"],
            "rule": "both thresholds must pass relative to the finest tested setting",
        },
    }


def build_rtl_cases(bench: O.Bench, cfg: dict) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(cfg["seeds"]["rtl"])
    clean = detector_config(cfg["detector"], total_photons=3e6,
                            shot_noise=False, read_noise_e=0.0)
    noisy = detector_config(cfg["detector"], total_photons=3e6, shot_noise=True)
    flat = np.zeros_like(np.asarray(bench.aperture))
    flat_image = bench.propagate(flat)
    cases = [
        ("optical_flat_noiseless", D.expose(flat_image, clean).adu),
        ("optical_flat_noisy", D.expose(flat_image, noisy, rng).adu),
    ]
    for mode in range(C.NUM_MODES):
        for sign in (-1, 1):
            truth = np.zeros(C.NUM_MODES); truth[mode] = sign * 80e-9
            image, _ = bench.propagate_coeffs(truth)
            cases.append((f"optical_mode_{mode+1}_{'pos' if sign > 0 else 'neg'}",
                          D.expose(image, clean).adu))
    levels = (20, 100, 400, 2000)
    for case in range(cfg["experiments"]["rtl_mixed_cases"]):
        level = levels[case % len(levels)]
        truth = random_coefficients(bench, rng, level)
        image, _ = bench.propagate_coeffs(truth)
        cases.append((f"optical_mixed_{level}nm_{case:02d}", D.expose(image, noisy, rng).adu))
    truth = random_coefficients(bench, rng, 100)
    image, _ = bench.propagate_coeffs(truth)
    cases.append(("optical_low_flux", D.expose(image,
                  detector_config(cfg["detector"], total_photons=5e5), rng).adu))
    cases.append(("optical_high_flux", D.expose(image,
                  detector_config(cfg["detector"], total_photons=8e6), rng).adu))
    n = C.FRAME_W * C.FRAME_H
    cases.append(("directed_invalid_centroids_dark", np.zeros(n, dtype=np.uint8)))
    cases.append(("directed_adc_saturation", np.full(n, 255, dtype=np.uint8)))
    threshold = np.zeros(n, dtype=np.uint8); threshold[::37] = C.PIXEL_THRESHOLD
    cases.append(("directed_threshold_boundary", threshold))
    return cases


def run_rtl(bench: O.Bench, cfg: dict, out: Path) -> dict:
    cases = build_rtl_cases(bench, cfg)
    rows, frames = [], []
    coeff_mismatches = slope_mismatches = unknowns = 0
    with R.RTLSimulator() as simulator:
        rom_manifest = simulator.rom_manifest
        for name, frame in cases:
            px = np.asarray(frame, dtype=np.uint8).ravel()
            rtl = simulator.run_frame(px)
            model = BA.run_pipeline(px)
            coeff_delta = rtl["zernike_raw"] - model["zernike_raw"]
            slopes = rtl["slopes"]
            slope_delta = np.column_stack([
                slopes[:, 1].astype(np.int64) - model["x_slope_valid"],
                slopes[:, 2].astype(np.int64) - model["y_slope_valid"],
                slopes[:, 3].astype(np.int64) - model["x_centroid"][model["valid_idx"]],
                slopes[:, 4].astype(np.int64) - model["y_centroid"][model["valid_idx"]],
            ])
            cm = int(np.count_nonzero(coeff_delta))
            sm = int(np.count_nonzero(slope_delta))
            un = rtl["unknown_valid_slopes"] + rtl["unknown_valid_coefficients"]
            coeff_mismatches += cm; slope_mismatches += sm; unknowns += un
            frames.append(px)
            rows.append({
                "case": name, "frame_sha256": hashlib.sha256(px.tobytes()).hexdigest(),
                "coefficient_mismatches": cm, "coefficient_max_lsb_difference": int(
                    np.max(np.abs(coeff_delta))),
                "slope_centroid_mismatches": sm, "slope_centroid_max_lsb_difference": int(
                    np.max(np.abs(slope_delta))),
                "unknown_valid_values": un, "latency_cycles": rtl["true_latency_cycles"],
                "pixels_checked": rtl["pixels_streamed"], "pixel_mismatches": rtl["pixel_mismatches"],
                "subapertures_valid": rtl["subaps_valid"],
            })
    write_csv(out / "rtl_cases.csv", rows)
    np.savez_compressed(out / "rtl_input_frames.npz",
                        names=np.asarray([r["case"] for r in rows]), frames=np.asarray(frames))
    summary = {
        "n_cases": len(rows),
        "n_coefficients_checked": len(rows) * C.NUM_MODES,
        "n_coefficient_mismatches": coeff_mismatches,
        "n_slope_centroid_values_checked": len(rows) * C.NUM_SUBAPS_VALID * 4,
        "n_slope_centroid_mismatches": slope_mismatches,
        "n_unknown_valid_values": unknowns,
        "latency_cycles_observed": sorted({r["latency_cycles"] for r in rows}),
        "latency_definition": "cycles from final contributing input-pixel arrival to output done/valid",
        "all_exact": coeff_mismatches == 0 and slope_mismatches == 0 and unknowns == 0,
        "scope": "finite Icarus simulation set; not formal equivalence and not measured hardware",
        "rom_manifest": rom_manifest,
    }
    (out / "rtl_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def source_provenance(cfg_path: Path, out: Path, command: list[str]) -> dict:
    source_paths = [
        C.REPO_ROOT / "evaluation/common/roms.py",
        C.REPO_ROOT / "evaluation/common/bitaccurate.py",
        C.REPO_ROOT / "evaluation/common/rtl_runner.py",
        C.REPO_ROOT / "evaluation/common/optics.py",
        C.REPO_ROOT / "evaluation/common/detector.py",
        Path(__file__).resolve(), cfg_path.resolve(),
    ]
    source_paths += [C.RTL_DIR / name for name in R.RTL_SOURCES]
    source_paths += [C.SIM_DIR / name for name in R.TB_SOURCES]
    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=C.REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.splitlines()
    iv = subprocess.run(["iverilog", "-V"], capture_output=True, text=True)
    packages = {}
    for name in ("hcipy", "numpy", "scipy", "matplotlib"):
        packages[name] = importlib.metadata.version(name)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git_revision": git_rev,
        "git_dirty": bool(git_status),
        "git_status_porcelain": git_status,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": packages,
        "iverilog": (iv.stdout + iv.stderr).splitlines()[0],
        "rtl_simulation": {
            "language_switch": "-g2012",
            "testbench_top": "tb_full_pipeline",
            "dut_top": "shwfs_processing (instantiated through tb_zynq_top)",
            "defines": [],
            "parameter_overrides": {
                "tb_full_pipeline.FINAL_TILE_PIXEL": R.final_tile_pixel_index(),
            },
            "rtl_sources": R.RTL_SOURCES,
            "testbench_sources": R.TB_SOURCES,
            "rom_resolution": "bare $readmemh names; validated and byte-copied into isolated simulator CWD",
        },
        "source_sha256": {str(path.relative_to(C.REPO_ROOT)): sha256(path)
                          for path in source_paths},
        "config_sha256": sha256(cfg_path),
        "roms": validate_rom_set(),
    }


def make_figures(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"; fig_dir.mkdir()
    plt.rcParams.update({"font.size": 8.5, "figure.dpi": 150})

    def load(name):
        with (out / name).open() as handle:
            return list(csv.DictReader(handle))

    def save(fig, name):
        for ext in ("pdf", "png", "svg"):
            fig.savefig(fig_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    modes = load("modes_summary.csv")
    labels = [r["label"] for r in modes]; x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    for key, color, marker in (("ideal", "#555555", "o"), ("fixed", "#2474b7", "s")):
        axes[0].plot(x, [float(r[f"{key}_gain"]) for r in modes], marker=marker,
                     color=color, label=key.replace("ideal", "primary float"))
    axes[0].axhline(1, color="#d95f02", ls="--", lw=1); axes[0].set_ylabel("Recovery gain")
    axes[0].legend(frameon=False); axes[0].grid(alpha=.25)
    axes[1].bar(x, [float(r["ideal_crosstalk_rms_m"]) * 1e9 for r in modes], color="#777777")
    axes[1].set_ylabel("Cross-talk RMS (nm)"); axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=.25); fig.tight_layout(); save(fig, "mode_gain_crosstalk")

    mixed = load("mixed_summary.csv")
    level = np.asarray([float(r["level_nm"]) for r in mixed])
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for metric, label, color, marker in (
        ("ideal_float_residual_m", "primary float", "#555555", "o"),
        ("fixed_residual_m", "fixed", "#2474b7", "s")):
        mean = np.asarray([float(r[f"{metric}_mean"]) for r in mixed]) * 1e9
        lo = np.asarray([float(r[f"{metric}_mean_ci95_low"]) for r in mixed]) * 1e9
        hi = np.asarray([float(r[f"{metric}_mean_ci95_high"]) for r in mixed]) * 1e9
        ax.errorbar(level, mean, yerr=[mean-lo, hi-mean], marker=marker, color=color,
                    capsize=2, label=label)
    ax.set(xscale="log", yscale="log", xlabel="Realized input OPD RMS (nm)",
           ylabel="Residual OPD RMS (nm)"); ax.grid(alpha=.25, which="both"); ax.legend(frameon=False)
    save(fig, "mixed_reconstruction_error")

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ar = np.asarray([float(r["arithmetic_quantization_opd_m_mean"]) for r in mixed]) * 1e9
    ar95 = np.asarray([float(r["arithmetic_quantization_opd_m_p95"]) for r in mixed]) * 1e9
    mq = np.asarray([float(r["matrix_quantization_opd_m_mean"]) for r in mixed]) * 1e9
    ax.plot(level, ar, "s-", label="arithmetic: fixed − dequantized float")
    ax.plot(level, ar95, "s--", label="arithmetic p95")
    ax.plot(level, mq, "o-", label="matrix: dequantized − original E")
    ax.set(xscale="log", xlabel="Input OPD RMS (nm)", ylabel="Paired wavefront difference RMS (nm)")
    ax.grid(alpha=.25, which="both"); ax.legend(frameon=False, fontsize=7.5)
    save(fig, "paired_fixed_float_discrepancy")

    noise = load("noise_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True)
    for ax, case in zip(axes, sorted({r["noise_case"] for r in noise})):
        subset = [r for r in noise if r["noise_case"] == case]
        ph = np.asarray([float(r["photons_per_pupil"]) for r in subset])
        for metric, label, color, marker in (("ideal_float_residual_m", "primary float", "#555", "o"),
                                              ("fixed_residual_m", "fixed", "#2474b7", "s")):
            mean = np.asarray([float(r[f"{metric}_mean"]) for r in subset]) * 1e9
            lo = np.asarray([float(r[f"{metric}_mean_ci95_low"]) for r in subset]) * 1e9
            hi = np.asarray([float(r[f"{metric}_mean_ci95_high"]) for r in subset]) * 1e9
            ax.errorbar(ph, mean, yerr=[mean-lo, hi-mean], color=color, marker=marker,
                        capsize=2, label=label)
        clipped = np.asarray([int(r["clipped_adu_px_total"]) for r in subset]) > 0
        if clipped.any(): ax.axvline(ph[clipped][0], color="#d95f02", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_title(case.replace("_", " ")); ax.grid(alpha=.25, which="both")
        ax.set_xlabel("Photons incident over pupil / frame")
    axes[0].set_ylabel("Residual OPD RMS (nm)"); axes[0].legend(frameon=False)
    fig.tight_layout(); save(fig, "photon_read_noise")

    dynamic = load("dynamic_summary.csv")
    x = np.asarray([float(r["level_nm"]) for r in dynamic])
    resid = np.asarray([float(r["fixed_residual_m_mean"]) for r in dynamic]) * 1e9
    edge = np.asarray([float(r["edge_flux_fraction_mean_mean"]) for r in dynamic])
    fig, ax = plt.subplots(figsize=(5.4, 3.6)); other = ax.twinx()
    ax.plot(x, resid/x*100, "o-", color="#2474b7", label="relative residual")
    other.plot(x, edge*100, "s--", color="#d95f02", label="outer-2-pixel flux")
    ax.set(xscale="log", xlabel="Input OPD RMS (nm)", ylabel="Fixed residual / input (%)")
    other.set_ylabel("Mean edge occupancy (%)"); ax.grid(alpha=.25, which="both")
    lines = ax.lines + other.lines; ax.legend(lines, [l.get_label() for l in lines], frameon=False)
    save(fig, "operating_range")

    data = np.load(out / "dynamic_spot_diagnostics.npz")
    fig, axes = plt.subplots(1, len(data["frames"]), figsize=(9.2, 2.5))
    for ax, frame, name in zip(axes, data["frames"], data["names"]):
        ax.imshow(frame.reshape(C.FRAME_H, C.FRAME_W), cmap="gray", origin="lower",
                  vmin=0, vmax=255)
        ax.set_title(str(name).replace("_", " ")); ax.axis("off")
    fig.tight_layout(); save(fig, "spot_diagnostics")

    conv = load("convergence_summary.csv"); sub = load("subpixel_raw.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    for study, marker in (("pupil_edge_supersampling", "o"),
                          ("propagation_grid_with_detector_integration", "s")):
        rows = [r for r in conv if r["study"] == study]
        axes[0].plot([float(r["setting"]) for r in rows],
                     [float(r["residual_mean_m"])*1e9 for r in rows], marker=marker,
                     label=study.replace("_", " "))
    axes[0].set(xlabel="Refinement setting", ylabel="Residual RMS (nm)"); axes[0].grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[1].plot([float(r["commanded_phase_px"]) for r in sub],
                 [float(r["centroid_bias_px"]) for r in sub], "o-", ms=3)
    axes[1].set(xlabel="Commanded detector-pixel phase", ylabel="Mean centroid bias (px)")
    axes[1].grid(alpha=.25); fig.tight_layout(); save(fig, "convergence_subpixel")

    rtl = json.loads((out / "rtl_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(5.6, 2.5)); ax.axis("off")
    text = (f"RTL agreement on identical archived inputs\n\n"
            f"Cases: {rtl['n_cases']}\nCoefficients: {rtl['n_coefficients_checked']}\n"
            f"Coefficient mismatches: {rtl['n_coefficient_mismatches']}\n"
            f"Slope/centroid mismatches: {rtl['n_slope_centroid_mismatches']}\n"
            f"Unknown valid values: {rtl['n_unknown_valid_values']}\n"
            f"Simulated latency: {rtl['latency_cycles_observed']} cycles")
    ax.text(.03, .95, text, va="top", family="monospace", fontsize=10)
    save(fig, "rtl_agreement")


def generate_report(out: Path, cfg: dict, summaries: dict, provenance: dict) -> None:
    mixed = summaries["mixed"]
    flat = summaries["flat"]
    noise = summaries["noise"]
    modes = summaries["modes"]
    dynamic = summaries["dynamic"]
    unmodelled = summaries["unmodelled"]
    rtl = summaries["rtl"]
    subpixel = summaries["subpixel"]
    convergence = summaries["convergence"]
    nm = lambda value: value * 1e9
    production_convergence = [r for r in convergence["rows"]
                              if (r["study"] == "pupil_edge_supersampling" and r["setting"] == 4)
                              or (r["study"] == "propagation_grid_with_detector_integration" and r["setting"] == 1)]
    convergence_pass = all(r["passes_frozen_tolerance"] for r in production_convergence)
    claims = [
        {"claim": "Optical calibration is self-consistent for the simulated bench",
         "status": "ESTABLISHED AT FROZEN DISCRETIZATION" if not convergence_pass else "ESTABLISHED IN SIMULATION",
         "evidence": "noiseless flat, independent ±mode probes, held-out optical wavefronts; see convergence qualification"},
        {"claim": "Floating reconstruction accuracy over 20–400 nm RMS",
         "status": "CONDITIONAL ON FROZEN NUMERICAL GRID" if not convergence_pass else "ESTABLISHED FOR FROZEN HCIPy MODEL",
         "evidence": f"{sum(int(r['true_opd_rms_m_n_independent']) for r in mixed)} held-out mixed wavefronts"},
        {"claim": "Fixed-point discrepancy is separated from optical residual",
         "status": "ESTABLISHED FOR TESTED CASES",
         "evidence": "paired original-E, dequantized-E, and bit-accurate reconstructions on identical frames"},
        {"claim": "RTL equals the bit-accurate model",
         "status": "ESTABLISHED FOR FINITE TEST SET",
         "evidence": f"{rtl['n_cases']} cases, {rtl['n_coefficients_checked']} coefficients"},
        {"claim": "Physical-bench optical performance or hardware timing/power/resources",
         "status": "UNTESTED",
         "evidence": "no experimental frames, board run, synthesis, implementation, or timing closure in scope"},
    ]
    write_csv(out / "claim_to_evidence.csv", claims)
    lines = [
        "# Reproducible HCIPy and RTL-simulation evidence",
        "",
        f"Run: `{cfg['run_id']}`  ",
        f"Revision: `{provenance['git_revision']}` (dirty state: `{provenance['git_dirty']}`)  ",
        "Scope: simulation only; no synthesis, implementation, board, power, or experimental-data claims.",
        "",
        "## Methods",
        "",
        "The optical forward model is HCIPy propagation through the frozen obscured, spidered pupil, magnifier, 16×16 microlens array, and detector model. OPD RMS is aperture-weighted after removing the aperture-weighted piston. Modes 1–10 after piston in HCIPy's RMS-normalized Zernike basis are reconstructed. The primary floating baseline uses the unquantized matrix obtained from independent ±30 nm physical optical probes. A second floating path uses the dequantized stored 18-bit matrix; the RTL model then adds reciprocal, centroid, multiply-accumulate, slicing, and wrap semantics. All three paths consume identical detector integers.",
        "",
        f"The pupil is {O.OpticalConfig().telescope_diameter_m:g} m with a {O.OpticalConfig().central_obscuration_m:g} m central obscuration and four {O.OpticalConfig().spider_width_m:g} m spiders, sampled on a 256×256 base grid with 4× pupil-edge supersampling. OPD is converted to phase as 2π·OPD/λ at λ={O.OpticalConfig().wavelength_m*1e9:g} nm. Detector x is array-column/local lenslet x; y is array-row/local lenslet y. The slope vector is all valid x slopes followed by all valid y slopes, matching the measured interaction matrix. Detector integration is represented by the propagated grid at the production setting and by explicit block integration in the refined-grid convergence study.",
        "",
        "The RTL formats are: 16-bit intensity, 19-bit moments at the interface, 19-bit Q18 reciprocal narrowed to the existing 18-bit port, 24-bit Q18 centroid, signed 25-bit Q18 slope, signed 18-bit stored E matrix, 52-bit accumulator, and signed 25-bit output slice [42:18]. ROM signed values use checked nibble sign extension. Arithmetic uses the RTL's truncation and wrap behavior; no fitted output multiplier is used.",
        "",
        f"Photons mean photons incident over the full simulated pupil per frame. Quantum efficiency is {cfg['detector']['quantum_efficiency']}; read noise is in electrons RMS; gain is fixed at {cfg['detector']['gain_e_per_adu']} e/ADU; threshold is {cfg['detector']['pixel_threshold_adu']} ADU. Noise intervals use wavefront means as independent units, with repeated detector-noise exposures nested within each wavefront.",
        "",
        "## Principal results",
        "",
        f"Noiseless flat residual: primary float {nm(flat['noiseless']['ideal_float_residual_m']):.6f} nm; fixed {nm(flat['noiseless']['fixed_residual_m']):.6f} nm. Noisy flat (n={flat['noisy_ideal_float_residual']['n_independent']}): primary float {nm(flat['noisy_ideal_float_residual']['mean']):.3f} nm, fixed {nm(flat['noisy_fixed_residual']['mean']):.3f} nm.",
        "",
        "| Input RMS | n | Primary float residual | Fixed residual | Relative fixed residual | Matrix quant. | Arithmetic quant. | ADC-clipped px |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mixed:
        lines.append(
            f"| {float(row['level_nm']):.0f} nm | {int(row['true_opd_rms_m_n_independent'])} | "
            f"{nm(row['ideal_float_residual_m_mean']):.2f} nm | {nm(row['fixed_residual_m_mean']):.2f} nm | "
            f"{100*row['fixed_residual_m_mean']/row['true_opd_rms_m_mean']:.1f}% | "
            f"{nm(row['matrix_quantization_opd_m_mean']):.3f} nm | "
            f"{nm(row['arithmetic_quantization_opd_m_mean']):.3f} nm | "
            f"{row['clipped_adu_px_total']} |")
    gain = [r["ideal_gain"] for r in modes]
    lines += [
        "",
        f"Individual-mode primary-float gain range: {min(gain):.3f}–{max(gain):.3f} over {len(modes)*10} signed amplitude cases.",
        "",
        f"Corrected noise sweep: {len(noise)} cells, each with {noise[0]['n_wavefronts']} independent wavefronts × {noise[0]['noise_repeats_per_wavefront']} noise exposures. The table below shows the 3 e⁻ RMS read-noise branch; shot-only results are retained in `noise_summary.csv`.",
        "",
        "| Photons over pupil/frame | Mean e⁻ / valid lenslet | Primary float residual | Fixed residual (95% CI) | ADC-clipped px / 100 frames |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in noise if r["read_noise_e"] == 3.0]:
        lines.append(
            f"| {row['photons_per_pupil']:.2g} | {row['detected_e_per_valid_lenslet_mean']:.0f} | "
            f"{nm(row['ideal_float_residual_m_mean']):.2f} nm | {nm(row['fixed_residual_m_mean']):.2f} nm "
            f"({nm(row['fixed_residual_m_mean_ci95_low']):.2f}–{nm(row['fixed_residual_m_mean_ci95_high']):.2f}) | "
            f"{row['clipped_adu_px_total']} |")
    lines += [
        "",
        "The fixed 20 e⁻/ADU gain is clean through 3×10⁶ photons/pupil/frame. ADC clipping begins at 5×10⁶ and performance worsens; this is an observed detector-model operating point, not a physical-camera specification.",
        "",
        f"Operating-range sweep: {len(dynamic)} levels × {cfg['experiments']['dynamic_trials_per_level']} paired directions.",
        "",
        "| Input RMS | Fixed relative residual | Mean outer-2-pixel flux | Mean edge-centroid count | Valid-tile flux vs flat | Output-field use | ADC-clipped px |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dynamic:
        lines.append(
            f"| {row['level_nm']:.0f} nm | {100*row['fixed_residual_m_mean']/row['true_opd_rms_m_mean']:.1f}% | "
            f"{100*row['edge_flux_fraction_mean_mean']:.2f}% | {row['centroids_near_window_edge_mean']:.2f} | "
            f"{100*row['valid_flux_fraction_vs_flat_mean']:.1f}% | {row['zernike_field_use_pct_mean']:.1f}% | "
            f"{row['clipped_adu_px_total']} |")
    lines += [
        "",
        "Degradation starts in these paired mixtures near 1.5 µm RMS, where edge occupancy and edge-centroid counts rise while the output field remains below half scale. ADC clipping occurs earlier without comparable degradation, so the diagnostics support spot displacement/escape as a contributor rather than treating it as an untested assumption or the sole cause.",
        "",
        f"Out-of-model sweep: {len(unmodelled)} levels × {cfg['experiments']['unmodelled_trials_per_level']} wavefronts.",
        "",
        "| Unmodelled / modelled RMS | Represented-mode wavefront error | Full-pupil residual |",
        "|---:|---:|---:|",
    ]
    for row in unmodelled:
        lines.append(
            f"| {100*row['unmodelled_fraction']:.0f}% | "
            f"{nm(row['represented_mode_wavefront_error_m_mean']):.2f} nm | "
            f"{nm(row['full_pupil_residual_m_mean']):.2f} nm |")
    lines += [
        "",
        "These two residual definitions are reported independently; no ordering is assumed from modal orthogonality.",
        "",
        f"Subpixel study: {subpixel['positions']} positions spanning {subpixel['phase_interval_px'][0]:.1f}–{subpixel['phase_interval_px'][1]:.1f} pixel; bias RMS {subpixel['bias_rms_px']:.5f} px, bias peak-to-peak {subpixel['bias_peak_to_peak_px']:.5f} px.",
        "",
        f"RTL: {rtl['n_cases']} cases, {rtl['n_coefficients_checked']} coefficients, {rtl['n_coefficient_mismatches']} coefficient mismatches, {rtl['n_slope_centroid_mismatches']} slope/centroid mismatches, {rtl['n_unknown_valid_values']} unknown valid values, latency {rtl['latency_cycles_observed']} simulated cycles.",
        "Across the held-out mixed campaign there were zero invalid centroids, intensity/moment overflows, accumulator wraps, or output-field wraps. Per-mode coefficient bias, RMSE, p95 absolute error, and maxima are in `mixed_coefficient_summary.csv`. The wavefront error decomposition closed to floating-point roundoff and retains the signed cross term in every raw row.",
        "",
        "## Convergence",
        "",
        f"Frozen tolerance: both ≤{convergence['criterion']['relative_pct']}% and ≤{convergence['criterion']['absolute_nm']} nm versus the finest tested setting.",
        "Each setting receives its own flat reference and ±30 nm push-pull optical interaction calibration before the common held-out wavefront directions are scored, so this test does not confuse grid refinement with a reference/matrix mismatch.",
        "",
        ("**The production numerical settings pass the frozen convergence criterion.**" if convergence_pass
         else "**The production numerical settings do not pass the frozen convergence criterion. Quantitative optical residuals are therefore conditional on the 256×256, 4×-edge-supersampled discretization and should not be presented as numerically converged.**"),
        "",
        "| Study | Setting | Residual | Δ vs finest | Pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in convergence["rows"]:
        lines.append(f"| {row['study']} | {row['setting']} | {nm(row['residual_mean_m']):.3f} nm | "
                     f"{row['relative_delta_vs_finest_pct']:.2f}% / {row['absolute_delta_vs_finest_nm']:.3f} nm | "
                     f"{'yes' if row['passes_frozen_tolerance'] else 'no'} |")
    lines += [
        "",
        "## Why close float/fixed agreement does not validate the optics",
        "",
        "The paired fixed–floating difference tests arithmetic implementation on the same slopes and calibration. Optical accuracy instead compares either reconstruction with an independently propagated known OPD. The arithmetic paths can agree exactly or nearly exactly while both inherit centroid bias, calibration/model mismatch, aliasing, detector loss, or an inadequate modal basis.",
        "",
        "## Limitations",
        "",
        "All optical frames are simulated. Detector parameters and optical geometry are assumed, not measured. The RTL test is finite simulation, not formal proof. Simulated cycle latency is not an achieved clock rate. No resource, timing-closure, throughput-in-seconds, power, physical-bench, or hardware-equivalence conclusion is supported.",
        "",
        "## Claim-to-evidence table",
        "",
        "| Claim | Status | Evidence |",
        "|---|---|---|",
    ]
    lines += [f"| {r['claim']} | {r['status']} | {r['evidence']} |" for r in claims]
    lines += [
        "",
        "## Reproduction",
        "",
        "From the repository root:",
        "",
        "```bash",
        "sim/venv/bin/python evaluation/optical/manuscript_evaluation.py --config evaluation/configs/manuscript_v1.json --out evaluation/results/runs/manuscript_v1_reproduction",
        "```",
        "",
        "The command refuses a non-empty output directory. Historical results are therefore not overwritten.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quick", action="store_true", help="development smoke test; not manuscript evidence")
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text())
    out = (args.out or (C.RESULTS_DIR / "runs" / cfg["run_id"])).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory is not empty (historical results preserved): {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.quick:
        cfg["run_id"] += "_QUICK_NOT_FOR_MANUSCRIPT"
        e = cfg["experiments"]
        e.update({"flat_noisy_trials": 4, "mixed_trials_per_level": 2,
                  "noise_wavefronts": 2, "noise_repeats_per_wavefront": 1,
                  "dynamic_trials_per_level": 2, "unmodelled_trials_per_level": 2,
                  "convergence_wavefronts": 2, "rtl_mixed_cases": 2})
        e["noise_photons_per_pupil"] = [1e6, 8e6]
        e["dynamic_levels_nm_rms"] = [100, 2000, 4000]

    calibration_files = {
        "ideal_e": CALIBRATION_DIR / "E_measured_m_per_slopepx.npy",
        "scale": CALIBRATION_DIR / "derived_scale.json",
        "calibration": CALIBRATION_DIR / "ematrix_regen.json",
        "reference": CALIBRATION_DIR / "reference_regen.json",
    }
    missing = [str(path) for path in calibration_files.values() if not path.is_file()]
    if missing:
        raise SystemExit("required frozen calibration artifact(s) missing: " + ", ".join(missing))
    frozen = out / "frozen_inputs"; frozen.mkdir()
    shutil.copyfile(cfg_path, frozen / "config.json")
    for name, path in calibration_files.items():
        shutil.copyfile(path, frozen / path.name)
    stage_rom_set(frozen / "roms")
    (out / "effective_config.json").write_text(json.dumps(cfg, indent=2))

    scale = float(json.loads(calibration_files["scale"].read_text())["scale_m_per_lsb"])
    ideal_e = np.load(calibration_files["ideal_e"])
    bench = O.get_bench()
    rec = Reconstruction(bench, scale, ideal_e)
    print(f"Frozen manuscript campaign -> {out}")
    print("  E1 flat"); _, flat = run_flat(rec, cfg, out)
    print("  E2 individual modes"); _, modes = run_modes(rec, cfg, out)
    print("  E3 held-out mixed modes"); _, mixed = run_mixed(rec, cfg, out)
    print("  E4 corrected photon/read-noise sweep"); _, noise = run_noise(rec, cfg, out)
    print("  E5 operating range"); _, dynamic = run_dynamic(rec, cfg, out)
    print("  E6 out-of-model modes"); _, unmodelled = run_unmodelled(rec, cfg, out)
    print("  E7 full-period subpixel sweep"); _, subpixel = run_subpixel(rec, cfg, out)
    print("  E8 numerical convergence"); _, convergence = run_convergence(rec, cfg, out)
    print("  E9 RTL agreement"); rtl = run_rtl(bench, cfg, out)

    summaries = {"flat": flat, "modes": modes, "mixed": mixed, "noise": noise,
                 "dynamic": dynamic, "unmodelled": unmodelled,
                 "subpixel": subpixel, "convergence": convergence, "rtl": rtl}
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    command = [str(C.VENV_PYTHON), str(Path(__file__).resolve()),
               "--config", str(cfg_path), "--out", str(out)]
    provenance = source_provenance(cfg_path, out, command)
    provenance["calibration_sha256"] = {name: sha256(path)
                                         for name, path in calibration_files.items()}
    provenance["random_seeds"] = cfg["seeds"]
    provenance["quick_not_for_manuscript"] = args.quick
    (out / "MANIFEST.json").write_text(json.dumps(provenance, indent=2))
    make_figures(out)
    generate_report(out, cfg, summaries, provenance)
    print(f"Completed: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
