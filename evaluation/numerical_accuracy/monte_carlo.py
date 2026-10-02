#!/usr/bin/env python3
"""
Monte Carlo numerical accuracy: FPGA fixed-point vs floating point.

Compares the bit-accurate model of the RTL (evaluation/common/bitaccurate.py,
verified to reproduce RTL outputs exactly) against the identical algorithm in
float64. This isolates QUANTISATION error from algorithmic or optical error.

Quantities evaluated per case:
    centroid x / y      (pixels)
    slope    x / y      (pixels)
    Zernike coefficients (metres), aggregate and per-mode

Wavefront conversion uses only constants present in src/shwfs_utils.py; see
config.centroid_px_to_wavefront_m.

Outputs
    results/accuracy_raw_zernike.csv    per-case, per-mode errors
    results/accuracy_summary.csv        aggregate statistics per quantity
    results/zernike_per_mode.csv        per-mode Zernike statistics
    results/accuracy_params.json        all parameters, for reproducibility
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import bitaccurate as BA          # noqa: E402
from evaluation.common import config as C                # noqa: E402
from evaluation.common import framegen as FG             # noqa: E402

PERCENTILES = (50, 95, 99)


def error_stats(err: np.ndarray, ref: np.ndarray | None = None,
                label: str = "") -> dict:
    """Standard error statistics for one quantity."""
    err = np.asarray(err, dtype=np.float64).ravel()
    finite = err[np.isfinite(err)]
    abs_err = np.abs(finite)

    stats = {
        "quantity": label,
        "n": int(finite.size),
        "bias_mean_signed_error": float(np.mean(finite)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(finite ** 2))),
        "std_error": float(np.std(finite)),
        "max_abs_error": float(np.max(abs_err)) if abs_err.size else float("nan"),
    }
    for p in PERCENTILES:
        stats[f"p{p}_abs_error"] = float(np.percentile(abs_err, p)) if abs_err.size else float("nan")

    if ref is not None:
        ref = np.asarray(ref, dtype=np.float64).ravel()
        mask = np.isfinite(ref) & (np.abs(ref) > 0)
        if mask.any():
            rel = np.abs(err[mask] / ref[mask])
            stats["rel_error_median"] = float(np.median(rel))
            stats["rel_error_p95"] = float(np.percentile(rel, 95))
            stats["rms_relative"] = float(
                np.sqrt(np.mean(err[mask] ** 2)) / np.sqrt(np.mean(ref[mask] ** 2))
            )
        else:
            stats["rel_error_median"] = float("nan")
            stats["rel_error_p95"] = float("nan")
            stats["rms_relative"] = float("nan")
    return stats


def run(n_cases: int, seed: int, scale_m: float, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    idx = BA.valid_subaps()
    scale_px = float(1 << C.CENTROID_FRAC)

    cen_err, slope_err = [], []
    cen_ref, slope_ref = [], []
    z_fixed = np.zeros((n_cases, C.NUM_MODES))
    z_float = np.zeros((n_cases, C.NUM_MODES))
    z_true = np.zeros((n_cases, C.NUM_MODES))

    t0 = time.time()
    for i in range(n_cases):
        case = FG.make_case(rng, scale_m=scale_m)
        frame = case["frame"]

        fx = BA.run_pipeline(frame)
        fl = BA.run_float_reference(frame)

        # Centroids: fixed-point Q18 -> pixels, vs exact float
        cf_x = fx["x_centroid"][idx] / scale_px
        cf_y = fx["y_centroid"][idx] / scale_px
        cen_err.append(np.concatenate([cf_x - fl["x_centroid_px"][idx],
                                       cf_y - fl["y_centroid_px"][idx]]))
        cen_ref.append(np.concatenate([fl["x_centroid_px"][idx],
                                       fl["y_centroid_px"][idx]]))

        sf_x = fx["x_slope_valid"] / scale_px
        sf_y = fx["y_slope_valid"] / scale_px
        slope_err.append(np.concatenate([sf_x - fl["x_slope_px"][idx],
                                         sf_y - fl["y_slope_px"][idx]]))
        slope_ref.append(np.concatenate([fl["x_slope_px"][idx],
                                         fl["y_slope_px"][idx]]))

        z_fixed[i] = fx["zernike_m"]
        z_float[i] = fl["zernike_m"]
        z_true[i] = case["true_coeffs_m"]

        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{n_cases}  ({rate:.0f} cases/s)", flush=True)

    elapsed = time.time() - t0
    cen_err = np.concatenate(cen_err); cen_ref = np.concatenate(cen_ref)
    slope_err = np.concatenate(slope_err); slope_ref = np.concatenate(slope_ref)
    z_err = z_fixed - z_float

    # ---------------- summary ----------------
    rows = [
        error_stats(cen_err, cen_ref, "centroid_px"),
        error_stats(slope_err, slope_ref, "slope_px"),
        error_stats(z_err, z_float, "zernike_m_aggregate"),
        error_stats(z_fixed - z_true, z_true, "zernike_m_vs_true_forward_model"),
    ]

    # Wavefront equivalent of the centroid error (uses only repo constants)
    wf_rmse = C.centroid_px_to_wavefront_m(rows[0]["rmse"])
    wf_p95 = C.centroid_px_to_wavefront_m(rows[0]["p95_abs_error"])
    rows.append({
        "quantity": "centroid_wavefront_equivalent_m",
        "n": rows[0]["n"],
        "bias_mean_signed_error": float("nan"),
        "mae": float("nan"),
        "rmse": wf_rmse,
        "std_error": float("nan"),
        "max_abs_error": C.centroid_px_to_wavefront_m(rows[0]["max_abs_error"]),
        "p50_abs_error": C.centroid_px_to_wavefront_m(rows[0]["p50_abs_error"]),
        "p95_abs_error": wf_p95,
        "p99_abs_error": C.centroid_px_to_wavefront_m(rows[0]["p99_abs_error"]),
    })

    per_mode = []
    for m in range(C.NUM_MODES):
        s = error_stats(z_err[:, m], z_float[:, m], C.MODE_LABELS[m])
        s["mode_index"] = m + 1
        s["rmse_waves"] = s["rmse"] / C.WAVELENGTH_M
        s["ref_rms_m"] = float(np.sqrt(np.mean(z_float[:, m] ** 2)))
        per_mode.append(s)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "accuracy_summary.csv", rows)
    _write_csv(out_dir / "zernike_per_mode.csv", per_mode)

    # Raw per-case data for plotting
    with (out_dir / "accuracy_raw_zernike.csv").open("w") as fh:
        fh.write("case,mode,mode_label,fixed_m,float_m,true_m,error_m\n")
        for i in range(n_cases):
            for m in range(C.NUM_MODES):
                fh.write(f"{i},{m+1},{C.MODE_LABELS[m]},{z_fixed[i,m]:.12e},"
                         f"{z_float[i,m]:.12e},{z_true[i,m]:.12e},{z_err[i,m]:.12e}\n")

    params = {
        "n_cases": n_cases,
        "seed": seed,
        "coefficient_scale_m": scale_m,
        "spot_sigma_px": FG.DEFAULT_SPOT_SIGMA_PX,
        "spot_sum_adu": FG.DEFAULT_SPOT_SUM_ADU,
        "pixel_threshold": C.PIXEL_THRESHOLD,
        "num_modes": C.NUM_MODES,
        "num_subaps_valid": C.NUM_SUBAPS_VALID,
        "zernike_shift": C.ZERNIKE_SHIFT,
        "meters_per_lsb": C.METERS_PER_LSB,
        "wavelength_m": C.WAVELENGTH_M,
        "elapsed_s": elapsed,
        "reference": "float64, identical algorithm (isolates quantisation)",
        "forward_model": "pinv(E) from e_matrix_{x,y}.mem",
    }
    (out_dir / "accuracy_params.json").write_text(json.dumps(params, indent=2))

    print(f"\n{n_cases} cases in {elapsed:.1f} s")
    print(f"  centroid RMSE : {rows[0]['rmse']:.4e} px "
          f"(= {wf_rmse*1e9:.3f} nm wavefront)")
    print(f"  slope    RMSE : {rows[1]['rmse']:.4e} px")
    print(f"  zernike  RMSE : {rows[2]['rmse']:.4e} m "
          f"({rows[2]['rms_relative']:.2%} of signal)")
    return {"summary": rows, "per_mode": per_mode, "params": params}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(_fmt(r.get(k, "")) for k in keys) + "\n")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.12e}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--cases", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED)
    ap.add_argument("--scale", type=float, default=6e-8,
                    help="coefficient std-dev in metres")
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()
    print(f"Monte Carlo: {a.cases} cases, seed {a.seed}")
    run(a.cases, a.seed, a.scale, a.out)


if __name__ == "__main__":
    main()
