#!/usr/bin/env python3
"""
STEP 4 -- Physically realistic noise sweep.

Question: is the FPGA's fixed-point quantisation significant relative to
photon noise, and if so, above what flux?

Method
------
For each flux level and each random wavefront:

  1. draw a physically reasonable Zernike vector (same distribution as the
     Monte Carlo campaign)
  2. forward-model it to per-subaperture slopes via pinv(E)
  3. render noiseless spots carrying N_photons per subaperture
  4. apply POISSON photon noise in photon units
  5. digitise to the 8-bit frame the sensor actually delivers
  6. process that identical frame two ways:
        A. float64, exact arithmetic
        B. bit-accurate FPGA fixed point
  7. score BOTH against the known ground truth

Because A and B see the *same* noisy frame, their difference is purely the
FPGA's arithmetic.

Stated assumptions (not claims about a specific detector)
--------------------------------------------------------
The repository defines no conversion gain (e-/ADU), full well, or read noise.
So this sweep works in PHOTON counts and holds the *digitised* signal fixed at
the level measured in the repository's own frame (spot sum ~812 ADU, peak ~201).
Sweeping N_photons at fixed digitisation is equivalent to sweeping SNR at
constant ADC scale; the implied gain is N_photons/812 e-/ADU and is reported
per level. No read noise or background is modelled, because the repository
defines neither -- so these curves are a PHOTON-NOISE-ONLY bound, and a real
detector will sit above them.

Outputs
    results/noise_sweep.csv          per-level summary
    results/noise_sweep_raw.csv      per-realisation observations
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

# Digitised spot level measured from rtl/hcipy_image_rotating.mem.
ADU_SUM = FG.DEFAULT_SPOT_SUM_ADU          # 812
DEFAULT_FLUXES = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]


def render_noisy(x_slope, y_slope, n_photons, rng) -> np.ndarray:
    """
    Render spots carrying n_photons per subaperture, apply Poisson noise, then
    digitise to 8 bits at the repository's measured ADU scale.
    """
    # quantise=False: render in photon units, unclipped. Clipping here would
    # destroy the spot at high flux before noise and digitisation are applied.
    photons = FG.render_frame(x_slope, y_slope, spot_sum=float(n_photons),
                              quantise=False)
    noisy = rng.poisson(np.clip(photons, 0, None)).astype(np.float64)
    gain = n_photons / ADU_SUM                 # photons per ADU
    adu = np.rint(noisy / gain)
    return np.clip(adu, 0, 255).astype(np.int64)


def score(frame, true_coeffs, true_sx, true_sy, idx, scale_px):
    """Return (float_metrics, fixed_metrics) against ground truth."""
    fx = BA.run_pipeline(frame)
    fl = BA.run_float_reference(frame)

    ref_x, ref_y = BA.slopes_ref()
    true_cx = ref_x[idx] / scale_px + true_sx
    true_cy = ref_y[idx] / scale_px + true_sy

    out = {}
    for tag, cx, cy, sx, sy, zern in (
        ("float", fl["x_centroid_px"][idx], fl["y_centroid_px"][idx],
         fl["x_slope_px"][idx], fl["y_slope_px"][idx], fl["zernike_m"]),
        ("fixed", fx["x_centroid"][idx] / scale_px, fx["y_centroid"][idx] / scale_px,
         fx["x_slope_valid"] / scale_px, fx["y_slope_valid"] / scale_px,
         fx["zernike_m"]),
    ):
        cen = np.concatenate([cx - true_cx, cy - true_cy])
        slp = np.concatenate([sx - true_sx, sy - true_sy])
        zer = zern - true_coeffs
        out[tag] = {
            "centroid_rms_px": float(np.sqrt(np.mean(cen ** 2))),
            "slope_rms_px": float(np.sqrt(np.mean(slp ** 2))),
            "zernike_rmse_m": float(np.sqrt(np.mean(zer ** 2))),
            # Wavefront RMS from the modal coefficient errors: the Zernike basis
            # is orthonormal, so the quadrature sum is the wavefront RMS.
            "wavefront_rms_m": float(np.sqrt(np.sum(zer ** 2))),
        }
    return out


def run(fluxes, n_real, seed, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    idx = BA.valid_subaps()
    scale_px = float(1 << C.CENTROID_FRAC)

    raw_rows, summary_rows = [], []
    t0 = time.time()

    for flux in fluxes:
        acc = {"float": [], "fixed": []}
        for k in range(n_real):
            coeffs = FG.random_coefficients(rng, 6e-8)
            sx, sy = FG.coeffs_to_slopes(coeffs)
            frame = render_noisy(sx, sy, flux, rng)
            m = score(frame, coeffs, sx, sy, idx, scale_px)
            for tag in ("float", "fixed"):
                acc[tag].append(m[tag])
            raw_rows.append({"flux_photons": flux, "realisation": k,
                             **{f"{t}_{k2}": v for t, d in m.items()
                                for k2, v in d.items()}})
        row = {"flux_photons": flux,
               "implied_gain_photons_per_adu": flux / ADU_SUM,
               "n_realisations": n_real}
        for tag in ("float", "fixed"):
            for metric in ("centroid_rms_px", "slope_rms_px",
                           "zernike_rmse_m", "wavefront_rms_m"):
                vals = np.array([d[metric] for d in acc[tag]])
                row[f"{tag}_{metric}"] = float(vals.mean())
                row[f"{tag}_{metric}_sem"] = float(vals.std(ddof=1) / np.sqrt(vals.size))
                row[f"{tag}_{metric}_ci95"] = float(
                    1.96 * vals.std(ddof=1) / np.sqrt(vals.size))
        for metric in ("centroid_rms_px", "zernike_rmse_m", "wavefront_rms_m"):
            f_, x_ = row[f"float_{metric}"], row[f"fixed_{metric}"]
            row[f"penalty_{metric}"] = x_ / f_ if f_ else float("nan")
            row[f"degradation_pct_{metric}"] = 100.0 * (x_ / f_ - 1.0) if f_ else float("nan")
        summary_rows.append(row)
        print(f"  flux {flux:9.0f} ph/subap  "
              f"centroid float {row['float_centroid_rms_px']:.5f} px | "
              f"fixed {row['fixed_centroid_rms_px']:.5f} px | "
              f"penalty {row['penalty_centroid_rms_px']:.3f}x", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    _csv(out_dir / "noise_sweep.csv", summary_rows)
    _csv(out_dir / "noise_sweep_raw.csv", raw_rows)

    # Crossover: lowest flux at which the FPGA penalty exceeds 10 %.
    crossover = None
    for r in summary_rows:
        if r["degradation_pct_centroid_rms_px"] > 10.0:
            crossover = r["flux_photons"]
            break

    meta = {
        "seed": seed, "n_realisations_per_level": n_real,
        "fluxes_photons": list(fluxes),
        "adu_sum_held_constant": ADU_SUM,
        "crossover_flux_10pct_penalty": crossover,
        "noise_model": "Poisson photon noise only; no read noise or background "
                       "(repository defines neither)",
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "noise_sweep_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  {len(fluxes)} levels x {n_real} realisations in "
          f"{meta['elapsed_s']:.1f}s")
    if crossover:
        print(f"  FPGA penalty exceeds 10% above ~{crossover:.0f} photons/subap")
    else:
        print("  FPGA penalty stays below 10% across the whole swept range")
    return meta


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(
                f"{r[k]:.10e}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                for k in keys) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--realisations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED + 10)
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()
    print(f"Noise sweep: {len(DEFAULT_FLUXES)} flux levels x "
          f"{a.realisations} realisations")
    run(DEFAULT_FLUXES, a.realisations, a.seed, a.out)


if __name__ == "__main__":
    main()
