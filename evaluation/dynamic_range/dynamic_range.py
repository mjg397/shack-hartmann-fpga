#!/usr/bin/env python3
"""
STEP 5 -- Dynamic range / aberration magnitude.

Question: at what aberration magnitude does the design stop reconstructing
accurately, and which physical or arithmetic limit binds first?

Sweeps aberration magnitude and records, for every case, the candidate limits
so the data show which one is reached first rather than a threshold being
asserted:

  * maximum centroid displacement from the reference position
  * whether any spot centre left its 16x16 subaperture
  * whether the intensity accumulator exceeded its assertion bound (65280)
  * whether the moment accumulators exceeded theirs (489600)
  * whether any Zernike output saturated the signed 25-bit field
  * reconstruction RMSE against ground truth

Both individual Zernike modes and randomised multimode wavefronts are swept.

Outputs
    results/dynamic_range.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import bitaccurate as BA          # noqa: E402
from evaluation.common import config as C                # noqa: E402
from evaluation.common import framegen as FG             # noqa: E402

SCALE_PX = float(1 << C.CENTROID_FRAC)
ZERN_MAX = (1 << (C.ZERNIKE_BITS - 1)) - 1      # 2^24 - 1


def evaluate(coeffs, rng=None) -> dict:
    sx, sy = FG.coeffs_to_slopes(coeffs)
    frame = FG.render_frame(sx, sy)

    idx = BA.valid_subaps()
    ref_x, ref_y = BA.slopes_ref()

    # Intended spot centres in local tile coordinates
    cx = ref_x[idx] / SCALE_PX + sx
    cy = ref_y[idx] / SCALE_PX + sy
    out_of_tile = int(np.sum((cx < 0) | (cx > C.SUBAP_PIXELS - 1) |
                             (cy < 0) | (cy > C.SUBAP_PIXELS - 1)))

    fx = BA.run_pipeline(frame)
    fl = BA.run_float_reference(frame)

    sI, xI, yI = BA.accumulate(frame)
    zern_raw = fx["zernike_raw"]

    err = fx["zernike_m"] - coeffs
    return {
        "max_abs_slope_px": float(np.max(np.abs(np.concatenate([sx, sy])))),
        "max_centroid_disp_px": float(np.max(np.abs(np.concatenate([cx, cy])
                                                    - (C.SUBAP_PIXELS - 1) / 2))),
        "spots_out_of_subaperture": out_of_tile,
        "sI_max": int(sI.max()),
        "sI_overflow": int(sI.max() > C.SI_MAX),
        "moment_max": int(max(xI.max(), yI.max())),
        "moment_overflow": int(max(xI.max(), yI.max()) > C.XI_MAX),
        "zernike_abs_max_lsb": int(np.max(np.abs(zern_raw))),
        "zernike_saturated": int(np.max(np.abs(zern_raw)) >= ZERN_MAX),
        "zernike_rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "zernike_rmse_nm": float(np.sqrt(np.mean(err ** 2)) * 1e9),
        "rel_rmse": float(np.sqrt(np.mean(err ** 2)) /
                          np.sqrt(np.mean(coeffs ** 2)))
        if np.any(coeffs) else float("nan"),
        "fixed_vs_float_rmse_m": float(
            np.sqrt(np.mean((fx["zernike_m"] - fl["zernike_m"]) ** 2))),
    }


def run(mags, n_random: int, seed: int, out_dir: Path) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []

    # --- individual modes ---
    for m in range(C.NUM_MODES):
        for mag in mags:
            coeffs = np.zeros(C.NUM_MODES)
            coeffs[m] = mag
            r = evaluate(coeffs)
            rows.append({"family": "single_mode", "mode": C.MODE_LABELS[m],
                         "magnitude_m": mag, "magnitude_nm": mag * 1e9,
                         "magnitude_waves": mag / C.WAVELENGTH_M, **r})

    # --- randomised multimode ---
    for mag in mags:
        for k in range(n_random):
            v = FG.random_coefficients(rng, 1.0)
            v = v / np.sqrt(np.mean(v ** 2)) * mag        # set RMS to mag
            r = evaluate(v)
            rows.append({"family": "multimode", "mode": f"random_{k}",
                         "magnitude_m": mag, "magnitude_nm": mag * 1e9,
                         "magnitude_waves": mag / C.WAVELENGTH_M, **r})

    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (out_dir / "dynamic_range.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.8e}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")

    # --- report which limit binds first, from the data ---
    print(f"\n{'magnitude(nm)':>14} {'rel RMSE':>10} {'out-of-tile':>12} "
          f"{'sI ovf':>7} {'mom ovf':>8} {'zern sat':>9}")
    for mag in mags:
        sub = [r for r in rows if r["magnitude_m"] == mag and
               r["family"] == "multimode"]
        if not sub:
            continue
        print(f"{mag*1e9:14.1f} {np.mean([r['rel_rmse'] for r in sub]):10.4f} "
              f"{np.mean([r['spots_out_of_subaperture'] for r in sub]):12.1f} "
              f"{sum(r['sI_overflow'] for r in sub):7d} "
              f"{sum(r['moment_overflow'] for r in sub):8d} "
              f"{sum(r['zernike_saturated'] for r in sub):9d}")

    first = {}
    for r in rows:
        for lim in ("spots_out_of_subaperture", "sI_overflow",
                    "moment_overflow", "zernike_saturated"):
            if r[lim] and lim not in first:
                first[lim] = r["magnitude_m"]
    print("\n  first magnitude (m) at which each limit is observed:")
    for lim in ("spots_out_of_subaperture", "sI_overflow",
                "moment_overflow", "zernike_saturated"):
        v = first.get(lim)
        print(f"    {lim:28s} {'never in swept range' if v is None else f'{v:.3e}'}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED + 30)
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()
    mags = np.logspace(np.log10(1e-9), np.log10(1e-6), 16)
    print(f"Dynamic range: {len(mags)} magnitudes, "
          f"{C.NUM_MODES} single modes + {a.random} multimode each")
    run(list(mags), a.random, a.seed, a.out)


if __name__ == "__main__":
    main()
