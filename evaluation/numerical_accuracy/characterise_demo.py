#!/usr/bin/env python3
"""
STEP 11 -- Characterise the repository's own HCIPy demonstration frame.

Primary sources, documented explicitly:
    sim/rtl/hcipy_image_rotating.mem   (the streamed frame)
    sim/rtl/subaperture_bitmap.mem     (the pupil mask)

Both are byte-identical to the copies in rtl/ and, for the image, to
sim/data/hcipy_image_rotating.hex.

Reports flux, spot, centroid and slope statistics for this one concrete frame,
plus float-vs-fixed and float-vs-RTL comparisons, so the paper has a worked
example alongside the Monte Carlo aggregates.

Outputs
    results/demo_frame_characterisation.csv
    results/demo_frame_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import bitaccurate as BA          # noqa: E402
from evaluation.common import config as C                # noqa: E402

SCALE_PX = float(1 << C.CENTROID_FRAC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--with-rtl", action="store_true",
                    help="also run the RTL simulation for comparison")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    img_path = C.RTL_DIR / "hcipy_image_rotating.mem"
    px = np.array([int(t, 16) for t in img_path.read_text().split()],
                  dtype=np.int64)
    frame = px.reshape(C.FRAME_H, C.FRAME_W)

    idx = BA.valid_subaps()
    fx = BA.run_pipeline(px)
    fl = BA.run_float_reference(px)
    sI, xI, yI = BA.accumulate(px)

    # ---- per-subaperture spot statistics ----
    xi = np.arange(C.SUBAP_PIXELS)
    rows, widths, peaks = [], [], []
    for k, tile in enumerate(idx):
        r, c = divmod(int(tile), C.NUM_LENSLETS)
        t = frame[r*16:(r+1)*16, c*16:(c+1)*16].copy()
        t[t < C.PIXEL_THRESHOLD] = 0
        s = t.sum()
        if s <= 0:
            continue
        mx = (t.sum(0) * xi).sum() / s
        var = (t.sum(0) * (xi - mx) ** 2).sum() / s
        sigma = float(np.sqrt(var))
        widths.append(sigma)
        peaks.append(int(t.max()))
        rows.append({
            "subap_index": int(tile), "row": r, "col": c,
            "flux_adu": int(s), "peak_adu": int(t.max()),
            "spot_sigma_px": sigma,
            "fwhm_px": 2.3548 * sigma,
            "centroid_x_float_px": float(fl["x_centroid_px"][tile]),
            "centroid_y_float_px": float(fl["y_centroid_px"][tile]),
            "centroid_x_fixed_px": float(fx["x_centroid"][tile] / SCALE_PX),
            "centroid_y_fixed_px": float(fx["y_centroid"][tile] / SCALE_PX),
            "slope_x_float_px": float(fl["x_slope_px"][tile]),
            "slope_y_float_px": float(fl["y_slope_px"][tile]),
        })

    keys = list(rows[0].keys())
    with (a.out / "demo_frame_characterisation.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.8e}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")

    cen_err = np.concatenate([
        fx["x_centroid"][idx] / SCALE_PX - fl["x_centroid_px"][idx],
        fx["y_centroid"][idx] / SCALE_PX - fl["y_centroid_px"][idx]])
    z_err = fx["zernike_m"] - fl["zernike_m"]
    flux = np.array([r["flux_adu"] for r in rows])
    slopes = np.concatenate([[r["slope_x_float_px"] for r in rows],
                             [r["slope_y_float_px"] for r in rows]])

    summary = {
        "image_source": str(img_path.relative_to(C.REPO_ROOT)),
        "bitmap_source": str((C.RTL_DIR / "subaperture_bitmap.mem")
                             .relative_to(C.REPO_ROOT)),
        "frame_shape": [C.FRAME_H, C.FRAME_W],
        "valid_subapertures": int(idx.size),
        "subapertures_with_flux": len(rows),
        "pixel_threshold_adu": C.PIXEL_THRESHOLD,
        "flux_adu_min": int(flux.min()), "flux_adu_max": int(flux.max()),
        "flux_adu_median": float(np.median(flux)),
        "peak_adu_min": int(min(peaks)), "peak_adu_max": int(max(peaks)),
        "peak_adu_median": float(np.median(peaks)),
        "spot_sigma_px_min": float(np.min(widths)),
        "spot_sigma_px_max": float(np.max(widths)),
        "spot_sigma_px_median": float(np.median(widths)),
        "fwhm_px_median": float(np.median(widths) * 2.3548),
        "nyquist_note": "sigma < 1 px means the spot is undersampled; plain CoG "
                        "then carries systematic pixelation error",
        "slope_px_min": float(slopes.min()), "slope_px_max": float(slopes.max()),
        "slope_px_rms": float(np.sqrt(np.mean(slopes ** 2))),
        "centroid_fixed_vs_float_rms_px": float(np.sqrt(np.mean(cen_err ** 2))),
        "centroid_fixed_vs_float_max_px": float(np.max(np.abs(cen_err))),
        "zernike_fixed_vs_float_rmse_m": float(np.sqrt(np.mean(z_err ** 2))),
        "zernike_raw_lsb": [int(v) for v in fx["zernike_raw"]],
        "zernike_fixed_m": [float(v) for v in fx["zernike_m"]],
        "zernike_float_m": [float(v) for v in fl["zernike_m"]],
    }

    if a.with_rtl:
        from evaluation.common import rtl_runner as R
        with R.RTLSimulator() as sim:
            rtl = sim.run_frame(px)
        summary["rtl_zernike_raw"] = [int(v) for v in rtl["zernike_raw"]]
        summary["rtl_matches_bitaccurate"] = bool(
            np.array_equal(rtl["zernike_raw"], fx["zernike_raw"]))
        summary["rtl_latency_cycles"] = rtl["true_latency_cycles"]

    (a.out / "demo_frame_summary.json").write_text(json.dumps(summary, indent=2))

    print("Demo frame characterisation")
    print(f"  source            : {summary['image_source']}")
    print(f"  valid subaps      : {summary['valid_subapertures']} "
          f"({summary['subapertures_with_flux']} with flux)")
    print(f"  flux (ADU)        : {summary['flux_adu_min']} .. "
          f"{summary['flux_adu_max']}  median {summary['flux_adu_median']:.0f}")
    print(f"  peak (ADU)        : {summary['peak_adu_min']} .. "
          f"{summary['peak_adu_max']}  median {summary['peak_adu_median']:.0f}")
    print(f"  spot sigma (px)   : {summary['spot_sigma_px_min']:.3f} .. "
          f"{summary['spot_sigma_px_max']:.3f}  median "
          f"{summary['spot_sigma_px_median']:.3f}")
    print(f"  FWHM (px)         : {summary['fwhm_px_median']:.3f}")
    print(f"  slope range (px)  : {summary['slope_px_min']:.3f} .. "
          f"{summary['slope_px_max']:.3f}  rms {summary['slope_px_rms']:.4f}")
    print(f"  centroid fx-vs-fl : {summary['centroid_fixed_vs_float_rms_px']:.6f} px RMS")
    print(f"  zernike  fx-vs-fl : {summary['zernike_fixed_vs_float_rmse_m']:.4e} m")
    if a.with_rtl:
        print(f"  RTL == bit-accurate: {summary['rtl_matches_bitaccurate']}")
        print(f"  RTL latency        : {summary['rtl_latency_cycles']} cycles")


if __name__ == "__main__":
    main()
