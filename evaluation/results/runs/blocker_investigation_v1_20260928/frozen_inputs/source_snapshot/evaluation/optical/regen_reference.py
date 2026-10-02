#!/usr/bin/env python3
"""
Regenerate the flat-wavefront reference centroids from THIS optical bench.

The baseline `slopes_ref_{x,y}.mem` does not match this bench: measured at
0.011 px RMS on undigitised images, which appears as an ~8.3 nm systematic
OPD bias on every frame. That is a calibration mismatch, not an arithmetic one,
so it is fixable without touching the datapath.

Two reference conventions are supported and recorded explicitly:

  * flat (zero-OPD) wavefront through the same cached HCIPy bench
  * same explicit detector model and operating point
  * same PIXEL_THRESHOLD (5 ADU) applied before accumulation
  * ``exact`` (the historical/default convention): divide the integer moments
    exactly and round the quotient to Q18;
  * ``rtl``: use the reciprocal LUT and multiplier exactly as the RTL does.

The old documentation incorrectly called the first convention identical to the
hardware.  It is not: ``xI * round(2^18 / sI)`` is generally different from
``round(2^18 * xI / sI)``.  Keeping the choice explicit prevents a reference
calibration from silently hiding that arithmetic difference.

Averaging over several noiseless exposures is pointless (they are identical),
so the noiseless frame is used directly; optionally a noisy average is offered
for comparison, which is what a real calibration would do.

Writes a candidate ROM pair. It does NOT overwrite the baseline unless --apply
is given, and the baseline is preserved in evaluation/results/baseline_v1/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import bitaccurate as BA          # noqa: E402
from evaluation.common import config as C                # noqa: E402
from evaluation.common import detector as D              # noqa: E402
from evaluation.common import optics as O                # noqa: E402

SEED = 4242
HEX_DIGITS = 7          # 25 bits -> 7 nibbles, matching the existing files


def thresholded_cog(adu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exactly the accumulator's arithmetic, in float: threshold then CoG."""
    img = np.asarray(adu, dtype=np.float64).reshape(C.FRAME_H, C.FRAME_W)
    xi = np.arange(C.SUBAP_PIXELS)
    cx = np.zeros(C.NUM_SUBAPS_GRID)
    cy = np.zeros(C.NUM_SUBAPS_GRID)
    for t in range(C.NUM_SUBAPS_GRID):
        r, c = divmod(t, C.NUM_LENSLETS)
        tile = img[r*16:(r+1)*16, c*16:(c+1)*16].copy()
        tile[tile < C.PIXEL_THRESHOLD] = 0.0
        s = tile.sum()
        if s > 0:
            cx[t] = (tile.sum(axis=0) * xi).sum() / s
            cy[t] = (tile.sum(axis=1) * xi).sum() / s
    return cx, cy


def quantise_q18(centroid_px: np.ndarray) -> np.ndarray:
    """Q18, signed 25-bit -- the format slope_calculation's ROM expects."""
    q = np.rint(centroid_px * (1 << C.CENTROID_FRAC)).astype(np.int64)
    lim = 1 << (C.SLOPE_BITS - 1)
    if np.any(np.abs(q) >= lim):
        raise SystemExit("reference exceeds signed 25-bit range")
    return q


def reference_centroids_q18(adu: np.ndarray, arithmetic: str) -> tuple[np.ndarray, np.ndarray]:
    """Return flat centroids in the selected, explicitly named convention."""
    if arithmetic == "exact":
        cx, cy = thresholded_cog(adu)
        return quantise_q18(cx), quantise_q18(cy)
    if arithmetic == "rtl":
        s_i, x_i, y_i = BA.accumulate(adu)
        _, reciprocal_port = BA.reciprocal(s_i)
        return BA.centroid(x_i, reciprocal_port), BA.centroid(y_i, reciprocal_port)
    raise ValueError(f"unknown centroid arithmetic {arithmetic!r}")


def write_mem(path: Path, vals: np.ndarray) -> None:
    mask = (1 << (HEX_DIGITS * 4)) - 1
    path.write_text("".join(f"{int(v) & mask:0{HEX_DIGITS}X}\n" for v in vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the new reference into rtl/ (baseline is preserved)")
    ap.add_argument("--noisy-average", type=int, default=0,
                    help="average N noisy exposures instead of one noiseless frame")
    ap.add_argument(
        "--centroid-arithmetic", choices=("exact", "rtl"), default="exact",
        help=("reference convention: historical rounded exact quotient, or the "
              "reciprocal-LUT/multiplier RTL result"),
    )
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR / "optical")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    bench = O.get_bench()
    flat = np.zeros_like(np.asarray(bench.aperture), dtype=float)
    img = bench.propagate(flat)

    if a.noisy_average > 0:
        rng = np.random.default_rng(SEED)
        det = D.DetectorConfig()
        acc_x = np.zeros(C.NUM_SUBAPS_GRID); acc_y = np.zeros(C.NUM_SUBAPS_GRID)
        for _ in range(a.noisy_average):
            qx, qy = reference_centroids_q18(
                D.expose(img, det, rng).adu, a.centroid_arithmetic)
            acc_x += qx; acc_y += qy
        new_x = np.rint(acc_x / a.noisy_average).astype(np.int64)
        new_y = np.rint(acc_y / a.noisy_average).astype(np.int64)
        cx = new_x / float(1 << C.CENTROID_FRAC)
        cy = new_y / float(1 << C.CENTROID_FRAC)
        method = f"mean of {a.noisy_average} noisy exposures"
    else:
        det = D.DetectorConfig(shot_noise=False, read_noise_e=0.0)
        new_x, new_y = reference_centroids_q18(
            D.expose(img, det).adu, a.centroid_arithmetic)
        cx = new_x / float(1 << C.CENTROID_FRAC)
        cy = new_y / float(1 << C.CENTROID_FRAC)
        method = "single noiseless exposure"

    old_x, old_y = BA.slopes_ref()
    idx = BA.valid_subaps()
    s = float(1 << C.CENTROID_FRAC)

    dx = (new_x[idx] - old_x[idx]) / s
    dy = (new_y[idx] - old_y[idx]) / s
    rec = {
        "method": method,
        "centroid_arithmetic": a.centroid_arithmetic,
        "detector": D.config_record(det),
        "n_valid": int(idx.size),
        "new_centroid_px_range": [float(cx[idx].min()), float(cx[idx].max())],
        "old_centroid_px_range": [float(old_x[idx].min()/s), float(old_x[idx].max()/s)],
        "delta_x_px_rms": float(np.sqrt((dx**2).mean())),
        "delta_y_px_rms": float(np.sqrt((dy**2).mean())),
        "delta_x_px_max": float(np.abs(dx).max()),
        "delta_max_px": float(max(np.abs(dx).max(), np.abs(dy).max())),
        "format": f"Q{C.CENTROID_FRAC}, signed {C.SLOPE_BITS}-bit",
        "applied_to_rtl": bool(a.apply),
    }
    (a.out / "reference_regen.json").write_text(json.dumps(rec, indent=2))

    cand = a.out / "candidate_reference"
    cand.mkdir(exist_ok=True)
    write_mem(cand / "slopes_ref_x.mem", new_x)
    write_mem(cand / "slopes_ref_y.mem", new_y)

    print(f"  method            : {method}")
    print(f"  new centroid range: {cx[idx].min():.4f} .. {cx[idx].max():.4f} px")
    print(f"  old centroid range: {old_x[idx].min()/s:.4f} .. {old_x[idx].max()/s:.4f} px")
    print(f"  change vs baseline: {rec['delta_x_px_rms']:.4f} px RMS (x), "
          f"max {rec['delta_max_px']:.4f} px")
    print(f"  wrote candidate   : {cand}")

    if a.apply:
        for n in ("slopes_ref_x.mem", "slopes_ref_y.mem"):
            shutil.copy(cand / n, C.RTL_DIR / n)
        BA.slopes_ref.cache_clear()
        print(f"  APPLIED to {C.RTL_DIR} (baseline kept in results/baseline_v1/)")


if __name__ == "__main__":
    main()
