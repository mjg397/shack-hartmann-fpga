#!/usr/bin/env python3
"""
Loader and compatibility gate for externally acquired Shack-Hartmann data.

NO external dataset is present. This module exists so that when one is obtained,
it can be checked and used without hand-written glue -- and, more importantly,
so that INCOMPATIBLE data is refused rather than silently resampled.

The reconstruction matrix in e_matrix_{x,y}.mem is calibrated for one specific
pupil and one specific set of 168 subapertures. Feeding it spots from a
different geometry returns numbers that look like coefficients and mean nothing.
This gate exists to prevent exactly that.

See docs/EVALUATION_REPORT.md section 4.10 for what a dataset must provide.

Usage
    python evaluation/real_data/load_external.py --frame raw.npy \
        [--reference flat.npy] [--truth coeffs.txt] [--force-float-only]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import bitaccurate as BA          # noqa: E402
from evaluation.common import config as C                # noqa: E402


@dataclass
class Compatibility:
    ok: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def load_frame(path: Path) -> np.ndarray:
    """Load a raw frame from .npy, .npz, .txt or .mem (hex-per-line)."""
    s = path.suffix.lower()
    if s == ".npy":
        arr = np.load(path)
    elif s == ".npz":
        z = np.load(path)
        arr = z[list(z.keys())[0]]
    elif s in (".mem", ".hex"):
        arr = np.array([int(t, 16) for t in path.read_text().split()])
    else:
        arr = np.loadtxt(path)
    return np.asarray(arr)


def check(frame: np.ndarray, *, bit_depth: int | None = None) -> Compatibility:
    """
    Strict compatibility gate against the DUT's fixed geometry.

    Never resamples, crops or pads. A failure here means the data cannot be run
    through the RTL path, full stop.
    """
    c = Compatibility()
    arr = np.asarray(frame)

    if arr.ndim == 1:
        n = arr.size
        side = int(round(np.sqrt(n)))
        if side * side != n:
            c.fail(f"flat array of {n} values is not square")
        else:
            arr = arr.reshape(side, side)

    if arr.ndim != 2:
        c.fail(f"expected a 2-D frame, got shape {arr.shape}")
        return c

    if arr.shape != (C.FRAME_H, C.FRAME_W):
        c.fail(f"frame is {arr.shape}, DUT requires "
               f"({C.FRAME_H}, {C.FRAME_W}); resampling would invalidate the "
               f"calibration matrix, so this is refused")

    vmax = float(np.nanmax(arr)) if arr.size else 0.0
    if vmax > 255:
        c.fail(f"peak value {vmax:.0f} exceeds 8-bit range; the DUT input is "
               f"8-bit unsigned. Rescaling changes the threshold behaviour "
               f"(PIXEL_THRESHOLD={C.PIXEL_THRESHOLD} ADU) and must be a "
               f"documented decision, not an automatic one")
    if vmax <= C.PIXEL_THRESHOLD:
        c.fail(f"peak value {vmax:.0f} is at or below the pixel threshold "
               f"({C.PIXEL_THRESHOLD}); every pixel would be zeroed")

    if bit_depth is not None and bit_depth != 8:
        c.warn(f"source is {bit_depth}-bit; the DUT consumes 8-bit. The "
               f"conversion must be stated in the paper")

    c.warn("The reconstruction matrix e_matrix_{x,y}.mem and the reference "
           "slopes slopes_ref_{x,y}.mem are calibrated for THIS optical setup. "
           "Unless the external data comes from an optically equivalent system, "
           "reconstructed coefficients are not meaningful even when the frame "
           "size matches.")
    c.warn("subaperture_bitmap.mem defines 168 valid subapertures at fixed "
           "positions. Confirm the external pupil matches before quoting "
           "reconstruction accuracy.")
    return c


def evaluate(frame: np.ndarray, truth: np.ndarray | None = None,
             float_only: bool = False) -> dict:
    """Run the available analyses on a compatible frame."""
    px = np.asarray(frame, dtype=np.int64).ravel()
    out: dict = {}

    fl = BA.run_float_reference(px)
    out["float_zernike_m"] = fl["zernike_m"]

    if not float_only:
        fx = BA.run_pipeline(px)
        out["fixed_zernike_m"] = fx["zernike_m"]
        out["fixed_vs_float_rmse_m"] = float(
            np.sqrt(np.mean((fx["zernike_m"] - fl["zernike_m"]) ** 2)))

    if truth is not None:
        t = np.asarray(truth, dtype=np.float64)
        if t.size == C.NUM_MODES:
            out["float_vs_truth_rmse_m"] = float(
                np.sqrt(np.mean((fl["zernike_m"] - t) ** 2)))
            if not float_only:
                out["fixed_vs_truth_rmse_m"] = float(
                    np.sqrt(np.mean((out["fixed_zernike_m"] - t) ** 2)))
        else:
            out["truth_error"] = (f"ground truth has {t.size} values, "
                                  f"expected {C.NUM_MODES}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--truth", type=Path)
    ap.add_argument("--bit-depth", type=int)
    ap.add_argument("--force-float-only", action="store_true",
                    help="run only the float centroid path (geometry differs)")
    a = ap.parse_args()

    frame = load_frame(a.frame)
    c = check(frame, bit_depth=a.bit_depth)

    print(f"Compatibility check for {a.frame}")
    for w in c.warnings:
        print(f"  WARNING: {w}")
    for f in c.failures:
        print(f"  FAIL   : {f}")

    if not c.ok:
        print("\nREFUSED: the data is not compatible with the DUT geometry.")
        print("Resampling to fit would invalidate the calibration matrix and "
              "produce meaningless coefficients.")
        print("See docs/EVALUATION_REPORT.md section 4.10.")
        sys.exit(2)

    truth = np.loadtxt(a.truth) if a.truth else None
    res = evaluate(frame, truth, float_only=a.force_float_only)
    print("\nResults:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    if a.force_float_only:
        print("\nNOTE: float-only mode. This is a centroiding sanity check, "
              "NOT validation of the RTL.")


if __name__ == "__main__":
    main()
