#!/usr/bin/env python3
"""
Generate directed test frames for the SystemVerilog verification environment.

The randomised frames are built inside the testbench in SystemVerilog and stress
arithmetic corners. These directed frames are the complement: physically
meaningful wavefronts produced by the repository's own optical forward model
(pinv of the E matrix), so the DUT is exercised on the data it will actually
see, not just on numerically extreme inputs.

Cases:
    zero_slope        spots exactly on their reference positions
    near_zero_slope   sub-LSB offsets -- the sign-change boundary
    pos_slope         all-positive displacement
    neg_slope         all-negative displacement
    large_offset      large but still in-tile
    pure_mode_N       one Zernike mode at a time (10 frames)
    dim_spots         intensity just above PIXEL_THRESHOLD
    bright_spots      pixels driven to saturation
    dark_frame        zero flux -- exercises the sI == 0 reciprocal path

Writes each frame as a .mem and a manifest the testbench reads via +TESTS=.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.common import config as C            # noqa: E402
from evaluation.common import framegen as FG         # noqa: E402


def write_mem(path: Path, frame: np.ndarray) -> None:
    path.write_text("".join(f"{int(v):02X}\n" for v in np.asarray(frame).ravel()))


def build(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = C.NUM_SUBAPS_VALID
    zeros = np.zeros(n)
    cases: list[tuple[str, np.ndarray]] = []

    cases.append(("zero_slope", FG.render_frame(zeros, zeros)))

    eps = 0.5 / (1 << C.CENTROID_FRAC)
    cases.append(("near_zero_slope",
                  FG.render_frame(np.full(n, eps), np.full(n, -eps))))

    cases.append(("pos_slope", FG.render_frame(np.full(n, 2.5), np.full(n, 2.5))))
    cases.append(("neg_slope", FG.render_frame(np.full(n, -2.5), np.full(n, -2.5))))
    cases.append(("large_offset",
                  FG.render_frame(np.full(n, 4.5), np.full(n, -4.0))))

    for m in range(C.NUM_MODES):
        coeffs = np.zeros(C.NUM_MODES)
        coeffs[m] = 5e-8
        sx, sy = FG.coeffs_to_slopes(coeffs)
        cases.append((f"pure_mode_{m+1}", FG.render_frame(sx, sy)))

    cases.append(("dim_spots", FG.render_frame(zeros, zeros, spot_sum=8.0 * 16)))
    cases.append(("bright_spots", FG.render_frame(zeros, zeros, spot_sum=4000.0)))
    cases.append(("dark_frame",
                  np.zeros(C.FRAME_W * C.FRAME_H, dtype=np.int64)))

    names = []
    for name, frame in cases:
        write_mem(out_dir / f"{name}.mem", frame)
        names.append(f"{name}.mem")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="work directory the simulator runs in")
    a = ap.parse_args()
    names = build(a.out)
    (a.out / "directed.f").write_text("\n".join(names) + "\n")
    print(f"generated {len(names)} directed frames -> {a.out}/directed.f")
    for nm in names:
        print(f"  {nm}")


if __name__ == "__main__":
    main()
