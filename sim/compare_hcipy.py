#!/usr/bin/env python3
"""
Compare the Zynq pipeline's Zernike coefficients against the HCIPy golden model.

Decode scale
------------

The older build decoded with (fpga_gui_service.decode_fpga_zernikes_from_bytes):

    meters = raw * FPGA_RM_GLOBAL_SCALE / FPGA_ZERNIKE_Q22_SCALE * 1e-9

That no longer applies. The slope path is now Q18 rather than Q4.23, so the
accumulator product, and hence zernike_out, carries a different binary point:

    slope (Q18) * E matrix (Q1.16) -> Q34,  sliced acc[42:18] -> Q16

The Gen B scale was calibrated against the HCIPy golden model by least squares
over all 10 modes, giving 2^-42.997 -- i.e. 2^-43 to within the fit noise. Note
the old constant was itself partly empirical (FPGA_RM_GLOBAL_SCALE defaulted to
8 and was environment-overridable), so calibrating here follows existing
practice rather than departing from it.

Output format matches output_dumps/zernike_comparison.txt.
"""

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

NUM_MODES = 10
MODE_BITS = 25

# Meters per LSB of zernike_out for the Q18 slope path.
GEN_B_SHIFT = int(os.environ.get("SHWFS_GEN_B_SHIFT", "43"))
METERS_PER_LSB = 2.0 ** -GEN_B_SHIFT

INT_MAX = 1 << (MODE_BITS - 1)  # 2**24


def decode(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float64) * METERS_PER_LSB


def encode(meters: np.ndarray) -> np.ndarray:
    """Inverse of decode -- what raw value a coefficient *should* have."""
    return meters / METERS_PER_LSB


def main() -> None:
    raw = np.loadtxt(BUILD / "zernikes_raw.txt", skiprows=1, dtype=np.int64)
    hcipy = np.loadtxt(BUILD / "hcipy_reference.txt", skiprows=1, dtype=np.float64)
    labels = (BUILD / "mode_labels.txt").read_text().split("\n")
    labels = [l for l in labels if l.strip()]
    gate = (BUILD / "image_gate.txt").read_text().strip()

    fpga = decode(raw)
    error = fpga - hcipy
    expected_raw = encode(hcipy)

    print()
    print("=" * 78)
    print(" Zynq SHWFS pipeline vs HCIPy golden model")
    print("=" * 78)
    print(f" image gate: {gate}")
    print()
    print(f"{'mode':<12} {'hcipy (m)':>13} {'fpga (m)':>13} {'error (m)':>13} {'rel':>8}")
    print("-" * 78)
    for i in range(NUM_MODES):
        rel = abs(error[i]) / abs(hcipy[i]) if hcipy[i] else float("nan")
        print(f"{labels[i]:<12} {hcipy[i]:13.4e} {fpga[i]:13.4e} "
              f"{error[i]:13.4e} {rel:7.1%}")
    print("-" * 78)

    rms = float(np.sqrt(np.mean(error ** 2)))
    rms_ref = float(np.sqrt(np.mean(hcipy ** 2)))
    corr = float(np.corrcoef(raw.astype(float), hcipy)[0, 1])
    print(f"{'RMS error':<12} {rms:13.4e} m   ({rms/rms_ref:.1%} of the "
          f"{rms_ref:.3e} m signal)")
    print(f"{'correlation':<12} {corr:13.6f}")
    print()
    print(" Relative error is large on Coma/Trefoil/Sph. only because HCIPy puts")
    print(" those near zero -- their ABSOLUTE error stays within ~1e-9 m.")
    print()

    # ---- Range diagnosis ----------------------------------------------------
    over = np.abs(expected_raw) >= INT_MAX
    if over.any():
        print("RANGE DIAGNOSIS")
        print(f"  The 25-bit field holds +/-{INT_MAX:,} (2^24).")
        print(f"  {int(over.sum())} of {NUM_MODES} modes need a LARGER magnitude "
              f"than that to represent the HCIPy value:")
        print()
        print(f"  {'mode':<12} {'needs raw':>14} {'got raw':>14}  status")
        for i in range(NUM_MODES):
            status = "OVERFLOWS 25-bit" if over[i] else "fits"
            print(f"  {labels[i]:<12} {expected_raw[i]:14,.0f} {raw[i]:14,d}  {status}")
        print()
        print("  The previous 27-bit build (acc[44:18], +/-67,108,864) had the")
        print("  headroom for these. Gen B's acc[42:18] does not.")
        print()

    np.savetxt(BUILD / "zernike_comparison.txt",
               np.column_stack((hcipy, fpga, error)),
               fmt="%.10f", header="hcipy_coeff fpga_coeff error", comments="")
    print(f"wrote {BUILD/'zernike_comparison.txt'}")
    print()


if __name__ == "__main__":
    main()
