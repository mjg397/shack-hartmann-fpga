#!/usr/bin/env python3
"""
Generate the HCIPy golden model for the Zynq pipeline comparison.

Mirrors the headless call pattern in fpga_gui_service._build_simulation_bundle
(demo_image_path=None skips the VLT scene assets, which are not in this tree).

Critically, this also GATES the comparison: it quantizes the aberrated detector
image exactly as the FPGA input was produced and diffs it against the .mem file
the simulation actually streams. If those differ, the golden model describes a
different wavefront than the RTL sees and any coefficient comparison is
meaningless -- so the mismatch is reported loudly rather than silently ignored.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from shwfs_utils import (  # noqa: E402
    generate_shwfs_case,
    run_hcipy_estimation,
    run_fpga_like_estimation,
    quantize_shwfs_image,
)

NUM_LENSLETS = 16
NUM_ZERNIKE = 10

STREAMED_IMAGE = HERE.parent / "rtl" / "hcipy_image_rotating.mem"
OUT_DIR = HERE / "build"


def load_streamed_image() -> np.ndarray:
    tokens = STREAMED_IMAGE.read_text().split()
    return np.array([int(t, 16) for t in tokens], dtype=np.uint8)


def main() -> None:
    print("Generating HCIPy case (this takes a moment)...")
    simulation = generate_shwfs_case(
        true_coeffs=None,
        num_lenslets=NUM_LENSLETS,
        num_zernike=NUM_ZERNIKE,
        demo_image_path=None,
    )

    estimation = run_hcipy_estimation(
        image=simulation["image_aber"],
        estimator=simulation["estimator"],
        reference_slopes=simulation["slopes_ref"],
        reconstruction_matrix=simulation["reconstruction_matrix"],
        zernike_modes=simulation["zernike_modes"],
        aperture=simulation["aperture"],
        measured_opd_field=simulation["input_opd_field"],
        shwfs=simulation["shwfs"],
    )

    hcipy_coeffs = np.asarray(estimation["estimated_coeffs"], dtype=np.float64)
    true_coeffs = np.asarray(simulation["true_coeffs"], dtype=np.float64)
    labels = list(simulation["mode_labels"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Image equality gate -------------------------------------------------
    generated = np.asarray(quantize_shwfs_image(simulation["image_aber"])).ravel()
    streamed = load_streamed_image()

    print()
    print("Image equality gate:")
    print(f"  generated : {generated.size} px, max={generated.max()}")
    print(f"  streamed  : {streamed.size} px, max={streamed.max()}")

    images_match = False
    if generated.size != streamed.size:
        print("  MISMATCH: different pixel counts")
    else:
        diff = generated.astype(int) - streamed.astype(int)
        n_diff = int(np.count_nonzero(diff))
        images_match = n_diff == 0
        if images_match:
            print("  MATCH: generated image is identical to the streamed .mem")
        else:
            print(f"  MISMATCH: {n_diff} of {diff.size} pixels differ "
                  f"({100.0 * n_diff / diff.size:.1f}%), "
                  f"max |delta| = {int(np.abs(diff).max())}")

    # ---- Host-side FPGA mirror, for isolating front-end vs back-end ---------
    # run_fpga_like_estimation reproduces the FPGA's CoG centroid and slope
    # procedure in Python. Comparing its centroids against the RTL's separates
    # a centroid/slope bug from an E-matrix reconstruction bug.
    host = run_fpga_like_estimation(simulation["image_aber"], NUM_LENSLETS, 16)
    np.savetxt(OUT_DIR / "host_centroids_q4_23.txt",
               host["centroids_q4_23"], fmt="%d", header="x y", comments="")
    np.savetxt(OUT_DIR / "host_slopes_q4_23.txt",
               host["slopes_q4_23"], fmt="%d", header="x y", comments="")
    if host["estimated_coeffs"] is not None:
        np.savetxt(OUT_DIR / "host_coeffs.txt", np.asarray(host["estimated_coeffs"]),
                   fmt="%.12e", header="host_coeff_meters", comments="")

    np.savetxt(OUT_DIR / "hcipy_reference.txt",
               hcipy_coeffs, fmt="%.12e", header="hcipy_coeff_meters", comments="")
    np.savetxt(OUT_DIR / "true_coeffs.txt",
               true_coeffs, fmt="%.12e", header="true_coeff_meters", comments="")
    (OUT_DIR / "mode_labels.txt").write_text("\n".join(labels) + "\n")
    (OUT_DIR / "image_gate.txt").write_text(
        ("MATCH" if images_match else "MISMATCH") + "\n"
    )

    print()
    print(f"{'mode':<12} {'hcipy (m)':>16} {'true (m)':>16}")
    for label, h, t in zip(labels, hcipy_coeffs, true_coeffs):
        print(f"{label:<12} {h:16.3e} {t:16.3e}")
    print()
    print(f"wrote {OUT_DIR/'hcipy_reference.txt'}")

    if not images_match:
        print()
        print("WARNING: the golden model and the streamed image differ.")
        print("         Coefficient comparison will be reported but is NOT valid")
        print("         as an accuracy measure -- they describe different scenes.")


if __name__ == "__main__":
    main()
