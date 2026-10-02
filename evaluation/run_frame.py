#!/usr/bin/env python3
"""
Run ONE frame through the hardware, from any source.

This is the single entry point for testing individual images, whether they come
from HCIPy, from a stored file, or from an external dataset. It drives the real
RTL in simulation and checks it against both software models.

    # the repository's own demonstration frame
    run_frame.py --demo

    # generate a fresh HCIPy frame (slow: rebuilds the optical model, ~40 s)
    run_frame.py --hcipy
    run_frame.py --hcipy --coeffs 6e-8,4e-8,5e-8,3e-8,-2e-8,0,0,0,0,0

    # a fast synthetic frame from the repository's own forward model
    run_frame.py --synth --seed 7

    # any stored frame (.mem / .hex / .npy / .npz / .txt)
    run_frame.py --frame myframe.mem

    # an EXTERNAL dataset frame -- the compatibility gate applies
    run_frame.py --frame real.npy --external
    run_frame.py --frame real.npy --external --float-only

What it reports
    compatibility   (external frames only) geometry / bit-depth gate
    equivalence     RTL vs the bit-accurate model -- must be EXACT
    quantisation    bit-accurate fixed point vs float64, same algorithm
    accuracy        vs ground truth, when the source provides one
    latency         final-pixel-to-result, in cycles and ns

External frames without ground truth still give the first three. That is
numerical validation on real spot profiles, which is worth having -- it is not
an accuracy claim. See EVALUATION_REPORT.md section 4.10.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common import bitaccurate as BA          # noqa: E402
from evaluation.common import config as C                # noqa: E402
from evaluation.common import framegen as FG             # noqa: E402
from evaluation.common import rtl_runner as R            # noqa: E402
from evaluation.real_data import load_external as EXT     # noqa: E402


# ---------------------------------------------------------------- sources ----
def source_demo() -> tuple[np.ndarray, np.ndarray | None, str]:
    path = C.RTL_DIR / "hcipy_image_rotating.mem"
    px = np.array([int(t, 16) for t in path.read_text().split()], dtype=np.int64)
    return px, None, f"demo frame ({path.name})"


def source_synth(seed: int, scale: float) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    case = FG.make_case(rng, scale_m=scale)
    return (case["frame"].ravel(), case["true_coeffs_m"],
            f"synthetic forward-model frame (seed {seed})")


def source_hcipy(coeffs: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate a frame through the full HCIPy optical model."""
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(C.SRC_DIR))
    from shwfs_utils import generate_aberrated_image, quantize_shwfs_image

    print("  generating via HCIPy (rebuilds the optical model, ~40 s)...")
    img = generate_aberrated_image(
        true_coeffs=coeffs, num_lenslets=C.NUM_LENSLETS,
        num_zernike=C.NUM_MODES, demo_image_path=None)
    px = np.asarray(quantize_shwfs_image(img)).ravel().astype(np.int64)
    used = coeffs if coeffs is not None else None
    return px, used, "HCIPy-generated frame"


def source_file(path: Path, external: bool) -> tuple[np.ndarray, None, str]:
    arr = EXT.load_frame(path)
    if external:
        c = EXT.check(arr)
        print("  compatibility gate:")
        for w in c.warnings:
            print(f"    WARNING {w}")
        for f in c.failures:
            print(f"    FAIL    {f}")
        if not c.ok:
            print("\n  REFUSED: incompatible with the DUT geometry. Resampling to")
            print("  fit would invalidate the calibration matrix and produce")
            print("  meaningless coefficients. See EVALUATION_REPORT.md 4.10.")
            sys.exit(2)
        print("    PASS")
    return np.asarray(arr, dtype=np.int64).ravel(), None, f"file {path.name}"


# ------------------------------------------------------------------ main -----
def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", action="store_true")
    g.add_argument("--hcipy", action="store_true")
    g.add_argument("--synth", action="store_true")
    g.add_argument("--frame", type=Path)
    ap.add_argument("--external", action="store_true",
                    help="apply the external-data compatibility gate")
    ap.add_argument("--float-only", action="store_true",
                    help="float centroiding only -- a sanity check, NOT RTL validation")
    ap.add_argument("--coeffs", type=str,
                    help="comma-separated Zernike coefficients in metres (--hcipy)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=6e-8)
    ap.add_argument("--truth", type=Path, help="ground-truth coefficients file")
    ap.add_argument("--json", type=Path, help="write results as JSON")
    a = ap.parse_args()

    print()
    print("=" * 66)
    print(" SHWFS single-frame test")
    print("=" * 66)

    coeffs = None
    if a.coeffs:
        coeffs = np.array([float(x) for x in a.coeffs.split(",")], dtype=np.float64)

    if a.demo:
        px, truth, label = source_demo()
    elif a.synth:
        px, truth, label = source_synth(a.seed, a.scale)
    elif a.hcipy:
        px, truth, label = source_hcipy(coeffs)
    else:
        px, truth, label = source_file(a.frame, a.external)

    if a.truth:
        truth = np.loadtxt(a.truth, dtype=np.float64).ravel()

    if px.size != C.FRAME_W * C.FRAME_H:
        print(f"  frame has {px.size} pixels, DUT needs {C.FRAME_W*C.FRAME_H}")
        sys.exit(2)

    nz = int((px > 0).sum())
    print(f"  source            : {label}")
    print(f"  frame             : {C.FRAME_W}x{C.FRAME_H}, "
          f"max={px.max()}, {nz} non-zero pixels")
    print()

    res: dict = {"source": label, "max_adu": int(px.max()), "nonzero_px": nz}

    # ---- software models ----
    fl = BA.run_float_reference(px)
    res["float_zernike_m"] = [float(v) for v in fl["zernike_m"]]

    if a.float_only:
        print("  FLOAT-ONLY MODE -- centroiding sanity check, not RTL validation")
        print(f"  float Zernike (m): "
              f"{np.array2string(fl['zernike_m'], precision=3)}")
        if a.json:
            a.json.write_text(json.dumps(res, indent=2))
        return

    fx = BA.run_pipeline(px)
    res["fixed_zernike_raw"] = [int(v) for v in fx["zernike_raw"]]
    res["fixed_zernike_m"] = [float(v) for v in fx["zernike_m"]]

    # ---- RTL ----
    print("  running RTL simulation...")
    with R.RTLSimulator() as sim:
        rtl = sim.run_frame(px)

    exact = bool(np.array_equal(rtl["zernike_raw"], fx["zernike_raw"]))
    res["rtl_zernike_raw"] = [int(v) for v in rtl["zernike_raw"]]
    res["rtl_matches_model"] = exact
    res["latency_cycles"] = rtl["true_latency_cycles"]
    res["latency_ns"] = rtl["true_latency_cycles"] * C.CLOCK_NS
    res["pixel_mismatches"] = rtl["pixel_mismatches"]

    q = fx["zernike_m"] - fl["zernike_m"]
    res["quantisation_rmse_m"] = float(np.sqrt(np.mean(q ** 2)))

    print()
    print(f"  {'EQUIVALENCE':<18} RTL vs bit-accurate model : "
          f"{'EXACT' if exact else 'MISMATCH'}")
    print(f"  {'':<18} pixel-stream mismatches   : {rtl['pixel_mismatches']}")
    print(f"  {'QUANTISATION':<18} fixed vs float RMSE       : "
          f"{res['quantisation_rmse_m']:.4e} m")
    print(f"  {'LATENCY':<18} final pixel -> result     : "
          f"{rtl['true_latency_cycles']} cycles "
          f"({res['latency_ns']:.1f} ns @ {C.CLOCK_MHZ:.0f} MHz)")

    if truth is not None and len(truth) == C.NUM_MODES:
        e_fx = fx["zernike_m"] - truth
        e_fl = fl["zernike_m"] - truth
        res["accuracy_fixed_rmse_m"] = float(np.sqrt(np.mean(e_fx ** 2)))
        res["accuracy_float_rmse_m"] = float(np.sqrt(np.mean(e_fl ** 2)))
        print(f"  {'ACCURACY':<18} vs ground truth, fixed    : "
              f"{res['accuracy_fixed_rmse_m']:.4e} m")
        print(f"  {'':<18} vs ground truth, float    : "
              f"{res['accuracy_float_rmse_m']:.4e} m")
    else:
        print(f"  {'ACCURACY':<18} no ground truth supplied  : "
              f"numerical checks only, no accuracy claim")

    print()
    print(f"  {'mode':<12}{'RTL (raw)':>12}{'fixed (m)':>14}{'float (m)':>14}")
    for i in range(C.NUM_MODES):
        print(f"  {C.MODE_LABELS[i]:<12}{int(rtl['zernike_raw'][i]):>12}"
              f"{fx['zernike_m'][i]:>14.4e}{fl['zernike_m'][i]:>14.4e}")

    print()
    ok = exact and rtl["pixel_mismatches"] == 0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 66)
    print()

    if a.json:
        a.json.write_text(json.dumps(res, indent=2))
        print(f"  wrote {a.json}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
