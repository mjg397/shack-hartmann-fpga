#!/usr/bin/env python3
"""
STEP 3 -- Error source ablation.

Question: WHICH fixed-point stage causes the observed error, and would a more
precise reciprocal materially improve final accuracy?

This is a SOFTWARE-ONLY ablation. The RTL is never modified. Each variant takes
the bit-accurate model and replaces exactly one quantisation step with exact
float arithmetic, leaving every other stage bit-accurate. The drop in error
attributable to that stage is then the difference from the all-fixed baseline.

Variants
    all_fixed              the shipping design (baseline)
    exact_reciprocal       LUT replaced by exact 1/sI, everything else fixed.
                           This is a BOUND, not a proposal: the reciprocal feeds
                           unsigned_mult's 18-bit `b` port, which is the DSP48E1
                           B-port width, so any replacement must also fit 18 bits.
    exact_centroid_mult    centroid product kept in full precision
    exact_slope            slope kept in float (no 25-bit truncation)
    exact_ematrix          E-matrix coefficients kept in float (no Q1.16)
    exact_output           output kept in full precision (no acc[42:18] slice)
    all_float              everything exact -- lower bound

Note on the accumulator: the thresholded CoG sums are EXACT integers in the RTL
(no rounding), so there is no accumulator quantisation to ablate. That is itself
a result and is reported as such rather than silently omitted.

Outputs
    results/error_ablation.csv
    results/error_ablation.tex
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
E_SCALE = float(1 << C.EMATRIX_FRAC)


def pipeline(frame, *, recip="lut", centroid_mult="fixed", slope="fixed",
             ematrix="fixed", output="fixed"):
    """
    Configurable pipeline. Every knob defaults to the shipping behaviour, so
    pipeline(frame) reproduces the bit-accurate model exactly.
    """
    sI, xI, yI = BA.accumulate(frame)          # exact integers in RTL too
    idx = BA.valid_subaps()
    sI_f = sI.astype(np.float64)

    # ---- reciprocal ----
    if recip == "lut":                          # shipping Q18 LUT, 18-bit port
        _, rec_port = BA.reciprocal(sI)
        rec = rec_port.astype(np.float64) / (1 << C.CENTROID_FRAC)
    elif recip == "exact":
        with np.errstate(divide="ignore", invalid="ignore"):
            rec = np.where(sI_f > 0, 1.0 / sI_f, 0.0)
    else:
        raise ValueError(recip)

    # ---- centroid product ----
    if centroid_mult == "fixed" and recip == "lut":
        cx = BA.centroid(xI, BA.reciprocal(sI)[1]).astype(np.float64) / SCALE_PX
        cy = BA.centroid(yI, BA.reciprocal(sI)[1]).astype(np.float64) / SCALE_PX
    else:
        cx = xI.astype(np.float64) * rec
        cy = yI.astype(np.float64) * rec

    # ---- slope ----
    ref_x, ref_y = BA.slopes_ref()
    rx = ref_x[: C.NUM_SUBAPS_GRID] / SCALE_PX
    ry = ref_y[: C.NUM_SUBAPS_GRID] / SCALE_PX
    if slope == "fixed" and centroid_mult == "fixed" and recip == "lut":
        sx = BA.slope(BA.centroid(xI, BA.reciprocal(sI)[1]),
                      ref_x[: C.NUM_SUBAPS_GRID]).astype(np.float64) / SCALE_PX
        sy = BA.slope(BA.centroid(yI, BA.reciprocal(sI)[1]),
                      ref_y[: C.NUM_SUBAPS_GRID]).astype(np.float64) / SCALE_PX
    else:
        sx, sy = cx - rx, cy - ry

    # ---- reconstruction ----
    ex, ey = BA.ematrix()
    exf, eyf = ex / E_SCALE, ey / E_SCALE
    if ematrix == "dither":
        # The pre-rounding E coefficients are NOT recoverable from the .mem
        # files, so "exact E" cannot be constructed. Instead perturb E by
        # +/-0.5 LSB of Q1.16 -- the rounding interval it was quantised into.
        # The resulting spread BOUNDS the contribution of E quantisation.
        rs = np.random.default_rng(0)
        exf = exf + (rs.random(exf.shape) - 0.5) / E_SCALE
        eyf = eyf + (rs.random(eyf.shape) - 0.5) / E_SCALE

    zern = exf @ sx[idx] + eyf @ sy[idx]
    zern_m = zern * (1 << C.CENTROID_FRAC) * (1 << C.EMATRIX_FRAC) \
        * (2.0 ** -C.ZERNIKE_SLICE_LOW) * C.METERS_PER_LSB

    # ---- output quantisation ----
    if output == "fixed":
        raw = np.rint(zern_m / C.METERS_PER_LSB)
        raw = np.clip(raw, -(1 << 24), (1 << 24) - 1)
        zern_m = raw * C.METERS_PER_LSB

    return {"cx": cx[idx], "cy": cy[idx], "sx": sx[idx], "sy": sy[idx],
            "zern_m": zern_m}


VARIANTS = [
    ("all_fixed", {}),
    ("exact_reciprocal", {"recip": "exact", "centroid_mult": "exact",
                          "slope": "exact"}),
    ("exact_slope_only", {"slope": "exact", "centroid_mult": "exact"}),
    ("exact_output_only", {"output": "exact"}),
    ("ematrix_dither_pm_half_lsb", {"ematrix": "dither"}),
    ("all_float", {"recip": "exact", "centroid_mult": "exact",
                   "slope": "exact", "output": "exact"}),
]


def run(n_cases: int, seed: int, out_dir: Path) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = BA.valid_subaps()

    cases = []
    for _ in range(n_cases):
        coeffs = FG.random_coefficients(rng, 6e-8)
        sx, sy = FG.coeffs_to_slopes(coeffs)
        cases.append((FG.render_frame(sx, sy), coeffs, sx, sy))

    ref_x, ref_y = BA.slopes_ref()
    rows = []
    for name, kw in VARIANTS:
        cen_err, zern_err = [], []
        for frame, coeffs, tsx, tsy in cases:
            r = pipeline(frame, **kw)
            tcx = ref_x[idx] / SCALE_PX + tsx
            tcy = ref_y[idx] / SCALE_PX + tsy
            cen_err.append(np.concatenate([r["cx"] - tcx, r["cy"] - tcy]))
            zern_err.append(r["zern_m"] - coeffs)
        cen_err = np.concatenate(cen_err)
        zern_err = np.concatenate(zern_err)
        cen_rms = float(np.sqrt(np.mean(cen_err ** 2)))
        rows.append({
            "variant": name,
            "centroid_rms_px": cen_rms,
            "centroid_wavefront_nm": C.centroid_px_to_wavefront_m(cen_rms) * 1e9,
            "zernike_rmse_m": float(np.sqrt(np.mean(zern_err ** 2))),
            "zernike_rmse_nm": float(np.sqrt(np.mean(zern_err ** 2))) * 1e9,
        })
        print(f"  {name:24s} centroid {cen_rms:.6f} px   "
              f"zernike {rows[-1]['zernike_rmse_nm']:7.3f} nm", flush=True)

    base = rows[0]
    floor = rows[-1]
    for r in rows:
        r["centroid_vs_baseline"] = r["centroid_rms_px"] / base["centroid_rms_px"]
        r["zernike_vs_baseline"] = r["zernike_rmse_m"] / base["zernike_rmse_m"]

    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (out_dir / "error_ablation.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.10e}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")

    lines = [r"% Error source ablation. Software-only; the RTL is unmodified.",
             r"\begin{tabular}{lrrr}", r"\hline",
             r"Variant & Centroid RMS (px) & Zernike RMSE (nm) & vs baseline \\",
             r"\hline"]
    for r in rows:
        lines.append(rf"{r['variant'].replace('_',' ')} & "
                     rf"{r['centroid_rms_px']:.5f} & "
                     rf"{r['zernike_rmse_nm']:.3f} & "
                     rf"{r['centroid_vs_baseline']:.3f} \\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_dir / "error_ablation.tex").write_text("\n".join(lines) + "\n")

    print()
    print(f"  baseline (all fixed)  : {base['centroid_rms_px']:.6f} px")
    print(f"  floor    (all float)  : {floor['centroid_rms_px']:.6f} px")
    recip = next(r for r in rows if r["variant"] == "exact_reciprocal")
    gain = base["centroid_rms_px"] / recip["centroid_rms_px"]
    print(f"  an EXACT reciprocal would improve centroid RMS by {gain:.2f}x")
    print("  (upper bound only -- any real replacement must fit the 18-bit")
    print("   DSP48 B port, so this gain is not fully attainable)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--cases", type=int, default=500)
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED + 20)
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()
    print(f"Error ablation: {a.cases} cases x {len(VARIANTS)} variants")
    run(a.cases, a.seed, a.out)


if __name__ == "__main__":
    main()
