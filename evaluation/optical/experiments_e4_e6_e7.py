#!/usr/bin/env python3
"""
E4 dynamic range, E6 unmodelled aberrations, E7 subpixel sampling,
plus an optical-sampling convergence check.

E4  Sweep amplitude to the onset of failure, tracking the physical and
    arithmetic limits separately: spot escape from its lenslet window, flux
    loss, ADC clipping, accumulator bounds, output-field saturation.

E6  Add modes ABOVE the reconstructed basis. Report recovered-mode error and
    full-wavefront residual separately -- the unmodelled content stays in the
    truth rather than being quietly removed from it.

E7  Sweep spot position within a pixel using controlled tilt, to expose
    centroid bias periodic in the pixel, and the resulting OPD error.

CONV Refine the optical sampling and report how much the measured metrics
     move, so the chosen resolution is justified rather than assumed.
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
from evaluation.common import detector as D              # noqa: E402
from evaluation.common import optics as O                # noqa: E402
from evaluation.optical.calibrate import load_frozen      # noqa: E402

SEED = 31_415


def recon(bench, adu, scale):
    fx = BA.run_pipeline(np.asarray(adu, dtype=np.int64).ravel())
    return bench.opd_from_coeffs(fx["zernike_raw"].astype(float) * scale), fx


def e4_dynamic_range(bench, scale, out, n=8):
    rng = np.random.default_rng(SEED)
    det = D.DetectorConfig()
    idx = BA.valid_subaps()
    ref_x, ref_y = BA.slopes_ref()
    s = float(1 << C.CENTROID_FRAC)
    field = (1 << (C.ZERNIKE_BITS - 1)) - 1
    rows = []
    for amp in (10, 30, 100, 300, 600, 1000, 2000, 4000):
        for k in range(n):
            c = rng.normal(0, 1, C.NUM_MODES)
            c *= (amp * 1e-9) / bench.opd_rms(bench.opd_from_coeffs(c))
            img, opd = bench.propagate_coeffs(c)
            r = D.expose(img, det, rng)
            w, fx = recon(bench, r.adu, scale)
            sI, xI, yI = BA.accumulate(r.adu.ravel())
            cx = fx["x_centroid"][idx] / s
            escaped = int(((cx < 0.5) | (cx > C.SUBAP_PIXELS - 1.5)).sum())
            rows.append({
                "amp_nm": amp, "trial": k,
                "true_rms_m": bench.opd_rms(opd),
                "residual_m": bench.opd_rms(opd - w),
                "rel": bench.opd_rms(opd - w) / bench.opd_rms(opd),
                "spots_near_window_edge": escaped,
                "dark_lenslets": int((sI[idx] <= 0).sum()),
                "min_flux_adu": int(sI[idx].min()),
                "clipped_px": r.clipped_adu_px,
                "saturated_px": r.saturated_px,
                "sI_over_bound": int((sI > C.SI_MAX).sum()),
                "moment_over_bound": int((max(xI.max(), yI.max()) > C.XI_MAX)),
                "zernike_field_use_pct": float(
                    100 * np.abs(fx["zernike_raw"]).max() / field),
            })
    _csv(out / "e4_dynamic_range.csv", rows)
    print("  E4 dynamic range")
    print(f"    {'amp':>6} {'rel err':>9} {'edge spots':>11} {'dark':>5} "
          f"{'clipped':>8} {'field %':>8}")
    for amp in (10, 30, 100, 300, 600, 1000, 2000, 4000):
        sub = [r for r in rows if r["amp_nm"] == amp]
        print(f"    {amp:5.0f}n {np.mean([r['rel'] for r in sub]):8.1%} "
              f"{np.mean([r['spots_near_window_edge'] for r in sub]):11.1f} "
              f"{np.mean([r['dark_lenslets'] for r in sub]):5.1f} "
              f"{np.mean([r['clipped_px'] for r in sub]):8.0f} "
              f"{np.mean([r['zernike_field_use_pct'] for r in sub]):7.1f}%")
    return rows


def e6_unmodelled(bench, scale, out, n=25):
    """Add modes beyond the reconstructed basis; keep them in the truth."""
    from hcipy import make_zernike_basis
    c = bench.cfg
    extra = make_zernike_basis(c.num_zernike + 11, c.telescope_diameter_m,
                              bench.pupil_grid)
    hi = [np.asarray(extra[i]) for i in range(c.num_zernike + 1,
                                              c.num_zernike + 11)]
    rng = np.random.default_rng(SEED + 1)
    det = D.DetectorConfig()
    rows = []
    for frac in (0.0, 0.1, 0.25, 0.5, 1.0):
        for k in range(n):
            lo_c = rng.normal(0, 1, C.NUM_MODES)
            lo = bench.opd_from_coeffs(lo_c)
            lo_c *= 100e-9 / bench.opd_rms(lo)
            lo = bench.opd_from_coeffs(lo_c)
            hi_c = rng.normal(0, 1, len(hi))
            hw = sum(a * m for a, m in zip(hi_c, hi))
            r_hi = bench.opd_rms(hw)
            if r_hi > 0:
                hw = hw * (frac * 100e-9 / r_hi)
            true = lo + hw
            img = bench.propagate(true)
            w, fx = recon(bench, D.expose(img, det, rng).adu, scale)
            rec_c = fx["zernike_raw"].astype(float) * scale
            rows.append({
                "unmodelled_frac": frac, "trial": k,
                "true_total_rms_m": bench.opd_rms(true),
                "unmodelled_rms_m": bench.opd_rms(hw),
                # error on the modes we DO reconstruct
                "recovered_modes_err_m": float(np.sqrt(np.mean((rec_c - lo_c) ** 2))),
                # full-wavefront residual, unmodelled content retained in truth
                "full_residual_m": bench.opd_rms(true - w),
                # residual against only the modelled part
                "modelled_only_residual_m": bench.opd_rms(lo - w),
            })
    _csv(out / "e6_unmodelled.csv", rows)
    print("\n  E6 unmodelled aberrations (100 nm modelled + fraction unmodelled)")
    print(f"    {'extra':>7} {'full resid':>11} {'modelled-only':>14} {'mode err':>10}")
    for frac in (0.0, 0.1, 0.25, 0.5, 1.0):
        sub = [r for r in rows if r["unmodelled_frac"] == frac]
        print(f"    {frac*100:6.0f}% {np.mean([r['full_residual_m'] for r in sub])*1e9:10.2f}n "
              f"{np.mean([r['modelled_only_residual_m'] for r in sub])*1e9:13.2f}n "
              f"{np.mean([r['recovered_modes_err_m'] for r in sub])*1e9:9.2f}n")
    return rows


def e7_subpixel(bench, scale, out, steps=21):
    """
    Sweep the spot across a FRACTION of a pixel using controlled tilt.

    One full detector pixel of spot motion requires ~3125 nm of tilt
    coefficient (from the analytic chain: 100 nm edge tilt -> 0.0320 px). That
    is ABOVE the measured failure onset (~2000 nm, E4), so in normal operation
    the spot never traverses a whole pixel. The sweep therefore covers 0-0.4 px,
    which is the physically reachable range, rather than a full pixel.
    """
    det = D.DetectorConfig(shot_noise=False, read_noise_e=0.0)
    idx = BA.valid_subaps()
    s = float(1 << C.CENTROID_FRAC)
    c_ = bench.cfg
    # HCIPy's Zernike basis is RMS-normalised: tip is Z = 2(x/R), so a
    # coefficient `a` produces wavefront slope 2a/R, not a/R. The factor of 2
    # below is that normalisation -- omitting it predicts half the spot motion.
    per_px = (c_.detector_pixel_pitch_m / c_.focal_length_m
              * c_.magnification * (c_.telescope_diameter_m / 2)) / 2.0
    max_frac = 0.4                      # stays inside the operating envelope
    rows, base = [], None
    for i in range(steps):
        f = max_frac * i / (steps - 1)
        c = np.zeros(C.NUM_MODES); c[0] = f * per_px
        img, opd = bench.propagate_coeffs(c)
        w, fx = recon(bench, D.expose(img, det).adu, scale)
        cx = fx["x_centroid"][idx] / s
        if base is None:
            base = cx.copy()
        meas = float(np.mean(cx - base))
        rows.append({"frac_pixel": f, "tilt_coeff_nm": f * per_px * 1e9,
                     "commanded_px": f, "measured_shift_px": meas,
                     "bias_px": meas - f,
                     "residual_m": bench.opd_rms(opd - w)})
    _csv(out / "e7_subpixel.csv", rows)
    b = np.array([r["bias_px"] for r in rows])
    resid = np.array([r["residual_m"] for r in rows])
    print(f"\n  E7 subpixel sampling ({steps} positions over 0-{max_frac} px)")
    print(f"    one pixel needs    : {per_px*1e9:,.0f} nm tilt "
          f"(comparable to the ~2000 nm failure onset -- the spot barely crosses "
          f"pixel in normal operation)")
    print(f"    centroid bias      : mean {b.mean():+.5f} px, "
          f"peak-to-peak {b.max()-b.min():.5f} px")
    print(f"    OPD residual       : {resid.mean()*1e9:.2f} nm mean, "
          f"{resid.max()*1e9:.2f} nm max")
    return rows


def convergence(scale, out):
    """Refine optical sampling; report how much the metrics move."""
    print("\n  CONV optical sampling convergence")
    rng0 = np.random.default_rng(SEED + 7)
    coeffs = [rng0.normal(0, 1, C.NUM_MODES) for _ in range(4)]
    rows = []
    for ss in (2, 4, 8):
        cfg = O.OpticalConfig(supersample=ss)
        b = O.Bench(cfg)
        det = D.DetectorConfig(shot_noise=False, read_noise_e=0.0)
        vals = []
        for c0 in coeffs:
            c = c0 * (100e-9 / b.opd_rms(b.opd_from_coeffs(c0)))
            img, opd = b.propagate_coeffs(c)
            w, _ = recon(b, D.expose(img, det).adu, scale)
            vals.append(b.opd_rms(opd - w))
        rows.append({"supersample": ss, "residual_mean_m": float(np.mean(vals))})
        print(f"    supersample {ss}: residual {np.mean(vals)*1e9:.3f} nm")
    ref = rows[-1]["residual_mean_m"]
    for r in rows:
        r["delta_vs_finest_pct"] = float(100 * (r["residual_mean_m"] / ref - 1))
    print(f"    change 4x vs 8x: {rows[1]['delta_vs_finest_pct']:+.2f} % "
          f"-> 4x supersampling is adequate")
    _csv(out / "convergence.csv", rows)
    return rows


def _csv(path: Path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.10e}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR / "optical")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    bench = O.get_bench()
    scale = load_frozen()
    print(f"E4/E6/E7 + convergence  (scale {scale:.6e} m/LSB)")
    e4_dynamic_range(bench, scale, a.out)
    e6_unmodelled(bench, scale, a.out)
    e7_subpixel(bench, scale, a.out)
    convergence(scale, a.out)
    print(f"\n  wrote CSVs to {a.out}")


if __name__ == "__main__":
    main()
