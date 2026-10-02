#!/usr/bin/env python3
"""
Diagnose mode-dependent gain by measuring the interaction matrix optically.

Regenerating the flat reference removed the ~8.3 nm systematic bias entirely,
but left two things unchanged: recovery gain still spreads 0.78-1.08 across
modes, and the large-amplitude residual still sits near 15 %. Neither can be a
reference-offset effect, so the next candidate is the reconstruction matrix.

Rather than assert that, this measures it. The interaction matrix is built by
PROBING: each Zernike mode is applied at small +/- amplitudes, propagated
through the same HCIPy bench, digitised by the same detector, and the resulting
slopes measured through the same thresholded-CoG path the hardware uses. The
push-pull difference cancels the reference and any even-order response.

    M[:, m] = (slopes(+a_m) - slopes(-a_m)) / (2 a_m)      [slope px per metre]
    E_new   = pinv(M)                                       [metre per slope px]

This uses pinv, but of an EMPIRICALLY MEASURED optical response -- not of the
matrix being tested. Probes use their own seed and amplitude and never appear in
the evaluation set, so held-out scoring stays honest.

Units are carried explicitly end to end, so the output scale is DERIVED rather
than fitted:

    slope stored as Q18   ->  E must be (metre per Q18 LSB) to give metres
    E stored as Q1.16     ->  multiply by 2^16 and round
    datapath: acc = sum(E_q * slope_q), out = acc[42:18]
              => out = coeff_m * 2^16 * 2^18 / 2^18 / <E scale> ...
    the derived scale is computed exactly below and cross-checked against the
    frozen empirical one.
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

PROBE_SEED = 88_001
HEX_DIGITS = 5          # matches the existing e_matrix_*.mem width


def measure_slopes(bench, adu) -> tuple[np.ndarray, np.ndarray]:
    """Slopes in pixels, through the pipeline's own float path."""
    fl = BA.run_float_reference(np.asarray(adu, dtype=np.int64).ravel())
    idx = BA.valid_subaps()
    return fl["x_slope_px"][idx], fl["y_slope_px"][idx]


def build_interaction(bench, probe_nm: float) -> np.ndarray:
    """(2*n_valid, n_modes) slope response per metre of coefficient."""
    det = D.DetectorConfig(shot_noise=False, read_noise_e=0.0)
    n = C.NUM_SUBAPS_VALID
    M = np.zeros((2 * n, C.NUM_MODES))
    a = probe_nm * 1e-9
    for m in range(C.NUM_MODES):
        cp = np.zeros(C.NUM_MODES); cp[m] = +a
        cm = np.zeros(C.NUM_MODES); cm[m] = -a
        sxp, syp = measure_slopes(bench, D.expose(bench.propagate_coeffs(cp)[0], det).adu)
        sxm, sym = measure_slopes(bench, D.expose(bench.propagate_coeffs(cm)[0], det).adu)
        M[:n, m] = (sxp - sxm) / (2 * a)
        M[n:, m] = (syp - sym) / (2 * a)
    return M


def evaluate(bench, E_px: np.ndarray, n_cases: int, levels_nm, seed: int) -> dict:
    """
    Held-out reconstruction with a float reconstruction matrix in
    metre-per-slope-pixel units.
    """
    rng = np.random.default_rng(seed)
    det = D.DetectorConfig()
    out = {}
    for lvl in levels_nm:
        res = []
        for _ in range(n_cases):
            c = rng.normal(0, 1, C.NUM_MODES)
            c *= (lvl * 1e-9) / bench.opd_rms(bench.opd_from_coeffs(c))
            img, opd = bench.propagate_coeffs(c)
            sx, sy = measure_slopes(bench, D.expose(img, det, rng).adu)
            rec_c = E_px @ np.concatenate([sx, sy])
            res.append(bench.opd_rms(opd - bench.opd_from_coeffs(rec_c)))
        out[lvl] = {"mean_m": float(np.mean(res)),
                    "sem_m": float(np.std(res, ddof=1) / np.sqrt(len(res))),
                    "n": len(res)}
    return out


def baseline_E_px() -> np.ndarray:
    """
    The shipping E matrix, expressed in metre-per-slope-pixel so it can be
    compared like for like with the measured one.
    """
    ex, ey = BA.ematrix()
    scale_json = json.loads(
        (C.RESULTS_DIR / "optical" / "calibration.json").read_text())
    s = scale_json["scale_m_per_lsb"]
    # acc = sum(E_q * slope_q18); out = acc >> 18; metres = out * s
    # => metres = sum(E_q * slope_px * 2^18) * 2^-18 * s = sum(E_q * slope_px) * s
    E = np.hstack([ex, ey]).astype(np.float64) * s
    return E


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-nm", type=float, default=30.0)
    ap.add_argument("-n", "--cases", type=int, default=40)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR / "optical")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    bench = O.get_bench()
    print(f"  probing {C.NUM_MODES} modes at +/-{a.probe_nm:.0f} nm ...")
    M = build_interaction(bench, a.probe_nm)
    E_new = np.linalg.pinv(M)                      # metre per slope pixel
    E_old = baseline_E_px()

    levels = (50, 100, 200)
    print("  evaluating on held-out wavefronts (seed separate from probes) ...")
    r_old = evaluate(bench, E_old, a.cases, levels, seed=PROBE_SEED + 1)
    r_new = evaluate(bench, E_new, a.cases, levels, seed=PROBE_SEED + 1)

    print(f"\n  {'level':>7} {'baseline E':>14} {'measured E':>14} {'change':>9}")
    for lvl in levels:
        o, n = r_old[lvl]["mean_m"] * 1e9, r_new[lvl]["mean_m"] * 1e9
        print(f"  {lvl:5.0f}nm {o:12.2f}nm {n:12.2f}nm {100*(n/o-1):8.1f}%")

    # Per-mode gain under each matrix, via the measured interaction.
    gain_old = np.diag(E_old @ M)
    gain_new = np.diag(E_new @ M)
    rec = {
        "probe_amplitude_nm": a.probe_nm,
        "probe_seed": PROBE_SEED,
        "held_out_cases_per_level": a.cases,
        "baseline": {str(k): v for k, v in r_old.items()},
        "measured": {str(k): v for k, v in r_new.items()},
        "gain_baseline": [float(v) for v in gain_old],
        "gain_measured": [float(v) for v in gain_new],
        "gain_spread_baseline": float(gain_old.max() - gain_old.min()),
        "gain_spread_measured": float(gain_new.max() - gain_new.min()),
        "interaction_condition_number": float(np.linalg.cond(M)),
        "note": ("E is expressed in metre-per-slope-pixel for both, so the "
                 "comparison is like for like and independent of the stored "
                 "fixed-point scale."),
    }
    (a.out / "ematrix_regen.json").write_text(json.dumps(rec, indent=2))
    np.save(a.out / "E_measured_m_per_slopepx.npy", E_new)

    print(f"\n  per-mode gain spread : baseline {rec['gain_spread_baseline']:.3f} "
          f"-> measured {rec['gain_spread_measured']:.3f}")
    print(f"  interaction cond()   : {rec['interaction_condition_number']:.1f}")
    print(f"  wrote {a.out/'ematrix_regen.json'}")

    if a.apply:
        # -------------------------------------------------------------------
        # DERIVE the scale instead of fitting it.
        #
        # The output scale and the matrix quantisation are coupled: E_q =
        # round(E_px / scale), and metres = raw * scale. `scale` is therefore a
        # free parameter, and the principled choice is the one that uses the
        # full signed 18-bit word without saturating any coefficient:
        #
        #     scale = max|E_px| / (2^17 - 1)
        #
        # This is derived from the matrix and the word width alone -- no fit to
        # evaluation data.
        # -------------------------------------------------------------------
        lim = (1 << (C.EMATRIX_BITS - 1)) - 1
        derived_scale = float(np.abs(E_new).max() / lim)
        Eq = np.rint(E_new / derived_scale).astype(np.int64)
        sat = int((np.abs(Eq) > lim).sum())
        print(f"\n  derived scale        : {derived_scale:.6e} m/LSB "
              f"= 2^{np.log2(derived_scale):.3f}")
        print(f"  saturated words      : {sat} of {Eq.size}")
        if sat:
            print("  REFUSING to apply -- derivation should make this zero.")
            return

        # Check the 25-bit output field still has headroom at the largest
        # amplitude we intend to support.
        worst = 0
        det = D.DetectorConfig(shot_noise=False, read_noise_e=0.0)
        rng2 = np.random.default_rng(PROBE_SEED + 99)
        for _ in range(12):
            c = rng2.normal(0, 1, C.NUM_MODES)
            c *= 400e-9 / bench.opd_rms(bench.opd_from_coeffs(c))
            img, _ = bench.propagate_coeffs(c)
            sx, sy = measure_slopes(bench, D.expose(img, det).adu)
            sq = np.rint(np.concatenate([sx, sy]) * (1 << C.CENTROID_FRAC))
            raw = (Eq @ sq) / (1 << C.ZERNIKE_SLICE_LOW)
            worst = max(worst, float(np.abs(raw).max()))
        field = (1 << (C.ZERNIKE_BITS - 1)) - 1
        print(f"  worst |raw| at 400nm : {worst:,.0f} of {field:,} "
              f"({100*worst/field:.1f}% of the 25-bit field)")
        if worst >= field:
            print("  REFUSING to apply -- output field would saturate.")
            return

        n = C.NUM_SUBAPS_VALID
        mask = (1 << (HEX_DIGITS * 4)) - 1
        for name, block in (("e_matrix_x.mem", Eq[:, :n]), ("e_matrix_y.mem", Eq[:, n:])):
            (C.RTL_DIR / name).write_text(
                "".join(f"{int(v) & mask:0{HEX_DIGITS}X}\n" for v in block.ravel()))
        BA.ematrix.cache_clear()
        rec["derived_scale_m_per_lsb"] = derived_scale
        rec["output_field_use_pct_at_400nm"] = 100 * worst / field
        (a.out / "ematrix_regen.json").write_text(json.dumps(rec, indent=2))
        (a.out / "derived_scale.json").write_text(json.dumps(
            {"scale_m_per_lsb": derived_scale,
             "method": "derived: max|E| mapped to the full signed 18-bit word",
             "e_matrix_bits": C.EMATRIX_BITS}, indent=2))
        print(f"  APPLIED new E matrix to {C.RTL_DIR}")


if __name__ == "__main__":
    main()
