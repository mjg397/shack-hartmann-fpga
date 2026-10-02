#!/usr/bin/env python3
"""
STEP 6 -- verify the arithmetic discrepancy directly on the pupil.

The claim "fixed point adds ~0.2 nm" was read off a pooled ratio of two
residuals. That is weak: two RMS values can be nearly equal while the underlying
wavefronts differ. This computes the PAIRED quantity directly, per frame:

    RMS(W_fixed - W_float)      on the actual pupil, aperture-weighted

and reports it alongside, but separately from:

    RMS(W_float - W_true)
    RMS(W_fixed - W_true)

No quadrature subtraction is used anywhere: paired differences are computed
frame by frame, so no covariance assumption is needed.

Distributions are reported by amplitude and by signal level, not pooled, so a
small mean cannot hide a heavy tail.

Also reconciles against the earlier arithmetic-only figure (fixed vs float
coefficients on identical pixels), which measured a different quantity.
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

SEED = 60_606


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--per-cell", type=int, default=40)
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR / "optical")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    bench = O.get_bench()
    scale = load_frozen()
    rng = np.random.default_rng(SEED)

    amps = (20, 100, 400)                       # nm RMS
    photons = (1e6, 3e6)                        # inside the clean envelope
    rows = []

    for amp in amps:
        for ph in photons:
            det = D.DetectorConfig(total_photons=ph)
            for k in range(a.per_cell):
                c = rng.normal(0, 1, C.NUM_MODES)
                c *= (amp * 1e-9) / bench.opd_rms(bench.opd_from_coeffs(c))
                img, opd = bench.propagate_coeffs(c)
                adu = D.expose(img, det, rng).adu.ravel()

                fx = BA.run_pipeline(adu)
                fl = BA.run_float_reference(adu)
                w_fixed = bench.opd_from_coeffs(
                    fx["zernike_raw"].astype(float) * scale)
                w_float = bench.opd_from_coeffs(
                    fl["zernike_m"] / C.METERS_PER_LSB * scale)

                rows.append({
                    "amp_nm": amp, "photons": ph, "trial": k,
                    "true_rms_m": bench.opd_rms(opd),
                    "float_vs_true_m": bench.opd_rms(opd - w_float),
                    "fixed_vs_true_m": bench.opd_rms(opd - w_fixed),
                    # THE paired quantity, computed per frame on the pupil
                    "fixed_vs_float_m": bench.opd_rms(w_fixed - w_float),
                })

    keys = list(rows[0].keys())
    with (a.out / "arithmetic_discrepancy.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.10e}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")

    print(f"  {'amp':>6} {'photons':>9} {'n':>4} "
          f"{'float-true':>11} {'fixed-true':>11} "
          f"{'|fixed-float| mean':>19} {'p95':>9} {'max':>9}")
    summary = []
    for amp in amps:
        for ph in photons:
            sub = [r for r in rows if r["amp_nm"] == amp and r["photons"] == ph]
            d = np.array([r["fixed_vs_float_m"] for r in sub]) * 1e9
            ft = np.array([r["float_vs_true_m"] for r in sub]) * 1e9
            xt = np.array([r["fixed_vs_true_m"] for r in sub]) * 1e9
            s = {"amp_nm": amp, "photons": ph, "n": len(sub),
                 "float_vs_true_mean_nm": float(ft.mean()),
                 "fixed_vs_true_mean_nm": float(xt.mean()),
                 "fixed_vs_float_mean_nm": float(d.mean()),
                 "fixed_vs_float_p95_nm": float(np.percentile(d, 95)),
                 "fixed_vs_float_max_nm": float(d.max()),
                 "ratio_to_total": float(d.mean() / ft.mean())}
            summary.append(s)
            print(f"  {amp:5.0f}n {ph:9.1e} {len(sub):4d} "
                  f"{ft.mean():10.3f}n {xt.mean():10.3f}n "
                  f"{d.mean():18.4f}n {np.percentile(d,95):8.4f}n {d.max():8.4f}n")

    # Reconciliation with the earlier arithmetic-only measurement.
    recon = {
        "note": ("the earlier ~0.2 nm figure came from comparing fixed and float "
                 "COEFFICIENTS on identical pixels, then converting. This measures "
                 "RMS(W_fixed - W_float) on the pupil per frame, which is the "
                 "quantity that matters for a wavefront claim. They are different "
                 "quantities and need not agree numerically."),
        "paired_method": "per-frame difference of reconstructed OPD maps; no quadrature",
    }
    (a.out / "arithmetic_discrepancy.json").write_text(
        json.dumps({"summary": summary, "reconciliation": recon,
                    "scale_m_per_lsb": scale, "seed": SEED}, indent=2))
    print(f"\n  wrote {a.out/'arithmetic_discrepancy.csv'}")


if __name__ == "__main__":
    main()
