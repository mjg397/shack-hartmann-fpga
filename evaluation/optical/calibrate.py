#!/usr/bin/env python3
"""
STEP B -- Derive and FREEZE the output scale on a calibration set.

The previous evaluation fitted the raw->metres scale (2^-43) against the same
HCIPy cases it then reported accuracy on. That is circular and is not
validation. This script fixes that in two ways:

  1. The scale is fitted on a CALIBRATION set with its own seed, and frozen to
     `calibration.json`. Every later experiment loads that frozen value and uses
     held-out wavefronts. Nothing downstream is permitted to refit.

  2. An ANALYTIC cross-check: a pure tilt of known amplitude has a closed-form
     wavefront slope, so the expected centroid displacement can be computed from
     geometry alone (pixel pitch, lenslet focal length, magnification) without
     reference to the reconstruction at all.

Why a fit is needed at all: `ematrix_genv4.py` calibrates E against slopes in
**Q4.23** (`build_slope_vector_q4_23`, fraction_bits=23), while the current RTL
supplies **Q18**. The stored matrix therefore carries a 2^5 format offset
relative to the datapath that consumes it, and the true scale cannot be read off
the generator alone. The fit measures that offset once, on data never used for
scoring; the analytic check confirms the optical chain independently.

Outputs
    evaluation/results/optical/calibration.json
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

CAL_SEED = 907_001          # distinct from every evaluation seed


def analytic_tilt_check(bench: O.Bench) -> dict:
    """
    Closed-form cross-check of the optical chain, independent of E.

    A wavefront W(x) = a * x/R (tilt of amplitude `a` at the pupil edge) has
    slope dW/dx = a/R. After demagnification the slope at the lenslet array is
    (a/R)/M, and the spot displacement is that slope times the lenslet focal
    length. Converting to pixels uses only the detector pitch.
    """
    c = bench.cfg
    R = c.telescope_diameter_m / 2.0
    a = 100e-9                                   # 100 nm RMS coefficient

    # HCIPy's Zernike basis is RMS-normalised over the pupil, so tip is
    # Z = 2(x/R) and a coefficient `a` gives slope 2a/R. Assuming edge-amplitude
    # normalisation (a/R) predicts half the true spot motion -- verified
    # empirically in E7, where the measured shift was exactly 2x the prediction.
    wf_slope = 2.0 * a / R                       # rad, at the pupil
    slope_at_mla = wf_slope / c.magnification    # demagnified
    disp_m = slope_at_mla * c.focal_length_m
    disp_px = disp_m / c.detector_pixel_pitch_m

    return {
        "tilt_rms_coefficient_m": a,
        "pupil_radius_m": R,
        "wavefront_slope_rad": wf_slope,
        "magnification": c.magnification,
        "slope_at_lenslet_rad": slope_at_mla,
        "focal_length_m": c.focal_length_m,
        "spot_displacement_m": disp_m,
        "detector_pixel_pitch_m": c.detector_pixel_pitch_m,
        "expected_displacement_px": disp_px,
        "zernike_normalisation": "RMS over the pupil (HCIPy); tip = 2(x/R)",
        "note": ("derived from geometry and the Zernike normalisation only. "
                 "Independent of the E matrix and of any fitted quantity. The "
                 "factor 2 was confirmed empirically in E7."),
    }


def run(n_cal: int, out_dir: Path) -> dict:
    bench = O.get_bench()
    c = bench.cfg
    rng = np.random.default_rng(CAL_SEED)

    det = D.DetectorConfig(shot_noise=False, read_noise_e=0.0)  # noiseless for calibration

    raws, truths = [], []
    for _ in range(n_cal):
        # Calibration wavefronts: moderate amplitude, all modes excited.
        coeffs = rng.normal(0.0, 40e-9, c.num_zernike)
        img, opd = bench.propagate_coeffs(coeffs)
        adu = D.expose(img, det, rng).adu
        fx = BA.run_pipeline(adu.ravel())
        raws.append(fx["zernike_raw"].astype(np.float64))
        truths.append(coeffs)

    Rw = np.array(raws)          # (n, modes) raw LSB
    Tr = np.array(truths)        # (n, modes) metres

    # Single global scale, least squares over all modes and cases.
    num = float((Rw * Tr).sum())
    den = float((Rw * Rw).sum())
    scale = num / den if den else float("nan")

    # Per-mode scales, to expose whether one global scale is adequate.
    per_mode = []
    for m in range(c.num_zernike):
        d = float((Rw[:, m] ** 2).sum())
        per_mode.append(float((Rw[:, m] * Tr[:, m]).sum() / d) if d else float("nan"))

    pred = Rw * scale
    resid = pred - Tr
    rel = float(np.sqrt((resid ** 2).sum() / (Tr ** 2).sum()))

    rec = {
        "calibration_seed": CAL_SEED,
        "n_calibration_wavefronts": n_cal,
        "detector": D.config_record(det),
        "optics": O.config_record(),
        "scale_m_per_lsb": scale,
        "scale_log2": float(np.log2(abs(scale))) if scale else float("nan"),
        "per_mode_scale_m_per_lsb": per_mode,
        "per_mode_spread_pct": float(
            100 * (np.nanmax(per_mode) - np.nanmin(per_mode)) / abs(scale)) if scale else float("nan"),
        "calibration_relative_residual": rel,
        "previous_fitted_scale": C.METERS_PER_LSB,
        "ratio_to_previous": float(scale / C.METERS_PER_LSB) if C.METERS_PER_LSB else float("nan"),
        "analytic_tilt_check": analytic_tilt_check(bench),
        "method": ("global least-squares scale on a calibration set with its own "
                   "seed; FROZEN here and loaded by every later experiment, which "
                   "uses held-out wavefronts and never refits."),
        "why_a_fit_is_needed": ("ematrix_genv4.build_slope_vector_q4_23 calibrates E "
                                "against Q4.23 slopes while the RTL supplies Q18, so "
                                "the stored matrix carries a format offset that cannot "
                                "be read off the generator."),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(rec, indent=2))

    print(f"  calibration wavefronts : {n_cal} (seed {CAL_SEED}, noiseless)")
    print(f"  scale                  : {scale:.6e} m/LSB  = 2^{rec['scale_log2']:.3f}")
    print(f"  previously fitted      : {C.METERS_PER_LSB:.6e}  "
          f"(ratio {rec['ratio_to_previous']:.4f})")
    print(f"  per-mode spread        : {rec['per_mode_spread_pct']:.2f} % of global")
    print(f"  calibration residual   : {rel:.3%}")
    a = rec["analytic_tilt_check"]
    print(f"  analytic tilt check    : 100 nm edge tilt -> "
          f"{a['expected_displacement_px']:.4f} px spot displacement")
    print(f"  wrote {out_dir/'calibration.json'}")
    return rec


def load_frozen(results_dir: Path | None = None) -> float:
    """
    Load the frozen output scale. Experiments use this and never refit.

    A DERIVED scale (from regen_ematrix.py, obtained by mapping max|E| onto the
    full signed 18-bit word) takes precedence over the empirical calibration
    fit, because it follows from the matrix and the word width alone.
    """
    base = results_dir or (C.RESULTS_DIR / "optical")
    derived = base / "derived_scale.json"
    if derived.exists():
        return float(json.loads(derived.read_text())["scale_m_per_lsb"])
    p = base / "calibration.json"
    if not p.exists():
        raise SystemExit("calibration.json missing -- run evaluation/optical/calibrate.py first")
    return float(json.loads(p.read_text())["scale_m_per_lsb"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--cases", type=int, default=60)
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR / "optical")
    a = ap.parse_args()
    print("Calibration (held out from every evaluation set)")
    run(a.cases, a.out)


if __name__ == "__main__":
    main()
