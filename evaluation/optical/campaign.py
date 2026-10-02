#!/usr/bin/env python3
"""
HCIPy optical validation campaign.

Every wavefront is an INDEPENDENT optical input: a known OPD propagated through
the HCIPy Shack-Hartmann model and digitised by an explicit detector. No case
uses pinv(E), so the forward and inverse models are genuinely different and the
reconstruction claim is not circular.

The output scale is LOADED FROZEN from calibrate.py, which fitted it on a
separate calibration set. Nothing here refits it.

Three comparison levels, on identical digitised pixels:
    float    float64 centroiding + modal reconstruction
    fixed    bit-accurate model of the RTL arithmetic
    (RTL itself is exercised separately by rtl_optical_subset.py)

Metrics are computed on the ACTUAL pupil, aperture-weighted, piston removed --
not from the Euclidean norm of the coefficient vector, which is not the
wavefront RMS on an obscured, spidered pupil.

Experiments
    flat        E1  noiseless bias + noisy repeatability
    modes       E2  per-mode gain, offset, nonlinearity, cross-talk
    mixed       E3  held-out mixed-mode wavefronts at declared RMS levels
    sweep       E5  signal level sweep, fixed gain, shot and shot+read
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import bitaccurate as BA          # noqa: E402
from evaluation.common import config as C                # noqa: E402
from evaluation.common import detector as D              # noqa: E402
from evaluation.common import optics as O                # noqa: E402
from evaluation.optical.calibrate import load_frozen      # noqa: E402

SEED_FLAT, SEED_MODES, SEED_MIXED, SEED_SWEEP = 5101, 5202, 5303, 5404


class Runner:
    def __init__(self, out: Path):
        self.bench = O.get_bench()
        self.scale = load_frozen()
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.idx = BA.valid_subaps()

    def reconstruct(self, adu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(float coefficients, fixed coefficients) in metres, same pixels."""
        px = np.asarray(adu, dtype=np.int64).ravel()
        fx = BA.run_pipeline(px)
        fl = BA.run_float_reference(px)
        fixed_m = fx["zernike_raw"].astype(np.float64) * self.scale
        # Float path scaled by the same physical constant, so the only
        # difference between the two is the fixed-point arithmetic.
        ratio = C.METERS_PER_LSB
        float_m = fl["zernike_m"] / ratio * self.scale
        return float_m, fixed_m

    def opd_metrics(self, true_opd, coeffs_f, coeffs_x) -> dict:
        b = self.bench
        rec_f = b.opd_from_coeffs(coeffs_f)
        rec_x = b.opd_from_coeffs(coeffs_x)
        return {
            "true_opd_rms_m": b.opd_rms(true_opd),
            "float_opd_residual_m": b.opd_rms(true_opd - rec_f),
            "fixed_opd_residual_m": b.opd_rms(true_opd - rec_x),
            # Paired difference, NOT a quadrature subtraction.
            "fixed_minus_float_opd_m": b.opd_rms(rec_x - rec_f),
        }

    # ---------------------------------------------------------------- E1 ----
    def flat(self, n_noisy: int) -> dict:
        b, rng = self.bench, np.random.default_rng(SEED_FLAT)
        flat = np.zeros_like(np.asarray(b.aperture), dtype=float)
        img = b.propagate(flat)

        clean = D.DetectorConfig(shot_noise=False, read_noise_e=0.0)
        f0, x0 = self.reconstruct(D.expose(img, clean).adu)
        m0 = self.opd_metrics(flat, f0, x0)

        det = D.DetectorConfig()
        rows = []
        for i in range(n_noisy):
            adu = D.expose(img, det, rng).adu
            f, x = self.reconstruct(adu)
            rows.append({"trial": i, **{f"float_c{k}": v for k, v in enumerate(f)},
                         **self.opd_metrics(flat, f, x)})
        arr = lambda k: np.array([r[k] for r in rows])

        res = {
            "noiseless_bias_coeffs_m": [float(v) for v in f0],
            "noiseless_float_residual_m": m0["float_opd_residual_m"],
            "noiseless_fixed_residual_m": m0["fixed_opd_residual_m"],
            "n_noisy_trials": n_noisy,
            "noisy_float_residual_mean_m": float(arr("float_opd_residual_m").mean()),
            "noisy_float_residual_sd_m": float(arr("float_opd_residual_m").std(ddof=1)),
            "noisy_fixed_residual_mean_m": float(arr("fixed_opd_residual_m").mean()),
            "repeatability_sd_coeff0_m": float(arr("float_c0").std(ddof=1)),
            "detector": D.config_record(det),
            "interpretation": ("the noiseless residual is systematic bias -- the "
                               "stored reference slopes do not exactly match this "
                               "bench's flat wavefront; the noisy spread is "
                               "repeatability"),
        }
        _csv(self.out / "flat_trials.csv", rows)
        return res

    # ---------------------------------------------------------------- E2 ----
    def modes(self, amps_nm) -> dict:
        b = self.bench
        det = D.DetectorConfig(shot_noise=False, read_noise_e=0.0)
        rows = []
        for m in range(C.NUM_MODES):
            for a_nm in amps_nm:
                for sign in (+1, -1):
                    c = np.zeros(C.NUM_MODES)
                    c[m] = sign * a_nm * 1e-9
                    img, opd = b.propagate_coeffs(c)
                    f, x = self.reconstruct(D.expose(img, det).adu)
                    rows.append({
                        "mode": m + 1, "label": C.MODE_LABELS[m],
                        "amp_nm": sign * a_nm,
                        "input_opd_rms_m": b.opd_rms(opd),
                        "float_recovered_m": float(f[m]),
                        "fixed_recovered_m": float(x[m]),
                        "float_crosstalk_rms_m": float(
                            np.sqrt(np.mean(np.delete(f, m) ** 2))),
                        **self.opd_metrics(opd, f, x)})
        _csv(self.out / "mode_recovery.csv", rows)

        summary = []
        for m in range(C.NUM_MODES):
            sub = [r for r in rows if r["mode"] == m + 1]
            x_in = np.array([r["amp_nm"] * 1e-9 for r in sub])
            y = np.array([r["float_recovered_m"] for r in sub])
            A = np.vstack([x_in, np.ones_like(x_in)]).T
            gain, offset = np.linalg.lstsq(A, y, rcond=None)[0]
            resid = y - (gain * x_in + offset)
            summary.append({
                "mode": m + 1, "label": C.MODE_LABELS[m],
                "gain": float(gain), "offset_m": float(offset),
                "nonlinearity_rms_m": float(np.sqrt(np.mean(resid ** 2))),
                "crosstalk_rms_m": float(np.mean(
                    [r["float_crosstalk_rms_m"] for r in sub])),
            })
        _csv(self.out / "mode_summary.csv", summary)
        return {"per_mode": summary,
                "gain_min": min(s["gain"] for s in summary),
                "gain_max": max(s["gain"] for s in summary),
                "amplitudes_nm": list(amps_nm)}

    # ---------------------------------------------------------------- E3 ----
    def mixed(self, n_per_level: int, levels_nm) -> dict:
        b, rng = self.bench, np.random.default_rng(SEED_MIXED)
        det = D.DetectorConfig()
        rows = []
        for lvl in levels_nm:
            got = 0
            while got < n_per_level:
                c = rng.normal(0.0, 1.0, C.NUM_MODES)
                opd = b.opd_from_coeffs(c)
                r = b.opd_rms(opd)
                if r <= 0:
                    continue
                c *= (lvl * 1e-9) / r             # set ACTUAL pupil RMS
                img, opd = b.propagate_coeffs(c)
                f, x = self.reconstruct(D.expose(img, det, rng).adu)
                rows.append({"level_nm": lvl, "trial": got,
                             "true_coeffs_m": ";".join(f"{v:.6e}" for v in c),
                             **self.opd_metrics(opd, f, x)})
                got += 1
        _csv(self.out / "mixed_mode.csv", rows)

        summary = []
        for lvl in levels_nm:
            sub = [r for r in rows if r["level_nm"] == lvl]
            fr = np.array([r["float_opd_residual_m"] for r in sub])
            fx = np.array([r["fixed_opd_residual_m"] for r in sub])
            df = np.array([r["fixed_minus_float_opd_m"] for r in sub])
            tr = np.array([r["true_opd_rms_m"] for r in sub])
            n = len(sub)
            summary.append({
                "level_nm": lvl, "n": n,
                "true_rms_mean_m": float(tr.mean()),
                "float_residual_mean_m": float(fr.mean()),
                "float_residual_sem_m": float(fr.std(ddof=1) / np.sqrt(n)),
                "float_residual_p95_m": float(np.percentile(fr, 95)),
                "fixed_residual_mean_m": float(fx.mean()),
                "fixed_residual_sem_m": float(fx.std(ddof=1) / np.sqrt(n)),
                "fixed_residual_p95_m": float(np.percentile(fx, 95)),
                "fixed_minus_float_mean_m": float(df.mean()),
                "float_rel_error": float(fr.mean() / tr.mean()),
                "fixed_rel_error": float(fx.mean() / tr.mean()),
            })
        _csv(self.out / "mixed_summary.csv", summary)
        return {"levels": summary, "n_total": len(rows)}

    # ---------------------------------------------------------------- E5 ----
    def sweep(self, n_real: int, photons, read_noise_cases) -> dict:
        b, rng = self.bench, np.random.default_rng(SEED_SWEEP)
        # A fixed, declared set of wavefronts, reused at every signal level so
        # the level is the only thing that changes.
        base = []
        for _ in range(8):
            c = rng.normal(0.0, 1.0, C.NUM_MODES)
            opd0 = b.opd_from_coeffs(c)
            c *= (80e-9) / b.opd_rms(opd0)
            img, opd = b.propagate_coeffs(c)
            base.append((img, opd))

        rows = []
        for rn_label, rn in read_noise_cases:
            for ph in photons:
                det = D.DetectorConfig(total_photons=ph, read_noise_e=rn)
                fr, fx, df, sat, clip = [], [], [], 0, 0
                for i in range(n_real):
                    img, opd = base[i % len(base)]
                    r = D.expose(img, det, rng)
                    sat += r.saturated_px; clip += r.clipped_adu_px
                    f, x = self.reconstruct(r.adu)
                    m = self.opd_metrics(opd, f, x)
                    fr.append(m["float_opd_residual_m"])
                    fx.append(m["fixed_opd_residual_m"])
                    df.append(m["fixed_minus_float_opd_m"])
                fr, fx, df = map(np.array, (fr, fx, df))
                rows.append({
                    "noise_case": rn_label, "photons": ph,
                    "read_noise_e": rn,
                    "e_per_lenslet": D.signal_per_lenslet_e(det),
                    "n": n_real,
                    "float_residual_mean_m": float(fr.mean()),
                    "float_residual_sem_m": float(fr.std(ddof=1) / np.sqrt(n_real)),
                    "fixed_residual_mean_m": float(fx.mean()),
                    "fixed_residual_sem_m": float(fx.std(ddof=1) / np.sqrt(n_real)),
                    "fixed_minus_float_mean_m": float(df.mean()),
                    "penalty_ratio": float(fx.mean() / fr.mean()),
                    "saturated_px_total": sat, "clipped_adu_total": clip,
                })
                print(f"    {rn_label:14s} {ph:9.1e} ph  "
                      f"float {fr.mean()*1e9:7.2f} nm  fixed {fx.mean()*1e9:7.2f} nm  "
                      f"ratio {fx.mean()/fr.mean():.3f}", flush=True)
        _csv(self.out / "signal_sweep.csv", rows)
        return {"rows": rows}


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(
                f"{r[k]:.10e}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                for k in keys) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="small sample sizes")
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR / "optical")
    a = ap.parse_args()
    p = a.pilot

    t0 = time.time()
    r = Runner(a.out)
    print(f"Optical campaign  (frozen scale {r.scale:.6e} m/LSB)")
    print(f"  pupil pixels {int(r.bench.pupil_mask.sum())}, "
          f"{C.NUM_SUBAPS_VALID} lenslets, {C.NUM_MODES} modes")

    out = {"frozen_scale_m_per_lsb": r.scale,
           "optics": O.config_record(),
           "pilot": p}

    print("\n E1 flat wavefront")
    out["flat"] = r.flat(10 if p else 100)
    f = out["flat"]
    print(f"    noiseless bias residual : {f['noiseless_float_residual_m']*1e9:.3f} nm (float)")
    print(f"    noisy residual          : {f['noisy_float_residual_mean_m']*1e9:.3f} "
          f"+/- {f['noisy_float_residual_sd_m']*1e9:.3f} nm")

    print("\n E2 single-mode recovery")
    amps = (20, 60) if p else (10, 20, 40, 80, 150)
    out["modes"] = r.modes(amps)
    print(f"    gain range across modes : {out['modes']['gain_min']:.3f} .. "
          f"{out['modes']['gain_max']:.3f}")

    print("\n E3 mixed-mode held-out")
    lv = (40, 120) if p else (20, 50, 100, 200, 400)
    out["mixed"] = r.mixed(5 if p else 60, lv)
    for s in out["mixed"]["levels"]:
        print(f"    {s['level_nm']:4.0f} nm RMS (n={s['n']:3d}): float "
              f"{s['float_residual_mean_m']*1e9:7.2f} nm  fixed "
              f"{s['fixed_residual_mean_m']*1e9:7.2f} nm  "
              f"rel {s['fixed_rel_error']:.1%}")

    print("\n E5 signal sweep (fixed gain, no per-frame normalisation)")
    ph = (1e6, 5e6) if p else (5e5, 1e6, 2e6, 3e6, 5e6, 8e6, 1.2e7, 2e7)
    cases = [("shot_only", 0.0)] if p else [("shot_only", 0.0), ("shot+read3e", 3.0)]
    out["sweep"] = r.sweep(10 if p else 100, ph, cases)

    (a.out / "campaign_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  elapsed {time.time()-t0:.1f} s")
    print(f"  wrote {a.out}/campaign_summary.json")


if __name__ == "__main__":
    main()
