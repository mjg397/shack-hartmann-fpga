#!/usr/bin/env python3
"""
RTL vs reference regression.

Two independent checks, deliberately kept separate:

  A. EQUIVALENCE  -- RTL vs the bit-accurate software model.
     Requires EXACT equality on every Zernike coefficient, slope and centroid.
     Any difference is a real discrepancy, never expected quantisation.

  B. NUMERICAL    -- bit-accurate fixed point vs float64.
     Reports error magnitudes. Quantisation differences are measured, not
     treated as failures.

Reuses the existing Icarus simulation (sim) through
evaluation.common.rtl_runner. cocotb is not used: the repository's
rtl/verification/test_*.py files are print-only scaffolds and cocotb is
not installed in this environment.

Case classes
    random_nominal         randomised physically reasonable aberrations
    zero_slope             spots exactly on the reference positions
    near_zero_slope        sub-LSB offsets
    positive_slope         all-positive displacements
    negative_slope         all-negative displacements
    large_offset           large but in-tile centroid offsets
    pure_mode_k            one Zernike mode at a time
    dim_spots              intensity just above PIXEL_THRESHOLD
    bright_spots           pixels driven to saturation (255)
    dark_frame             zero intensity -- exercises the sI == 0 reciprocal path

Outputs
    results/rtl_regression.csv
    results/rtl_regression_summary.json
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
from evaluation.common import framegen as FG             # noqa: E402
from evaluation.common import rtl_runner as R            # noqa: E402


def _zero_slopes() -> tuple[np.ndarray, np.ndarray]:
    n = C.NUM_SUBAPS_VALID
    return np.zeros(n), np.zeros(n)


def build_cases(n_random: int, rng: np.random.Generator) -> list[dict]:
    cases: list[dict] = []
    n = C.NUM_SUBAPS_VALID

    for i in range(n_random):
        c = FG.make_case(rng, scale_m=6e-8)
        cases.append({"name": f"random_nominal_{i:03d}", "cls": "random_nominal",
                      "frame": c["frame"], "true": c["true_coeffs_m"]})

    zx, zy = _zero_slopes()
    cases.append({"name": "zero_slope", "cls": "zero_slope",
                  "frame": FG.render_frame(zx, zy), "true": None})

    eps = 1.0 / (1 << C.CENTROID_FRAC) * 0.5          # half an LSB
    cases.append({"name": "near_zero_slope", "cls": "near_zero_slope",
                  "frame": FG.render_frame(np.full(n, eps), np.full(n, -eps)),
                  "true": None})

    for tag, mag in (("positive_slope", 2.5), ("negative_slope", -2.5)):
        cases.append({"name": tag, "cls": tag,
                      "frame": FG.render_frame(np.full(n, mag), np.full(n, mag)),
                      "true": None})

    # Large but still in-tile: references sit at 5.7-9.3 px inside a 16 px tile.
    cases.append({"name": "large_offset", "cls": "large_offset",
                  "frame": FG.render_frame(np.full(n, 4.5), np.full(n, -4.0)),
                  "true": None})

    for m in range(C.NUM_MODES):
        coeffs = np.zeros(C.NUM_MODES)
        coeffs[m] = 5e-8
        sx, sy = FG.coeffs_to_slopes(coeffs)
        cases.append({"name": f"pure_mode_{m+1}", "cls": "pure_mode",
                      "frame": FG.render_frame(sx, sy), "true": coeffs})

    cases.append({"name": "dim_spots", "cls": "dim_spots",
                  "frame": FG.render_frame(*_zero_slopes(), spot_sum=8.0 * 16),
                  "true": None})
    cases.append({"name": "bright_spots", "cls": "bright_spots",
                  "frame": FG.render_frame(*_zero_slopes(), spot_sum=4000.0),
                  "true": None})
    cases.append({"name": "dark_frame", "cls": "dark_frame",
                  "frame": np.zeros(C.FRAME_W * C.FRAME_H, dtype=np.int64),
                  "true": None})
    return cases


def run(n_random: int, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    cases = build_cases(n_random, rng)
    print(f"Regression: {len(cases)} cases "
          f"({n_random} random + {len(cases)-n_random} directed)")

    rows = []
    n_equiv_fail = 0
    idx = BA.valid_subaps()
    scale_px = float(1 << C.CENTROID_FRAC)

    with R.RTLSimulator() as sim:
        for k, case in enumerate(cases):
            frame = case["frame"]
            rtl = sim.run_frame(frame)
            fx = BA.run_pipeline(frame)
            fl = BA.run_float_reference(frame)

            # ---- A: exact equivalence ----
            z_equal = bool(np.array_equal(rtl["zernike_raw"], fx["zernike_raw"]))
            z_maxdiff = int(np.max(np.abs(rtl["zernike_raw"] - fx["zernike_raw"])))

            s_equal, s_maxdiff = True, 0
            if rtl["slopes"].size:
                rs = rtl["slopes"]
                if rs.shape[0] == fx["x_slope_valid"].size:
                    dx = rs[:, 1].astype(np.int64) - fx["x_slope_valid"]
                    dy = rs[:, 2].astype(np.int64) - fx["y_slope_valid"]
                    dc = rs[:, 3].astype(np.int64) - fx["x_centroid"][idx]
                    s_maxdiff = int(max(np.abs(dx).max(), np.abs(dy).max(),
                                        np.abs(dc).max()))
                    s_equal = s_maxdiff == 0
                else:
                    s_equal = False

            equivalent = z_equal and s_equal
            if not equivalent:
                n_equiv_fail += 1

            # ---- B: fixed vs float ----
            z_err = fx["zernike_m"] - fl["zernike_m"]
            cen_err = np.concatenate([
                fx["x_centroid"][idx] / scale_px - fl["x_centroid_px"][idx],
                fx["y_centroid"][idx] / scale_px - fl["y_centroid_px"][idx]])

            rows.append({
                "case": case["name"],
                "class": case["cls"],
                "equivalence_pass": int(equivalent),
                "zernike_exact_match": int(z_equal),
                "zernike_max_lsb_diff": z_maxdiff,
                "slope_centroid_exact_match": int(s_equal),
                "slope_centroid_max_lsb_diff": s_maxdiff,
                # Measured from the ARRIVAL of the final contributing pixel, so
                # this agrees with latency_sweep.py. rtl["latency_cycles"] is
                # anchored 2 cycles later at the accumulator completion event.
                "latency_cycles": rtl["true_latency_cycles"],
                "pixel_mismatches": rtl["pixel_mismatches"],
                "subaps_valid": rtl["subaps_valid"],
                "zernike_rmse_m": float(np.sqrt(np.mean(z_err ** 2))),
                "zernike_max_abs_err_m": float(np.max(np.abs(z_err))),
                "centroid_rmse_px": float(np.sqrt(np.mean(cen_err ** 2))),
                "centroid_max_abs_err_px": float(np.max(np.abs(cen_err))),
            })
            flag = "OK " if equivalent else "FAIL"
            print(f"  [{k+1:3d}/{len(cases)}] {flag} {case['name']:<22} "
                  f"lat={rtl['true_latency_cycles']}cy "
                  f"zRMSE={rows[-1]['zernike_rmse_m']:.2e}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (out_dir / "rtl_equivalence.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(
                f"{r[k]:.12e}" if isinstance(r[k], float) else str(r[k])
                for k in keys) + "\n")

    lat = [r["latency_cycles"] for r in rows if r["latency_cycles"] > 0]

    # Coefficient-level accounting: the equivalence claim is about individual
    # reconstructed coefficients, not just whole cases.
    n_coeff = len(rows) * C.NUM_MODES
    n_coeff_mismatch = sum(C.NUM_MODES for r in rows
                           if not r["zernike_exact_match"])
    n_slope = sum(r["subaps_valid"] * 3 for r in rows)   # x, y, centroid
    n_slope_mismatch = sum(r["subaps_valid"] * 3 for r in rows
                           if not r["slope_centroid_exact_match"])

    summary = {
        "n_cases": len(rows),
        "n_coefficients_checked": n_coeff,
        "n_coefficients_exact": n_coeff - n_coeff_mismatch,
        "n_coefficients_mismatched": n_coeff_mismatch,
        "coefficient_mismatch_rate": n_coeff_mismatch / n_coeff if n_coeff else 0.0,
        "n_slope_centroid_values_checked": n_slope,
        "n_slope_centroid_mismatched": n_slope_mismatch,
        "n_random": n_random,
        "seed": seed,
        "equivalence_failures": n_equiv_fail,
        "equivalence_pass_rate": 1.0 - n_equiv_fail / len(rows),
        "all_exact": n_equiv_fail == 0,
        "unique_latencies_cycles": sorted(set(lat)),
        "zernike_rmse_m_median": float(np.median([r["zernike_rmse_m"] for r in rows])),
        "centroid_rmse_px_median": float(np.median([r["centroid_rmse_px"] for r in rows])),
        "pixel_mismatch_total": int(sum(r["pixel_mismatches"] for r in rows)),
    }
    (out_dir / "rtl_equivalence_summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "rtl_equivalence_summary.csv").open("w") as fh:
        fh.write("metric,value\n")
        for k, v in summary.items():
            fh.write(f"{k},{v}\n")

    tex = [r"% RTL vs bit-accurate model: exact equivalence check.",
           r"\begin{tabular}{lr}", r"\hline", r"Quantity & Value \\", r"\hline",
           rf"Test cases & {summary['n_cases']} \\",
           rf"Coefficients checked & {summary['n_coefficients_checked']:,} \\",
           rf"Exact matches & {summary['n_coefficients_exact']:,} \\",
           rf"Mismatches & {summary['n_coefficients_mismatched']} \\",
           rf"Mismatch rate & {summary['coefficient_mismatch_rate']:.3e} \\",
           rf"Slope/centroid values checked & "
           rf"{summary['n_slope_centroid_values_checked']:,} \\",
           rf"Slope/centroid mismatches & "
           rf"{summary['n_slope_centroid_mismatched']} \\",
           r"\hline", r"\end{tabular}"]
    (out_dir / "rtl_equivalence.tex").write_text("\n".join(tex) + "\n")

    print(f"\n  coefficients checked : {summary['n_coefficients_checked']:,}")
    print(f"  exact matches        : {summary['n_coefficients_exact']:,}")
    print(f"  mismatches           : {summary['n_coefficients_mismatched']} "
          f"(rate {summary['coefficient_mismatch_rate']:.3e})")
    print(f"\nA. equivalence : {len(rows)-n_equiv_fail}/{len(rows)} exact "
          f"({'PASS' if n_equiv_fail == 0 else 'FAIL'})")
    print(f"B. numerical   : median Zernike RMSE "
          f"{summary['zernike_rmse_m_median']:.3e} m, "
          f"centroid {summary['centroid_rmse_px_median']:.3e} px")
    print(f"   latencies observed: {summary['unique_latencies_cycles']} cycles")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--random", type=int, default=100)
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED + 1)
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()
    run(a.random, a.seed, a.out)


if __name__ == "__main__":
    main()
