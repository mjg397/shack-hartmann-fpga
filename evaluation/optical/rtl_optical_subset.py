#!/usr/bin/env python3
"""
STEP G -- run the ACTUAL RTL on genuine optical frames.

The software campaign uses the bit-accurate model for speed. This script
establishes that the model is entitled to stand in for the RTL on the very
frames the campaign uses, by running a representative optical subset through
Icarus and comparing exact output words.

Cases are HCIPy-propagated optical wavefronts, not pinv(E) constructions, plus
directed edge cases. Every comparison is exact: any mismatch is a real
discrepancy.

Passing a finite set of tests supports correctness on those tests. It is not a
proof of equivalence for all possible inputs, and is not claimed as one.

Outputs
    evaluation/results/optical/rtl_optical_subset.csv
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
from evaluation.common import rtl_runner as R            # noqa: E402

SEED = 7711


def build_cases(bench, n_optical: int, rng) -> list[tuple[str, np.ndarray]]:
    det = D.DetectorConfig()
    clean = D.DetectorConfig(shot_noise=False, read_noise_e=0.0)
    cases: list[tuple[str, np.ndarray]] = []

    flat = np.zeros_like(np.asarray(bench.aperture), dtype=float)
    cases.append(("optical_flat_noiseless", D.expose(bench.propagate(flat), clean).adu))
    cases.append(("optical_flat_noisy", D.expose(bench.propagate(flat), det, rng).adu))

    for m in range(C.NUM_MODES):
        c = np.zeros(C.NUM_MODES); c[m] = 80e-9
        img, _ = bench.propagate_coeffs(c)
        cases.append((f"optical_mode{m+1}", D.expose(img, clean).adu))

    for i in range(n_optical):
        c = rng.normal(0.0, 1.0, C.NUM_MODES)
        c *= (100e-9) / bench.opd_rms(bench.opd_from_coeffs(c))
        img, _ = bench.propagate_coeffs(c)
        cases.append((f"optical_mixed_{i:02d}", D.expose(img, det, rng).adu))

    # Directed arithmetic / protocol edges -- cheap, not optical.
    n = C.FRAME_W * C.FRAME_H
    cases.append(("edge_zero_intensity", np.zeros(n, dtype=np.int64)))
    cases.append(("edge_max_pixels", np.full(n, 255, dtype=np.int64)))
    dim = np.zeros(n, dtype=np.int64); dim[::37] = 6      # just above threshold
    cases.append(("edge_dim_lenslets", dim))
    fp = np.zeros(n, dtype=np.int64); fp[0] = 255; fp[-1] = 255
    cases.append(("edge_first_last_pixel", fp))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--optical", type=int, default=12)
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR / "optical")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    bench = O.get_bench()
    rng = np.random.default_rng(SEED)
    cases = build_cases(bench, a.optical, rng)
    print(f"RTL on optical frames: {len(cases)} cases "
          f"({sum(1 for n,_ in cases if n.startswith('optical'))} optical, "
          f"{sum(1 for n,_ in cases if n.startswith('edge'))} directed)")

    rows, mism, lat = [], 0, set()
    with R.RTLSimulator() as sim:
        for name, adu in cases:
            px = np.asarray(adu, dtype=np.int64).ravel()
            rtl = sim.run_frame(px)
            model = BA.run_pipeline(px)["zernike_raw"]
            exact = bool(np.array_equal(rtl["zernike_raw"], model))
            maxdiff = int(np.max(np.abs(rtl["zernike_raw"] - model)))
            if not exact:
                mism += 1
            lat.add(rtl["true_latency_cycles"])
            rows.append({"case": name, "exact": int(exact),
                         "max_lsb_diff": maxdiff,
                         "latency_cycles": rtl["true_latency_cycles"],
                         "pixel_mismatches": rtl["pixel_mismatches"],
                         "subaps_valid": rtl["subaps_valid"]})
            print(f"  {'OK  ' if exact else 'FAIL'} {name:26s} "
                  f"lat={rtl['true_latency_cycles']}cy", flush=True)

    keys = list(rows[0].keys())
    with (a.out / "rtl_optical_subset.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    n_coeff = len(rows) * C.NUM_MODES
    summary = {
        "n_cases": len(rows),
        "n_optical_cases": sum(1 for r in rows if r["case"].startswith("optical")),
        "n_directed_cases": sum(1 for r in rows if r["case"].startswith("edge")),
        "coefficients_compared": n_coeff,
        "case_mismatches": mism,
        "all_exact": mism == 0,
        "latency_cycles_observed": sorted(int(x) for x in lat),
        "seed": SEED,
        "scope": ("supports correctness on these tests; not a proof of "
                  "equivalence for all inputs"),
    }
    (a.out / "rtl_optical_subset.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  cases {len(rows)}, coefficients {n_coeff}, mismatches {mism}")
    print(f"  latency observed: {summary['latency_cycles_observed']} cycles")


if __name__ == "__main__":
    main()
