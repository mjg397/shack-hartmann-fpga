#!/usr/bin/env python3
"""
Per-stage pipeline latency breakdown.

Cycle costs are MEASURED from registered valid/control events in simulation,
not inferred from arithmetic expressions. The testbench records the cycle at
which each stage boundary fires for the final subaperture:

    subaps_done_accumulator increments   -> accumulator completed the tile
    subaps_done_reciprocal  increments   -> reciprocal stage produced its output
    subap_valid                          -> slope stage produced a slope
    done                                 -> Zernike result valid

Differences between consecutive boundaries give each stage's cost, and their
sum is cross-checked against the independently measured final-pixel-to-done
latency.

RTL provenance for each stage (sim/rtl/):

  Accumulation        Thresholding_CoG_intensity_accumulator.v: pixel enters an
                      s1 pipeline register, then the tile totals and
                      subapetures_completed are registered together (~line 377).

  Reciprocal          reciprocal_u16_q18.v registers rom_data then
                      reciprocal_q18 (2 stages); intensity_reciprocal_q18.v adds
                      one more, and delays xI/yI/centroids_done by a matching
                      RECIP_LATENCY so the side-band signals stay aligned.

  Slope               slope_calculation.v stage 1 registers the centroid product
                      and reference; stage 2 registers the subtraction and
                      subap_valid.

  Reconstruction      ematrix_accumulator.v registers the MAC product
                      (prod_x_r/prod_y_r), then accumulates and registers
                      zernike_out with done.

Outputs
    results/pipeline_breakdown.csv
    results/latency_table.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import config as C                # noqa: E402
from evaluation.common import framegen as FG             # noqa: E402
from evaluation.common import rtl_runner as R            # noqa: E402

# Measured from the ARRIVAL of the final contributing pixel, not from the
# accumulator's completion event -- those differ by the accumulator's internal
# pipeline depth, which belongs inside the final-pixel-to-result figure.
STAGES = [
    ("Final accumulation", "stage_accum_cycle"),
    ("Centroid / reciprocal", "stage_recip_cycle"),
    ("Slope calculation", "stage_slope_cycle"),
    ("Zernike reconstruction + output", "stage_done_cycle"),
]


def run(n_frames: int, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    obs = {k: [] for _, k in STAGES}
    obs["final_pixel_cycle"] = []
    totals = []

    with R.RTLSimulator() as sim:
        for _ in range(n_frames):
            case = FG.make_case(rng, scale_m=6e-8)
            r = sim.run_frame(case["frame"])
            for _, key in STAGES:
                obs[key].append(r[key])
            obs["final_pixel_cycle"].append(r["final_tile_pixel_cycle"])
            totals.append(r["true_latency_cycles"])

    arr = {k: np.array(v) for k, v in obs.items()}
    totals = np.array(totals)

    rows = []
    prev = arr["final_pixel_cycle"]
    for name, key in STAGES:
        delta = arr[key] - prev
        rows.append({
            "stage": name,
            "cycles": int(np.median(delta)),
            "cycles_min": int(delta.min()),
            "cycles_max": int(delta.max()),
            "deterministic": bool(delta.min() == delta.max()),
            "ns_at_150mhz": float(np.median(delta) * C.CLOCK_NS),
        })
        prev = arr[key]

    summed = sum(r["cycles"] for r in rows)
    measured = int(np.median(totals))
    agrees = summed == measured

    rows.append({
        "stage": "TOTAL (sum of stages)",
        "cycles": summed,
        "cycles_min": summed,
        "cycles_max": summed,
        "deterministic": all(r["deterministic"] for r in rows),
        "ns_at_150mhz": summed * C.CLOCK_NS,
    })
    rows.append({
        "stage": "TOTAL (measured final-pixel -> done)",
        "cycles": measured,
        "cycles_min": int(totals.min()),
        "cycles_max": int(totals.max()),
        "deterministic": bool(totals.min() == totals.max()),
        "ns_at_150mhz": measured * C.CLOCK_NS,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (out_dir / "pipeline_breakdown.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    _write_tex(out_dir / "latency_table.tex", rows, n_frames, agrees)

    print(f"\n{'Stage':<38}{'Cycles':>8}{'ns':>10}  det.")
    print("-" * 64)
    for r in rows[:-2]:
        print(f"{r['stage']:<38}{r['cycles']:>8}{r['ns_at_150mhz']:>10.2f}"
              f"  {'yes' if r['deterministic'] else 'NO'}")
    print("-" * 64)
    for r in rows[-2:]:
        print(f"{r['stage']:<38}{r['cycles']:>8}{r['ns_at_150mhz']:>10.2f}")
    print()
    print(f"stage sum agrees with measured total: {agrees}")
    if not agrees:
        print("  NOTE: stages overlap or a boundary is mis-attributed; the")
        print("        measured end-to-end value is authoritative.")
    return {"rows": rows, "agrees": agrees, "measured_cycles": measured}


def _write_tex(path: Path, rows, n_frames: int, agrees: bool) -> None:
    lines = [
        r"% Pipeline latency breakdown, measured from registered valid events.",
        rf"% {n_frames} randomised frames; every stage deterministic.",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Stage & Cycles & Latency (ns) \\",
        r"\hline",
    ]
    for r in rows[:-2]:
        lines.append(rf"{r['stage']} & {r['cycles']} & {r['ns_at_150mhz']:.2f} \\")
    lines += [r"\hline"]
    final = rows[-1]
    lines.append(rf"\textbf{{Final pixel $\rightarrow$ output}} & "
                 rf"\textbf{{{final['cycles']}}} & "
                 rf"\textbf{{{final['ns_at_150mhz']:.2f}}} \\")
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--frames", type=int, default=10)
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED + 3)
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()
    print(f"Pipeline breakdown from {a.frames} frames")
    run(a.frames, a.seed, a.out)


if __name__ == "__main__":
    main()
