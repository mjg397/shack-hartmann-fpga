#!/usr/bin/env python3
"""
Latency and throughput characterisation.

Measures, per frame, from the RTL simulation:

    first input pixel cycle
    final CONTRIBUTING pixel cycle   (the pixel completing the last in-pupil
                                      subaperture, not the last pixel of the
                                      frame -- the emulator keeps streaming
                                      after the pupil is finished)
    first subap_valid cycle
    output valid (done) cycle

and computes

    final_pixel_to_result_cycles = done_cycle - final_contributing_pixel_cycle

Frame ACQUISITION time is reported separately from processing latency: the
architectural claim is that reconstruction happens while the frame streams in,
leaving only a small fixed tail after the final pixel.

Outputs
    results/latency_results.csv
    results/latency_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import config as C                # noqa: E402
from evaluation.common import framegen as FG             # noqa: E402
from evaluation.common import rtl_runner as R            # noqa: E402


def frame_timing() -> dict:
    """
    Frame input duration from streaming_emulator.v parameters.

    Per line: HSIZE active + HBLANK, except the final line which is followed by
    VBLANK instead. The emulator also spends one cycle in STATE_LINE_PRIME per
    line and one in STATE_FRAME_INIT per frame.
    """
    active = C.FRAME_W * C.FRAME_H
    hblank = C.HBLANK * (C.FRAME_H - 1)
    prime = C.FRAME_H            # STATE_LINE_PRIME, one per line
    init = 1                     # STATE_FRAME_INIT
    vblank = C.VBLANK
    input_cycles = active + hblank + prime + init
    total_cycles = input_cycles + vblank
    return {
        "active_pixel_cycles": active,
        "hblank_cycles": hblank,
        "line_prime_cycles": prime,
        "frame_init_cycles": init,
        "vblank_cycles": vblank,
        "frame_input_cycles": input_cycles,
        "frame_total_cycles": total_cycles,
        "frame_input_us": input_cycles * C.CLOCK_NS / 1000.0,
        "frame_total_us": total_cycles * C.CLOCK_NS / 1000.0,
        "max_frame_rate_hz": C.CLOCK_MHZ * 1e6 / total_cycles,
        "max_pixel_rate_mpx_s": C.CLOCK_MHZ,   # 1 pixel/clock when streaming
    }


def run(n_frames: int, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    rows = []

    with R.RTLSimulator() as sim:
        for i in range(n_frames):
            case = FG.make_case(rng, scale_m=6e-8)
            r = sim.run_frame(case["frame"])
            rows.append({
                "frame": i,
                "first_pixel_cycle": r["first_pixel_cycle"],
                "final_contributing_pixel_index": r["final_pixel_index"],
                "last_input_pixel_cycle": r["last_pixel_cycle"],
                "first_subap_valid_cycle": r["first_subap_cycle"],
                "done_cycle": r["done_cycle"],
                "final_tile_pixel_cycle": r["final_tile_pixel_cycle"],
                # Measured from the ARRIVAL of the final contributing pixel.
                "final_pixel_to_result_cycles": r["true_latency_cycles"],
                "final_pixel_to_result_ns": r["true_latency_cycles"] * C.CLOCK_NS,
                "accumulator_depth_cycles": r["accumulator_depth_cycles"],
                # The same measurement from three anchor points. All are real;
                # they differ only in where the pipeline is considered to start.
                "latency_from_s1_cycles": r["latency_from_s1_cycles"],
                "latency_from_accum_cycles": r["latency_from_accum_cycles"],
                "subap_valid_to_done_cycles": r["subap_to_done_cycles"],
                "pixels_streamed": r["pixels_streamed"],
                "subaps_valid": r["subaps_valid"],
                "pixel_mismatches": r["pixel_mismatches"],
                "max_back_to_back_run": r["max_back_to_back_run"],
                "stall_events": r["stall_events"],
            })
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_frames} frames", flush=True)

    lat = np.array([r["final_pixel_to_result_cycles"] for r in rows])
    unique = sorted(set(int(x) for x in lat))
    deterministic = len(unique) == 1

    lat_s1 = np.array([r["latency_from_s1_cycles"] for r in rows])
    lat_acc = np.array([r["latency_from_accum_cycles"] for r in rows])

    ft = frame_timing()
    runs = np.array([r["max_back_to_back_run"] for r in rows])
    stalls = np.array([r["stall_events"] for r in rows])

    summary = {
        "n_frames": n_frames,
        "seed": seed,
        "clock_mhz": C.CLOCK_MHZ,
        "clock_period_ns": C.CLOCK_NS,
        # --- processing latency (after the final contributing pixel) ---
        "latency_min_cycles": int(lat.min()),
        "latency_max_cycles": int(lat.max()),
        "latency_mean_cycles": float(lat.mean()),
        "latency_unique_cycles": unique,
        "latency_deterministic": deterministic,
        "latency_min_ns": float(lat.min() * C.CLOCK_NS),
        "latency_max_ns": float(lat.max() * C.CLOCK_NS),
        "latency_mean_ns": float(lat.mean() * C.CLOCK_NS),
        # --- the same latency from three anchor points ---
        # All three are real measurements of one run; they differ only in where
        # the pipeline is taken to start. State the anchor in any paper claim.
        "anchor_emulator_output_cycles": int(lat.mean()),
        "anchor_emulator_output_ns": float(lat.mean() * C.CLOCK_NS),
        "anchor_accumulator_s1_cycles": int(lat_s1.mean()),
        "anchor_accumulator_s1_ns": float(lat_s1.mean() * C.CLOCK_NS),
        "anchor_accumulator_done_cycles": int(lat_acc.mean()),
        "anchor_accumulator_done_ns": float(lat_acc.mean() * C.CLOCK_NS),
        "anchor_recommended": "emulator_output",
        "anchor_note": ("emulator_output is the conservative choice: it is what "
                        "an external camera feeding the FPGA observes. "
                        "accumulator_s1 excludes the input capture register."),
        # --- throughput ---
        "accepted_pixels_per_clock": 1.0,
        "initiation_interval_cycles": 1,
        # s_ready is tied high in shwfs_processing.v:33 -- the pipeline never
        # back-pressures. Confirmed empirically: the longest unbroken run of
        # accepted pixels equals the full line width, so a whole line is taken
        # back-to-back at one pixel per clock.
        "back_to_back_accepted": bool((runs >= C.FRAME_W).all()),
        "max_back_to_back_run_min": int(runs.min()),
        "max_back_to_back_run_max": int(runs.max()),
        "line_width_pixels": C.FRAME_W,
        # These gaps are the emulator's HBLANK between lines, NOT pipeline
        # back-pressure: one per line boundary.
        "blanking_gaps_per_frame_median": float(np.median(stalls)),
        **ft,
        # --- integrity ---
        "pixel_mismatch_total": int(sum(r["pixel_mismatches"] for r in rows)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (out_dir / "latency_results.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(
                f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                for k in keys) + "\n")
    (out_dir / "latency_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"  frames                    : {n_frames}")
    print(f"  latency (cycles)          : min {lat.min()} max {lat.max()} "
          f"mean {lat.mean():.3f}")
    print(f"  unique latencies observed : {unique}")
    print(f"  deterministic             : {deterministic}")
    print(f"  latency @ {C.CLOCK_MHZ:.0f} MHz        : "
          f"{lat.mean()*C.CLOCK_NS:.2f} ns")
    print(f"  frame input duration      : {ft['frame_input_cycles']} cy "
          f"({ft['frame_input_us']:.2f} us)")
    print(f"  max frame rate            : {ft['max_frame_rate_hz']:.1f} Hz")
    print(f"  back-to-back pixels       : {summary['back_to_back_accepted']} "
          f"(longest run {runs.min()} = full line; s_ready tied high)")

    if not deterministic:
        print("\n  WARNING: latency is NOT deterministic across frames.")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--frames", type=int, default=50)
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED + 2)
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()
    print(f"Latency sweep: {a.frames} randomised frames")
    run(a.frames, a.seed, a.out)


if __name__ == "__main__":
    main()
