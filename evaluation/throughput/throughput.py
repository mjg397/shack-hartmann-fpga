#!/usr/bin/env python3
"""
STEP 8 -- Streaming throughput, measured rather than assumed.

Derives everything from the RTL simulation via latency_summary.json (produced by
evaluation/latency/latency_sweep.py), which records per-frame:

    longest unbroken run of accepted pixels
    interruptions during the active region
    first/last accepted pixel cycles

Distinctions preserved deliberately:
  * PROCESSING throughput (what the FPGA sustains) is not a detector frame rate.
    The theoretical frames/s below is what the pipeline could consume given a
    source able to supply pixels at one per clock. It is NOT a claim that any
    detector runs at that rate.
  * Blanking gaps are the streaming_emulator's HBLANK, not pipeline
    back-pressure. s_ready is tied high in shwfs_processing.v:33, so the
    pipeline never stalls the source.

Outputs
    results/throughput.csv
    results/throughput.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import config as C  # noqa: E402


def build(summary: dict) -> list[dict]:
    px_per_s = C.CLOCK_MHZ * 1e6 * summary["accepted_pixels_per_clock"]
    return [
        {"metric": "clock_frequency_mhz", "value": C.CLOCK_MHZ,
         "note": "target; see implementation results for closure evidence"},
        {"metric": "accepted_pixels_per_clock",
         "value": summary["accepted_pixels_per_clock"],
         "note": "measured: longest unbroken accepted run equals the line width"},
        {"metric": "initiation_interval_cycles",
         "value": summary["initiation_interval_cycles"], "note": "one pixel per clock"},
        {"metric": "back_to_back_accepted",
         "value": int(summary["back_to_back_accepted"]),
         "note": "s_ready tied high (shwfs_processing.v:33); no back-pressure"},
        {"metric": "longest_back_to_back_run_px",
         "value": summary["max_back_to_back_run_min"],
         "note": f"equals frame width {C.FRAME_W}"},
        {"metric": "blanking_gaps_per_frame",
         "value": summary.get("blanking_gaps_per_frame_median", float("nan")),
         "note": "emulator HBLANK between lines, NOT pipeline stalls"},
        {"metric": "pixels_per_second", "value": px_per_s,
         "note": "processing throughput at the target clock"},
        {"metric": "frame_active_pixels", "value": C.FRAME_W * C.FRAME_H,
         "note": "256x256"},
        {"metric": "frame_input_cycles", "value": summary["frame_input_cycles"],
         "note": "active + HBLANK + line prime + frame init"},
        {"metric": "frame_input_time_us", "value": summary["frame_input_us"],
         "note": "minimum streaming input time for one frame"},
        {"metric": "frame_total_cycles", "value": summary["frame_total_cycles"],
         "note": "including VBLANK"},
        {"metric": "max_processing_frame_rate_hz",
         "value": summary["max_frame_rate_hz"],
         "note": "PROCESSING capability only; not a detector frame rate"},
        {"metric": "final_pixel_to_result_cycles",
         "value": summary["latency_mean_cycles"],
         "note": "emulator-output anchor; tail latency, separate from acquisition"},
        {"metric": "final_pixel_to_result_ns",
         "value": summary["latency_mean_ns"], "note": "at the target clock"},
        {"metric": "acquisition_to_tail_ratio",
         "value": summary["frame_input_cycles"] / summary["latency_mean_cycles"],
         "note": "how much larger frame acquisition is than the processing tail"},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()

    p = a.results / "latency_summary.json"
    if not p.exists():
        print("latency_summary.json missing -- run evaluation/latency/latency_sweep.py")
        sys.exit(1)
    s = json.loads(p.read_text())
    rows = build(s)

    with (a.results / "throughput.csv").open("w") as fh:
        fh.write("metric,value,note\n")
        for r in rows:
            fh.write(f"{r['metric']},{r['value']},\"{r['note']}\"\n")

    fmt = {
        "clock_frequency_mhz": ("Clock frequency", "{:.0f} MHz"),
        "accepted_pixels_per_clock": ("Accepted pixels per clock", "{:.0f}"),
        "initiation_interval_cycles": ("Initiation interval", "{:.0f} cycle"),
        "pixels_per_second": ("Sustained pixel rate", "{:.3g} pixel/s"),
        "frame_input_time_us": ("Frame input time ($256\\times256$)", "{:.1f}~$\\mu$s"),
        "max_processing_frame_rate_hz": ("Max processing frame rate", "{:.0f} Hz"),
        "final_pixel_to_result_cycles": ("Final-pixel-to-result", "{:.0f} cycles"),
        "final_pixel_to_result_ns": ("Final-pixel-to-result", "{:.1f}~ns"),
    }
    lines = [r"% Throughput. Processing capability, NOT a detector frame rate.",
             r"\begin{tabular}{ll}", r"\hline", r"Metric & Value \\", r"\hline"]
    for r in rows:
        if r["metric"] in fmt:
            label, f = fmt[r["metric"]]
            lines.append(rf"{label} & {f.format(float(r['value']))} \\")
    lines += [r"\hline", r"\end{tabular}"]
    (a.results / "throughput.tex").write_text("\n".join(lines) + "\n")

    print("Throughput (measured):")
    for r in rows:
        v = r["value"]
        vs = f"{v:.4g}" if isinstance(v, float) else str(v)
        print(f"  {r['metric']:34s} {vs}")


if __name__ == "__main__":
    main()
