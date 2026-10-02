#!/usr/bin/env python3
"""
LaTeX-ready summary tables built from the CSV/JSON results.

Writes
    results/accuracy_table.tex
    results/per_mode_table.tex
    results/throughput_table.tex
    results/rtl_regression_table.tex

results/latency_table.tex is written by latency/pipeline_breakdown.py.
results/implementation_results.tex by vivado_reports/parse_reports.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import config as C  # noqa: E402


def _rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return [r for r in csv.DictReader(fh)
                if r and not next(iter(r.values())).startswith("#")]


def _f(v, fmt="{:.3e}") -> str:
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def accuracy_table(results: Path) -> str:
    rows = _rows(results / "accuracy_summary.csv")
    want = {
        "centroid_px": ("Centroid", "px"),
        "slope_px": ("Slope", "px"),
        "zernike_m_aggregate": ("Zernike coefficient", "m"),
    }
    out = [
        r"% Fixed-point vs floating-point accuracy (identical algorithm).",
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Quantity & Unit & Bias & MAE & RMSE & $p_{95}$ & Max \\",
        r"\hline",
    ]
    for r in rows:
        key = r["quantity"]
        if key not in want:
            continue
        label, unit = want[key]
        out.append(
            rf"{label} & {unit} & {_f(r['bias_mean_signed_error'])} & "
            rf"{_f(r['mae'])} & {_f(r['rmse'])} & "
            rf"{_f(r['p95_abs_error'])} & {_f(r['max_abs_error'])} \\")
    out += [r"\hline", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def per_mode_table(results: Path) -> str:
    rows = _rows(results / "zernike_per_mode.csv")
    out = [
        r"% Per-mode Zernike accuracy, fixed point vs floating point.",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Mode & Bias (nm) & MAE (nm) & RMSE (nm) & $p_{99}$ (nm) & RMSE ($\lambda$) \\",
        r"\hline",
    ]
    for r in rows:
        out.append(
            rf"{r['quantity']} & {_f(float(r['bias_mean_signed_error'])*1e9, '{:.3f}')} & "
            rf"{_f(float(r['mae'])*1e9, '{:.3f}')} & "
            rf"{_f(float(r['rmse'])*1e9, '{:.3f}')} & "
            rf"{_f(float(r['p99_abs_error'])*1e9, '{:.3f}')} & "
            rf"{_f(r['rmse_waves'], '{:.2e}')} \\")
    out += [r"\hline", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def throughput_table(results: Path) -> str:
    p = results / "latency_summary.json"
    if not p.exists():
        return "% latency_summary.json missing -- run evaluation/latency/latency_sweep.py\n"
    s = json.loads(p.read_text())
    rows = [
        ("Clock frequency", f"{s['clock_mhz']:.0f} MHz"),
        ("Accepted pixels per clock", f"{s['accepted_pixels_per_clock']:.0f}"),
        ("Initiation interval", f"{s['initiation_interval_cycles']} cycle"),
        ("Back-to-back acceptance", "yes" if s["back_to_back_accepted"] else "no"),
        ("Sustained pixel rate", f"{s['max_pixel_rate_mpx_s']:.0f} Mpixel/s"),
        ("Frame input duration ($256\\times256$)",
         f"{s['frame_input_cycles']:,} cycles ({s['frame_input_us']:.1f}~$\\mu$s)"),
        ("Frame period incl. blanking",
         f"{s['frame_total_cycles']:,} cycles ({s['frame_total_us']:.1f}~$\\mu$s)"),
        ("Maximum frame rate", f"{s['max_frame_rate_hz']:.0f} Hz"),
        ("Final-pixel-to-result latency",
         f"{s['latency_mean_cycles']:.0f} cycles ({s['latency_mean_ns']:.1f}~ns)"),
        ("Latency determinism",
         f"{len(s['latency_unique_cycles'])} unique value over "
         f"{s['n_frames']} frames"),
    ]
    out = [r"% Throughput and latency. Acquisition is SEPARATE from the",
           r"% post-final-pixel processing tail.",
           r"\begin{tabular}{ll}", r"\hline",
           r"Metric & Value \\", r"\hline"]
    out += [rf"{k} & {v} \\" for k, v in rows]
    out += [r"\hline", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def regression_table(results: Path) -> str:
    p = results / "rtl_equivalence_summary.json"
    if not p.exists():
        return "% rtl_regression_summary.json missing\n"
    s = json.loads(p.read_text())
    rows = [
        ("Test cases", f"{s['n_cases']}"),
        ("RTL vs bit-accurate model", "exact on all cases"
            if s["all_exact"] else f"{s['equivalence_failures']} FAILURES"),
        ("Equivalence failures", f"{s['equivalence_failures']}"),
        ("Pixel-stream mismatches", f"{s['pixel_mismatch_total']}"),
        ("Median Zernike RMSE (fixed vs float)",
         f"{s['zernike_rmse_m_median']:.3e} m"),
        ("Median centroid RMSE", f"{s['centroid_rmse_px_median']:.3e} px"),
        ("Observed latencies",
         ", ".join(str(x) for x in s["unique_latencies_cycles"]) + " cycles"),
    ]
    out = [r"% RTL regression: A exact equivalence, B numerical error.",
           r"\begin{tabular}{ll}", r"\hline",
           r"Check & Result \\", r"\hline"]
    out += [rf"{k} & {v} \\" for k, v in rows]
    out += [r"\hline", r"\end{tabular}"]
    return "\n".join(out) + "\n"


PRIOR_WORK_TEMPLATE = r"""% Prior-work comparison -- TEMPLATE, DELIBERATELY EMPTY.
%
% Every cell must be filled from a cited publication that you have read. Do not
% populate this from memory or from secondary sources.
%
% Latency figures are NOT comparable unless the anchor matches. This design's
% figure is 9 cycles / 60.0 ns measured from the final contributing pixel
% ARRIVING at the processing input (see EVALUATION_REPORT.md section 4.6). Published figures
% variously measure from end-of-frame, from end-of-exposure, or include readout
% -- record the definition in its own column, and do not compare across
% different definitions.
%
% Do not claim superiority on any row where the configurations differ in
% subaperture count, mode count, or reconstruction method.
\begin{tabular}{llllrrlll}
\hline
Work & FPGA & WFS config. & Recon. method & Modes & $f$ (MHz) &
Latency & Latency definition & Resources \\
\hline
This work & (device TBD) & $16\times16$, 168 valid & Modal MAC (dense E) & 10 &
150\textsuperscript{*} & 60.0~ns & final pixel arrival $\rightarrow$ output &
(TBD) \\
\hline
 &  &  &  &  &  &  &  &  \\
 &  &  &  &  &  &  &  &  \\
 &  &  &  &  &  &  &  &  \\
\hline
\end{tabular}

% \textsuperscript{*} target clock; timing closure not yet verified -- no
% post-implementation report exists. See EVALUATION_REPORT.md section 4.9.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()
    a.results.mkdir(parents=True, exist_ok=True)

    (a.results / "prior_work_comparison_TEMPLATE.tex").write_text(PRIOR_WORK_TEMPLATE)
    print("  prior_work_comparison_TEMPLATE.tex (empty by design)")

    jobs = [
        ("accuracy_table.tex", accuracy_table),
        ("per_mode_table.tex", per_mode_table),
        ("zernike_accuracy_table.tex", per_mode_table),
        ("throughput_table.tex", throughput_table),
        ("rtl_regression_table.tex", regression_table),
    ]
    print("Generating LaTeX tables:")
    for name, fn in jobs:
        try:
            (a.results / name).write_text(fn(a.results))
            print(f"  {name}")
        except FileNotFoundError as e:
            print(f"  {name}: SKIPPED ({e.filename} missing)")


if __name__ == "__main__":
    main()
