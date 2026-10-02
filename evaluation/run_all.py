#!/usr/bin/env python3
"""
Run the complete evaluation campaign and regenerate everything in results/.

    sim/venv/bin/python evaluation/run_all.py
    ...                                       evaluation/run_all.py --quick
    ...                                       evaluation/run_all.py --skip-rtl

The venv is required: it carries hcipy, numpy, scipy and matplotlib.
Full run is roughly 6 minutes; --quick is under one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common import config as C  # noqa: E402

HERE = Path(__file__).resolve().parent


def step(name: str, script: Path, args: list[str], needs_rtl=False):
    print("\n" + "=" * 72)
    print(f" {name}")
    print("=" * 72)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(script)] + args)
    dt = time.time() - t0
    ok = proc.returncode == 0
    print(f"  -> {'ok' if ok else 'FAILED'} in {dt:.1f}s")
    return name, ok, dt, needs_rtl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-rtl", action="store_true",
                    help="skip steps that invoke the RTL simulator")
    a = ap.parse_args()
    C.ensure_dirs()

    q = a.quick
    res = []

    # ---- software-only experiments ----
    res.append(step("STEP 2  Monte Carlo numerical accuracy",
                    HERE / "numerical_accuracy" / "monte_carlo.py",
                    ["-n", "200" if q else "10000"]))
    res.append(step("STEP 11 Demo frame characterisation",
                    HERE / "numerical_accuracy" / "characterise_demo.py",
                    [] if q else ["--with-rtl"]))
    res.append(step("STEP 3  Error source ablation",
                    HERE / "error_ablation" / "ablation.py",
                    ["-n", "50" if q else "300"]))
    res.append(step("STEP 4  Noise sweep",
                    HERE / "noise_sweep" / "noise_sweep.py",
                    ["-n", "20" if q else "300"]))
    res.append(step("STEP 5  Dynamic range",
                    HERE / "dynamic_range" / "dynamic_range.py",
                    ["-n", "5" if q else "15"]))

    # ---- RTL-in-the-loop experiments ----
    if not a.skip_rtl:
        res.append(step("STEP 1  RTL exact equivalence",
                        HERE / "rtl_equivalence" / "regression.py",
                        ["-n", "10" if q else "150"], needs_rtl=True))
        res.append(step("STEP 6  Latency characterisation",
                        HERE / "latency" / "latency_sweep.py",
                        ["-n", "5" if q else "50"], needs_rtl=True))
        res.append(step("STEP 7  Pipeline latency breakdown",
                        HERE / "latency" / "pipeline_breakdown.py",
                        ["-n", "3" if q else "10"], needs_rtl=True))

    # ---- HCIPy optical campaign (independent optical inputs) ----
    res.append(step("OPT-B  Calibrate + freeze output scale",
                    HERE / "optical" / "calibrate.py",
                    ["-n", "20" if q else "60"]))
    res.append(step("OPT-E  Optical campaign (E1,E2,E3,E5)",
                    HERE / "optical" / "campaign.py",
                    ["--pilot"] if q else []))
    if not a.skip_rtl:
        res.append(step("OPT-G  RTL on optical frames",
                        HERE / "optical" / "rtl_optical_subset.py",
                        ["-n", "3" if q else "10"], needs_rtl=True))
    res.append(step("OPT-6  Paired arithmetic discrepancy",
                    HERE / "optical" / "arithmetic_discrepancy.py",
                    ["-n", "8" if q else "40"]))
    if not q:
        res.append(step("OPT-E4/6/7  Dynamic range, unmodelled, subpixel",
                        HERE / "optical" / "experiments_e4_e6_e7.py", []))
    res.append(step("OPT-K  Optical figures",
                    HERE / "optical" / "make_optical_figures.py", []))

    # ---- derived outputs ----
    res.append(step("STEP 8  Throughput", HERE / "throughput" / "throughput.py", []))
    res.append(step("STEP 9  Vivado report extraction",
                    HERE / "implementation" / "parse_reports.py", []))
    res.append(step("STEP 12 Figures", HERE / "plotting" / "make_figures.py", []))
    res.append(step("STEP 13 LaTeX tables", HERE / "plotting" / "make_tables.py", []))

    print("\n" + "=" * 72)
    print(" CAMPAIGN SUMMARY")
    print("=" * 72)
    for name, ok, dt, _ in res:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<46} {dt:6.1f}s")

    failed = [n for n, ok, _, _ in res if not ok]
    print(f"\n  results        : {C.RESULTS_DIR}")
    print(f"  full report    : {C.EVAL_DIR/'EVALUATION_REPORT.md'}")

    if q or a.skip_rtl:
        print()
        print("  " + "!" * 66)
        if q:
            print("  !! --quick: results/ now holds SMALL-SAMPLE numbers that do")
            print("  !! NOT match EVALUATION_REPORT.md. Re-run without --quick")
            print("  !! before quoting anything.")
        if a.skip_rtl:
            print("  !! --skip-rtl: RTL equivalence, latency and pipeline results")
            print("  !! in results/ are STALE (carried over from a previous run).")
        print("  " + "!" * 66)
    print("\n  NOT PRODUCED (no data exists):")
    print("    - FPGA resource / timing results: no Vivado report anywhere")
    print("    - experimental-data validation  : no compatible dataset")
    if failed:
        print(f"\n  {len(failed)} step(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
