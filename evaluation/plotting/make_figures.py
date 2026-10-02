#!/usr/bin/env python3
"""
Paper figures, generated from the CSVs in results/.

Design notes:
  * 10 Zernike modes exceeds the 8-slot categorical palette, and cycling hues is
    not permitted, so the coefficient scatter POOLS all modes into one series
    rather than inventing colours. Per-mode detail is carried by the per-mode
    RMSE figure and the CSV, not by 10 hues.
  * Per-mode RMSE encodes magnitude, so it uses one hue (sequential), not
    categorical colours.
  * Single axis everywhere; recessive grid and spines; thin marks.
  * Latency is deterministic, so it gets a stage breakdown rather than a
    histogram of one repeated value -- and is shown against frame acquisition
    time, which is the architectural point.

Every figure is written as both PDF and PNG.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import config as C  # noqa: E402

# Validated palette slots 1 and 2 (see dataviz references/palette.md)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#1a1a19"
MUTED = "#8a8a87"
GRID = "#e3e3e0"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def _save(fig, name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def fig_coefficient_comparison(raw: dict, out: Path) -> None:
    """Fixed-point vs floating-point, all modes pooled, with y=x reference."""
    fixed = raw["fixed_m"] * 1e9      # nm
    flt = raw["float_m"] * 1e9

    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    lim = float(max(np.abs(flt).max(), np.abs(fixed).max())) * 1.05

    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1.0, zorder=1,
            label="$y=x$")
    ax.scatter(flt, fixed, s=3, c=BLUE, alpha=0.25, linewidths=0, zorder=2,
               label=f"coefficients ($n$={fixed.size:,})")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Floating-point reference (nm)")
    ax.set_ylabel("FPGA fixed-point (nm)")
    ax.grid(True)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    _tidy(ax)
    _save(fig, "coefficient_comparison", out)


def fig_error_distribution(raw: dict, out: Path) -> None:
    """Histogram and CDF of the reconstruction error."""
    err = raw["error_m"] * 1e9        # nm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))

    ax1.hist(err, bins=80, color=BLUE, alpha=0.85, linewidth=0)
    ax1.set_xlabel("Error (nm)")
    ax1.set_ylabel("Count")
    ax1.grid(True, axis="y")
    _tidy(ax1)

    a = np.sort(np.abs(err))
    cdf = np.arange(1, a.size + 1) / a.size
    ax2.plot(a, cdf, color=BLUE, lw=2.0)
    for q, style in ((0.95, ":"), (0.99, "--")):
        v = float(np.quantile(a, q))
        ax2.axvline(v, color=ORANGE, lw=1.0, ls=style)
        ax2.annotate(f"p{int(q*100)} = {v:.2f} nm", xy=(v, q),
                     xytext=(4, -14), textcoords="offset points",
                     fontsize=7, color=ORANGE)
    ax2.set_xlabel("Absolute error (nm)")
    ax2.set_ylabel("CDF")
    ax2.set_ylim(0, 1.02)
    ax2.grid(True)
    _tidy(ax2)

    fig.tight_layout()
    _save(fig, "error_distribution", out)


def fig_per_mode_rmse(per_mode: list[dict], out: Path) -> None:
    """Per-mode RMSE -- magnitude, so one hue."""
    labels = [r["quantity"] for r in per_mode]
    rmse = np.array([float(r["rmse"]) for r in per_mode]) * 1e9

    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.bar(np.arange(len(labels)), rmse, color=BLUE, width=0.62, linewidth=0)
    for i, v in enumerate(rmse):
        ax.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (nm)")
    ax.set_ylim(0, rmse.max() * 1.18)
    ax.grid(True, axis="y")
    _tidy(ax)
    _save(fig, "per_mode_rmse", out)


def fig_latency_breakdown(stages: list[dict], lat: dict, out: Path) -> None:
    """
    Stage breakdown, and the tail latency against frame acquisition time --
    the architectural claim that computation overlaps streaming.
    """
    rows = [r for r in stages if not r["stage"].startswith("TOTAL")]
    names = [r["stage"] for r in rows]
    cycles = [int(r["cycles"]) for r in rows]
    total = sum(cycles)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5),
                                   gridspec_kw={"width_ratios": [1.45, 1]})

    left = 0.0
    shades = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]   # one hue, ordinal
    for n, c, sh in zip(names, cycles, shades):
        ax1.barh([0], [c], left=left, color=sh, height=0.5, linewidth=0)
        ax1.annotate(f"{c}", xy=(left + c / 2, 0), ha="center", va="center",
                     fontsize=8, color=INK if sh != "#1c5cab" else "white")
        left += c
    ax1.set_xlim(0, total)
    ax1.set_ylim(-0.5, 0.9)
    ax1.set_yticks([])
    ax1.set_xlabel("Cycles after final contributing pixel")
    ax1.set_title(f"{total} cycles = {total*C.CLOCK_NS:.1f} ns "
                  f"@ {C.CLOCK_MHZ:.0f} MHz", fontsize=8.5, color=INK, pad=16)
    handles = [plt.Rectangle((0, 0), 1, 1, color=s) for s in shades]
    ax1.legend(handles, names, frameon=False, fontsize=6.6, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, -0.38))
    _tidy(ax1)
    ax1.spines["left"].set_visible(False)

    acq = float(lat["frame_input_cycles"])
    proc = float(total)
    ax2.bar([0, 1], [acq, proc], color=[MUTED, BLUE], width=0.55, linewidth=0)
    ax2.set_yscale("log")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Frame\nacquisition", "Processing\ntail"], fontsize=8)
    ax2.set_ylabel("Cycles (log)")
    for i, v in enumerate([acq, proc]):
        ax2.annotate(f"{v:,.0f}", xy=(i, v), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=7.5)
    ax2.grid(True, axis="y")
    _tidy(ax2)

    fig.tight_layout()
    _save(fig, "latency_breakdown", out)


def fig_noise_sweep(rows: list[dict], out: Path) -> None:
    """
    Centroid error vs photon flux: floating point against FPGA fixed point.
    Two series, so a legend is present and both are directly labelled.
    """
    flux = np.array([float(r["flux_photons"]) for r in rows])
    fl = np.array([float(r["float_centroid_rms_px"]) for r in rows])
    fx = np.array([float(r["fixed_centroid_rms_px"]) for r in rows])
    fl_ci = np.array([float(r["float_centroid_rms_px_ci95"]) for r in rows])
    fx_ci = np.array([float(r["fixed_centroid_rms_px_ci95"]) for r in rows])
    pen = np.array([float(r["degradation_pct_centroid_rms_px"]) for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    ax1.errorbar(flux, fl, yerr=fl_ci, color=MUTED, lw=2.0, marker="o", ms=4,
                 capsize=2, label="Floating point")
    ax1.errorbar(flux, fx, yerr=fx_ci, color=BLUE, lw=2.0, marker="s", ms=4,
                 capsize=2, label="FPGA fixed point")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Photons per subaperture")
    ax1.set_ylabel("Centroid RMS error (px)")
    ax1.grid(True, which="both")
    ax1.legend(frameon=False, fontsize=7.5, loc="lower left")
    _tidy(ax1)

    ax2.semilogx(flux, pen, color=ORANGE, lw=2.0, marker="o", ms=4)
    ax2.axhline(10, color=MUTED, lw=1.0, ls=":")
    ax2.annotate("10 %", xy=(flux[0], 10), xytext=(2, 3),
                 textcoords="offset points", fontsize=7, color=MUTED)
    ax2.set_xlabel("Photons per subaperture")
    ax2.set_ylabel("FPGA penalty (%)")
    ax2.grid(True, which="both")
    _tidy(ax2)

    fig.tight_layout()
    _save(fig, "noise_sweep", out)


def fig_dynamic_range(rows: list[dict], out: Path) -> None:
    """Relative reconstruction error vs aberration magnitude."""
    sub = [r for r in rows if r["family"] == "multimode"]
    mags = sorted({float(r["magnitude_m"]) for r in sub})
    rel, oot = [], []
    for m in mags:
        g = [r for r in sub if float(r["magnitude_m"]) == m]
        rel.append(np.mean([float(r["rel_rmse"]) for r in g]))
        oot.append(np.mean([float(r["spots_out_of_subaperture"]) for r in g]))
    mags = np.array(mags) * 1e9
    rel = np.array(rel); oot = np.array(oot)

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.loglog(mags, rel * 100, color=BLUE, lw=2.0, marker="o", ms=4,
              label="Relative RMSE")
    first_oot = mags[np.argmax(oot > 0)] if (oot > 0).any() else None
    if first_oot is not None:
        ax.axvline(first_oot, color=ORANGE, lw=1.2, ls="--")
        ax.annotate("spots leave\nsubaperture", xy=(first_oot, rel.max() * 100),
                    xytext=(-62, -6), textcoords="offset points",
                    fontsize=7, color=ORANGE)
    ax.set_xlabel("Input wavefront RMS (nm)")
    ax.set_ylabel("Relative reconstruction RMSE (%)")
    ax.grid(True, which="both")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    _tidy(ax)
    _save(fig, "dynamic_range", out)


def _read_csv(path: Path) -> list[dict]:
    import csv
    with path.open() as fh:
        return [r for r in csv.DictReader(fh) if not r[next(iter(r))].startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--figures", type=Path, default=C.FIGURES_DIR)
    a = ap.parse_args()

    print("Generating figures:")

    raw_path = a.results / "accuracy_raw_zernike.csv"
    if raw_path.exists():
        d = np.genfromtxt(raw_path, delimiter=",", names=True,
                          dtype=None, encoding="utf-8")
        raw = {"fixed_m": d["fixed_m"], "float_m": d["float_m"],
               "error_m": d["error_m"]}
        fig_coefficient_comparison(raw, a.figures)
        fig_error_distribution(raw, a.figures)
    else:
        print("  (skipped scatter/CDF -- run accuracy/monte_carlo.py first)")

    pm_path = a.results / "zernike_per_mode.csv"
    if pm_path.exists():
        fig_per_mode_rmse(_read_csv(pm_path), a.figures)
    else:
        print("  (skipped per-mode RMSE)")

    ns = a.results / "noise_sweep.csv"
    if ns.exists():
        fig_noise_sweep(_read_csv(ns), a.figures)
    else:
        print("  (skipped noise sweep -- run noise_sweep/noise_sweep.py)")

    dr = a.results / "dynamic_range.csv"
    if dr.exists():
        fig_dynamic_range(_read_csv(dr), a.figures)
    else:
        print("  (skipped dynamic range -- run dynamic_range/dynamic_range.py)")

    pb = a.results / "pipeline_breakdown.csv"
    ls = a.results / "latency_summary.json"
    if pb.exists() and ls.exists():
        fig_latency_breakdown(_read_csv(pb), json.loads(ls.read_text()), a.figures)
    else:
        print("  (skipped latency figure -- run latency/ scripts first)")


if __name__ == "__main__":
    main()
