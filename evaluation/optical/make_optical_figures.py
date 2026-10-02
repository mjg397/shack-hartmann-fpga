#!/usr/bin/env python3
"""Figures for the HCIPy optical campaign. PDF + SVG + PNG, CSV already saved."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.common import config as C          # noqa: E402
from evaluation.common import optics as O          # noqa: E402
from evaluation.common import detector as D        # noqa: E402
from evaluation.common import bitaccurate as BA    # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#8a8a87", "#e3e3e0"
plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def tidy(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name, out):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig); print(f"  {name}.pdf / .svg / .png")


def rows(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def fig_amplitude(res, out):
    r = rows(res / "mixed_summary.csv")
    x = np.array([float(v["true_rms_mean_m"]) for v in r]) * 1e9
    fl = np.array([float(v["float_residual_mean_m"]) for v in r]) * 1e9
    fx = np.array([float(v["fixed_residual_mean_m"]) for v in r]) * 1e9
    fe = np.array([float(v["float_residual_sem_m"]) for v in r]) * 1e9
    xe = np.array([float(v["fixed_residual_sem_m"]) for v in r]) * 1e9
    n = [v["n"] for v in r]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.9))
    a1.errorbar(x, fl, yerr=fe, color=MUTED, lw=2, marker="o", ms=4, capsize=2,
                label="Floating point")
    a1.errorbar(x, fx, yerr=xe, color=BLUE, lw=2, marker="s", ms=4, capsize=2,
                label="FPGA fixed point")
    a1.plot(x, x, color=ORANGE, lw=1, ls=":", label="residual = input (no recovery)")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("Input wavefront RMS (nm)")
    a1.set_ylabel("OPD residual RMS (nm)")
    a1.grid(True, which="both"); a1.legend(frameon=False, fontsize=7)
    tidy(a1)

    a2.semilogx(x, fl / x * 100, color=MUTED, lw=2, marker="o", ms=4)
    a2.set_xlabel("Input wavefront RMS (nm)")
    a2.set_ylabel("Relative residual (%)")
    a2.grid(True, which="both"); tidy(a2)
    for xi, yi, ni in zip(x, fl / x * 100, n):
        a2.annotate(f"n={ni}", xy=(xi, yi), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=6, color=MUTED)
    fig.tight_layout(); save(fig, "optical_error_vs_amplitude", out)


def fig_signal(res, out):
    r = [v for v in rows(res / "signal_sweep.csv") if v["noise_case"] == "shot_only"]
    ph = np.array([float(v["photons"]) for v in r])
    fl = np.array([float(v["float_residual_mean_m"]) for v in r]) * 1e9
    fx = np.array([float(v["fixed_residual_mean_m"]) for v in r]) * 1e9
    clip = np.array([float(v["clipped_adu_total"]) for v in r])

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ok = clip == 0
    ax.errorbar(ph[ok], fl[ok], color=MUTED, lw=2, marker="o", ms=4,
                label="Floating point")
    ax.errorbar(ph[ok], fx[ok], color=BLUE, lw=2, marker="s", ms=4,
                label="FPGA fixed point")
    if (~ok).any():
        ax.plot(ph[~ok], fl[~ok], color=MUTED, lw=1.2, ls="--", marker="o",
                ms=3, alpha=0.55)
        ax.plot(ph[~ok], fx[~ok], color=BLUE, lw=1.2, ls="--", marker="s",
                ms=3, alpha=0.55)
        edge = ph[~ok].min()
        ax.axvline(edge, color=ORANGE, lw=1.2, ls="--")
        ax.annotate("ADC clipping\nbeyond here", xy=(edge, fl.max()),
                    xytext=(6, -4), textcoords="offset points",
                    fontsize=7, color=ORANGE)
    ax.set_xscale("log"); ax.set_xlabel("Photons over the pupil")
    ax.set_ylabel("OPD residual RMS (nm)")
    ax.grid(True, which="both"); ax.legend(frameon=False, fontsize=7.5)
    tidy(ax); fig.tight_layout(); save(fig, "optical_error_vs_signal", out)


def fig_modes(res, out):
    r = rows(res / "mode_summary.csv")
    lab = [v["label"] for v in r]
    gain = np.array([float(v["gain"]) for v in r])
    xt = np.array([float(v["crosstalk_rms_m"]) for v in r]) * 1e9
    i = np.arange(len(lab))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 4.0), sharex=True)
    a1.bar(i, gain, color=BLUE, width=0.62, linewidth=0)
    a1.axhline(1.0, color=ORANGE, lw=1.2, ls="--")
    a1.annotate("ideal gain = 1", xy=(len(lab) - 1, 1.0), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=7, color=ORANGE)
    a1.set_ylabel("Recovery gain"); a1.grid(True, axis="y"); tidy(a1)
    a2.bar(i, xt, color=MUTED, width=0.62, linewidth=0)
    a2.set_ylabel("Cross-talk RMS (nm)"); a2.grid(True, axis="y"); tidy(a2)
    a2.set_xticks(i); a2.set_xticklabels(lab, rotation=35, ha="right", fontsize=8)
    fig.tight_layout(); save(fig, "optical_mode_recovery", out)


def fig_wavefront(out):
    """True / reconstructed / residual OPD maps on a shared scale."""
    b = O.get_bench()
    rng = np.random.default_rng(31337)
    c = rng.normal(0, 1, C.NUM_MODES)
    c *= 150e-9 / b.opd_rms(b.opd_from_coeffs(c))
    img, opd = b.propagate_coeffs(c)
    adu = D.expose(img, D.DetectorConfig(shot_noise=False, read_noise_e=0.0)).adu

    scale = json.loads((C.RESULTS_DIR / "optical" / "calibration.json").read_text())["scale_m_per_lsb"]
    fx = BA.run_pipeline(adu.ravel())
    rec = b.opd_from_coeffs(fx["zernike_raw"].astype(float) * scale)

    n = b.cfg.num_pupil_pixels
    mask = b.pupil_mask.reshape(n, n)
    fmt = lambda a: np.where(mask, np.asarray(a).reshape(n, n), np.nan) * 1e9
    T, R = fmt(opd), fmt(rec)
    T = T - np.nanmean(T); R = R - np.nanmean(R)
    E = T - R
    lim = np.nanmax(np.abs(T))

    fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.7))
    spot = np.asarray(adu).reshape(C.FRAME_H, C.FRAME_W)
    axs[0].imshow(spot, cmap="gray", origin="lower")
    for k in range(1, 16):
        axs[0].axhline(k * 16, color=ORANGE, lw=0.3, alpha=0.6)
        axs[0].axvline(k * 16, color=ORANGE, lw=0.3, alpha=0.6)
    axs[0].set_title("Detector (lenslet grid)", fontsize=8.5); axs[0].axis("off")
    for ax, d, t in ((axs[1], T, "Known input OPD"),
                     (axs[2], R, "FPGA reconstruction"),
                     (axs[3], E, "Residual")):
        im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        ax.set_title(t, fontsize=8.5); ax.axis("off")
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.02, pad=0.01)
    cb.set_label("OPD (nm)", fontsize=8)
    save(fig, "optical_wavefront_maps", out)


def main() -> None:
    res = C.RESULTS_DIR / "optical"
    out = C.FIGURES_DIR / "optical"
    print("Optical figures:")
    if (res / "mixed_summary.csv").exists(): fig_amplitude(res, out)
    if (res / "signal_sweep.csv").exists(): fig_signal(res, out)
    if (res / "mode_summary.csv").exists(): fig_modes(res, out)
    fig_wavefront(out)


if __name__ == "__main__":
    main()
