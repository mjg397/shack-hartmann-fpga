"""
Plot FPGA Verilog simulation results alongside Python model predictions.

Takes the raw Q4.23 Zernike outputs from the FPGA pipeline and produces
publication-quality comparison figures matching the HCIPy visualization style.

Usage:
    python plot_fpga_results.py
"""
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils
from hcipy import Field, Wavefront, NoiselessDetector, Magnifier

# ── Configuration ──────────────────────────────────────────────────────────
G_SCALE = 8          # Global scale factor (power of 2)
Q_FRAC_BITS = 22     # Fractional bits in Q4.22 output

MODE_NAMES = [
    "Tilt X", "Tilt Y", "Defocus", "Astig 45", "Astig 0",
    "Coma X", "Coma Y", "Trefoil X", "Trefoil Y", "Sph.",
]

# FPGA Verilog simulation output (raw Q4.23 signed integers)
FPGA_RAW = np.array([
    32606488,   # Tilt X
    20598286,   # Tilt Y
    31438578,   # Defocus
    18360958,   # Astig 45
   -14435202,   # Astig 0
     1194566,   # Coma X
     1290705,   # Coma Y
      388962,   # Trefoil X
      -68723,   # Trefoil Y
     1173552,   # Sph.
])

# ── Derive nm values ──────────────────────────────────────────────────────
fpga_nm = (FPGA_RAW / (1 << Q_FRAC_BITS)) * G_SCALE

# ── Rebuild the pixel-slope RM (same pipeline as build_rm_and_test.py) ────
case = shwfs_utils.generate_shwfs_case(
    num_lenslets=16,
    num_zernike=10,
    demo_image_path=None,
)
zernike_modes = case["zernike_modes"]
aperture      = case["aperture"]
wavelength    = case["wavelength"]
shwfs_optics  = case["shwfs"]
true_coeffs   = case["true_coeffs"]
true_nm       = true_coeffs * 1e9

telescope_diameter = 8.0
magnification = 5e-3 / telescope_diameter
magnifier = Magnifier(magnification)

valid_mask = case["fpga_subaperture_mask"].ravel()
n_valid = int(valid_mask.sum())

SCALE_Q23 = 1 << 23

def propagate_and_detect(opd_field):
    phase = 2.0 * np.pi / wavelength * opd_field
    aber_ap = Field(aperture * np.exp(1j * np.asarray(phase)), aperture.grid)
    wf = Wavefront(aber_ap, wavelength)
    wf.total_power = 1
    wf_out = shwfs_optics(magnifier(wf))
    camera = NoiselessDetector(wf_out.electric_field.grid)
    camera.integrate(wf_out, 1)
    return camera.read_out()

def get_centroids_q23(image):
    quantized = shwfs_utils.quantize_shwfs_image(image)
    est = shwfs_utils.run_fpga_like_estimation(quantized, 16, 16)
    return est["centroids_q4_23"]

def get_pixel_slopes(opd_field, ref_q23):
    img = propagate_and_detect(opd_field)
    c_q23 = get_centroids_q23(img)
    return (c_q23 - ref_q23).astype(np.float64) / SCALE_Q23

# Reference centroids
img_ref = propagate_and_detect(aperture * 0.0)
ref_centroids_q23 = get_centroids_q23(img_ref)

# Interaction matrix in pixel-slope domain
probe_amp = 0.05 * wavelength
IM_rows = []
for mode in zernike_modes:
    s_p = get_pixel_slopes(aperture * mode * (+probe_amp), ref_centroids_q23)
    s_m = get_pixel_slopes(aperture * mode * (-probe_amp), ref_centroids_q23)
    ds_da = (s_p - s_m) / (2.0 * probe_amp)
    ds_valid = ds_da[valid_mask]
    row = np.concatenate([ds_valid[:, 0], ds_valid[:, 1]])
    IM_rows.append(row)
IM = np.array(IM_rows)

# Reconstruction matrix
rcond = 1e-3
U, s, Vt = np.linalg.svd(IM, full_matrices=False)
s_reg = s / (s**2 + (rcond * s.max())**2)
RM = (Vt.T * s_reg) @ U.T

# Float prediction
aber_slopes_px = get_pixel_slopes(case["input_opd_field"], ref_centroids_q23)
slope_vec = np.concatenate([
    aber_slopes_px[valid_mask, 0],
    aber_slopes_px[valid_mask, 1],
])
float_pred_m  = slope_vec @ RM
float_pred_nm = float_pred_m * 1e9

print(f"Pipeline rebuilt: {n_valid} valid subapertures")
print(f"Float prediction (nm): {np.round(float_pred_nm, 2)}")
print(f"FPGA output     (nm): {np.round(fpga_nm, 2)}")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Bar chart — True vs Float vs FPGA
# ═══════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(14, 6))
fig1.patch.set_facecolor('#0d1117')
ax1.set_facecolor('#161b22')

x = np.arange(len(MODE_NAMES))
bar_width = 0.25

bars_true = ax1.bar(x - bar_width, true_nm, bar_width, label='Ground Truth',
                    color='#58a6ff', edgecolor='#58a6ff', alpha=0.9, zorder=3)
bars_float = ax1.bar(x, float_pred_nm, bar_width, label='Python Model',
                     color='#f0883e', edgecolor='#f0883e', alpha=0.9, zorder=3)
bars_fpga = ax1.bar(x + bar_width, fpga_nm, bar_width, label='FPGA (Verilog)',
                    color='#3fb950', edgecolor='#3fb950', alpha=0.9, zorder=3)

ax1.set_xticks(x)
ax1.set_xticklabels(MODE_NAMES, rotation=45, ha='right', fontsize=11, color='#c9d1d9')
ax1.set_ylabel('Coefficient [nm]', fontsize=13, color='#c9d1d9')
ax1.set_title('Zernike Coefficient Reconstruction: Ground Truth vs Python vs FPGA',
              fontsize=15, fontweight='bold', color='#f0f6fc', pad=15)
ax1.axhline(0, color='#484f58', linewidth=0.8, zorder=1)
ax1.legend(fontsize=11, loc='upper right', facecolor='#21262d', edgecolor='#30363d',
           labelcolor='#c9d1d9')
ax1.tick_params(colors='#8b949e')
ax1.spines['bottom'].set_color('#30363d')
ax1.spines['left'].set_color('#30363d')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', color='#21262d', linewidth=0.5, zorder=0)

plt.tight_layout()
fig1.savefig('fpga_zernike_comparison.png', dpi=200, bbox_inches='tight',
             facecolor=fig1.get_facecolor())
print(f"Saved: fpga_zernike_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Reconstruction error analysis (2x2 grid)
# ═══════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(16, 12))
fig2.patch.set_facecolor('#0d1117')
gs = GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.3)

active_mask = np.abs(true_nm) > 0.01
error_nm = fpga_nm - true_nm

# Panel (a): Residual error per mode
ax2a = fig2.add_subplot(gs[0, 0])
ax2a.set_facecolor('#161b22')
ax2a.bar(x, error_nm, 0.6, color='#da3633', alpha=0.85, edgecolor='#f85149', zorder=3)
ax2a.set_xticks(x)
ax2a.set_xticklabels(MODE_NAMES, rotation=45, ha='right', fontsize=9, color='#c9d1d9')
ax2a.set_ylabel('Error [nm]', fontsize=11, color='#c9d1d9')
ax2a.set_title('(a) FPGA Reconstruction Error per Mode', fontsize=13,
               fontweight='bold', color='#f0f6fc')
ax2a.axhline(0, color='#484f58', linewidth=0.8, zorder=1)
ax2a.tick_params(colors='#8b949e')
ax2a.spines['bottom'].set_color('#30363d')
ax2a.spines['left'].set_color('#30363d')
ax2a.spines['top'].set_visible(False)
ax2a.spines['right'].set_visible(False)
ax2a.grid(axis='y', color='#21262d', linewidth=0.5, zorder=0)

# Panel (b): FPGA vs Python float parity
ax2b = fig2.add_subplot(gs[0, 1])
ax2b.set_facecolor('#161b22')
fpga_vs_float_err = fpga_nm - float_pred_nm
max_err = np.max(np.abs(fpga_vs_float_err))
ax2b.bar(x, fpga_vs_float_err, 0.6, color='#8957e5', alpha=0.85,
         edgecolor='#a371f7', zorder=3)
ax2b.set_xticks(x)
ax2b.set_xticklabels(MODE_NAMES, rotation=45, ha='right', fontsize=9, color='#c9d1d9')
ax2b.set_ylabel('FPGA − Python [nm]', fontsize=11, color='#c9d1d9')
ax2b.set_title(f'(b) FPGA vs Python Parity  (max |Δ| = {max_err:.4f} nm)',
               fontsize=13, fontweight='bold', color='#f0f6fc')
ax2b.axhline(0, color='#484f58', linewidth=0.8, zorder=1)
ax2b.tick_params(colors='#8b949e')
ax2b.spines['bottom'].set_color('#30363d')
ax2b.spines['left'].set_color('#30363d')
ax2b.spines['top'].set_visible(False)
ax2b.spines['right'].set_visible(False)
ax2b.grid(axis='y', color='#21262d', linewidth=0.5, zorder=0)

# Panel (c): Gain ratio for active modes
ax2c = fig2.add_subplot(gs[1, 0])
ax2c.set_facecolor('#161b22')
gain_fpga  = np.full(10, np.nan)
gain_float = np.full(10, np.nan)
gain_fpga[active_mask]  = fpga_nm[active_mask] / true_nm[active_mask]
gain_float[active_mask] = float_pred_nm[active_mask] / true_nm[active_mask]
active_idx = np.where(active_mask)[0]

ax2c.bar(x[active_mask] - 0.2, gain_float[active_mask], 0.35,
         label='Python', color='#f0883e', alpha=0.9, edgecolor='#f0883e', zorder=3)
ax2c.bar(x[active_mask] + 0.2, gain_fpga[active_mask], 0.35,
         label='FPGA', color='#3fb950', alpha=0.9, edgecolor='#3fb950', zorder=3)
ax2c.axhline(1.0, color='#58a6ff', linewidth=1.2, linestyle='--', zorder=2, label='Ideal (1.0)')
ax2c.set_xticks(x[active_mask])
ax2c.set_xticklabels(np.array(MODE_NAMES)[active_mask], rotation=45,
                     ha='right', fontsize=9, color='#c9d1d9')
ax2c.set_ylabel('Gain (estimated / true)', fontsize=11, color='#c9d1d9')
ax2c.set_title('(c) Reconstruction Gain (Active Modes)', fontsize=13,
               fontweight='bold', color='#f0f6fc')
ax2c.legend(fontsize=10, facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
ax2c.tick_params(colors='#8b949e')
ax2c.spines['bottom'].set_color('#30363d')
ax2c.spines['left'].set_color('#30363d')
ax2c.spines['top'].set_visible(False)
ax2c.spines['right'].set_visible(False)
ax2c.grid(axis='y', color='#21262d', linewidth=0.5, zorder=0)

# Panel (d): Crosstalk — leakage into zero-truth modes
ax2d = fig2.add_subplot(gs[1, 1])
ax2d.set_facecolor('#161b22')
inactive_mask = ~active_mask
if np.any(inactive_mask):
    ax2d.bar(x[inactive_mask], fpga_nm[inactive_mask], 0.6,
             color='#f778ba', alpha=0.85, edgecolor='#ff7eb6', zorder=3)
    ax2d.set_xticks(x[inactive_mask])
    ax2d.set_xticklabels(np.array(MODE_NAMES)[inactive_mask], rotation=45,
                         ha='right', fontsize=9, color='#c9d1d9')
ax2d.set_ylabel('Leakage [nm]', fontsize=11, color='#c9d1d9')
ax2d.set_title('(d) Crosstalk: FPGA Output for Zero-Truth Modes', fontsize=13,
               fontweight='bold', color='#f0f6fc')
ax2d.axhline(0, color='#484f58', linewidth=0.8, zorder=1)
ax2d.tick_params(colors='#8b949e')
ax2d.spines['bottom'].set_color('#30363d')
ax2d.spines['left'].set_color('#30363d')
ax2d.spines['top'].set_visible(False)
ax2d.spines['right'].set_visible(False)
ax2d.grid(axis='y', color='#21262d', linewidth=0.5, zorder=0)

# Stats annotation
rms_error = np.sqrt(np.mean(error_nm[active_mask]**2))
rms_crosstalk = np.sqrt(np.mean(fpga_nm[inactive_mask]**2)) if np.any(inactive_mask) else 0
fig2.suptitle(
    f'FPGA Reconstruction Analysis  —  {n_valid} subapertures, G = {G_SCALE}, '
    f'RMS error = {rms_error:.2f} nm, RMS crosstalk = {rms_crosstalk:.2f} nm',
    fontsize=14, fontweight='bold', color='#f0f6fc', y=0.98,
)
fig2.savefig('fpga_error_analysis.png', dpi=200, bbox_inches='tight',
             facecolor=fig2.get_facecolor())
print(f"Saved: fpga_error_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Summary table as figure
# ═══════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(12, 5))
fig3.patch.set_facecolor('#0d1117')
ax3.set_facecolor('#0d1117')
ax3.axis('off')

table_data = []
for i in range(10):
    ratio = fpga_nm[i] / true_nm[i] if abs(true_nm[i]) > 0.01 else "—"
    if isinstance(ratio, float):
        ratio = f"{ratio:.4f}"
    table_data.append([
        MODE_NAMES[i],
        f"{true_nm[i]:.2f}",
        f"{float_pred_nm[i]:.2f}",
        f"{FPGA_RAW[i]:,d}",
        f"{fpga_nm[i]:.2f}",
        ratio,
    ])

col_labels = ['Mode', 'True (nm)', 'Python (nm)', 'FPGA Raw (Q4.22)', 'FPGA (nm)', 'Gain']
table = ax3.table(cellText=table_data, colLabels=col_labels,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)

for key, cell in table.get_celld().items():
    cell.set_edgecolor('#30363d')
    if key[0] == 0:
        cell.set_facecolor('#21262d')
        cell.set_text_props(color='#58a6ff', fontweight='bold')
    else:
        cell.set_facecolor('#161b22')
        cell.set_text_props(color='#c9d1d9')

ax3.set_title('FPGA Zernike Reconstruction — Full Results Table',
              fontsize=14, fontweight='bold', color='#f0f6fc', pad=20)
fig3.savefig('fpga_results_table.png', dpi=200, bbox_inches='tight',
             facecolor=fig3.get_facecolor())
print(f"Saved: fpga_results_table.png")

plt.close('all')
print("\nAll figures saved successfully.")
