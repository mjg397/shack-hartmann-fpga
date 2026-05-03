"""
generate_ematrix.py — Standalone FPGA E-Matrix & Calibration Generator

Generates ALL hex files needed by the FPGA Verilog simulation:
  - e_matrix_x.hex, e_matrix_y.hex   (reconstruction matrix, Q1.16, 18-bit signed)
  - subaperture_bitmap.hex            (256-bit validity mask)
  - slopes_ref_x.hex, slopes_ref_y.hex (reference centroids, Q4.23, 28-bit)
  - hcipy_image_rotating.hex          (test aberrated image, 8-bit)

Usage:
    python generate_ematrix.py

Output directory: ../full_pipeline_sim/data/
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils
from hcipy import Field, Wavefront, NoiselessDetector, Magnifier

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
NUM_LENSLETS   = 16
NUM_ZERNIKE    = 10
OUTPUT_DIR     = Path('../full_pipeline_sim/data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Build the optical model
# ═══════════════════════════════════════════════════════════════════════════
print("Step 1: Building optical model...")
case = shwfs_utils.generate_shwfs_case(
    num_lenslets=NUM_LENSLETS,
    num_zernike=NUM_ZERNIKE,
    demo_image_path=None,
)

zernike_modes = case["zernike_modes"]
aperture      = case["aperture"]
wavelength    = case["wavelength"]
shwfs_optics  = case["shwfs"]
true_coeffs   = case["true_coeffs"]

telescope_diameter = 8.0
magnification = 5e-3 / telescope_diameter
magnifier = Magnifier(magnification)

valid_mask = case["fpga_subaperture_mask"].ravel()
n_valid = int(valid_mask.sum())
print(f"  Valid subapertures: {n_valid}")

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Write subaperture bitmap
# ═══════════════════════════════════════════════════════════════════════════
print("Step 2: Writing subaperture bitmap...")
bitmap_bin_lsb_first = "".join(["1" if b else "0" for b in valid_mask])
bitmap_hex = hex(int(bitmap_bin_lsb_first[::-1], 2))[2:].zfill(64).upper()

with open(OUTPUT_DIR / 'subaperture_bitmap.hex', 'w') as f:
    f.write(f"{bitmap_hex}\n")
print(f"  Wrote subaperture_bitmap.hex ({n_valid} valid)")

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Compute reference centroids
# ═══════════════════════════════════════════════════════════════════════════
print("Step 3: Computing reference centroids...")
SCALE_Q23 = 1 << 23

def propagate_and_detect(opd_field):
    """Propagate an OPD field through the SHWFS and return the detector image."""
    phase = 2.0 * np.pi / wavelength * opd_field
    aber_ap = Field(aperture * np.exp(1j * np.asarray(phase)), aperture.grid)
    wf = Wavefront(aber_ap, wavelength)
    wf.total_power = 1
    wf_out = shwfs_optics(magnifier(wf))
    camera = NoiselessDetector(wf_out.electric_field.grid)
    camera.integrate(wf_out, 1)
    return camera.read_out()

def get_centroids_q23(image):
    """Get Q4.23 centroids from a detector image."""
    quantized = shwfs_utils.quantize_shwfs_image(image)
    est = shwfs_utils.run_fpga_like_estimation(quantized, 16, 16)
    return est["centroids_q4_23"]  # (256, 2) int64

def get_pixel_slopes(opd_field, ref_q23):
    """Get pixel-domain slopes relative to a reference."""
    img = propagate_and_detect(opd_field)
    c_q23 = get_centroids_q23(img)
    return (c_q23 - ref_q23).astype(np.float64) / SCALE_Q23

# Reference image (flat wavefront)
img_ref = propagate_and_detect(aperture * 0.0)
ref_centroids_q23 = get_centroids_q23(img_ref)
print(f"  Non-zero ref centroids: {np.count_nonzero(ref_centroids_q23)} / {ref_centroids_q23.size}")

# Write reference centroids
def write_hex_28(filename, values):
    with open(filename, 'w') as f:
        for v in values:
            f.write(f"{int(v) & 0xFFFFFFF:07X}\n")

write_hex_28(OUTPUT_DIR / 'slopes_ref_x.hex', ref_centroids_q23[:, 0])
write_hex_28(OUTPUT_DIR / 'slopes_ref_y.hex', ref_centroids_q23[:, 1])
print(f"  Wrote slopes_ref_x/y.hex (256 entries each)")

# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Build interaction matrix (pixel-slope domain)
# ═══════════════════════════════════════════════════════════════════════════
print("Step 4: Building interaction matrix...")
probe_amp = 0.05 * wavelength
IM_rows = []

for mi, mode in enumerate(zernike_modes):
    s_p = get_pixel_slopes(aperture * mode * (+probe_amp), ref_centroids_q23)
    s_m = get_pixel_slopes(aperture * mode * (-probe_amp), ref_centroids_q23)
    ds_da = (s_p - s_m) / (2.0 * probe_amp)
    ds_valid = ds_da[valid_mask]
    row = np.concatenate([ds_valid[:, 0], ds_valid[:, 1]])
    IM_rows.append(row)
    print(f"  Mode {mi+1:2d}: |ds/da|_max = {np.max(np.abs(row)):.2f} px/m")

IM = np.array(IM_rows)
print(f"  IM shape: {IM.shape}, condition number: {np.linalg.cond(IM):.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Invert → Reconstruction Matrix
# ═══════════════════════════════════════════════════════════════════════════
print("Step 5: Computing reconstruction matrix (Tikhonov-regularized SVD)...")
rcond = 1e-3
U, s, Vt = np.linalg.svd(IM, full_matrices=False)
s_reg = s / (s**2 + (rcond * s.max())**2)
RM = (Vt.T * s_reg) @ U.T  # shape: (2*n_valid, 10)
print(f"  RM shape: {RM.shape}, range: [{RM.min():.4e}, {RM.max():.4e}]")

# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Quantize RM with global scale factor G (power of 2)
# ═══════════════════════════════════════════════════════════════════════════
print("Step 6: Quantizing reconstruction matrix...")
RM_nm = RM * 1e9                       # nm per pixel-slope
raw_q16 = RM_nm * (1 << 16)           # ideal Q1.16 values
max_abs = np.max(np.abs(raw_q16))
SIGNED_18_MAX = 131071.0               # 2^17 - 1

# Global scale factor: shrink e-matrix entries to fit in 18 bits
G_exact = max_abs / SIGNED_18_MAX
G_pow2  = int(2 ** np.ceil(np.log2(max(G_exact, 1))))
G_shift = int(np.log2(G_pow2))

print(f"  Global scale factor G = {G_pow2} (2^{G_shift})")
print(f"  max |raw_q16| = {max_abs:.1f}")
print(f"  after /G: max = {max_abs / G_pow2:.1f} (limit: {SIGNED_18_MAX:.0f})")

# Quantize and write
e_x_vals = []
e_y_vals = []

for m in range(NUM_ZERNIKE):
    col = RM_nm[:, m]
    ex = col[:n_valid]
    ey = col[n_valid:]
    e_x_vals.extend(np.round(ex * (1 << 16) / G_pow2).astype(np.int64))
    e_y_vals.extend(np.round(ey * (1 << 16) / G_pow2).astype(np.int64))

max_after = max(max(abs(int(v)) for v in e_x_vals), max(abs(int(v)) for v in e_y_vals))
print(f"  Quantized max: {max_after} (limit: {int(SIGNED_18_MAX)})")

def write_hex_18(filename, values):
    with open(filename, 'w') as f:
        for v in values:
            v = int(max(-131072, min(131071, v)))
            f.write(f"{v & 0x3FFFF:05X}\n")

write_hex_18(OUTPUT_DIR / 'e_matrix_x.hex', e_x_vals)
write_hex_18(OUTPUT_DIR / 'e_matrix_y.hex', e_y_vals)
print(f"  Wrote {len(e_x_vals)} entries each to e_matrix_x/y.hex")

# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Export aberrated test image
# ═══════════════════════════════════════════════════════════════════════════
print("Step 7: Exporting test aberrated image...")
img_aber = propagate_and_detect(case["input_opd_field"])
quantized_aber = shwfs_utils.quantize_shwfs_image(img_aber)
with open(OUTPUT_DIR / 'hcipy_image_rotating.hex', 'w') as f:
    for val in quantized_aber:
        f.write(f"{int(val):02X}\n")
print(f"  Wrote {len(quantized_aber)} pixels to hcipy_image_rotating.hex")

# ═══════════════════════════════════════════════════════════════════════════
# Step 8: Bit-accurate FPGA prediction (verification)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"Bit-accurate FPGA prediction  (G = {G_pow2}, shift = {G_shift})")
print(f"{'='*70}")

# Float prediction
aber_slopes_px = get_pixel_slopes(case["input_opd_field"], ref_centroids_q23)
slope_vec = np.concatenate([
    aber_slopes_px[valid_mask, 0],
    aber_slopes_px[valid_mask, 1],
])
float_pred_m  = slope_vec @ RM
float_pred_nm = float_pred_m * 1e9
true_nm       = true_coeffs * 1e9

mode_names = ["Tilt X","Tilt Y","Defocus","Astig 45","Astig 0",
              "Coma X","Coma Y","Trefoil X","Trefoil Y","Sph."]

print(f"\n{'Mode':<12s}  {'True (nm)':>12s}  {'Float pred (nm)':>15s}")
print("-" * 45)
for i in range(NUM_ZERNIKE):
    print(f"{mode_names[i]:<12s}  {true_nm[i]:12.4f}  {float_pred_nm[i]:15.4f}")

# Bit-accurate integer prediction
aber_slopes_q23_int = np.round(
    get_pixel_slopes(case["input_opd_field"], ref_centroids_q23) * SCALE_Q23
).astype(np.int64)
sx_q23 = aber_slopes_q23_int[valid_mask, 0]
sy_q23 = aber_slopes_q23_int[valid_mask, 1]

e_x_arr = np.array(e_x_vals).reshape(NUM_ZERNIKE, n_valid)
e_y_arr = np.array(e_y_vals).reshape(NUM_ZERNIKE, n_valid)

print(f"\n{'Mode':<12s}  {'Raw Q4.22':>12s}  {'nm (×G)':>12s}  {'True (nm)':>12s}  {'Ratio':>8s}")
print("-" * 64)
for m in range(NUM_ZERNIKE):
    acc = np.int64(0)
    for s_idx in range(n_valid):
        acc += np.int64(e_x_arr[m, s_idx]) * np.int64(sx_q23[s_idx])
        acc += np.int64(e_y_arr[m, s_idx]) * np.int64(sy_q23[s_idx])
    raw_out = (acc >> 17) & ((1 << 27) - 1)
    if raw_out & (1 << 26):
        raw_out -= (1 << 27)
    nm_val = (raw_out / (1 << 22)) * G_pow2
    t = true_nm[m]
    ratio = nm_val / t if abs(t) > 0.001 else float('nan')
    print(f"{mode_names[m]:<12s}  {raw_out:12d}  {nm_val:12.4f}  {t:12.4f}  {ratio:8.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Valid subapertures: {n_valid}")
print(f"  Verilog parameter:  NUM_SUBS = {n_valid}")
print(f"  Verilog parameter:  NUM_SLOPES = {2*n_valid}")
print(f"  Scale factor:       G = {G_pow2} (2^{G_shift})")
print(f"  Conversion:         nm = raw_fpga_output / 2^{22-G_shift}")
print(f"  Or equivalently:    nm = (raw / 2^22) * {G_pow2}")
print(f"\nAll files written to: {OUTPUT_DIR.resolve()}")
print("Done!")
