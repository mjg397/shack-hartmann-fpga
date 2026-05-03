"""
Build the FPGA Reconstruction Matrix with a global scale factor.

The e-matrix values (Q1.16, 18-bit signed) overflow when we try to encode
the full nm-per-pixel RM. Solution: divide all e-matrix entries by a global
scale factor G so they fit in 18 bits. The FPGA output is then:

    coeff_nm = (raw_fpga_output / 2^22) * G

This preserves relative precision across all entries.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils
from hcipy import Field, Wavefront, NoiselessDetector, Magnifier

# ── 1. Build the optical model ─────────────────────────────────────────────
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

telescope_diameter = 8.0
sh_diameter = 5e-3
magnification = sh_diameter / telescope_diameter
magnifier = Magnifier(magnification)

wf_ref = Wavefront(aperture, wavelength)
wf_ref.total_power = 1
wf_out_ref = shwfs_optics(magnifier(wf_ref))
detector_grid = wf_out_ref.electric_field.grid
camera = NoiselessDetector(detector_grid)

# ── 2. Get and write bitmap ────────────────────────────────────────────────
valid_mask = case["fpga_subaperture_mask"].ravel()
n_valid = int(valid_mask.sum())
print(f"Bitmap: {n_valid} valid subapertures")

bitmap_bin_lsb_first = "".join(["1" if b else "0" for b in valid_mask])
bitmap_hex = hex(int(bitmap_bin_lsb_first[::-1], 2))[2:].zfill(64).upper()

output_dir = Path('../simulation/data')
with open(output_dir / 'subaperture_bitmap.hex', 'w') as f:
    f.write(f"{bitmap_hex}\n")
print(f"Wrote bitmap to subaperture_bitmap.hex")

# ── 3. Helpers ─────────────────────────────────────────────────────────────
SCALE_Q23 = 1 << 23

def propagate_and_detect(opd_field):
    phase = 2.0 * np.pi / wavelength * opd_field
    aber_ap = Field(aperture * np.exp(1j * np.asarray(phase)), aperture.grid)
    wf = Wavefront(aber_ap, wavelength)
    wf.total_power = 1
    camera.integrate(shwfs_optics(magnifier(wf)), 1)
    return camera.read_out()

def get_centroids_q23(image):
    quantized = shwfs_utils.quantize_shwfs_image(image)
    est = shwfs_utils.run_fpga_like_estimation(
        quantized, num_subapertures_side=16, subaperture_pixels=16
    )
    return est["centroids_q4_23"]  # (256, 2) int64

def get_pixel_slopes(opd_field, ref_q23):
    img = propagate_and_detect(opd_field)
    c_q23 = get_centroids_q23(img)
    return (c_q23 - ref_q23).astype(np.float64) / SCALE_Q23

# ── 4. Reference centroids ────────────────────────────────────────────────
img_ref = propagate_and_detect(aperture * 0.0)
ref_centroids_q23 = get_centroids_q23(img_ref)
print(f"Non-zero ref centroids: {np.count_nonzero(ref_centroids_q23)} / {ref_centroids_q23.size}")

# ── 5. Build interaction matrix (pixel slopes) ────────────────────────────
probe_amp = 0.05 * wavelength
IM_rows = []

for mi, mode in enumerate(zernike_modes):
    s_p = get_pixel_slopes(aperture * mode * (+probe_amp), ref_centroids_q23)
    s_m = get_pixel_slopes(aperture * mode * (-probe_amp), ref_centroids_q23)
    ds_da = (s_p - s_m) / (2.0 * probe_amp)
    ds_valid = ds_da[valid_mask]
    row = np.concatenate([ds_valid[:, 0], ds_valid[:, 1]])
    IM_rows.append(row)
    print(f"  mode {mi+1}: |ds/da|_max = {np.max(np.abs(row)):.2f} px/m")

IM = np.array(IM_rows)
print(f"\nIM shape: {IM.shape}, cond = {np.linalg.cond(IM):.1f}")

# ── 6. Invert → RM (metres-of-OPD per pixel-slope) ───────────────────────
rcond = 1e-3
U, s, Vt = np.linalg.svd(IM, full_matrices=False)
s_reg = s / (s**2 + (rcond * s.max())**2)
RM = (Vt.T * s_reg) @ U.T  # (2*n_valid, 10)
print(f"RM shape: {RM.shape}, range: [{RM.min():.4e}, {RM.max():.4e}]")

# ── 7. Float forward test ────────────────────────────────────────────────
aber_slopes_px = get_pixel_slopes(case["input_opd_field"], ref_centroids_q23)
slope_vec = np.concatenate([
    aber_slopes_px[valid_mask, 0],
    aber_slopes_px[valid_mask, 1],
])
predicted_m  = slope_vec @ RM
predicted_nm = predicted_m * 1e9
true_nm      = true_coeffs * 1e9

mode_names = ["Tilt X","Tilt Y","Defocus","Astig 45","Astig 0",
              "Coma X","Coma Y","Trefoil X","Trefoil Y","Sph."]

print(f"\n{'Mode':<12s}  {'True (nm)':>12s}  {'Float pred (nm)':>15s}")
print("-" * 45)
for i in range(10):
    print(f"{mode_names[i]:<12s}  {true_nm[i]:12.4f}  {predicted_nm[i]:15.4f}")

# ── 8. Quantize RM with global scale factor ──────────────────────────────
RM_nm = RM * 1e9                       # nm per pixel-slope
raw_q16 = RM_nm * (1 << 16)           # what we'd LIKE to write
max_abs = np.max(np.abs(raw_q16))
SIGNED_18_MAX = 131071.0               # 2^17 - 1

# Global scale factor: how much we shrink the e-matrix to fit 18 bits
# Use a power-of-2 for the scale factor so FPGA can just shift
G_exact = max_abs / SIGNED_18_MAX
G_pow2  = int(2 ** np.ceil(np.log2(G_exact)))  # round up to next power of 2
G_shift = int(np.log2(G_pow2))

print(f"\nGlobal scale factor G = {G_pow2} (2^{G_shift})")
print(f"  max |raw_q16| = {max_abs:.1f}")
print(f"  after /G: max = {max_abs / G_pow2:.1f} (limit: {SIGNED_18_MAX:.0f})")
print(f"  To get nm from FPGA output: raw / 2^22 * {G_pow2}")
print(f"  Or equivalently: raw >> (22 - {G_shift}) = raw >> {22 - G_shift}")

# Quantize
e_x_vals = []
e_y_vals = []
NUM_MODES = 10

for m in range(NUM_MODES):
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

output_dir = Path('../simulation/data')
write_hex_18(output_dir / 'e_matrix_x.hex', e_x_vals)
write_hex_18(output_dir / 'e_matrix_y.hex', e_y_vals)
print(f"Wrote {len(e_x_vals)} entries each to e_matrix_x/y.hex")

# ── 9. Export reference centroids ────────────────────────────────────────
def write_hex_28(filename, values):
    with open(filename, 'w') as f:
        for v in values:
            f.write(f"{int(v) & 0xFFFFFFF:07X}\n")

write_hex_28(output_dir / 'slopes_ref_x.hex', ref_centroids_q23[:, 0])
write_hex_28(output_dir / 'slopes_ref_y.hex', ref_centroids_q23[:, 1])
print("Wrote 256 reference centroids")

# ── 9b. Export aberrated image ───────────────────────────────────────────
img_aber = propagate_and_detect(case["input_opd_field"])
quantized_aber = shwfs_utils.quantize_shwfs_image(img_aber)
with open(output_dir / 'hcipy_image_rotating.hex', 'w') as f:
    for val in quantized_aber:
        f.write(f"{int(val):02X}\n")
print(f"Wrote {len(quantized_aber)} pixels to hcipy_image_rotating.hex")

# ── 10. Bit-accurate FPGA prediction ────────────────────────────────────
print(f"\n{'='*60}")
print(f"Bit-accurate FPGA prediction  (G = {G_pow2}, shift = {G_shift})")
print(f"{'='*60}")

aber_slopes_q23_int = np.round(
    get_pixel_slopes(case["input_opd_field"], ref_centroids_q23) * SCALE_Q23
).astype(np.int64)
sx_q23 = aber_slopes_q23_int[valid_mask, 0]
sy_q23 = aber_slopes_q23_int[valid_mask, 1]

e_x_arr = np.array(e_x_vals).reshape(NUM_MODES, n_valid)
e_y_arr = np.array(e_y_vals).reshape(NUM_MODES, n_valid)

print(f"\n{'Mode':<12s}  {'Raw Q4.22':>12s}  {'nm (×G)':>12s}  {'True (nm)':>12s}  {'Ratio':>8s}")
print("-" * 64)
for m in range(NUM_MODES):
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

print("\nDone! Re-run Verilog simulation, then divide raw output by 2^22")
print(f"and multiply by {G_pow2} to get nanometers.")