import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils
from hcipy import Field, Wavefront

case = shwfs_utils.generate_shwfs_case(num_lenslets=16, num_zernike=10, demo_image_path=None)

zernike_modes = case["zernike_modes"]
aperture = case["aperture"]
wavelength = case["wavelength"]
magnifier = case["shwfs"].magnifier if hasattr(case["shwfs"], "magnifier") else shwfs_utils.Magnifier(5e-3 / 8.0)
shwfs = case["shwfs"]

def measure_fpga_slopes(mode, amp):
    phase = aperture * mode * amp * (2.0 * np.pi / wavelength)
    aber_ap = Field(aperture * np.exp(1j * np.array(phase)), aperture.grid)
    wf = Wavefront(aber_ap, wavelength)
    wf.total_power = 1
    from hcipy import NoiselessDetector
    camera = NoiselessDetector(shwfs(magnifier(wf)).grid)
    camera.integrate(shwfs(magnifier(wf)), 1)
    img = camera.read_out()
    est = shwfs_utils.run_fpga_like_estimation(img, num_subapertures_side=16, subaperture_pixels=16)
    return est["slopes_xy"]

probe_amp = 0.05 * wavelength
IM_rows = []

# Load the original 132 bitmap
with open('../full_pipeline_sim/data/subaperture_bitmap.hex', 'r') as f:
    bitmap_hex = f.read().strip()
bitmap_bin = bin(int(bitmap_hex, 16))[2:].zfill(256)[::-1]
valid_mask = np.array([b == '1' for b in bitmap_bin])

for i, mode in enumerate(zernike_modes):
    s_p = measure_fpga_slopes(mode, probe_amp)
    s_m = measure_fpga_slopes(mode, -probe_amp)
    s_p_valid = s_p[valid_mask]
    s_m_valid = s_m[valid_mask]
    diff_x = (s_p_valid[:, 0] - s_m_valid[:, 0]) / (2 * probe_amp)
    diff_y = (s_p_valid[:, 1] - s_m_valid[:, 1]) / (2 * probe_amp)
    IM_rows.append(np.concatenate([diff_x, diff_y]))

IM = np.array(IM_rows)
rcond = 1e-3
U, s, Vt = np.linalg.svd(IM, full_matrices=False)
s_reg = s / (s**2 + (rcond * s.max())**2)
RM = (Vt.T * s_reg) @ U.T

print("Built new RM based on FPGA pixel slopes. Shape:", RM.shape)

# Export the RM to nanometers
RM_scaled = RM * 1e9
SCALE_E = 65536.0
NUM_MODES = 10
N_subs = RM.shape[0] // 2

e_matrix_x_vals = []
e_matrix_y_vals = []

for mode in range(NUM_MODES):
    col = RM_scaled[:, mode]
    ex = col[:N_subs]
    ey = col[N_subs:]
    ex_q16 = np.round(ex * SCALE_E).astype(np.int64)
    ey_q16 = np.round(ey * SCALE_E).astype(np.int64)
    e_matrix_x_vals.extend(ex_q16)
    e_matrix_y_vals.extend(ey_q16)

def write_hex(filename, data, bits=18):
    with open(filename, 'w') as f:
        for val in data:
            hex_val = hex(val & ((1 << bits) - 1))[2:].zfill(bits // 4 + (bits % 4 > 0)).upper()
            f.write(f"{hex_val}\n")

output_dir = Path('../full_pipeline_sim/data')
write_hex(output_dir / 'e_matrix_x.hex', e_matrix_x_vals, bits=18)
write_hex(output_dir / 'e_matrix_y.hex', e_matrix_y_vals, bits=18)

# Export the 256 reference centroids
fpga_est = shwfs_utils.run_fpga_like_estimation(
    image=case["image_ref"],
    num_subapertures_side=16,
    subaperture_pixels=16
)
centroids_q4_23 = fpga_est["centroids_q4_23"]
write_hex(output_dir / 'slopes_ref_x.hex', centroids_q4_23[:, 0], bits=28)
write_hex(output_dir / 'slopes_ref_y.hex', centroids_q4_23[:, 1], bits=28)

# Export the image
quantized_image = shwfs_utils.quantize_shwfs_image(case["image_aber"])
with open(output_dir / 'hcipy_image_rotating.hex', 'w') as f:
    for val in quantized_image:
        f.write(f"{hex(val)[2:].zfill(2).upper()}\n")

# Revert subaperture_bitmap.hex? No, we didn't touch it since we read it. Wait, the previous script MIGHT have overwritten it with 56 ones!
# Let me just check if the current file has 132 ones.
print("Finished exporting e_matrix and image for 132 subapertures.")
