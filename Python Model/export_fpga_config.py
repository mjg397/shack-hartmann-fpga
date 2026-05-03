import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils

# Generate the full case
case = shwfs_utils.generate_shwfs_case(
    num_lenslets=16,
    num_zernike=10,
    demo_image_path=None
)

# 1. Export slopes_ref to hex
# slopes_ref from HCIPy is shape (2, N_subs) where N_subs ~ 132
# For FPGA, we need a 256-element array for all subapertures.
slopes_ref_hcipy = case["slopes_ref"] # (2, N_subs)
valid_mask = case["valid_subaperture_mask"].ravel() # length 256

slopes_ref_x = np.zeros(256)
slopes_ref_y = np.zeros(256)

slopes_ref_x[valid_mask] = slopes_ref_hcipy[0, :]
slopes_ref_y[valid_mask] = slopes_ref_hcipy[1, :]

# Convert to Q4.23 format (multiply by 2^23)
SCALE = 8388608.0

slopes_ref_x_q23 = np.round(slopes_ref_x * SCALE).astype(np.int32)
slopes_ref_y_q23 = np.round(slopes_ref_y * SCALE).astype(np.int32)

# Write to hex files
def write_hex(filename, data):
    with open(filename, 'w') as f:
        for val in data:
            # 28-bit hex for Q4.23 + sign
            hex_val = hex(val & ((1 << 28) - 1))[2:].zfill(7).upper()
            f.write(f"{hex_val}\n")

output_dir = Path('../full_pipeline_sim/data')
output_dir.mkdir(parents=True, exist_ok=True)

write_hex(output_dir / 'slopes_ref_x.hex', slopes_ref_x_q23)
write_hex(output_dir / 'slopes_ref_y.hex', slopes_ref_y_q23)
print("Exported slopes_ref_x.hex and slopes_ref_y.hex")

# 2. Scale RM and export to e_matrix
RM = case["reconstruction_matrix"] # (2*N_subs, NUM_ZERNIKE)
# The output of the FPGA dot product should be in Nanometers.
# RM is currently in meters / slope_unit.
# We multiply RM by 1e9 so the output is in nanometers.
RM_scaled = RM * 1e9

# Export e_matrix. Format: Q1.16, 18-bit signed
SCALE_E = 65536.0
NUM_MODES = 10
N_subs = RM.shape[0] // 2

e_matrix_x_hex = []
e_matrix_y_hex = []

# RM columns are the Zernike modes
for mode in range(NUM_MODES):
    col = RM_scaled[:, mode]
    ex = col[:N_subs]
    ey = col[N_subs:]
    
    ex_q16 = np.round(ex * SCALE_E).astype(np.int32)
    ey_q16 = np.round(ey * SCALE_E).astype(np.int32)
    
    for val in ex_q16:
        e_matrix_x_hex.append(hex(val & ((1 << 18) - 1))[2:].zfill(5).upper())
    for val in ey_q16:
        e_matrix_y_hex.append(hex(val & ((1 << 18) - 1))[2:].zfill(5).upper())

with open(output_dir / 'e_matrix_x.hex', 'w') as f:
    for h in e_matrix_x_hex:
        f.write(f"{h}\n")

with open(output_dir / 'e_matrix_y.hex', 'w') as f:
    for h in e_matrix_y_hex:
        f.write(f"{h}\n")

print("Exported scaled e_matrix_x.hex and e_matrix_y.hex (in nanometers)")

# Also, let's export the image to ensure it's the right one
quantized_image = shwfs_utils.quantize_shwfs_image(case["image_aber"])
with open(output_dir / 'hcipy_image_rotating.hex', 'w') as f:
    for val in quantized_image:
        f.write(f"{hex(val)[2:].zfill(2).upper()}\n")
print("Exported new hcipy_image_rotating.hex")
