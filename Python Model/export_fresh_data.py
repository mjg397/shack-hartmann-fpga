import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils

case = shwfs_utils.generate_shwfs_case(
    num_lenslets=16,
    num_zernike=10,
    demo_image_path=None
)

output_dir = Path('../full_pipeline_sim/data')
output_dir.mkdir(parents=True, exist_ok=True)

def write_hex(filename, data, bits=28):
    with open(filename, 'w') as f:
        for val in data:
            hex_val = hex(val & ((1 << bits) - 1))[2:].zfill(bits // 4 + (bits % 4 > 0)).upper()
            f.write(f"{hex_val}\n")

# 1. 256 reference centroids in Q4.23
fpga_est = shwfs_utils.run_fpga_like_estimation(
    image=case["image_ref"],
    num_subapertures_side=16,
    subaperture_pixels=16
)
centroids_q4_23 = fpga_est["centroids_q4_23"]
write_hex(output_dir / 'slopes_ref_x.hex', centroids_q4_23[:, 0], bits=28)
write_hex(output_dir / 'slopes_ref_y.hex', centroids_q4_23[:, 1], bits=28)

# 2. Scale the NEW Reconstruction Matrix (RM) to Nanometers and export to Q1.16
RM = case["reconstruction_matrix"] # shape (2 * N_subs, NUM_MODES)
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

write_hex(output_dir / 'e_matrix_x.hex', e_matrix_x_vals, bits=18)
write_hex(output_dir / 'e_matrix_y.hex', e_matrix_y_vals, bits=18)

# 3. Export the subaperture bitmap
bitmap_bool = case["fpga_subaperture_mask"].ravel()
bitmap_int = 0
for i, val in enumerate(bitmap_bool):
    if val:
        bitmap_int |= (1 << i)

with open(output_dir / 'subaperture_bitmap.hex', 'w') as f:
    f.write(hex(bitmap_int)[2:].upper() + '\n')

# 4. Export aberrated image
quantized_image = shwfs_utils.quantize_shwfs_image(case["image_aber"])
with open(output_dir / 'hcipy_image_rotating.hex', 'w') as f:
    for val in quantized_image:
        f.write(f"{hex(val)[2:].zfill(2).upper()}\n")

print(f"Exported all files! N_subs = {N_subs}")
