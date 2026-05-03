import numpy as np

# Load FPGA generated valid slopes
def read_hex(filename):
    with open(filename, 'r') as f:
        return [int(line.strip(), 16) for line in f.readlines()]

def to_signed(val, bits):
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val

# Load RM from our generated python file
rm_x = [to_signed(v, 18) for v in read_hex('../full_pipeline_sim/data/e_matrix_x.hex')]
rm_y = [to_signed(v, 18) for v in read_hex('../full_pipeline_sim/data/e_matrix_y.hex')]

rm_x = np.array(rm_x).reshape(10, 56)
rm_y = np.array(rm_y).reshape(10, 56)

# Instead of extracting from the image, let's load what the FPGA actually got for slopes.
# But we don't have the FPGA slopes.
# Let's print what Python expects for the slopes.
import sys
from pathlib import Path
sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils

case = shwfs_utils.generate_shwfs_case(num_lenslets=16, num_zernike=10, demo_image_path=None)
est = shwfs_utils.run_fpga_like_estimation(case["image_aber"], num_subapertures_side=16, subaperture_pixels=16)

ref_est = shwfs_utils.run_fpga_like_estimation(case["image_ref"], num_subapertures_side=16, subaperture_pixels=16)

# FPGA mask
with open('../full_pipeline_sim/data/subaperture_bitmap.hex', 'r') as f:
    bitmap_bin = bin(int(f.read().strip(), 16))[2:].zfill(256)[::-1]
valid_mask = np.array([b == '1' for b in bitmap_bin])

x_slope_q23 = est["centroids_q4_23"][:, 0] - ref_est["centroids_q4_23"][:, 0]
y_slope_q23 = est["centroids_q4_23"][:, 1] - ref_est["centroids_q4_23"][:, 1]

x_valid = x_slope_q23[valid_mask]
y_valid = y_slope_q23[valid_mask]

# FPGA MAC simulation
zernike_out = np.zeros(10, dtype=np.int64)

for m in range(10):
    acc = 0
    for s in range(56):
        prod_x = np.int64(rm_x[m, s]) * np.int64(x_valid[s])
        prod_y = np.int64(rm_y[m, s]) * np.int64(y_valid[s])
        acc += prod_x + prod_y
    # extract [43:17]
    zernike_out[m] = (acc >> 17) & ((1<<27)-1)
    zernike_out[m] = to_signed(zernike_out[m], 27)

print("Python-simulated FPGA outputs:")
for m in range(10):
    print(f"Mode {m+1}: {zernike_out[m]}")
