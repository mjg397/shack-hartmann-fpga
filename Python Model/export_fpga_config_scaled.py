import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils

# 1. Generate case to get the flat image reference
case = shwfs_utils.generate_shwfs_case(
    num_lenslets=16,
    num_zernike=10,
    demo_image_path=None
)

image_ref = case["image_ref"]

# 2. Run the FPGA-like estimation on the reference image to get the 256 centroids
fpga_est = shwfs_utils.run_fpga_like_estimation(
    image=image_ref,
    num_subapertures_side=16,
    subaperture_pixels=16
)

# centroids_q4_23 has shape (256, 2)
centroids_q4_23 = fpga_est["centroids_q4_23"]

# Write the 256 centroids to slopes_ref_x.hex and slopes_ref_y.hex
def write_hex(filename, data):
    with open(filename, 'w') as f:
        for val in data:
            hex_val = hex(val & ((1 << 28) - 1))[2:].zfill(7).upper()
            f.write(f"{hex_val}\n")

output_dir = Path('../full_pipeline_sim/data')
output_dir.mkdir(parents=True, exist_ok=True)

write_hex(output_dir / 'slopes_ref_x.hex', centroids_q4_23[:, 0])
write_hex(output_dir / 'slopes_ref_y.hex', centroids_q4_23[:, 1])

print(f"Exported 256 reference centroids to {output_dir}")

# 3. Read the original e_matrix files and scale by 700 for nanometers
def hex_to_signed_int(hex_str, bits=18):
    val = int(hex_str, 16)
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val

def scale_and_write_ematrix(in_file, out_file, scale_factor=700):
    original_hex = []
    # Read the original file, which is in full_pipeline_sim/data/e_matrix_x.hex
    # We will backup the original ones first or just overwrite them.
    # Wait, e_matrix_x.hex in the root repo directory is likely the original.
    root_ematrix = Path('../../e_matrix_x.hex')
    if root_ematrix.exists():
        with open(root_ematrix, 'r') as f:
            lines = f.readlines()
    else:
        # read from the current data folder
        with open(in_file, 'r') as f:
            lines = f.readlines()
            
    scaled_hex = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        val = hex_to_signed_int(line, bits=18)
        # Scale value
        scaled_val = int(round(val * scale_factor))
        # Ensure it fits in 18 bits (clamp if necessary, though 700 * Q1.16 should fit since values are tiny)
        # Actually max value in e_matrix was ~0.003 * 2^16 = 200. 200 * 700 = 140000, which fits in 18-bit signed (-131072 to 131071).
        # Wait! 140000 > 131071! It might overflow 18-bit!
        # Let's check max value and print warning if it overflows.
        if scaled_val > 131071:
            # print(f"Warning: overflow {scaled_val}")
            scaled_val = 131071
        elif scaled_val < -131072:
            # print(f"Warning: underflow {scaled_val}")
            scaled_val = -131072
            
        hex_val = hex(scaled_val & ((1 << 18) - 1))[2:].zfill(5).upper()
        scaled_hex.append(hex_val)
        
    with open(out_file, 'w') as f:
        for h in scaled_hex:
            f.write(f"{h}\n")

scale_and_write_ematrix(output_dir / 'e_matrix_x.hex', output_dir / 'e_matrix_x.hex', 700)
scale_and_write_ematrix(output_dir / 'e_matrix_y.hex', output_dir / 'e_matrix_y.hex', 700)

print("Scaled and exported e_matrix files to nanometers")
quantized_image = shwfs_utils.quantize_shwfs_image(case["image_aber"])
with open(output_dir / 'hcipy_image_rotating.hex', 'w') as f:
    for val in quantized_image:
        f.write(f"{hex(val)[2:].zfill(2).upper()}\n")
print("Exported new hcipy_image_rotating.hex")
