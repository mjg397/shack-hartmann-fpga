import sys
from pathlib import Path
sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils
import numpy as np

case = shwfs_utils.generate_shwfs_case(num_lenslets=16, num_zernike=10, demo_image_path=None)
print("Image shape:", case["image_aber"].shape)

quantized_image = shwfs_utils.quantize_shwfs_image(case["image_aber"])
# Reshape to 256x256
img_2d = quantized_image.reshape((256, 256))

est = shwfs_utils.run_fpga_like_estimation(quantized_image, num_subapertures_side=16, subaperture_pixels=16)

# Test if row/col match
print("Est centroids [0]:", est["centroids_q4_23"][0])
