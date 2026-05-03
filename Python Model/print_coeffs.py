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
print("True coeffs (nm):", case["true_coeffs"] * 1e9)
print("Estimated coeffs (nm):", case["estimated_coeffs"] * 1e9)
