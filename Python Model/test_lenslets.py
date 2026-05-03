import sys
from pathlib import Path
sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils
case = shwfs_utils.generate_shwfs_case(num_lenslets=16)
print("valid mask shape:", case["valid_subaperture_mask"].shape)
print("slopes_ref shape:", case["slopes_ref"].shape)
print("fpga_subaperture_mask shape:", case["fpga_subaperture_mask"].shape)
print("fpga mask true count:", case["fpga_subaperture_mask"].sum())
