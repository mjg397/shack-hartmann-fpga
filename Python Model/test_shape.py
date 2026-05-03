import sys
from pathlib import Path
sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils
case = shwfs_utils.generate_shwfs_case(num_lenslets=16)
print("mla_grid shape:", case["shwfs"].mla_grid.shape)
