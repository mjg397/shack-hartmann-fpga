"""
Compare FPGA Zernike output (Q4.23 nanometers) against HCIPy reference (meters).

This script:
1. Reads the FPGA raw Q4.23 Zernike outputs from the simulation
2. Converts them to nanometers (divide by 2^23)
3. Loads the HCIPy true_coeffs and converts to nanometers
4. Prints a side-by-side comparison
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils

# ---------- FPGA simulation output (from vvp) ----------
fpga_raw_q4_23 = [
    -64803877,   # Mode 1  Tilt X
     10460598,   # Mode 2  Tilt Y
     33160733,   # Mode 3  Defocus
      7689076,   # Mode 4  Astig 45
      8044943,   # Mode 5  Astig 0
      7242553,   # Mode 6  Coma X
    -31705125,   # Mode 7  Coma Y
     44544504,   # Mode 8  Trefoil X
     49999790,   # Mode 9  Trefoil Y
     -8410883,   # Mode 10 Sph.
]

fpga_nm = [v / (1 << 23) for v in fpga_raw_q4_23]

# ---------- HCIPy reference ----------
case = shwfs_utils.generate_shwfs_case(
    num_lenslets=16,
    num_zernike=10,
    demo_image_path=None,
)

hcipy_true_coeffs_m = case["true_coeffs"]            # metres of OPD
hcipy_true_nm       = hcipy_true_coeffs_m * 1e9       # nanometers

mode_names = [
    "Tilt X",
    "Tilt Y",
    "Defocus",
    "Astig 45°",
    "Astig 0°",
    "Coma X",
    "Coma Y",
    "Trefoil X",
    "Trefoil Y",
    "Sph.",
]

print(f"{'Mode':<12s}  {'HCIPy True (nm)':>16s}  {'FPGA (nm)':>16s}  {'Ratio':>10s}")
print("-" * 60)
for i in range(10):
    h = hcipy_true_nm[i]
    f = fpga_nm[i]
    ratio = f / h if abs(h) > 1e-6 else float('nan')
    print(f"{mode_names[i]:<12s}  {h:16.6f}  {f:16.6f}  {ratio:10.4f}")
