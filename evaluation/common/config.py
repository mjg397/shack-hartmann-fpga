"""
Shared configuration for the evaluation suite.

Every fixed-point constant here was traced from the RTL in
sim/rtl/ -- not inferred from arithmetic expressions.
Source lines are cited so the paper can point at them.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parents[1]      # the evaluation/ folder
SIM_DIR = REPO_ROOT / "sim"
# Single source of truth for the DUT. rtl/ is the tree that is actually
# synthesised, so the evaluation and the hardware build can never diverge.
RTL_DIR = REPO_ROOT / "rtl"
SRC_DIR = REPO_ROOT / "src"

# All generated output lives inside evaluation/, so the whole campaign --
# scripts, documentation and results -- is one self-contained folder.
RESULTS_DIR = EVAL_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
VIVADO_REPORT_DIR = EVAL_DIR / "implementation"

VENV_PYTHON = SIM_DIR / "venv" / "bin" / "python"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
MASTER_SEED = 20260914

# --------------------------------------------------------------------------
# Array / frame geometry
# --------------------------------------------------------------------------
NUM_LENSLETS = 16                   # 16x16 lenslet grid
SUBAP_PIXELS = 16                   # 16x16 pixels per subaperture
FRAME_W = NUM_LENSLETS * SUBAP_PIXELS   # 256
FRAME_H = FRAME_W
NUM_SUBAPS_GRID = NUM_LENSLETS ** 2     # 256 tiles
NUM_SUBAPS_VALID = 168                  # ematrix_accumulator NUM_SUBS
NUM_MODES = 10                          # ematrix_accumulator NUM_MODES

# streaming_emulator.v parameters
HBLANK = 4
VBLANK = 152

CLOCK_MHZ = 150.0
CLOCK_NS = 1000.0 / CLOCK_MHZ       # 6.667 ns

# --------------------------------------------------------------------------
# Fixed-point formats (traced from RTL)
# --------------------------------------------------------------------------
# Thresholding_CoG_intensity_accumulator.v:28
PIXEL_THRESHOLD = 5
# Overflow assertions in that module (lines ~461,468,475)
SI_MAX = 65280
XI_MAX = 489600

# Port widths at the shwfs_processing.v instantiation boundary
SI_BITS = 16
XI_BITS = 19            # accumulator drives 19 bits
RECIP_OUT_BITS = 19     # intensity_reciprocal_q18.rI_out
RECIP_PORT_BITS = 18    # slope_calculation.rec_intensity  <-- 1 bit pruned
CENTROID_BITS = 24      # unsigned_mult output, Q18
CENTROID_FRAC = 18
SLOPE_BITS = 25         # signed; also the DSP48 operand limit
SLOPE_RAW_BITS = 26     # signed subtraction intermediate

# ematrix_accumulator.v fixed-point block comment (lines 36-45)
EMATRIX_BITS = 18       # Q1.16 signed
EMATRIX_FRAC = 16
PRODUCT_BITS = 43       # signed
ACC_BITS = 52           # signed
ZERNIKE_BITS = 25       # signed, acc[42:18]
ZERNIKE_SLICE_LOW = 18
ZERNIKE_SLICE_HIGH = 42

# --------------------------------------------------------------------------
# Output decode
# --------------------------------------------------------------------------
# Calibrated against the HCIPy golden model by least squares over all 10 modes
# (2^-42.997). See sim/README.md "Decode scale".
ZERNIKE_SHIFT = 43
METERS_PER_LSB = 2.0 ** -ZERNIKE_SHIFT

# --------------------------------------------------------------------------
# Optical constants (from src/shwfs_utils.py generate_shwfs_case)
# These are the ONLY physical constants used; anything not derivable from the
# repository is reported as unavailable rather than invented.
# --------------------------------------------------------------------------
TELESCOPE_DIAMETER_M = 8.0
CENTRAL_OBSCURATION_M = 1.2
OVERSIZING_FACTOR = 16.0 / 15.0
WAVELENGTH_M = 0.7e-6
F_NUMBER = 50.0
SH_DIAMETER_M = 5e-3 * OVERSIZING_FACTOR
LENSLET_DIAMETER_M = SH_DIAMETER_M / NUM_LENSLETS
FOCAL_LENGTH_M = F_NUMBER * LENSLET_DIAMETER_M
PIXEL_PITCH_M = SH_DIAMETER_M / FRAME_W

MODE_LABELS = [
    "Tilt X", "Tilt Y", "Defocus", "Astig 45", "Astig 0",
    "Coma X", "Coma Y", "Trefoil X", "Trefoil Y", "Sph.",
]


def centroid_px_to_wavefront_m(centroid_err_px: float) -> float:
    """
    Convert a centroid error in pixels to the equivalent wavefront error
    across one lenslet, in metres.

        displacement = err_px * pixel_pitch
        slope        = displacement / focal_length
        wavefront    = slope * lenslet_diameter
    """
    displacement = centroid_err_px * PIXEL_PITCH_M
    slope = displacement / FOCAL_LENGTH_M
    return slope * LENSLET_DIAMETER_M


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
