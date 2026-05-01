"""
Shack-Hartmann Wavefront Sensor simulation using HCIPy.

This script models the full SHWFS pipeline and is split into two stages:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  HOST (Python / HCIPy)                                              │
  │                                                                     │
  │  1. Build a VLT-like pupil and SHWFS optical model                 │
  │  2. Capture a flat-wavefront reference image                       │
  │  3. Select valid subapertures by flux thresholding                 │
  │  4. Build and invert a Zernike-to-slope interaction matrix (RM)    │
  │  5. Apply a known Zernike aberration to simulate an incoming WF    │
  │  6. Propagate through SHWFS and read out the detector image        │
  │                                                                     │
  │  ── DATA SENT TO FPGA ────────────────────────────────────────────│
  │     • image_aber   : 2-D detector pixel array (uint intensity)     │
  │     • slopes_ref   : (2 × N_subs) float reference centroid array   │
  │     • RM           : (2·N_subs × NUM_ZERNIKE) reconstruction matrix│
  │       (RM is pre-loaded into FPGA block RAM at configuration time) │
  └─────────────────────────────────────────────────────────────────────┘
           │                         │
           ▼  (PCIe / UART / JTAG)   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  FPGA                                                               │
  │                                                                     │
  │  7. Centroid estimation — compute (x, y) centroid per subaperture  │
  │     window → differential slope vector  (2 × N_subs)               │
  │  8. Wavefront reconstruction — fixed-point matrix-vector multiply  │
  │     RM.T @ slopes_delta  →  Zernike coefficient vector             │
  │                                                                     │
  │  ── DATA RETURNED FROM FPGA ─────────────────────────────────────│
  │     • estimated_coeffs : (NUM_ZERNIKE,) Zernike coefficient vector │
  └─────────────────────────────────────────────────────────────────────┘
           │
           ▼
  HOST resumes: reconstruct OPD map, display results, compare to truth
"""

from hcipy import (
    make_pupil_grid,
    make_obstructed_circular_aperture,
    evaluate_supersampled,
    Wavefront,
    Field,
    SquareShackHartmannWavefrontSensorOptics,
    ShackHartmannWavefrontSensorEstimator,
    NoiselessDetector,
    Magnifier,
    make_zernike_basis,
)

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import scipy.ndimage as ndimage
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shwfs_utils import generate_shwfs_visualizations, run_hcipy_estimation


# ===========================================================================
# HOST-SIDE CODE
# Everything below through section 6 runs on the host CPU using HCIPy.
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper: poke a single Zernike mode and return differential slopes
# (used only during offline calibration to build the interaction matrix)
# ---------------------------------------------------------------------------
def _measure_slopes(mode_field, amp, aperture, wavelength,
                    magnifier, shwfs, camera, shwfse, slopes_ref):
    phase = aperture * mode_field * amp * (2.0 * np.pi / wavelength)
    aber_ap = Field(aperture * np.exp(1j * np.array(phase)), aperture.grid)
    wf = Wavefront(aber_ap, wavelength)
    wf.total_power = 1
    camera.integrate(shwfs(magnifier(wf)), 1)
    img = camera.read_out()
    return shwfse.estimate([img]) - slopes_ref


# ---------------------------------------------------------------------------
# 1. Pupil (VLT-like: 8 m, 4 spiders, central obscuration)
#    HOST: optical model only — never sent to FPGA
# ---------------------------------------------------------------------------
telescope_diameter       = 8.0        # m
central_obscuration      = 1.2        # m
central_obscuration_ratio = central_obscuration / telescope_diameter
spider_width             = 0.05       # m
oversizing_factor        = 16 / 15

num_pupil_pixels    = int(240 * oversizing_factor)   # 256
pupil_grid_diameter = telescope_diameter * oversizing_factor

pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

aperture_gen = make_obstructed_circular_aperture(
    telescope_diameter,
    central_obscuration_ratio,
    num_spiders=4,
    spider_width=spider_width,
)
VLT_aperture = evaluate_supersampled(aperture_gen, pupil_grid, 4)

print(f"Pupil grid: {num_pupil_pixels}x{num_pupil_pixels} px")

# ---------------------------------------------------------------------------
# 2. SHWFS optics (16x16 lenslets, F/50, 5 mm beam)
#    HOST: optical propagation model — not run on FPGA
# ---------------------------------------------------------------------------
wavelength_wfs = 0.7e-6   # m
f_number       = 50
num_lenslets   = 16
sh_diameter    = 5e-3     # m

magnification = sh_diameter / telescope_diameter
magnifier     = Magnifier(magnification)

shwfs = SquareShackHartmannWavefrontSensorOptics(
    pupil_grid.scaled(magnification),
    f_number,
    num_lenslets,
    sh_diameter,
)

shwfse = ShackHartmannWavefrontSensorEstimator(
    shwfs.mla_grid,
    shwfs.micro_lens_array.mla_index,
)

print(f"SHWFS: {num_lenslets}x{num_lenslets} lenslets, F/{f_number}")

# ---------------------------------------------------------------------------
# 3. Reference image and valid subaperture mask
#    HOST: flat-wavefront calibration frame.
#
#    slopes_ref will be pre-loaded into FPGA registers/BRAM so the FPGA
#    can subtract it from every live centroid measurement.
# ---------------------------------------------------------------------------
wf_ref             = Wavefront(VLT_aperture, wavelength_wfs)
wf_ref.total_power = 1

# Derive the detector grid from the first propagation output
_wf_out_ref = shwfs(magnifier(wf_ref))
detector_grid = _wf_out_ref.electric_field.grid
camera = NoiselessDetector(detector_grid)

camera.integrate(_wf_out_ref, 1)
image_ref = camera.read_out()   # flat-field reference — captured once at startup

fluxes = ndimage.sum(
    image_ref,
    shwfse.mla_index,
    shwfse.estimation_subapertures,
)
flux_limit = fluxes.max() * 0.5

good_subs = shwfs.mla_grid.zeros(dtype='bool')
good_subs[shwfse.estimation_subapertures[fluxes > flux_limit]] = True

shwfse = ShackHartmannWavefrontSensorEstimator(
    shwfs.mla_grid,
    shwfs.micro_lens_array.mla_index,
    good_subs,
)

# slopes_ref: (2 × N_subs) array — SENT TO FPGA and stored in BRAM
# The FPGA subtracts slopes_ref from every new centroid measurement.
slopes_ref = shwfse.estimate([image_ref])
N_subs     = slopes_ref.shape[1]
print(f"Valid subapertures: {N_subs}")

# ---------------------------------------------------------------------------
# 4. Zernike basis and interaction / reconstruction matrix
#    HOST: offline calibration step — not repeated at runtime.
#
#    The resulting reconstruction matrix RM  (shape: 2·N_subs × NUM_ZERNIKE)
#    is the primary FPGA payload.  It is quantised to fixed-point and
#    pre-loaded into FPGA block RAM before operation begins.
#    RM never changes unless the instrument is re-calibrated.
# ---------------------------------------------------------------------------
NUM_ZERNIKE = 10

mode_labels = [
    "Tilt X", "Tilt Y", "Defocus",
    "Astig 45°", "Astig 0°",
    "Coma X", "Coma Y",
    "Trefoil X", "Trefoil Y",
    "Sph.",
][:NUM_ZERNIKE]

zernike_basis = make_zernike_basis(
    NUM_ZERNIKE + 1,   # piston (Noll 1) + NUM_ZERNIKE modes
    telescope_diameter,
    pupil_grid,
    starting_mode=1,

)
zernike_modes = [zernike_basis[i] for i in range(1, NUM_ZERNIKE + 1)]

probe_amp = 0.05 * wavelength_wfs
print(f"Building {NUM_ZERNIKE}-mode interaction matrix …")

IM_rows = []
for i, mode in enumerate(zernike_modes):
    s_p = _measure_slopes(mode,  probe_amp, VLT_aperture, wavelength_wfs,
                          magnifier, shwfs, camera, shwfse, slopes_ref)
    s_m = _measure_slopes(mode, -probe_amp, VLT_aperture, wavelength_wfs,
                          magnifier, shwfs, camera, shwfse, slopes_ref)
    IM_rows.append((s_p - s_m).ravel() / (2.0 * probe_amp))
    print(f"  mode {i+1:2d}/{NUM_ZERNIKE}", end='\r')

print()
IM = np.array(IM_rows)   # (NUM_ZERNIKE, 2*N_subs)

# Tikhonov pseudo-inverse
rcond     = 1e-3
U, s, Vt  = np.linalg.svd(IM, full_matrices=False)
s_reg     = s / (s**2 + (rcond * s.max())**2)
RM        = (Vt.T * s_reg) @ U.T   # (2*N_subs, NUM_ZERNIKE)

print(f"Reconstruction matrix: {RM.shape}")
# >>> RM is SENT TO FPGA here (e.g. written to block RAM via JTAG/PCIe DMA)
# Shape: ({RM.shape[0]} rows × {RM.shape[1]} cols) = 2·N_subs × NUM_ZERNIKE
# In hardware this is a fixed-point matrix stored in M10K/URAM blocks.
# Each output Zernike coefficient is one dot-product of a RM column with
# the flattened slope vector.

# ---------------------------------------------------------------------------
# 5. Apply known Zernike aberration
#    HOST: HCIPy simulates the effect of atmospheric/optical aberrations.
#    In real operation this step does not exist in software — the physical
#    wavefront arriving at the sensor is already aberrated.  HCIPy is only
#    used here to generate a realistic test image to feed into the FPGA.
# ---------------------------------------------------------------------------
true_coeffs        = np.zeros(NUM_ZERNIKE)
true_coeffs[0]     =  0.10 * wavelength_wfs   # tilt X
true_coeffs[1]     =  0.07 * wavelength_wfs   # tilt Y
true_coeffs[2]     =  0.08 * wavelength_wfs   # defocus
true_coeffs[3]     =  0.05 * wavelength_wfs   # astigmatism 45°
true_coeffs[4]     = -0.04 * wavelength_wfs   # astigmatism 0°

opd_map   = sum(c * m for c, m in zip(true_coeffs, zernike_modes))
opd_field = VLT_aperture * opd_map

phase_map   = opd_field * (2.0 * np.pi / wavelength_wfs)
aber_ap     = Field(VLT_aperture * np.exp(1j * np.array(phase_map)), pupil_grid)
wf_aber     = Wavefront(aber_ap, wavelength_wfs)
wf_aber.total_power = 1

# ---------------------------------------------------------------------------
# 6. Capture aberrated SHWFS image
#    HOST: HCIPy propagates the aberrated wavefront and simulates the
#    detector readout.  The resulting pixel array (image_aber) is what
#    a real camera would produce and is the primary input to the FPGA.
# ---------------------------------------------------------------------------
camera.integrate(shwfs(magnifier(wf_aber)), 1)
image_aber = camera.read_out()
# ^^^ image_aber is the 1-D pixel array (256×256 = 65536 values) that
#     would be streamed from the detector into the FPGA over a parallel
#     pixel bus or via DMA transfer.

# ==========================================================================
# >>>>> DATA TRANSFER TO FPGA <<<<<
#
#   Sent once at calibration time (stored in BRAM):
#     slopes_ref  : float32 array, shape (2, N_subs)  — reference centroids
#     RM          : float32 array, shape (2*N_subs, NUM_ZERNIKE) — recon matrix
#
#   Sent each frame at runtime:
#     image_aber  : pixel intensity array, shape (detector_pixels,)
#                   e.g. uint16, streamed row-by-row from the sensor
#
# ==========================================================================

# ---------------------------------------------------------------------------
# The following slope estimation and reconstruction replicate what the FPGA
# will compute in hardware.  They are retained here for result verification
# and display purposes only.
# ---------------------------------------------------------------------------

# -- FPGA STAGE 1: Centroid estimation ------------------------------------
#
# How the 256×256 image is divided into subaperture windows:
#
#   The SHWFS has 40 lenslets along each axis.  The SPHERE design uses 6
#   detector pixels per lenslet, so the natural detector size is
#   40 × 6 = 240 px.  This simulation oversamples by 16/15 (→ 256 px total)
#   but the 6 px/lenslet relationship is preserved.
#
#   The 256×256 detector is therefore logically tiled as a 40×40 grid of
#   non-overlapping ~6×6 pixel windows, one per lenslet:
#
#     col 0      col 1      col 2    …   col 39
#   ┌──────────┬──────────┬──────────┬─────────────┐  ← row 0
#   │  sub 0   │  sub 1   │  sub 2   │  …  sub 39  │  (6 px tall)
#   ├──────────┼──────────┼──────────┼─────────────┤  ← row 1
#   │  sub 40  │  sub 41  │  sub 42  │  …  sub 79  │
#   ├──────────┼──────────┼──────────┼─────────────┤
#   │   …      │   …      │   …      │     …       │
#   └──────────┴──────────┴──────────┴─────────────┘  ← row 39
#
#   Only N_subs (~1216) of the 1600 windows fall inside the circular pupil
#   and have enough flux to be used (selected in step 3 above).
#
# What the centroid computation does inside each window:
#
#   Given an M×M pixel window with intensities I[r,c]:
#
#     total_flux = Σ_{r,c}  I[r,c]
#     cx         = Σ_{r,c}  c · I[r,c]  /  total_flux   (x-centroid)
#     cy         = Σ_{r,c}  r · I[r,c]  /  total_flux   (y-centroid)
#
#   cx and cy give the sub-pixel position of the focused spot within the
#   window.  When the wavefront is flat (unaberrated), each spot sits at
#   the centre of its window.  A local wavefront tilt shifts the spot,
#   and that shift is the local wavefront slope.
#
#   In hardware this is:  2 × (M²) multiply-accumulate operations per
#   subaperture, all computed in parallel for all N_subs windows each frame.
#
# --- FPGA fixed-point design note: reciprocal LUT ---
#
#   Proposed scheme:
#     - Represent each pixel as a fractional fixed-point value clamped to
#       [0.0, 1.0]  (e.g. 0.8 unsigned fixed-point:  8 fractional bits,
#       integer part always 0).  This preserves the Gaussian intensity
#       weighting that makes centroid estimation sub-pixel accurate.
#     - total_flux = Σ I[r,c] is then a fixed-point number in [0.0, 36.0],
#       represented as 6.K format (6 integer bits to hold up to 36, K
#       fractional bits inherited from the pixel representation).
#     - Clamp total_flux to a minimum of 1.0 to avoid divide-by-zero for
#       windows that lie outside the pupil / receive no light.
#     - Store 1/total_flux in a lookup table with entries in 1.8 unsigned
#       fixed-point (1 integer bit + 8 fractional bits, range [0, ~2.0),
#       resolution 1/256 ≈ 0.0039).  All reciprocals fit because
#       1/total_flux ∈ [1/36, 1.0], which is always < 1.996.
#
#   Practical approach:  accumulate pixel values in full precision (0.8) for
#   centroid accuracy, then truncate total_flux to fewer fractional bits
#   (e.g. 4) before the LUT lookup to keep the table small.  The truncation
#   error is at most 1/16 of a flux unit, which causes a centroid error of
#   at most  Δc ≈ (1/16) × (d_centroid/d_flux)  — negligible in practice.
#
#   Reciprocal output precision:
#     The 1.8 format has resolution 1/256.  The worst-case quantisation
#     error on the reciprocal is  ε = 1/(2×256) ≈ 0.2%.  For total_flux = 1
#     (minimum), 1/1 = 1.0 is exactly representable.  For total_flux = 36
#     (maximum), 1/36 ≈ 0.02778 → stored as 7/256 = 0.02734, error ≈ 1.6%.
#
# Differential slopes (the quantity that encodes wavefront error):
#
#   slopes_delta = [cx, cy] − slopes_ref
#
#   slopes_ref (from step 3) holds the centroid positions for a flat
#   wavefront.  Subtracting it removes any systematic offset (e.g. lenslet
#   misalignment) and leaves only the aberration-induced shift.
#   This subtraction is done in-line on the FPGA immediately after each
#   centroid is computed.  The result is a (2 × N_subs) slope vector.
estimation = run_hcipy_estimation(
    image=image_aber,
    estimator=shwfse,
    reference_slopes=slopes_ref,
    reconstruction_matrix=RM,
    zernike_modes=zernike_modes,
    aperture=VLT_aperture,
    measured_opd_field=opd_field,
    shwfs=shwfs,
)

slopes_aber = estimation["slopes_aber"]
slopes_delta = estimation["slopes_delta"]   # (2, N_subs) — replicated by FPGA

# ---------------------------------------------------------------------------
# 7. Reconstruct Zernike coefficients
#    FPGA STAGE 2: matrix-vector multiply.
#    The FPGA computes  c = RM.T @ s  where:
#      s  = flattened slope vector,  length 2·N_subs  (from centroid stage)
#      RM = pre-loaded reconstruction matrix in block RAM
#      c  = NUM_ZERNIKE Zernike coefficients  (the output)
#
#    Implementation: NUM_ZERNIKE parallel dot-product units, each
#    accumulating 2·N_subs fixed-point multiply-add operations.
#    One result vector is produced per detector frame.
# ---------------------------------------------------------------------------
estimated_coeffs = RM.T @ slopes_delta.ravel()
# ^^^ This line represents the FPGA's core computation.
#     In hardware this is a pipelined fixed-point MAC array.

# ==========================================================================
# >>>>> DATA RETURNED FROM FPGA <<<<<
#
#   estimated_coeffs : float/fixed-point array, shape (NUM_ZERNIKE,)
#                      Zernike mode amplitudes in metres of OPD.
#                      Returned to the host over UART/PCIe/JTAG per frame.
#
# ==========================================================================

print("Estimated coeffs: \n", estimated_coeffs)
print("Slopes: \n", slopes_aber)
np.set_printoptions(threshold=np.inf, linewidth=200, precision=6, suppress=True)
print(np.asarray(slopes_aber))

# ===========================================================================
# HOST-SIDE CODE (resumes)
# Results returned from the FPGA are displayed and compared to ground truth.
# ===========================================================================

recon_opd_field = estimation["reconstructed_opd_field"]
residual_field = estimation["residual_field"]

figure_info = generate_shwfs_visualizations(
    image_ref=image_ref,
    image_aber=image_aber,
    estimation=estimation,
    aperture=VLT_aperture,
    input_opd_field=opd_field,
    true_coeffs=true_coeffs,
    mode_labels=mode_labels,
    wavelength=wavelength_wfs,
    num_lenslets=num_lenslets,
    results_path="shwfs_results.png",
    ao_demo_path="shwfs_ao_demo.png",
    show_plots=True,
)

if figure_info["results_path"] is not None:
    print(f"Figure saved to {figure_info['results_path']}")
if figure_info["ao_demo_path"] is not None:
    print(f"AO demo figure saved to {figure_info['ao_demo_path']}")
residual_rms = figure_info["residual_rms"]
print(f"\nResidual OPD RMS inside pupil = {residual_rms*1e9:.2f} nm")
print(
    f"Strehl (aberrated): {figure_info['strehl_ab']:.3f}   |   "
    f"Strehl (corrected): {figure_info['strehl_corr']:.3f}"
)
