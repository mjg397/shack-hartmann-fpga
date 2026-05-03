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

# Ensure src is in path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import shwfs_utils
from shwfs_utils import generate_shwfs_visualizations, run_hcipy_estimation

import matplotlib
matplotlib.use('Agg') # Headless
import matplotlib.pyplot as plt

# ===========================================================================
# CONFIGURATION / FPGA DATA
# ===========================================================================
# FPGA raw output values (Q4.22) from the user's simulation
FPGA_RAW = np.array([
    4812849,   # Tilt X
    2838404,   # Tilt Y
    5116565,   # Defocus
    2444038,   # Astig 45
   -2463446,   # Astig 0
    -190754,   # Coma X
      45978,   # Coma Y
     185194,   # Trefoil X
    -128593,   # Trefoil Y
     371984,   # Sph.
])

# Scale factor to match the user's reference (HCIPy-like 58.7 nm for Tilt X)
# 4812849 / 2^22 * 51.15 = 58.7 nm
G_SCALE = 51.15

# ===========================================================================
# HOST-SIDE CODE
# ===========================================================================

def _measure_slopes(mode_field, amp, aperture, wavelength,
                    magnifier, shwfs, camera, shwfse, slopes_ref):
    phase = aperture * mode_field * amp * (2.0 * np.pi / wavelength)
    aber_ap = Field(aperture * np.exp(1j * np.array(phase)), aperture.grid)
    wf = Wavefront(aber_ap, wavelength)
    wf.total_power = 1
    camera.integrate(shwfs(magnifier(wf)), 1)
    img = camera.read_out()
    return shwfse.estimate([img]) - slopes_ref

# 1. Pupil
telescope_diameter       = 8.0
central_obscuration      = 1.2
central_obscuration_ratio = central_obscuration / telescope_diameter
spider_width             = 0.05
oversizing_factor        = 16 / 15

num_pupil_pixels    = int(240 * oversizing_factor)
pupil_grid_diameter = telescope_diameter * oversizing_factor
pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

aperture_gen = make_obstructed_circular_aperture(
    telescope_diameter, central_obscuration_ratio, num_spiders=4, spider_width=spider_width
)
VLT_aperture = evaluate_supersampled(aperture_gen, pupil_grid, 4)

# 2. SHWFS optics
wavelength_wfs = 0.7e-6
f_number       = 50
num_lenslets   = 16
sh_diameter    = 5e-3

magnification = sh_diameter / telescope_diameter
magnifier     = Magnifier(magnification)

shwfs = SquareShackHartmannWavefrontSensorOptics(
    pupil_grid.scaled(magnification), f_number, num_lenslets, sh_diameter
)
shwfse = ShackHartmannWavefrontSensorEstimator(
    shwfs.mla_grid, shwfs.micro_lens_array.mla_index
)

# 3. Reference image and valid subaperture mask
wf_ref = Wavefront(VLT_aperture, wavelength_wfs)
wf_ref.total_power = 1
_wf_out_ref = shwfs(magnifier(wf_ref))
detector_grid = _wf_out_ref.electric_field.grid
camera = NoiselessDetector(detector_grid)
camera.integrate(_wf_out_ref, 1)
image_ref = camera.read_out()

fluxes = ndimage.sum(image_ref, shwfse.mla_index, shwfse.estimation_subapertures)
flux_limit = fluxes.max() * 0.5
good_subs = shwfs.mla_grid.zeros(dtype='bool')
good_subs[shwfse.estimation_subapertures[fluxes > flux_limit]] = True
shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index, good_subs)

slopes_ref = shwfse.estimate([image_ref])
N_subs = slopes_ref.shape[1]
print(f"Valid subapertures: {N_subs}")

# 4. Zernike basis and RM
NUM_ZERNIKE = 10
mode_labels = ["Tilt X", "Tilt Y", "Defocus", "Astig 45", "Astig 0", "Coma X", "Coma Y", "Trefoil X", "Trefoil Y", "Sph."]

zernike_basis = make_zernike_basis(NUM_ZERNIKE + 1, telescope_diameter, pupil_grid, starting_mode=1)
zernike_modes = [zernike_basis[i] for i in range(1, NUM_ZERNIKE + 1)]

probe_amp = 0.05 * wavelength_wfs
IM_rows = []
for i, mode in enumerate(zernike_modes):
    s_p = _measure_slopes(mode,  probe_amp, VLT_aperture, wavelength_wfs, magnifier, shwfs, camera, shwfse, slopes_ref)
    s_m = _measure_slopes(mode, -probe_amp, VLT_aperture, wavelength_wfs, magnifier, shwfs, camera, shwfse, slopes_ref)
    IM_rows.append((s_p - s_m).ravel() / (2.0 * probe_amp))
IM = np.array(IM_rows)

rcond = 1e-3
U, s, Vt = np.linalg.svd(IM, full_matrices=False)
s_reg = s / (s**2 + (rcond * s.max())**2)
RM = (Vt.T * s_reg) @ U.T

# 5. Apply known aberration
true_coeffs = np.zeros(NUM_ZERNIKE)
true_coeffs[0] = 0.10 * wavelength_wfs   # tilt X
true_coeffs[1] = 0.07 * wavelength_wfs   # tilt Y
true_coeffs[2] = 0.08 * wavelength_wfs   # defocus
true_coeffs[3] = 0.05 * wavelength_wfs   # astig 45
true_coeffs[4] = -0.04 * wavelength_wfs  # astig 0

opd_map = sum(c * m for c, m in zip(true_coeffs, zernike_modes))
opd_field = VLT_aperture * opd_map
phase_map = opd_field * (2.0 * np.pi / wavelength_wfs)
aber_ap = Field(VLT_aperture * np.exp(1j * np.array(phase_map)), pupil_grid)
wf_aber = Wavefront(aber_ap, wavelength_wfs)
wf_aber.total_power = 1

# 6. Capture aberrated image
camera.integrate(shwfs(magnifier(wf_aber)), 1)
image_aber = camera.read_out()

# 7. Run estimation (HCIPy reference)
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

# >>> INJECT FPGA RESULTS <<<
# Convert FPGA raw (Q4.22) to metres using G_SCALE
fpga_metres = (FPGA_RAW.astype(float) / (1 << 22)) * G_SCALE * 1e-9 # convert nm to m
estimation["estimated_coeffs"] = fpga_metres

# Reconstruct OPD map from FPGA coefficients for visualization
recon_opd_field = sum(c * m for c, m in zip(fpga_metres, zernike_modes))
estimation["reconstructed_opd_field"] = VLT_aperture * recon_opd_field
estimation["residual_field"] = opd_field - (VLT_aperture * recon_opd_field)

print("FPGA coefficients (nm):", fpga_metres * 1e9)

# 8. Visualizations
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
    results_path="fpga_comparison_results.png",
    ao_demo_path="fpga_ao_demo.png",
    show_plots=False,
    figure_title="FPGA Hardware Reconstruction vs HCIPy Truth"
)

print(f"\nFigures saved: {figure_info['results_path']}, {figure_info['ao_demo_path']}")
print(f"Residual OPD RMS = {figure_info['residual_rms']*1e9:.2f} nm")
print(f"Strehl (aberrated): {figure_info['strehl_ab']:.3f}")
print(f"Strehl (corrected): {figure_info['strehl_corr']:.3f}")
