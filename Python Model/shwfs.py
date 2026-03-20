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
    imshow_field,
)

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


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
# 2. SHWFS optics (40x40 lenslets, F/50, 5 mm beam)
#    HOST: optical propagation model — not run on FPGA
# ---------------------------------------------------------------------------
wavelength_wfs = 0.7e-6   # m
f_number       = 50
num_lenslets   = 40
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
# Differential slopes (the quantity that encodes wavefront error):
#
#   slopes_delta = [cx, cy] − slopes_ref
#
#   slopes_ref (from step 3) holds the centroid positions for a flat
#   wavefront.  Subtracting it removes any systematic offset (e.g. lenslet
#   misalignment) and leaves only the aberration-induced shift.
#   This subtraction is done in-line on the FPGA immediately after each
#   centroid is computed.  The result is a (2 × N_subs) slope vector.
slopes_aber  = shwfse.estimate([image_aber])
slopes_delta = slopes_aber - slopes_ref   # (2, N_subs) — replicated by FPGA

sx = slopes_delta[0, :]
sy = slopes_delta[1, :]
sub_positions = shwfs.mla_grid.subset(shwfse.estimation_subapertures)

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

# HOST resumes: build OPD map from returned coefficients for display/logging.
recon_opd       = sum(c * m for c, m in zip(estimated_coeffs, zernike_modes))
recon_opd_field = VLT_aperture * recon_opd

residual_field  = opd_field - recon_opd_field

# ===========================================================================
# HOST-SIDE CODE (resumes)
# Results returned from the FPGA are displayed and compared to ground truth.
# ===========================================================================

# ---------------------------------------------------------------------------
# 8. Display
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Shack-Hartmann WFS Simulation (HCIPy)", fontsize=14, fontweight='bold')

# Row 0 ── sensor images and slope field ────────────────────────────────────
plt.sca(axes[0, 0])
imshow_field(image_ref, cmap='inferno')
plt.title("SHWFS image — flat wavefront")
plt.colorbar(label='counts')

plt.sca(axes[0, 1])
imshow_field(image_aber, cmap='inferno')
plt.title("SHWFS image — aberrated wavefront")
plt.colorbar(label='counts')

plt.sca(axes[0, 2])
imshow_field(image_aber, cmap='inferno', alpha=0.4)
# Estimate typical subaperture spacing from nearest-neighbour distances
_x = sub_positions.x
_y = sub_positions.y
_dx = np.diff(np.sort(np.unique(np.round(_x * 1e6).astype(int)))) * 1e-6
pitch = _dx[_dx > 0].min() if len(_dx[_dx > 0]) > 0 else (np.ptp(_x) / (num_lenslets - 1))
mag_max = np.hypot(sx, sy).max() + 1e-30
arrow_scale = pitch * 1.5 / mag_max
plt.quiver(
    sub_positions.x, sub_positions.y,
    sx * arrow_scale, sy * arrow_scale,
    color='cyan', scale=1, scale_units='xy', angles='xy', width=0.003,
)
plt.title("Differential slope field")

# Row 1 ── wavefronts and Zernike bar chart ─────────────────────────────────
vmax_nm = np.abs(opd_field[VLT_aperture > 0.5]).max() * 1e9

plt.sca(axes[1, 0])
imshow_field(opd_field * 1e9, cmap='RdBu', vmin=-vmax_nm, vmax=vmax_nm,
             mask=VLT_aperture)
plt.title("Input OPD [nm]")
plt.colorbar(label='nm')

plt.sca(axes[1, 1])
imshow_field(recon_opd_field * 1e9, cmap='RdBu', vmin=-vmax_nm, vmax=vmax_nm,
             mask=VLT_aperture)
plt.title("Reconstructed OPD [nm]")
plt.colorbar(label='nm')

plt.sca(axes[1, 2])
x     = np.arange(NUM_ZERNIKE)
width = 0.35
plt.bar(x - width / 2, true_coeffs * 1e9,      width, label='True',      color='steelblue')
plt.bar(x + width / 2, estimated_coeffs * 1e9,  width, label='Estimated', color='tomato')
plt.xticks(x, mode_labels, rotation=45, ha='right', fontsize=8)
plt.ylabel("Coefficient [nm]")
plt.title("Zernike coefficients")
plt.legend(loc='upper right')
plt.axhline(0, color='k', linewidth=0.6)

plt.tight_layout()
plt.savefig("shwfs_results.png", dpi=150, bbox_inches='tight')
print("Figure saved to shwfs_results.png")
plt.show()

# ---------------------------------------------------------------------------
# 9. Numerical summary
# ---------------------------------------------------------------------------
print("\n--- Zernike coefficient comparison ---")
print(f"{'Mode':<14} {'True [nm]':>12} {'Estimated [nm]':>15} {'Error [nm]':>12}")
print("-" * 58)
for label, tc, ec in zip(mode_labels, true_coeffs, estimated_coeffs):
    print(f"{label:<14} {tc*1e9:>12.2f} {ec*1e9:>15.2f} {(ec-tc)*1e9:>12.2f}")

residual_rms = np.std(residual_field[VLT_aperture > 0.5])
print(f"\nResidual OPD RMS inside pupil = {residual_rms*1e9:.2f} nm")
