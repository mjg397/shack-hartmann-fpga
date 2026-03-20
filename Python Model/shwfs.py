"""
Shack-Hartmann Wavefront Sensor simulation using HCIPy.

Pipeline:
  1. Build a VLT-like pupil
  2. Build SHWFS optics + noiseless detector
  3. Capture reference SHWFS image (flat wavefront), select valid subapertures
  4. Build a Zernike interaction / reconstruction matrix
  5. Apply a known mix of low-order Zernike aberrations
  6. Capture aberrated SHWFS image, estimate differential slope fields
  7. Reconstruct Zernike coefficients from slopes (Tikhonov least-squares)
  8. Display: flat SHWFS image, aberrated SHWFS image, slope field quiver,
     input OPD, reconstructed OPD, and a coefficient comparison bar chart
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


# ---------------------------------------------------------------------------
# Helper: poke a single Zernike mode and return differential slopes
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
# ---------------------------------------------------------------------------
wf_ref             = Wavefront(VLT_aperture, wavelength_wfs)
wf_ref.total_power = 1

# Derive the detector grid from the first propagation output
_wf_out_ref = shwfs(magnifier(wf_ref))
detector_grid = _wf_out_ref.electric_field.grid
camera = NoiselessDetector(detector_grid)

camera.integrate(_wf_out_ref, 1)
image_ref = camera.read_out()

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

slopes_ref = shwfse.estimate([image_ref])
N_subs     = slopes_ref.shape[1]
print(f"Valid subapertures: {N_subs}")

# ---------------------------------------------------------------------------
# 4. Zernike basis (Noll 2..11: tilt, defocus, astigmatism, coma, trefoil,
#    spherical) and interaction / reconstruction matrix
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

# ---------------------------------------------------------------------------
# 5. Apply known Zernike aberration
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
# 6. Capture aberrated SHWFS image and estimate slope field
# ---------------------------------------------------------------------------
camera.integrate(shwfs(magnifier(wf_aber)), 1)
image_aber = camera.read_out()

slopes_aber  = shwfse.estimate([image_aber])
slopes_delta = slopes_aber - slopes_ref   # (2, N_subs)

sx = slopes_delta[0, :]
sy = slopes_delta[1, :]
sub_positions = shwfs.mla_grid.subset(shwfse.estimation_subapertures)

# ---------------------------------------------------------------------------
# 7. Reconstruct Zernike coefficients
# ---------------------------------------------------------------------------
estimated_coeffs = RM.T @ slopes_delta.ravel()

recon_opd       = sum(c * m for c, m in zip(estimated_coeffs, zernike_modes))
recon_opd_field = VLT_aperture * recon_opd

residual_field  = opd_field - recon_opd_field

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
