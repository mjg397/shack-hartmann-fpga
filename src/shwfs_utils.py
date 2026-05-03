"""
Utilities for the Shack-Hartmann Wavefront Sensor HCIPy model.

Includes helpers to generate an aberrated SHWFS image, run the host-side HCIPy
estimation flow that mirrors the FPGA pipeline, and render the result figures.
"""

from hcipy import (
    make_pupil_grid,
    make_obstructed_circular_aperture,
    evaluate_supersampled,
    Wavefront,
    Field,
    CartesianGrid,
    SeparatedCoords,
    MicroLensArray,
    ShackHartmannWavefrontSensorOptics,
    ShackHartmannWavefrontSensorEstimator,
    NoiselessDetector,
    Magnifier,
    make_zernike_basis,
    imshow_field,
)

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import scipy.ndimage as ndimage
from pathlib import Path
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Private helpers for demo-image generation
# ---------------------------------------------------------------------------

def load_vlt_demo_scene(size=256):
    """Load a VLT image from ./VLT_Images and resize to (size, size) grayscale."""
    plt = _import_pyplot(interactive=False)

    image_dir = Path(__file__).resolve().parent.parent / "VLT_Images"
    for name in ("eso1322a.jpg", "eso1131a.jpg"):
        path = image_dir / name
        if path.exists():
            img = plt.imread(path)
            if img.ndim == 3:
                img = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            img = np.asarray(img, dtype=float)

            zoom_factor = min(size / img.shape[0], size / img.shape[1])
            resized = np.atleast_2d(np.asarray(ndimage.zoom(img, zoom_factor, order=1), dtype=float))
            resized_height = int(resized.shape[0])
            resized_width = int(resized.shape[1])

            out = np.zeros((size, size), dtype=float)
            y0 = max((size - resized_height) // 2, 0)
            x0 = max((size - resized_width) // 2, 0)
            copy_height = min(resized_height, size)
            copy_width = min(resized_width, size)
            out[y0:y0 + copy_height, x0:x0 + copy_width] = resized[:copy_height, :copy_width]

            out -= out.min()
            out /= out.max() + 1e-30
            print(f"Loaded VLT scene from {path}")
            return out

    raise FileNotFoundError("No VLT image found in ./VLT_Images")


def _estimate_subaperture_pitch(sub_positions, num_lenslets):
    """Estimate the subaperture spacing for quiver scaling."""
    x_coords = np.asarray(sub_positions.x)
    unique_x = np.unique(np.round(x_coords * 1e6).astype(int))
    delta_x = np.diff(np.sort(unique_x)) * 1e-6
    positive_delta_x = delta_x[delta_x > 0]
    if positive_delta_x.size > 0:
        return positive_delta_x.min()
    return np.ptp(x_coords) / max(num_lenslets - 1, 1)


def _import_pyplot(interactive=True):
    """Import pyplot with a non-interactive backend for file-only rendering."""
    import matplotlib

    if not interactive and matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    return plt


def _default_mode_labels(num_zernike):
    return [
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
    ][:num_zernike]


def _default_true_coeffs(num_zernike, wavelength):
    coeffs = np.zeros(num_zernike)
    if num_zernike > 0:
        coeffs[0] = 0.10 * wavelength
    if num_zernike > 1:
        coeffs[1] = 0.07 * wavelength
    if num_zernike > 2:
        coeffs[2] = 0.08 * wavelength
    if num_zernike > 3:
        coeffs[3] = 0.05 * wavelength
    if num_zernike > 4:
        coeffs[4] = -0.04 * wavelength
    return coeffs


def _measure_slopes(
    mode_field,
    amplitude,
    aperture,
    wavelength,
    magnifier,
    shwfs,
    camera,
    estimator,
    reference_slopes,
):
    phase = aperture * mode_field * amplitude * (2.0 * np.pi / wavelength)
    aberrated_aperture = Field(aperture * np.exp(1j * np.array(phase)), aperture.grid)
    wavefront = Wavefront(aberrated_aperture, wavelength)
    wavefront.total_power = 1
    camera.integrate(shwfs(magnifier(wavefront)), 1)
    image = camera.read_out()
    return estimator.estimate([image]) - reference_slopes


def _psf_from_opd(opd_2d, aperture_2d, wavelength):
    """PSF via Fraunhofer FFT from a 2-D OPD map and aperture mask."""
    phase = (2.0 * np.pi / wavelength) * opd_2d
    pupil = aperture_2d * np.exp(1j * phase)
    ef = np.fft.fftshift(np.fft.fft2(pupil))
    psf = np.abs(ef) ** 2
    psf /= psf.sum() + 1e-30
    return psf


def _convolve_with_psf(image, psf):
    """Circular convolution via FFT."""
    kernel = np.fft.ifftshift(psf)
    out = np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(kernel)).real
    return np.clip(out, 0.0, None)


def quantize_shwfs_image(image, out_min=0, out_max=255):
    """Match the server's detector-image quantization before sending pixels to the FPGA."""
    image_array = np.asarray(image)
    if image_array.dtype == np.uint8:
        return image_array.copy()

    image_array = np.asarray(image_array, dtype=np.float64)
    image_min = float(image_array.min())
    image_max = float(image_array.max())
    if image_max <= image_min:
        return np.zeros_like(image_array, dtype=np.uint8)

    return np.interp(image_array, (image_min, image_max), (out_min, out_max)).astype(np.uint8)


def _compute_reciprocal_q27(total_intensity):
    """Approximate the FPGA reciprocal output format using rounded Q0.27 values."""
    if total_intensity <= 0:
        return (1 << 27) - 1
    return min((1 << 27) - 1, int(round((1 << 27) / total_intensity)))


def _compute_centroid_q4_23(weighted_intensity, reciprocal_q27):
    """Mirror the FPGA centroid multiplier that converts 20.0 x 0.27 into Q4.23."""
    return (int(weighted_intensity) * int(reciprocal_q27)) >> 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_shwfs_case(
    true_coeffs=None,
    num_lenslets=16,
    num_zernike=10,
    demo_image_path="shwfs_aber_demo.png",
):
    """
    Build a full HCIPy SHWFS simulation case for transport, estimation, and plotting.

    Returns a dictionary containing the detector image, calibration products,
    optical model objects, and reference data needed for HCIPy/FPGA comparison.
    """
    plt = _import_pyplot(interactive=False)

    telescope_diameter = 8.0
    central_obscuration = 1.2
    spider_width = 0.05
    oversizing_factor = 16 / 15
    wavelength_wfs = 0.7e-6
    f_number = 50
    sh_diameter = 5e-3 * oversizing_factor

    num_pupil_pixels = int(240 * oversizing_factor)
    pupil_grid_diameter = telescope_diameter * oversizing_factor
    pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

    aperture_gen = make_obstructed_circular_aperture(
        telescope_diameter,
        central_obscuration / telescope_diameter,
        num_spiders=4,
        spider_width=spider_width,
    )
    aperture = evaluate_supersampled(aperture_gen, pupil_grid, 4)

    magnification = 5e-3 / telescope_diameter
    magnifier = Magnifier(magnification)
    
    # HCIPy's SquareShackHartmannWavefrontSensorOptics generates an asymmetric 
    # grid because of np.arange. We must manually generate a centered grid.
    lenslet_diameter = sh_diameter / num_lenslets
    x_centers = np.linspace(-sh_diameter/2 + lenslet_diameter/2, 
                             sh_diameter/2 - lenslet_diameter/2, 
                             num_lenslets)
    mla_grid = CartesianGrid(SeparatedCoords((x_centers, x_centers)))
    focal_length = f_number * lenslet_diameter
    micro_lens_array = MicroLensArray(pupil_grid.scaled(magnification), mla_grid, focal_length)
    
    shwfs = ShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), micro_lens_array)
    base_estimator = ShackHartmannWavefrontSensorEstimator(
        mla_grid,
        micro_lens_array.mla_index,
    )

    wf_ref = Wavefront(aperture, wavelength_wfs)
    wf_ref.total_power = 1
    wf_out_ref = shwfs(magnifier(wf_ref))
    camera = NoiselessDetector(wf_out_ref.electric_field.grid)
    camera.integrate(wf_out_ref, 1)
    image_ref = camera.read_out()

    fluxes = ndimage.sum(
        image_ref,
        base_estimator.mla_index,
        base_estimator.estimation_subapertures,
    )
    flux_limit = fluxes.max() * 0.5

    # Run a quick estimation on the reference image to find spot locations
    q_img = quantize_shwfs_image(image_ref)
    ref_est = run_fpga_like_estimation(q_img, 16, 16)
    c_q23 = ref_est["centroids_q4_23"].astype(float) / (1 << 23)
    c_x, c_y = c_q23[:, 0], c_q23[:, 1]
    
    # Keep only subapertures where the reference spot is centrally located 
    # (between 5.5 and 9.5 in the 16x16 block)
    valid_x = (c_x > 5.5) & (c_x < 9.5)
    valid_y = (c_y > 5.5) & (c_y < 9.5)
    
    combined_mask = (fluxes > flux_limit) & valid_x & valid_y
    valid_indices = base_estimator.estimation_subapertures[combined_mask]

    valid_subaperture_mask = shwfs.mla_grid.zeros(dtype="bool")
    valid_subaperture_mask[valid_indices] = True

    estimator = ShackHartmannWavefrontSensorEstimator(
        shwfs.mla_grid,
        shwfs.micro_lens_array.mla_index,
        valid_subaperture_mask,
    )
    slopes_ref = estimator.estimate([image_ref])

    zernike_basis = make_zernike_basis(
        num_zernike + 1,
        telescope_diameter,
        pupil_grid,
        starting_mode=1,
    )
    zernike_modes = [zernike_basis[i] for i in range(1, num_zernike + 1)]

    if true_coeffs is None:
        true_coeffs = _default_true_coeffs(num_zernike, wavelength_wfs)
    true_coeffs = np.asarray(true_coeffs, dtype=float)

    probe_amp = 0.05 * wavelength_wfs
    interaction_rows = []
    for mode in zernike_modes:
        slopes_plus = _measure_slopes(
            mode,
            probe_amp,
            aperture,
            wavelength_wfs,
            magnifier,
            shwfs,
            camera,
            estimator,
            slopes_ref,
        )
        slopes_minus = _measure_slopes(
            mode,
            -probe_amp,
            aperture,
            wavelength_wfs,
            magnifier,
            shwfs,
            camera,
            estimator,
            slopes_ref,
        )
        interaction_rows.append((slopes_plus - slopes_minus).ravel() / (2.0 * probe_amp))

    interaction_matrix = np.asarray(interaction_rows)
    rcond = 1e-3
    u_mat, singular_vals, vt_mat = np.linalg.svd(interaction_matrix, full_matrices=False)
    sigma_reg = singular_vals / (singular_vals**2 + (rcond * singular_vals.max())**2)
    reconstruction_matrix = (vt_mat.T * sigma_reg) @ u_mat.T

    opd_field = aperture * sum(c * m for c, m in zip(true_coeffs, zernike_modes))
    phase_map = opd_field * (2.0 * np.pi / wavelength_wfs)
    aberrated_aperture = Field(aperture * np.exp(1j * np.asarray(phase_map)), pupil_grid)
    wf_aber = Wavefront(aberrated_aperture, wavelength_wfs)
    wf_aber.total_power = 1

    camera.integrate(shwfs(magnifier(wf_aber)), 1)
    image_aber = camera.read_out()

    if demo_image_path is not None:
        Path(demo_image_path).parent.mkdir(parents=True, exist_ok=True)
        aperture_2d = np.asarray(aperture, dtype=float).reshape(num_pupil_pixels, num_pupil_pixels)
        opd_2d = np.asarray(opd_field, dtype=float).reshape(num_pupil_pixels, num_pupil_pixels)
        scene = load_vlt_demo_scene(num_pupil_pixels)
        psf_ab = _psf_from_opd(opd_2d, aperture_2d, wavelength_wfs)
        img_ab = _convolve_with_psf(scene, psf_ab)
        img_ab /= img_ab.max() + 1e-30

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(scene, cmap="gray", origin="lower")
        axes[0].set_title("Reference (VLT image)")
        axes[0].axis("off")
        axes[1].imshow(img_ab, cmap="gray", origin="lower")
        axes[1].set_title("Aberrated")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(demo_image_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "image_ref": image_ref,
        "image_aber": image_aber,
        "slopes_ref": np.asarray(slopes_ref),
        "reconstruction_matrix": np.asarray(reconstruction_matrix),
        "zernike_modes": zernike_modes,
        "true_coeffs": true_coeffs,
        "mode_labels": _default_mode_labels(num_zernike),
        "input_opd_field": opd_field,
        "aperture": aperture,
        "wavelength": wavelength_wfs,
        "num_lenslets": num_lenslets,
        "num_pupil_pixels": num_pupil_pixels,
        "shwfs": shwfs,
        "estimator": estimator,
        "valid_subaperture_mask": np.asarray(valid_subaperture_mask, dtype=bool),
        "valid_subaperture_indices": np.asarray(estimator.estimation_subapertures, dtype=int),
        "fpga_subaperture_mask": np.asarray(valid_subaperture_mask, dtype=bool),
    }


def expand_valid_subaperture_array(values, valid_indices, total_subapertures=None, fill_value=np.nan):
    """Expand valid-subaperture values using either integer indices or a boolean mask."""
    values = np.asarray(values)
    valid_indices = np.asarray(valid_indices)

    if valid_indices.dtype == bool:
        if values.shape[0] != int(valid_indices.sum()):
            raise ValueError(
                f"Boolean valid mask selects {int(valid_indices.sum())} entries, got {values.shape[0]} values"
            )
        expanded_shape = valid_indices.shape + values.shape[1:]
        expanded = np.full(expanded_shape, fill_value, dtype=np.float64)
        expanded[valid_indices] = values
        return expanded

    if total_subapertures is None:
        raise ValueError("total_subapertures is required when expanding integer indices")

    expanded_shape = (total_subapertures,) + values.shape[1:]
    expanded = np.full(expanded_shape, fill_value, dtype=np.float64)
    expanded[valid_indices.astype(int)] = values
    return expanded


def reshape_subaperture_xy(xy_values, num_lenslets):
    """Reshape flat per-lenslet XY data into a (num_lenslets, num_lenslets, 2) grid."""
    xy_values = np.asarray(xy_values, dtype=np.float64)
    return xy_values.reshape(num_lenslets, num_lenslets, 2)


def collapse_subaperture_mask(mask_grid, factor=2, reduction="any"):
    """Collapse a higher-resolution boolean mask to the FPGA lenslet grid."""
    mask_grid = np.asarray(mask_grid, dtype=bool)
    collapsed = mask_grid.reshape(
        mask_grid.shape[0] // factor,
        factor,
        mask_grid.shape[1] // factor,
        factor,
    )
    if reduction == "all":
        return collapsed.all(axis=(1, 3))
    return collapsed.any(axis=(1, 3))


def collapse_xy_grid(xy_grid, factor=2, valid_mask=None, fill_value=0.0):
    """Average a higher-resolution XY grid down to the FPGA lenslet grid."""
    xy_grid = np.asarray(xy_grid, dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.isfinite(xy_grid[..., 0]) & np.isfinite(xy_grid[..., 1])
    valid_mask = np.asarray(valid_mask, dtype=bool)

    out_rows = xy_grid.shape[0] // factor
    out_cols = xy_grid.shape[1] // factor
    collapsed = np.full((out_rows, out_cols, 2), fill_value, dtype=np.float64)

    for row_idx in range(out_rows):
        for col_idx in range(out_cols):
            row_slice = slice(row_idx * factor, (row_idx + 1) * factor)
            col_slice = slice(col_idx * factor, (col_idx + 1) * factor)
            block_mask = valid_mask[row_slice, col_slice]
            if not np.any(block_mask):
                continue
            block_values = xy_grid[row_slice, col_slice][block_mask]
            collapsed[row_idx, col_idx] = block_values.mean(axis=0)

    return collapsed


def make_fpga_subaperture_positions(shwfs, num_lenslets):
    """Construct 16x16 logical lenslet positions from the HCIPy 32x32 MLA grid."""
    x_grid = np.asarray(shwfs.mla_grid.x, dtype=np.float64).reshape(num_lenslets * 2, num_lenslets * 2)
    y_grid = np.asarray(shwfs.mla_grid.y, dtype=np.float64).reshape(num_lenslets * 2, num_lenslets * 2)
    x_pos = collapse_xy_grid(np.dstack((x_grid, x_grid)), factor=2)[..., 0].ravel()
    y_pos = collapse_xy_grid(np.dstack((y_grid, y_grid)), factor=2)[..., 0].ravel()
    return SimpleNamespace(x=x_pos, y=y_pos)


def build_fpga_estimation(
    slopes_xy,
    estimated_coeffs,
    zernike_modes,
    aperture,
    shwfs,
    num_lenslets=16,
    measured_opd_field=None,
):
    """
    Build an estimation dictionary for FPGA outputs using full-grid slope vectors.
    """
    slopes_xy = np.asarray(slopes_xy, dtype=np.float64)
    estimated_coeffs = np.asarray(estimated_coeffs, dtype=np.float64)
    reconstructed_zernike = sum(
        coeff * mode for coeff, mode in zip(estimated_coeffs, zernike_modes)
    )
    reconstructed_opd_field = aperture * reconstructed_zernike
    residual_field = None
    if measured_opd_field is not None:
        residual_field = measured_opd_field - reconstructed_opd_field

    return {
        "slopes_aber": slopes_xy.T,
        "slopes_delta": slopes_xy.T,
        "slope_x": slopes_xy[:, 0],
        "slope_y": slopes_xy[:, 1],
        "estimated_coeffs": estimated_coeffs,
        "reconstructed_zernike": reconstructed_zernike,
        "reconstructed_opd_field": reconstructed_opd_field,
        "residual_field": residual_field,
        "sub_positions": make_fpga_subaperture_positions(shwfs, num_lenslets),
    }


def run_fpga_like_estimation(
    image,
    num_subapertures_side=16,
    subaperture_pixels=16,
    slope_reference_pixels=None,
    fraction_bits=23,
    reconstruction_mask=None,
    reconstruction_matrix_q1_16=None,
):
    """
    Mirror the FPGA centroid and slope procedure on the quantized detector image.

    The FPGA processes every 16x16 subaperture tile on the transmitted 8-bit
    image, computes CoG centroids in local pixel coordinates, and subtracts a
    fixed 7.5-pixel reference (for a 16-pixel tile) to form slopes.
    """
    quantized_image = quantize_shwfs_image(image)
    detector_pixels = num_subapertures_side * subaperture_pixels
    image_grid = np.asarray(quantized_image, dtype=np.uint8).reshape(detector_pixels, detector_pixels)

    if slope_reference_pixels is None:
        slope_reference_pixels = (subaperture_pixels - 1) / 2.0

    scale = 1 << fraction_bits
    slope_reference_q4_23 = int(round(slope_reference_pixels * scale))

    x_indices = np.arange(subaperture_pixels, dtype=np.int64)[None, :]
    y_indices = np.arange(subaperture_pixels, dtype=np.int64)[:, None]

    centroids_q4_23 = np.zeros((num_subapertures_side * num_subapertures_side, 2), dtype=np.int64)
    slopes_q4_23 = np.zeros((num_subapertures_side * num_subapertures_side, 2), dtype=np.int64)

    for subap_row in range(num_subapertures_side):
        row_slice = slice(subap_row * subaperture_pixels, (subap_row + 1) * subaperture_pixels)
        for subap_col in range(num_subapertures_side):
            col_slice = slice(subap_col * subaperture_pixels, (subap_col + 1) * subaperture_pixels)
            tile = image_grid[row_slice, col_slice].astype(np.int64)
            total_intensity = int(tile.sum())
            x_weighted = int((tile * x_indices).sum())
            y_weighted = int((tile * y_indices).sum())

            reciprocal_q27 = _compute_reciprocal_q27(total_intensity)
            x_centroid_q4_23 = _compute_centroid_q4_23(x_weighted, reciprocal_q27) if total_intensity > 0 else 0
            y_centroid_q4_23 = _compute_centroid_q4_23(y_weighted, reciprocal_q27) if total_intensity > 0 else 0

            subap_index = subap_row * num_subapertures_side + subap_col
            centroids_q4_23[subap_index, 0] = x_centroid_q4_23
            centroids_q4_23[subap_index, 1] = y_centroid_q4_23
            slopes_q4_23[subap_index, 0] = x_centroid_q4_23 - slope_reference_q4_23
            slopes_q4_23[subap_index, 1] = y_centroid_q4_23 - slope_reference_q4_23

    centroids_xy = centroids_q4_23.astype(np.float64) / scale
    slopes_xy = slopes_q4_23.astype(np.float64) / scale

    estimated_coeffs_q4_23 = None
    estimated_coeffs = None
    if reconstruction_mask is not None and reconstruction_matrix_q1_16 is not None:
        reconstruction_mask = np.asarray(reconstruction_mask, dtype=bool).ravel()
        valid_slopes_q4_23 = slopes_q4_23[reconstruction_mask]
        slope_vector_q4_23 = np.empty(valid_slopes_q4_23.shape[0] * 2, dtype=np.int64)
        slope_vector_q4_23[0::2] = valid_slopes_q4_23[:, 0]
        slope_vector_q4_23[1::2] = valid_slopes_q4_23[:, 1]

        mac_sum_q5_39 = np.asarray(reconstruction_matrix_q1_16, dtype=np.int64) @ slope_vector_q4_23
        estimated_coeffs_q4_23 = mac_sum_q5_39 >> 17
        estimated_coeffs = estimated_coeffs_q4_23.astype(np.float64) / scale

    return {
        "quantized_image": quantized_image,
        "centroids_q4_23": centroids_q4_23,
        "slopes_q4_23": slopes_q4_23,
        "centroids_xy": centroids_xy,
        "slopes_xy": slopes_xy,
        "centroids_grid": centroids_xy.reshape(num_subapertures_side, num_subapertures_side, 2),
        "slopes_grid": slopes_xy.reshape(num_subapertures_side, num_subapertures_side, 2),
        "slope_reference_pixels": slope_reference_pixels,
        "estimated_coeffs_q4_23": estimated_coeffs_q4_23,
        "estimated_coeffs": estimated_coeffs,
    }

def run_hcipy_estimation(
    image,
    estimator,
    reference_slopes,
    reconstruction_matrix,
    zernike_modes,
    aperture,
    measured_opd_field=None,
    shwfs=None,
):
    """
    Run centroid estimation and Zernike reconstruction on a SHWFS frame.

    Parameters
    ----------
    image : hcipy.Field or np.ndarray
        Detector readout for the aberrated wavefront.
    estimator : ShackHartmannWavefrontSensorEstimator
        HCIPy estimator used to recover subaperture slopes.
    reference_slopes : np.ndarray
        Flat-wavefront centroid reference with shape (2, N_subs).
    reconstruction_matrix : np.ndarray
        Pseudo-inverse interaction matrix with shape (2*N_subs, N_modes).
    zernike_modes : sequence[hcipy.Field]
        Zernike basis fields used for reconstruction.
    aperture : hcipy.Field
        Pupil mask used to constrain reconstructed OPD.
    measured_opd_field : hcipy.Field or None
        Ground-truth input OPD. When provided, the residual OPD is returned.
    shwfs : SquareShackHartmannWavefrontSensorOptics or None
        SHWFS optics object. When provided, valid subaperture positions are
        returned for visualization.

    Returns
    -------
    dict
        Slope vectors, estimated coefficients, reconstructed OPD, and optional
        residual wavefront products.
    """
    slopes_aber = estimator.estimate([image])
    slopes_delta = slopes_aber - reference_slopes
    estimated_coeffs = reconstruction_matrix.T @ slopes_delta.ravel()

    reconstructed_zernike = sum(
        coeff * mode for coeff, mode in zip(estimated_coeffs, zernike_modes)
    )
    reconstructed_opd_field = aperture * reconstructed_zernike
    residual_field = None
    if measured_opd_field is not None:
        residual_field = measured_opd_field - reconstructed_opd_field

    sub_positions = None
    if shwfs is not None:
        sub_positions = shwfs.mla_grid.subset(estimator.estimation_subapertures)

    return {
        "slopes_aber": np.asarray(slopes_aber),
        "slopes_delta": np.asarray(slopes_delta),
        "slope_x": np.asarray(slopes_delta[0, :]),
        "slope_y": np.asarray(slopes_delta[1, :]),
        "estimated_coeffs": np.asarray(estimated_coeffs),
        "reconstructed_zernike": reconstructed_zernike,
        "reconstructed_opd_field": reconstructed_opd_field,
        "residual_field": residual_field,
        "sub_positions": sub_positions,
    }


def generate_shwfs_visualizations(
    *,
    image_ref,
    image_aber,
    estimation,
    aperture,
    input_opd_field,
    true_coeffs,
    mode_labels,
    wavelength,
    num_lenslets,
    results_path="shwfs_results.png",
    ao_demo_path="shwfs_ao_demo.png",
    show_plots=True,
    comparison_label="True",
    estimated_label="Estimated",
    figure_title="Shack-Hartmann WFS Simulation (HCIPy)",
):
    """
    Render the SHWFS result summary and AO imaging demo figures.

    Parameters
    ----------
    image_ref, image_aber : hcipy.Field or np.ndarray
        Flat and aberrated SHWFS detector images.
    estimation : dict
        Output from run_hcipy_estimation().
    aperture : hcipy.Field
        Pupil mask.
    input_opd_field : hcipy.Field
        Ground-truth OPD map applied to the wavefront.
    true_coeffs : np.ndarray
        Ground-truth Zernike coefficients in metres.
    mode_labels : sequence[str]
        Labels for the plotted Zernike modes.
    wavelength : float
        WFS wavelength in metres.
    num_lenslets : int
        Lenslets per axis, used for slope vector scaling.
    results_path, ao_demo_path : str or None
        Output filenames. Set either to None to skip saving that figure.
    show_plots : bool
        Whether to display the generated figures.

    Returns
    -------
    dict
        Saved figure paths and Strehl / residual metrics.
    """
    plt = _import_pyplot(interactive=show_plots)

    slope_x = estimation["slope_x"]
    slope_y = estimation["slope_y"]
    estimated_coeffs = estimation["estimated_coeffs"]
    reconstructed_opd_field = estimation["reconstructed_opd_field"]
    residual_field = estimation["residual_field"]
    sub_positions = estimation["sub_positions"]

    if residual_field is None:
        raise ValueError("measured_opd_field is required to visualize residual wavefronts")
    if sub_positions is None:
        raise ValueError("shwfs is required to visualize subaperture slopes")

    results_fig = None
    ao_fig = None

    results_fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    results_fig.suptitle(
        figure_title,
        fontsize=14,
        fontweight="bold",
    )

    plt.sca(axes[0, 0])
    imshow_field(image_ref, cmap="inferno")
    plt.title("SHWFS image — flat wavefront")
    plt.colorbar(label="counts")

    plt.sca(axes[0, 1])
    imshow_field(image_aber, cmap="inferno")
    plt.title("SHWFS image — aberrated wavefront")
    plt.colorbar(label="counts")

    plt.sca(axes[0, 2])
    imshow_field(image_aber, cmap="inferno", alpha=0.4)
    pitch = _estimate_subaperture_pitch(sub_positions, num_lenslets)
    max_magnitude = np.hypot(slope_x, slope_y).max() + 1e-30
    arrow_scale = pitch * 1.5 / max_magnitude
    plt.quiver(
        sub_positions.x,
        sub_positions.y,
        slope_x * arrow_scale,
        slope_y * arrow_scale,
        color="cyan",
        scale=1,
        scale_units="xy",
        angles="xy",
        width=0.003,
    )
    plt.title("Differential slope field")

    pupil_mask = aperture > 0.5
    vmax_nm = np.abs(input_opd_field[pupil_mask]).max() * 1e9

    plt.sca(axes[1, 0])
    imshow_field(input_opd_field * 1e9, cmap="RdBu", vmin=-vmax_nm, vmax=vmax_nm, mask=aperture)
    plt.title("Input OPD [nm]")
    plt.colorbar(label="nm")

    plt.sca(axes[1, 1])
    imshow_field(
        reconstructed_opd_field * 1e9,
        cmap="RdBu",
        vmin=-vmax_nm,
        vmax=vmax_nm,
        mask=aperture,
    )
    plt.title("Reconstructed OPD [nm]")
    plt.colorbar(label="nm")

    plt.sca(axes[1, 2])
    x_axis = np.arange(len(mode_labels))
    width = 0.35
    plt.bar(
        x_axis - width / 2,
        np.asarray(true_coeffs) * 1e9,
        width,
        label=comparison_label,
        color="steelblue",
    )
    plt.bar(
        x_axis + width / 2,
        estimated_coeffs * 1e9,
        width,
        label=estimated_label,
        color="tomato",
    )
    plt.xticks(x_axis, mode_labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Coefficient [nm]")
    plt.title("Zernike coefficients")
    plt.legend(loc="upper right")
    plt.axhline(0, color="k", linewidth=0.6)

    plt.tight_layout()
    if results_path is not None:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        results_fig.savefig(results_path, dpi=150, bbox_inches="tight")

    num_pupil_pixels = int(np.sqrt(np.asarray(aperture).size))
    aperture_mask = np.asarray(aperture, dtype=float).reshape(num_pupil_pixels, num_pupil_pixels)
    input_opd_2d = np.asarray(input_opd_field, dtype=float).reshape(num_pupil_pixels, num_pupil_pixels)
    residual_opd_2d = np.asarray(residual_field, dtype=float).reshape(num_pupil_pixels, num_pupil_pixels)

    scene = load_vlt_demo_scene(size=num_pupil_pixels)
    psf_dl = _psf_from_opd(np.zeros_like(input_opd_2d), aperture_mask, wavelength)
    psf_ab = _psf_from_opd(input_opd_2d, aperture_mask, wavelength)
    psf_corr = _psf_from_opd(residual_opd_2d, aperture_mask, wavelength)

    img_ab = _convolve_with_psf(scene, psf_ab)
    img_corr = _convolve_with_psf(scene, psf_corr)
    img_dl = _convolve_with_psf(scene, psf_dl)

    img_ab /= img_ab.max() + 1e-30
    img_corr /= img_corr.max() + 1e-30
    img_dl /= img_dl.max() + 1e-30

    strehl_ab = psf_ab.max() / (psf_dl.max() + 1e-30)
    strehl_corr = psf_corr.max() / (psf_dl.max() + 1e-30)

    ao_fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ao_fig.suptitle(
        "AO Imaging Demo: Aberrated vs Corrected Telescope Image",
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].imshow(scene, cmap="gray", origin="lower")
    axes[0, 0].set_title("Reference object (VLT image)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_ab, cmap="gray", origin="lower")
    axes[0, 1].set_title("Aberrated image")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_corr, cmap="gray", origin="lower")
    axes[0, 2].set_title("Corrected image (from reconstructed OPD)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(np.log10(psf_ab + 1e-12), cmap="magma", origin="lower")
    axes[1, 0].set_title("log10 PSF (aberrated)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.log10(psf_corr + 1e-12), cmap="magma", origin="lower")
    axes[1, 1].set_title("log10 PSF (corrected)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(np.log10(psf_dl + 1e-12), cmap="magma", origin="lower")
    axes[1, 2].set_title("log10 PSF (diffraction-limited)")
    axes[1, 2].axis("off")

    ao_fig.text(
        0.5,
        0.01,
        f"Strehl (aberrated): {strehl_ab:.3f}   |   Strehl (corrected): {strehl_corr:.3f}",
        ha="center",
        fontsize=11,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    if ao_demo_path is not None:
        Path(ao_demo_path).parent.mkdir(parents=True, exist_ok=True)
        ao_fig.savefig(ao_demo_path, dpi=150, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        if results_fig is not None:
            plt.close(results_fig)
        if ao_fig is not None:
            plt.close(ao_fig)

    return {
        "results_path": results_path,
        "ao_demo_path": ao_demo_path,
        "strehl_ab": strehl_ab,
        "strehl_corr": strehl_corr,
        "residual_rms": np.std(residual_field[pupil_mask]),
    }


def generate_aberrated_image(
    true_coeffs=None,
    num_lenslets=16,
    num_zernike=10,
    demo_image_path="shwfs_aber_demo.png",
):
    """
    Build SHWFS model, apply Zernike aberration, return detector pixel array.

    Parameters
    ----------
    true_coeffs : array-like or None
        Zernike coefficients (metres of OPD), length num_zernike.
        If None, a default set of low-order aberrations is used.
    num_lenslets : int
        Lenslet grid size (num_lenslets x num_lenslets).
    num_zernike : int
        Number of Zernike modes (Noll 2 .. num_zernike+1).
    demo_image_path : str or None
        Where to save the aberrated VLT demo image.  None to skip.

    Returns
    -------
    image_aber : np.ndarray, shape (num_pupil_pixels**2,)
        Flat detector pixel array from the aberrated SHWFS readout.
    """
    simulation = generate_shwfs_case(
        true_coeffs=true_coeffs,
        num_lenslets=num_lenslets,
        num_zernike=num_zernike,
        demo_image_path=demo_image_path,
    )
    image_aber = np.asarray(simulation["image_aber"])
    print(f"Generated {num_lenslets}x{num_lenslets} SHWFS image, {len(image_aber)} pixels")
    return image_aber


def parse_zernike_coefficient_file(path):
    """Load a text file of Zernike coefficients from HCIPy-side dumps."""
    coeffs = []
    coeff_path = Path(path)

    with coeff_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.lower().startswith("mode "):
                continue

            tokens = stripped.replace(",", " ").split()
            try:
                coeffs.append(float(tokens[-1]))
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse coefficient from line: {stripped!r}"
                ) from exc

    if not coeffs:
        raise ValueError(f"No coefficients found in {coeff_path}")

    return np.asarray(coeffs, dtype=np.float64)


def write_shwfs_hex(path, image):
    """Write a quantized SHWFS detector frame in image_rotating.hex format."""
    output_path = Path(path)
    quantized = quantize_shwfs_image(image)
    flat_values = np.asarray(quantized, dtype=np.uint8).reshape(-1)

    with output_path.open("w", encoding="ascii") as handle:
        for value in flat_values:
            handle.write(f"{int(value):02X}\n")

    return flat_values


def generate_shwfs_hex_from_coefficients(
    coefficients,
    output_path,
    num_lenslets=16,
    demo_image_path=None,
):
    """Generate an image_rotating.hex-style byte stream from Zernike coefficients."""
    coeffs = np.asarray(coefficients, dtype=np.float64)
    image = generate_aberrated_image(
        true_coeffs=coeffs,
        num_lenslets=num_lenslets,
        num_zernike=int(coeffs.size),
        demo_image_path=demo_image_path,
    )
    return write_shwfs_hex(output_path, image)


def generate_shwfs_hex_from_file(
    coefficients_path,
    output_path,
    num_lenslets=16,
    demo_image_path=None,
):
    """Parse coefficients from disk and write an image_rotating.hex-style file."""
    coeffs = parse_zernike_coefficient_file(coefficients_path)
    return generate_shwfs_hex_from_coefficients(
        coeffs,
        output_path,
        num_lenslets=num_lenslets,
        demo_image_path=demo_image_path,
    )


if __name__ == "__main__":
    image_aber = generate_aberrated_image()
    print(f"image_aber shape: {image_aber.shape}, "
          f"range: [{image_aber.min():.4e}, {image_aber.max():.4e}]")

