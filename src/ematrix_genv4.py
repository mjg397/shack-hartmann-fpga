#!/usr/bin/env python3
"""Generate an FPGA reconstruction matrix using an HDL-style CoG estimator.

This script keeps only the pieces needed for two tasks:

1. Emulate the centroid / slope arithmetic used by the HDL pipeline.
2. Use HCIPy to render Shack-Hartmann images and calibrate an interaction matrix
   with that emulated estimator.

The resulting reconstruction matrix can then be quantized and written in the
same split-ROM format used by the FPGA accumulator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from hcipy import (
    Field,
    Magnifier,
    NoiselessDetector,
    ShackHartmannWavefrontSensorEstimator,
    SquareShackHartmannWavefrontSensorOptics,
    Wavefront,
    evaluate_supersampled,
    make_obstructed_circular_aperture,
    make_pupil_grid,
    make_zernike_basis,
)
import scipy.ndimage as ndimage


# ---------------------------------------------------------------------------
# Default paths and system parameters
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_FULL_HEX_PATH = REPO_ROOT / "tmp" / "e_matrix_hdl_cog.hex"
DEFAULT_MIF_PATH = REPO_ROOT / "tmp" / "e_matrix_hdl_cog.mif"
DEFAULT_CSV_PATH = REPO_ROOT / "tmp" / "e_matrix_hdl_cog.csv"
DEFAULT_X_HEX_PATH = REPO_ROOT / "hdl" / "data" / "e_matrix_x.hex"
DEFAULT_Y_HEX_PATH = REPO_ROOT / "hdl" / "data" / "e_matrix_y.hex"

DEFAULT_TELESCOPE_DIAMETER = 8.0
DEFAULT_CENTRAL_OBSCURATION = 1.2
DEFAULT_SPIDER_WIDTH = 0.05
DEFAULT_OVERSIZING_FACTOR = 16 / 15
DEFAULT_WAVELENGTH_WFS = 0.7e-6
DEFAULT_F_NUMBER = 50.0
DEFAULT_SH_DIAMETER = 5e-3
DEFAULT_NUM_LENSLETS = 16
DEFAULT_NUM_ZERNIKE = 10
DEFAULT_SUBAPERTURE_PIXELS = 16
DEFAULT_SUBAP_PUPIL_RADIUS = 7.20
DEFAULT_PROBE_AMPLITUDE_WAVES = 0.05
DEFAULT_TIKHONOV_RCOND = 1e-3
DEFAULT_WORD_BITS = 18
DEFAULT_FRAC_BITS = 16
DEFAULT_COEFF_UNIT = "waves"
DEFAULT_ESTIMATED_HDL_PATH = REPO_ROOT / "full_pipeline_sim" / "data" / "hdl_estimated_zernikes_named.txt"
DEFAULT_ESTIMATED_HCIPY_PATH = REPO_ROOT / "full_pipeline_sim" / "data" / "hcipy_estimated_zernikes_named.txt"
DEFAULT_COMPARISON_PATH = REPO_ROOT / "full_pipeline_sim" / "data" / "hdl_hcipy_zernike_comparison.txt"


# ---------------------------------------------------------------------------
# Exact HDL reciprocal emulation helpers
# ---------------------------------------------------------------------------

# This LUT is copied from reciprocal_u16_q27.v. Values are Q0.16 seeds indexed
# by the top 8 fractional bits of the normalized 16-bit denominator.
RECIP_SEED_LUT_Q0_16 = (
    0x8020, 0x8060, 0x80A1, 0x80E2, 0x8123, 0x8164, 0x81A5, 0x81E7,
    0x8229, 0x826B, 0x82AE, 0x82F1, 0x8334, 0x8377, 0x83BB, 0x83FF,
    0x8443, 0x8488, 0x84CC, 0x8511, 0x8557, 0x859C, 0x85E2, 0x8628,
    0x866F, 0x86B6, 0x86FD, 0x8744, 0x878C, 0x87D3, 0x881C, 0x8864,
    0x88AD, 0x88F6, 0x8940, 0x8989, 0x89D3, 0x8A1E, 0x8A68, 0x8AB3,
    0x8AFF, 0x8B4A, 0x8B96, 0x8BE2, 0x8C2F, 0x8C7C, 0x8CC9, 0x8D17,
    0x8D65, 0x8DB3, 0x8E02, 0x8E51, 0x8EA0, 0x8EF0, 0x8F40, 0x8F90,
    0x8FE1, 0x9032, 0x9083, 0x90D5, 0x9127, 0x9179, 0x91CC, 0x921F,
    0x9273, 0x92C7, 0x931B, 0x9370, 0x93C5, 0x941B, 0x9470, 0x94C7,
    0x951D, 0x9574, 0x95CC, 0x9624, 0x967C, 0x96D5, 0x972E, 0x9787,
    0x97E1, 0x983B, 0x9896, 0x98F1, 0x994D, 0x99A9, 0x9A05, 0x9A62,
    0x9AC0, 0x9B1D, 0x9B7C, 0x9BDA, 0x9C39, 0x9C99, 0x9CF9, 0x9D59,
    0x9DBA, 0x9E1C, 0x9E7E, 0x9EE0, 0x9F43, 0x9FA6, 0xA00A, 0xA06E,
    0xA0D3, 0xA138, 0xA19E, 0xA204, 0xA26B, 0xA2D3, 0xA33A, 0xA3A3,
    0xA40C, 0xA475, 0xA4DF, 0xA549, 0xA5B4, 0xA620, 0xA68C, 0xA6F8,
    0xA766, 0xA7D3, 0xA842, 0xA8B1, 0xA920, 0xA990, 0xAA01, 0xAA72,
    0xAAE4, 0xAB56, 0xABC9, 0xAC3D, 0xACB1, 0xAD26, 0xAD9B, 0xAE11,
    0xAE88, 0xAEFF, 0xAF77, 0xAFF0, 0xB069, 0xB0E3, 0xB15D, 0xB1D8,
    0xB254, 0xB2D1, 0xB34E, 0xB3CC, 0xB44B, 0xB4CA, 0xB54A, 0xB5CB,
    0xB64C, 0xB6CE, 0xB751, 0xB7D5, 0xB859, 0xB8DE, 0xB964, 0xB9EB,
    0xBA72, 0xBAFB, 0xBB83, 0xBC0D, 0xBC98, 0xBD23, 0xBDAF, 0xBE3C,
    0xBECA, 0xBF59, 0xBFE8, 0xC078, 0xC109, 0xC19B, 0xC22E, 0xC2C2,
    0xC357, 0xC3EC, 0xC482, 0xC51A, 0xC5B2, 0xC64B, 0xC6E5, 0xC780,
    0xC81C, 0xC8B9, 0xC957, 0xC9F6, 0xCA96, 0xCB36, 0xCBD8, 0xCC7B,
    0xCD1F, 0xCDC4, 0xCE6A, 0xCF11, 0xCFB9, 0xD062, 0xD10C, 0xD1B7,
    0xD263, 0xD311, 0xD3BF, 0xD46F, 0xD520, 0xD5D2, 0xD685, 0xD73A,
    0xD7EF, 0xD8A6, 0xD95E, 0xDA17, 0xDAD1, 0xDB8D, 0xDC4A, 0xDD08,
    0xDDC8, 0xDE88, 0xDF4B, 0xE00E, 0xE0D3, 0xE199, 0xE260, 0xE329,
    0xE3F4, 0xE4BF, 0xE58C, 0xE65B, 0xE72B, 0xE7FC, 0xE8CF, 0xE9A4,
    0xEA7A, 0xEB51, 0xEC2A, 0xED05, 0xEDE1, 0xEEBF, 0xEF9F, 0xF080,
    0xF163, 0xF247, 0xF32D, 0xF415, 0xF4FF, 0xF5EA, 0xF6D7, 0xF7C6,
    0xF8B7, 0xF9A9, 0xFA9E, 0xFB94, 0xFC8C, 0xFD86, 0xFE82, 0xFF80,
)


def clz16(value: int) -> int:
    """Count leading zeros in a 16-bit unsigned integer."""
    masked = value & 0xFFFF
    if masked == 0:
        return 16
    return 16 - masked.bit_length()


def newton_step_q27(a_q1_26: int, x0_u27: int) -> int:
    """Mirror one fixed-point Newton step from newton_step_q27.v."""
    ax_mul = a_q1_26 * x0_u27
    ax_q1_26 = (ax_mul >> 27) & 0x0FFFFFFF
    ax_q1_26_clamped = min(ax_q1_26, 0x08000000)
    two_minus_ax_q1_26 = 0x08000000 - ax_q1_26_clamped

    prod_mul = x0_u27 * two_minus_ax_q1_26
    x1_rounded_ext = (prod_mul + 0x0000000002000000) >> 26
    return min(x1_rounded_ext, 0x07FFFFFF)


def reciprocal_u16_q27(total_intensity: int) -> int:
    """Exact Python port of reciprocal_u16_q27.v."""
    if total_intensity <= 0:
        return 0x07FFFFFF

    v_safe = total_intensity & 0xFFFF
    shift_left = clz16(v_safe)
    a_q1_15 = (v_safe << shift_left) & 0xFFFF
    a_q1_26 = a_q1_15 << 11

    lut_idx = (a_q1_15 >> 7) & 0xFF
    seed_q0_16 = RECIP_SEED_LUT_Q0_16[lut_idx]
    x0_u27 = seed_q0_16 << 11

    x1_q0_27 = newton_step_q27(a_q1_26, x0_u27)
    x2_q0_27 = newton_step_q27(a_q1_26, x1_q0_27)

    msb_index = 15 - shift_left
    denorm_round_bias = 0 if msb_index == 0 else (1 << (msb_index - 1))
    denorm_numer = x2_q0_27 + denorm_round_bias
    out_q0_27 = x2_q0_27 if msb_index == 0 else (denorm_numer >> msb_index)

    return min(out_q0_27, 0x07FFFFFF)


# ---------------------------------------------------------------------------
# HDL-style image quantization and CoG helpers
# ---------------------------------------------------------------------------


def quantize_shwfs_image(image: np.ndarray, out_min: int = 0, out_max: int = 255) -> np.ndarray:
    """Match the detector quantization used before pixels are streamed to the FPGA."""
    image_array = np.asarray(image)
    if image_array.dtype == np.uint8:
        return image_array.copy()

    image_array = np.asarray(image_array, dtype=np.float64)
    image_min = float(image_array.min())
    image_max = float(image_array.max())
    if image_max <= image_min:
        return np.zeros_like(image_array, dtype=np.uint8)

    return np.interp(image_array, (image_min, image_max), (out_min, out_max)).astype(np.uint8)


def compute_centroid_q4_23(weighted_intensity: int, reciprocal_q27: int) -> int:
    """Mirror the HDL centroid multiplier: 20.0 x 0.27 -> Q4.23."""
    return (int(weighted_intensity) * int(reciprocal_q27)) >> 4


def subaperture_inside_pupil(row: int, col: int, grid_size: int, radius: float) -> bool:
    """Return True when the entire unit-square subaperture lies inside the pupil."""
    center = grid_size / 2.0
    corners = (
        (col, row),
        (col + 1, row),
        (col, row + 1),
        (col + 1, row + 1),
    )
    return all(np.hypot(x - center, y - center) <= radius for x, y in corners)


def build_valid_subaperture_mask(grid_size: int, radius: float) -> np.ndarray:
    """Build the same row-major valid-subaperture mask used by the HDL bitmap."""
    return np.array(
        [
            subaperture_inside_pupil(row, col, grid_size, radius)
            for row in range(grid_size)
            for col in range(grid_size)
        ],
        dtype=bool,
    )


def build_slope_vector_q4_23(
    image: np.ndarray,
    valid_mask: np.ndarray,
    num_subapertures_side: int,
    subaperture_pixels: int,
    fraction_bits: int = 23,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Emulate the HDL path from image tiles to packed valid slope vector.

    The output ordering matches the accumulator expectation:
    [sx_0, sy_0, sx_1, sy_1, ...]
    """
    quantized_image = quantize_shwfs_image(image)
    detector_pixels = num_subapertures_side * subaperture_pixels
    image_grid = np.asarray(quantized_image, dtype=np.uint8).reshape(detector_pixels, detector_pixels)

    scale = 1 << fraction_bits
    slope_reference_q4_23 = int(round(((subaperture_pixels - 1) / 2.0) * scale))

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

            reciprocal_q27 = reciprocal_u16_q27(total_intensity)
            if total_intensity > 0:
                x_centroid_q4_23 = compute_centroid_q4_23(x_weighted, reciprocal_q27)
                y_centroid_q4_23 = compute_centroid_q4_23(y_weighted, reciprocal_q27)
            else:
                x_centroid_q4_23 = 0
                y_centroid_q4_23 = 0

            subap_index = subap_row * num_subapertures_side + subap_col
            centroids_q4_23[subap_index, 0] = x_centroid_q4_23
            centroids_q4_23[subap_index, 1] = y_centroid_q4_23
            slopes_q4_23[subap_index, 0] = x_centroid_q4_23 - slope_reference_q4_23
            slopes_q4_23[subap_index, 1] = y_centroid_q4_23 - slope_reference_q4_23

    valid_slopes_q4_23 = slopes_q4_23[np.asarray(valid_mask, dtype=bool).ravel()]
    slope_vector_q4_23 = np.empty(valid_slopes_q4_23.shape[0] * 2, dtype=np.int64)
    slope_vector_q4_23[0::2] = valid_slopes_q4_23[:, 0]
    slope_vector_q4_23[1::2] = valid_slopes_q4_23[:, 1]

    debug = {
        "quantized_image": quantized_image,
        "centroids_q4_23": centroids_q4_23,
        "slopes_q4_23": slopes_q4_23,
        "valid_slopes_q4_23": valid_slopes_q4_23,
    }
    return slope_vector_q4_23, debug


# ---------------------------------------------------------------------------
# HCIPy optics setup and response calibration helpers
# ---------------------------------------------------------------------------


def build_shwfs_optics(
    telescope_diameter: float,
    central_obscuration: float,
    spider_width: float,
    oversizing_factor: float,
    wavelength_wfs: float,
    f_number: float,
    sh_diameter: float,
    num_lenslets: int,
    num_zernike: int = DEFAULT_NUM_ZERNIKE,
) -> dict[str, object]:
    """Build the HCIPy optical objects needed to render SHWFS detector images."""
    num_pupil_pixels = int(240 * oversizing_factor)
    pupil_grid_diameter = telescope_diameter * oversizing_factor
    pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

    aperture_generator = make_obstructed_circular_aperture(
        telescope_diameter,
        central_obscuration / telescope_diameter,
        num_spiders=4,
        spider_width=spider_width,
    )
    aperture = evaluate_supersampled(aperture_generator, pupil_grid, 4)

    magnification = sh_diameter / telescope_diameter
    magnifier = Magnifier(magnification)
    shwfs = SquareShackHartmannWavefrontSensorOptics(
        pupil_grid.scaled(magnification),
        f_number,
        num_lenslets,
        sh_diameter,
    )

    reference_wavefront = Wavefront(aperture, wavelength_wfs)
    reference_wavefront.total_power = 1
    reference_wfs_wavefront = shwfs(magnifier(reference_wavefront))
    camera = NoiselessDetector(reference_wfs_wavefront.electric_field.grid)

    zernike_basis = make_zernike_basis(
        num_zernike + 1,
        telescope_diameter,
        pupil_grid,
        starting_mode=1,
    )

    return {
        "aperture": aperture,
        "pupil_grid": pupil_grid,
        "magnifier": magnifier,
        "shwfs": shwfs,
        "camera": camera,
        "zernike_basis": zernike_basis,
        "num_pupil_pixels": num_pupil_pixels,
        "wavelength_wfs": wavelength_wfs,
    }


def parse_zernike_coefficient_file(path: Path) -> np.ndarray:
    """Parse a coefficient file that may contain either bare floats or named values."""
    coefficients = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.lower().startswith("mode "):
                continue
            coefficients.append(float(line.split()[-1]))
    return np.asarray(coefficients, dtype=np.float64)


def write_named_coefficients(path: Path, coeffs: np.ndarray, mode_labels: list[str], header_prefix: str) -> None:
    """Write a human-readable named coefficient file for one estimator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"mode {header_prefix}_coeff\n")
        for label, coeff in zip(mode_labels, coeffs):
            handle.write(f"{label} {coeff:.10f}\n")


def write_estimator_comparison(
    path: Path,
    mode_labels: list[str],
    true_coeffs: np.ndarray,
    hdl_coeffs: np.ndarray,
    hcipy_coeffs: np.ndarray,
) -> None:
    """Write a side-by-side comparison of the true, HDL-emulated, and HCIPy estimates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("mode true_coeff hdl_coeff hcipy_coeff hdl_error hcipy_error\n")
        for label, true_coeff, hdl_coeff, hcipy_coeff in zip(mode_labels, true_coeffs, hdl_coeffs, hcipy_coeffs):
            handle.write(
                f"{label} {true_coeff:.10f} {hdl_coeff:.10f} {hcipy_coeff:.10f} "
                f"{(hdl_coeff - true_coeff):.10f} {(hcipy_coeff - true_coeff):.10f}\n"
            )


def build_mode_labels(num_zernike: int) -> list[str]:
    """Return readable labels for the first ten Noll modes, then generic fallbacks."""
    named_modes = [
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
    if num_zernike <= len(named_modes):
        return named_modes[:num_zernike]
    return named_modes + [f"Mode {index}" for index in range(len(named_modes) + 1, num_zernike + 1)]


def render_shwfs_image_for_opd(
    opd_field: Field,
    aperture: Field,
    wavelength_wfs: float,
    magnifier: Magnifier,
    shwfs: SquareShackHartmannWavefrontSensorOptics,
    camera: NoiselessDetector,
) -> np.ndarray:
    """Render one Shack-Hartmann detector image for a given OPD field."""
    phase_map = opd_field * (2.0 * np.pi / wavelength_wfs)
    aberrated_aperture = Field(aperture * np.exp(1j * np.asarray(phase_map)), aperture.grid)
    wavefront = Wavefront(aberrated_aperture, wavelength_wfs)
    wavefront.total_power = 1
    camera.integrate(shwfs(magnifier(wavefront)), 1)
    return camera.read_out()


def build_true_opd_field(aperture: Field, zernike_modes: list[Field], coeffs_meters: np.ndarray) -> Field:
    """Combine Zernike modes into one physical OPD field in meters."""
    return aperture * sum(coeff * mode for coeff, mode in zip(coeffs_meters, zernike_modes))


def build_hcipy_estimator(
    aperture: Field,
    wavelength_wfs: float,
    magnifier: Magnifier,
    shwfs: SquareShackHartmannWavefrontSensorOptics,
    camera: NoiselessDetector,
) -> tuple[ShackHartmannWavefrontSensorEstimator, np.ndarray, np.ndarray]:
    """Build the HCIPy centroid estimator and its flat-wave reference slopes."""
    base_estimator = ShackHartmannWavefrontSensorEstimator(
        shwfs.mla_grid,
        shwfs.micro_lens_array.mla_index,
    )

    reference_wavefront = Wavefront(aperture, wavelength_wfs)
    reference_wavefront.total_power = 1
    camera.integrate(shwfs(magnifier(reference_wavefront)), 1)
    image_ref = camera.read_out()

    fluxes = ndimage.sum(
        image_ref,
        base_estimator.mla_index,
        base_estimator.estimation_subapertures,
    )
    flux_limit = fluxes.max() * 0.5

    valid_subaperture_mask = shwfs.mla_grid.zeros(dtype="bool")
    valid_subaperture_mask[
        base_estimator.estimation_subapertures[fluxes > flux_limit]
    ] = True

    estimator = ShackHartmannWavefrontSensorEstimator(
        shwfs.mla_grid,
        shwfs.micro_lens_array.mla_index,
        valid_subaperture_mask,
    )
    slopes_ref = np.asarray(estimator.estimate([image_ref]))
    return estimator, slopes_ref, np.asarray(image_ref)


def calibrate_interaction_matrix_with_hdl_estimator(
    zernike_modes: list[Field],
    aperture: Field,
    wavelength_wfs: float,
    magnifier: Magnifier,
    shwfs: SquareShackHartmannWavefrontSensorOptics,
    camera: NoiselessDetector,
    valid_mask: np.ndarray,
    num_lenslets: int,
    subaperture_pixels: int,
    probe_amplitude_meters: float,
) -> np.ndarray:
    """Calibrate the HDL-style interaction matrix using HCIPy-rendered SH images.

    For each Zernike mode we render +probe and -probe images, run the HDL-style
    CoG estimator on each, and use a central finite difference to estimate the
    slope response per metre of modal amplitude.
    """
    interaction_rows = []

    for mode_field in zernike_modes:
        opd_plus = aperture * (probe_amplitude_meters * mode_field)
        opd_minus = aperture * (-probe_amplitude_meters * mode_field)

        image_plus = render_shwfs_image_for_opd(
            opd_plus,
            aperture,
            wavelength_wfs,
            magnifier,
            shwfs,
            camera,
        )
        image_minus = render_shwfs_image_for_opd(
            opd_minus,
            aperture,
            wavelength_wfs,
            magnifier,
            shwfs,
            camera,
        )

        slopes_plus_q4_23, _ = build_slope_vector_q4_23(
            image_plus,
            valid_mask,
            num_subapertures_side=num_lenslets,
            subaperture_pixels=subaperture_pixels,
        )
        slopes_minus_q4_23, _ = build_slope_vector_q4_23(
            image_minus,
            valid_mask,
            num_subapertures_side=num_lenslets,
            subaperture_pixels=subaperture_pixels,
        )

        # Convert back to floating Q4.23 slope units before dividing by the probe amplitude.
        slopes_plus = slopes_plus_q4_23.astype(np.float64) / (1 << 23)
        slopes_minus = slopes_minus_q4_23.astype(np.float64) / (1 << 23)
        interaction_rows.append((slopes_plus - slopes_minus) / (2.0 * probe_amplitude_meters))

    return np.asarray(interaction_rows, dtype=np.float64)


def estimate_coefficients_with_hdl_matrix(
    image: np.ndarray,
    e_matrix_quantized: np.ndarray,
    valid_mask: np.ndarray,
    num_lenslets: int,
    subaperture_pixels: int,
    coefficient_unit: str,
    wavelength_wfs: float,
    reference_slope_vector_q4_23: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Estimate coefficients from one SHWFS image using the HDL-emulated pipeline.

    If a flat-wave reference slope vector is supplied, it is subtracted before the
    reconstruction matrix is applied. This mirrors the differential slope quantity
    used during interaction-matrix calibration.
    """
    slope_vector_q4_23, debug = build_slope_vector_q4_23(
        image,
        valid_mask,
        num_subapertures_side=num_lenslets,
        subaperture_pixels=subaperture_pixels,
    )

    slopes_delta_q4_23 = slope_vector_q4_23.copy()
    if reference_slope_vector_q4_23 is not None:
        slopes_delta_q4_23 = slopes_delta_q4_23 - np.asarray(reference_slope_vector_q4_23, dtype=np.int64)

    mac_sum_q5_39 = np.asarray(e_matrix_quantized, dtype=np.int64) @ slopes_delta_q4_23
    estimated_coeffs_q4_23 = mac_sum_q5_39 >> 17
    estimated_coeffs_output_units = estimated_coeffs_q4_23.astype(np.float64) / (1 << 23)

    if coefficient_unit == "meters":
        estimated_coeffs_meters = estimated_coeffs_output_units
    elif coefficient_unit == "waves":
        estimated_coeffs_meters = estimated_coeffs_output_units * wavelength_wfs
    else:
        raise ValueError(f"Unsupported coefficient unit: {coefficient_unit}")

    return {
        "estimated_coeffs_q4_23": estimated_coeffs_q4_23,
        "estimated_coeffs_output_units": estimated_coeffs_output_units,
        "estimated_coeffs_meters": estimated_coeffs_meters,
        "slope_vector_q4_23": slope_vector_q4_23,
        "slopes_delta_q4_23": slopes_delta_q4_23,
        **debug,
    }


def estimate_coefficients_with_hcipy(
    image: np.ndarray,
    estimator: ShackHartmannWavefrontSensorEstimator,
    reference_slopes: np.ndarray,
    reconstruction_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    """Estimate coefficients using HCIPy's own centroid and reconstruction path."""
    slopes_aber = np.asarray(estimator.estimate([image]))
    slopes_delta = slopes_aber - np.asarray(reference_slopes)
    estimated_coeffs = reconstruction_matrix.T @ slopes_delta.ravel()
    return {
        "slopes_aber": slopes_aber,
        "slopes_delta": slopes_delta,
        "estimated_coeffs": np.asarray(estimated_coeffs, dtype=np.float64),
    }


def compare_against_hcipy(
    true_coeffs_meters: np.ndarray,
    zernike_modes: list[Field],
    optics: dict[str, object],
    e_matrix_quantized: np.ndarray,
    valid_mask: np.ndarray,
    coefficient_unit: str,
    num_lenslets: int,
    subaperture_pixels: int,
    subtract_reference_slopes: bool,
) -> dict[str, object]:
    """Run one synthetic aberration through both estimators and collect a comparison."""
    opd_field = build_true_opd_field(optics["aperture"], zernike_modes, true_coeffs_meters)
    image_aber = render_shwfs_image_for_opd(
        opd_field,
        optics["aperture"],
        optics["wavelength_wfs"],
        optics["magnifier"],
        optics["shwfs"],
        optics["camera"],
    )
    image_ref = render_shwfs_image_for_opd(
        optics["aperture"] * 0.0,
        optics["aperture"],
        optics["wavelength_wfs"],
        optics["magnifier"],
        optics["shwfs"],
        optics["camera"],
    )

    estimator, hcipy_reference_slopes, _ = build_hcipy_estimator(
        optics["aperture"],
        optics["wavelength_wfs"],
        optics["magnifier"],
        optics["shwfs"],
        optics["camera"],
    )
    hcipy_estimation = estimate_coefficients_with_hcipy(
        image_aber,
        estimator,
        hcipy_reference_slopes,
        optics["hcipy_reconstruction_matrix"],
    )

    reference_slope_vector_q4_23 = None
    if subtract_reference_slopes:
        reference_slope_vector_q4_23, _ = build_slope_vector_q4_23(
            image_ref,
            valid_mask,
            num_subapertures_side=num_lenslets,
            subaperture_pixels=subaperture_pixels,
        )

    hdl_estimation = estimate_coefficients_with_hdl_matrix(
        image_aber,
        e_matrix_quantized,
        valid_mask,
        num_lenslets,
        subaperture_pixels,
        coefficient_unit,
        optics["wavelength_wfs"],
        reference_slope_vector_q4_23=reference_slope_vector_q4_23,
    )

    return {
        "true_coeffs_meters": np.asarray(true_coeffs_meters, dtype=np.float64),
        "image_ref": np.asarray(image_ref),
        "image_aber": np.asarray(image_aber),
        "hcipy_estimation": hcipy_estimation,
        "hdl_estimation": hdl_estimation,
        "reference_slope_vector_q4_23": reference_slope_vector_q4_23,
    }


def compute_reconstruction_matrix(interaction_matrix: np.ndarray, rcond: float) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-invert the measured interaction matrix.

    The interaction matrix is stored as:
        rows    -> modes
        columns -> packed slope samples [sx_0, sy_0, ...]

    Its pseudo-inverse therefore has shape (2M, N). The FPGA accumulator wants an
    E matrix of shape (N, 2M), so we return both forms explicitly.
    """
    u_mat, singular_vals, vt_mat = np.linalg.svd(interaction_matrix, full_matrices=False)
    sigma_reg = singular_vals / (singular_vals**2 + (rcond * singular_vals.max())**2)
    pseudo_inverse = (vt_mat.T * sigma_reg) @ u_mat.T
    e_matrix = pseudo_inverse.T
    return pseudo_inverse, e_matrix


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def scale_coefficients_for_output(e_matrix_m_per_slope: np.ndarray, wavelength_wfs: float, coeff_unit: str) -> np.ndarray:
    """Convert the physical reconstructor into the requested output coefficient unit."""
    if coeff_unit == "meters":
        return e_matrix_m_per_slope
    if coeff_unit == "waves":
        return e_matrix_m_per_slope / wavelength_wfs
    raise ValueError(f"Unsupported coefficient unit: {coeff_unit}")


def quantize_e_matrix(e_matrix: np.ndarray, word_bits: int, frac_bits: int) -> tuple[np.ndarray, dict[str, int]]:
    """Quantize the floating-point E matrix into signed fixed-point integers."""
    scale = 1 << frac_bits
    int_min = -(1 << (word_bits - 1))
    int_max = (1 << (word_bits - 1)) - 1

    rounded = np.round(e_matrix * scale).astype(np.int64)
    saturations = int(np.count_nonzero((rounded < int_min) | (rounded > int_max)))
    quantized = np.clip(rounded, int_min, int_max)

    stats = {
        "nonzero_words": int(np.count_nonzero(quantized)),
        "saturated_words": saturations,
        "total_words": int(quantized.size),
    }
    return quantized, stats


def write_matrix_outputs(
    e_matrix_float: np.ndarray,
    e_matrix_quantized: np.ndarray,
    noll_indices: list[int],
    coeff_unit: str,
    frac_bits: int,
    word_bits: int,
    full_hex_path: Path,
    mif_path: Path,
    csv_path: Path,
    x_hex_path: Path,
    y_hex_path: Path,
) -> None:
    """Write full and split FPGA matrix files."""
    for path in (full_hex_path, mif_path, csv_path, x_hex_path, y_hex_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    word_mask = (1 << word_bits) - 1
    num_modes, num_slopes = e_matrix_quantized.shape
    num_subapertures = num_slopes // 2
    total_words = num_modes * num_slopes

    with full_hex_path.open("w", encoding="ascii") as handle:
        handle.write("// HDL-calibrated E reconstruction matrix\n")
        handle.write(f"// Shape : {num_modes} modes x {num_slopes} slopes = {total_words} words\n")
        handle.write(f"// Format: {word_bits}-bit two's complement, Q1.{frac_bits}\n")
        handle.write(f"// Coefficient unit: {coeff_unit}\n")
        handle.write(f"// Noll indices: {noll_indices}\n\n")
        for row in e_matrix_quantized:
            for value in row:
                handle.write(f"{value & word_mask:05X}\n")

    with mif_path.open("w", encoding="ascii") as handle:
        handle.write(f"WIDTH={word_bits};\n")
        handle.write(f"DEPTH={total_words};\n\n")
        handle.write("ADDRESS_RADIX=UNS;\n")
        handle.write("DATA_RADIX=HEX;\n\n")
        handle.write("CONTENT BEGIN\n")
        for mode_index, row in enumerate(e_matrix_quantized):
            for slope_index, value in enumerate(row):
                address = mode_index * num_slopes + slope_index
                handle.write(f"\t{address} : {value & word_mask:05X};\n")
        handle.write("END;\n")

    header_cols = [name for sub_index in range(num_subapertures) for name in (f"sx_{sub_index}", f"sy_{sub_index}")]
    np.savetxt(csv_path, e_matrix_float, delimiter=",", header=",".join(header_cols), comments="")

    with x_hex_path.open("w", encoding="ascii") as x_handle, y_hex_path.open("w", encoding="ascii") as y_handle:
        x_handle.write("// HDL-calibrated E matrix x-slope columns only\n")
        x_handle.write(f"// Shape : {num_modes} modes x {num_subapertures} x-cols = {num_modes * num_subapertures} words\n")
        x_handle.write(f"// Format: {word_bits}-bit two's complement, Q1.{frac_bits}\n")
        x_handle.write(f"// Coefficient unit: {coeff_unit}\n")
        x_handle.write(f"// Noll indices: {noll_indices}\n\n")

        y_handle.write("// HDL-calibrated E matrix y-slope columns only\n")
        y_handle.write(f"// Shape : {num_modes} modes x {num_subapertures} y-cols = {num_modes * num_subapertures} words\n")
        y_handle.write(f"// Format: {word_bits}-bit two's complement, Q1.{frac_bits}\n")
        y_handle.write(f"// Coefficient unit: {coeff_unit}\n")
        y_handle.write(f"// Noll indices: {noll_indices}\n\n")

        for row in e_matrix_quantized:
            for slope_index, value in enumerate(row):
                if slope_index % 2 == 0:
                    x_handle.write(f"{value & word_mask:05X}\n")
                else:
                    y_handle.write(f"{value & word_mask:05X}\n")


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for the standalone HDL-calibrated matrix generator."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Shack-Hartmann reconstruction matrix by combining HCIPy optics "
            "with an HDL-style CoG slope estimator."
        )
    )
    parser.add_argument("--num-lenslets", type=int, default=DEFAULT_NUM_LENSLETS)
    parser.add_argument("--num-zernike", type=int, default=DEFAULT_NUM_ZERNIKE)
    parser.add_argument("--subaperture-pixels", type=int, default=DEFAULT_SUBAPERTURE_PIXELS)
    parser.add_argument("--subap-radius", type=float, default=DEFAULT_SUBAP_PUPIL_RADIUS)
    parser.add_argument("--probe-amplitude-waves", type=float, default=DEFAULT_PROBE_AMPLITUDE_WAVES)
    parser.add_argument("--tikhonov-rcond", type=float, default=DEFAULT_TIKHONOV_RCOND)
    parser.add_argument("--coefficient-unit", choices=("meters", "waves"), default=DEFAULT_COEFF_UNIT)
    parser.add_argument("--word-bits", type=int, default=DEFAULT_WORD_BITS)
    parser.add_argument("--frac-bits", type=int, default=DEFAULT_FRAC_BITS)
    parser.add_argument("--full-hex-path", type=Path, default=DEFAULT_FULL_HEX_PATH)
    parser.add_argument("--mif-path", type=Path, default=DEFAULT_MIF_PATH)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--x-hex-path", type=Path, default=DEFAULT_X_HEX_PATH)
    parser.add_argument("--y-hex-path", type=Path, default=DEFAULT_Y_HEX_PATH)
    parser.add_argument("--compare-coefficients", type=Path, default=None)
    parser.add_argument("--subtract-reference-slopes", action="store_true")
    parser.add_argument("--hdl-estimated-output", type=Path, default=DEFAULT_ESTIMATED_HDL_PATH)
    parser.add_argument("--hcipy-estimated-output", type=Path, default=DEFAULT_ESTIMATED_HCIPY_PATH)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON_PATH)
    return parser


def main() -> int:
    """Run the HCIPy calibration loop with the HDL-style CoG estimator."""
    args = build_parser().parse_args()

    optics = build_shwfs_optics(
        telescope_diameter=DEFAULT_TELESCOPE_DIAMETER,
        central_obscuration=DEFAULT_CENTRAL_OBSCURATION,
        spider_width=DEFAULT_SPIDER_WIDTH,
        oversizing_factor=DEFAULT_OVERSIZING_FACTOR,
        wavelength_wfs=DEFAULT_WAVELENGTH_WFS,
        f_number=DEFAULT_F_NUMBER,
        sh_diameter=DEFAULT_SH_DIAMETER,
        num_lenslets=args.num_lenslets,
        num_zernike=args.num_zernike,
    )

    valid_mask = build_valid_subaperture_mask(args.num_lenslets, args.subap_radius)
    num_valid_subapertures = int(valid_mask.sum())

    noll_indices = list(range(2, args.num_zernike + 2))
    zernike_basis = make_zernike_basis(
        args.num_zernike + 1,
        DEFAULT_TELESCOPE_DIAMETER,
        optics["pupil_grid"],
        starting_mode=1,
    )
    zernike_modes = [zernike_basis[index] for index in range(1, args.num_zernike + 1)]

    # Build the HCIPy reference estimator as a comparison baseline. This is not
    # used to generate the HDL matrix; it is only used for side-by-side testing.
    hcipy_estimator, hcipy_reference_slopes, _ = build_hcipy_estimator(
        optics["aperture"],
        DEFAULT_WAVELENGTH_WFS,
        optics["magnifier"],
        optics["shwfs"],
        optics["camera"],
    )

    probe_amplitude_meters = args.probe_amplitude_waves * DEFAULT_WAVELENGTH_WFS
    hcipy_interaction_rows = []
    for mode_field in zernike_modes:
        opd_plus = optics["aperture"] * (probe_amplitude_meters * mode_field)
        opd_minus = optics["aperture"] * (-probe_amplitude_meters * mode_field)
        image_plus = render_shwfs_image_for_opd(
            opd_plus,
            optics["aperture"],
            DEFAULT_WAVELENGTH_WFS,
            optics["magnifier"],
            optics["shwfs"],
            optics["camera"],
        )
        image_minus = render_shwfs_image_for_opd(
            opd_minus,
            optics["aperture"],
            DEFAULT_WAVELENGTH_WFS,
            optics["magnifier"],
            optics["shwfs"],
            optics["camera"],
        )
        slopes_plus = np.asarray(hcipy_estimator.estimate([image_plus])) - hcipy_reference_slopes
        slopes_minus = np.asarray(hcipy_estimator.estimate([image_minus])) - hcipy_reference_slopes
        hcipy_interaction_rows.append((slopes_plus - slopes_minus).ravel() / (2.0 * probe_amplitude_meters))
    optics["hcipy_reconstruction_matrix"] = compute_reconstruction_matrix(
        np.asarray(hcipy_interaction_rows, dtype=np.float64),
        args.tikhonov_rcond,
    )[0]
    interaction_matrix = calibrate_interaction_matrix_with_hdl_estimator(
        zernike_modes=zernike_modes,
        aperture=optics["aperture"],
        wavelength_wfs=DEFAULT_WAVELENGTH_WFS,
        magnifier=optics["magnifier"],
        shwfs=optics["shwfs"],
        camera=optics["camera"],
        valid_mask=valid_mask,
        num_lenslets=args.num_lenslets,
        subaperture_pixels=args.subaperture_pixels,
        probe_amplitude_meters=probe_amplitude_meters,
    )

    _, e_matrix_meters = compute_reconstruction_matrix(interaction_matrix, args.tikhonov_rcond)
    e_matrix_output_units = scale_coefficients_for_output(
        e_matrix_meters,
        DEFAULT_WAVELENGTH_WFS,
        args.coefficient_unit,
    )
    e_matrix_quantized, quant_stats = quantize_e_matrix(
        e_matrix_output_units,
        args.word_bits,
        args.frac_bits,
    )

    write_matrix_outputs(
        e_matrix_float=e_matrix_output_units,
        e_matrix_quantized=e_matrix_quantized,
        noll_indices=noll_indices,
        coeff_unit=args.coefficient_unit,
        frac_bits=args.frac_bits,
        word_bits=args.word_bits,
        full_hex_path=args.full_hex_path,
        mif_path=args.mif_path,
        csv_path=args.csv_path,
        x_hex_path=args.x_hex_path,
        y_hex_path=args.y_hex_path,
    )

    print(f"Valid subapertures: {num_valid_subapertures}")
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    print(f"E matrix shape: {e_matrix_output_units.shape}")
    print(f"E coefficient unit: {args.coefficient_unit}")
    print(f"E abs max: {np.max(np.abs(e_matrix_output_units)):.12e}")
    print(f"Quantized nonzero words: {quant_stats['nonzero_words']} / {quant_stats['total_words']}")
    print(f"Quantized saturated words: {quant_stats['saturated_words']}")
    print(f"Wrote full HEX to {args.full_hex_path}")
    print(f"Wrote MIF to {args.mif_path}")
    print(f"Wrote CSV to {args.csv_path}")
    print(f"Wrote split X ROM to {args.x_hex_path}")
    print(f"Wrote split Y ROM to {args.y_hex_path}")

    if quant_stats["nonzero_words"] == 0:
        print("WARNING: all quantized E entries are zero in the selected output unit.")

    if args.compare_coefficients is not None:
        true_coeffs_meters = parse_zernike_coefficient_file(args.compare_coefficients)
        if true_coeffs_meters.size != args.num_zernike:
            raise ValueError(
                f"Expected {args.num_zernike} coefficients, got {true_coeffs_meters.size} in {args.compare_coefficients}"
            )

        comparison = compare_against_hcipy(
            true_coeffs_meters=true_coeffs_meters,
            zernike_modes=zernike_modes,
            optics=optics,
            e_matrix_quantized=e_matrix_quantized,
            valid_mask=valid_mask,
            coefficient_unit=args.coefficient_unit,
            num_lenslets=args.num_lenslets,
            subaperture_pixels=args.subaperture_pixels,
            subtract_reference_slopes=args.subtract_reference_slopes,
        )

        mode_labels = build_mode_labels(args.num_zernike)

        write_named_coefficients(
            args.hdl_estimated_output,
            comparison["hdl_estimation"]["estimated_coeffs_meters"],
            mode_labels,
            "hdl_estimated",
        )
        write_named_coefficients(
            args.hcipy_estimated_output,
            comparison["hcipy_estimation"]["estimated_coeffs"],
            mode_labels,
            "hcipy_estimated",
        )
        write_estimator_comparison(
            args.comparison_output,
            mode_labels,
            comparison["true_coeffs_meters"],
            comparison["hdl_estimation"]["estimated_coeffs_meters"],
            comparison["hcipy_estimation"]["estimated_coeffs"],
        )

        print(f"Wrote HDL estimates to {args.hdl_estimated_output}")
        print(f"Wrote HCIPy estimates to {args.hcipy_estimated_output}")
        print(f"Wrote estimator comparison to {args.comparison_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())