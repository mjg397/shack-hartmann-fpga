"""
Synthetic Shack-Hartmann frame generation for Monte Carlo evaluation.

Running the full HCIPy optical model 10,000 times is infeasible
(generate_aberrated_image rebuilds the optics per call, ~40 s => ~4.6 days), so
the Monte Carlo operates on synthesised frames instead. The forward model is
derived entirely from repository data:

    E                = the reconstruction matrix in e_matrix_{x,y}.mem  (Q1.16)
    slopes           = pinv(E) @ coefficients          (forward model)
    spot position    = reference slope + slope         (pixels, per subaperture)
    frame            = Gaussian spots rendered and quantised to 8 bits

Spot sigma and integrated intensity default to values MEASURED from the
repository's own hcipy_image_rotating.mem (sigma ~ 0.75 px, sum ~ 812 ADU), so
the synthetic frames match the real one's regime rather than an invented one.

A small set of true HCIPy frames is used separately (accuracy/physical_anchor.py)
to confirm the synthetic population is representative.
"""

from __future__ import annotations

import numpy as np

from . import bitaccurate as BA
from . import config as C

# Measured from rtl/hcipy_image_rotating.mem -- see README "Spot statistics".
DEFAULT_SPOT_SIGMA_PX = 0.75
DEFAULT_SPOT_SUM_ADU = 812.0


def forward_matrix() -> np.ndarray:
    """
    Pseudo-inverse of the reconstruction matrix: coefficients -> slopes.

    Returns shape (2*NUM_SUBAPS_VALID, NUM_MODES); the first half is x, the
    second y, matching the E-matrix layout.
    """
    ex, ey = BA.ematrix()
    scale = float(1 << C.EMATRIX_FRAC)
    E = np.hstack([ex / scale, ey / scale])          # (modes, 2*subaps)
    return np.linalg.pinv(E)                          # (2*subaps, modes)


# Metres per unit of the raw (unscaled) E-matrix reconstruction.
#
#   raw_reconstruction = sum (E/2^16) * slope_px
#   metres             = raw * 2^EMATRIX_FRAC * METERS_PER_LSB  =  2^-27
#
# Verified empirically on the repository's own frame: the ratio between
# run_pipeline (fixed) and run_float_reference (float) has median 2^-26.983.
COEFF_UNIT_M = (1 << C.EMATRIX_FRAC) * C.METERS_PER_LSB


def coeffs_to_slopes(coeffs_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward model: Zernike coefficients in METRES -> slopes in pixels.

    Coefficients are converted out of metres into the E matrix's internal unit
    before inversion, otherwise the round trip closes on the wrong scale.
    """
    internal = np.asarray(coeffs_m, dtype=np.float64) / COEFF_UNIT_M
    v = forward_matrix() @ internal
    n = C.NUM_SUBAPS_VALID
    return v[:n], v[n:]


def render_frame(
    x_slope_px: np.ndarray,
    y_slope_px: np.ndarray,
    spot_sigma: float = DEFAULT_SPOT_SIGMA_PX,
    spot_sum: float = DEFAULT_SPOT_SUM_ADU,
    valid_idx: np.ndarray | None = None,
    quantise: bool = True,
) -> np.ndarray:
    """
    Render a 256x256 frame from per-subaperture slopes.

    Spot centre for subaperture s is reference[s] + slope[s], in local tile
    coordinates. Only in-pupil tiles get a spot; the rest stay dark, matching
    the pupil bitmap the RTL gates on.

    quantise=True  -> round and clip to 8-bit ADU, i.e. what the sensor delivers.
    quantise=False -> return the raw float intensity, unclipped.

    The noise sweep MUST use quantise=False: it renders in PHOTON units, where
    spot_sum can be far above 255, and clipping there would destroy the spot
    before Poisson noise and digitisation are applied.
    """
    if valid_idx is None:
        valid_idx = BA.valid_subaps()

    ref_x, ref_y = BA.slopes_ref()
    scale = float(1 << C.CENTROID_FRAC)

    frame = np.zeros((C.FRAME_H, C.FRAME_W), dtype=np.float64)
    grid = np.arange(C.SUBAP_PIXELS, dtype=np.float64)

    two_sig_sq = 2.0 * spot_sigma * spot_sigma

    for k, tile in enumerate(valid_idx):
        r, c = divmod(int(tile), C.NUM_LENSLETS)
        cx = ref_x[tile] / scale + float(x_slope_px[k])
        cy = ref_y[tile] / scale + float(y_slope_px[k])

        gx = np.exp(-((grid - cx) ** 2) / two_sig_sq)
        gy = np.exp(-((grid - cy) ** 2) / two_sig_sq)
        spot = np.outer(gy, gx)
        total = spot.sum()
        if total > 0:
            spot *= spot_sum / total

        frame[r * C.SUBAP_PIXELS:(r + 1) * C.SUBAP_PIXELS,
              c * C.SUBAP_PIXELS:(c + 1) * C.SUBAP_PIXELS] = spot

    if not quantise:
        return frame
    return np.clip(np.rint(frame), 0, 255).astype(np.int64)


def random_coefficients(rng: np.random.Generator, scale_m: float = 6e-8) -> np.ndarray:
    """
    Draw a physically reasonable Zernike coefficient vector, in metres.

    The default scale matches the repository's own demo case, whose HCIPy
    coefficients span roughly +/-6.3e-8 m. Low-order modes get more power than
    high-order ones, as in real atmospheric turbulence.
    """
    falloff = 1.0 / np.sqrt(np.arange(1, C.NUM_MODES + 1))
    return rng.normal(0.0, scale_m, C.NUM_MODES) * falloff


def make_case(rng: np.random.Generator, **kw) -> dict:
    """Generate one Monte Carlo case: coefficients -> slopes -> frame."""
    coeffs = random_coefficients(rng, kw.pop("scale_m", 6e-8))
    sx, sy = coeffs_to_slopes(coeffs)
    frame = render_frame(sx, sy, **kw)
    return {
        "true_coeffs_m": coeffs,
        "true_x_slope_px": sx,
        "true_y_slope_px": sy,
        "frame": frame,
    }
