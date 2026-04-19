"""
Shack-Hartmann WFS — E Matrix (Reconstruction Matrix) Library

Provides `generate_e_matrix()` which returns E as a flat 1-D numpy array.
"""

import numpy as np
from math import factorial


_NOLL_TABLE = {
    1:(0,0),  2:(1,1),   3:(1,-1),
    4:(2,0),  5:(2,-2),  6:(2,2),
    7:(3,-1), 8:(3,1),   9:(3,-3), 10:(3,3),
    11:(4,0), 12:(4,2),  13:(4,-2), 14:(4,4),  15:(4,-4),
    16:(5,1), 17:(5,-1), 18:(5,3), 19:(5,-3), 20:(5,5), 21:(5,-5),
}


def _noll_to_nm(j):
    if j in _NOLL_TABLE:
        return _NOLL_TABLE[j]
    n = int(np.ceil((-1 + np.sqrt(1 + 8*(j-1))) / 2 - 0.5))
    pos = j - n * (n + 1) // 2
    m_list = []
    for ma in range(n % 2, n + 1, 2):
        m_list += [0] if ma == 0 else [-ma, ma]
    return n, m_list[pos - 1]


def _radial_poly(n, m, r):
    m_abs = abs(m)
    Rnm   = np.zeros_like(r, dtype=float)
    if (n - m_abs) % 2 != 0:
        return Rnm
    for k in range((n - m_abs) // 2 + 1):
        num = (-1)**k * factorial(n - k)
        den = (factorial(k)
               * factorial((n + m_abs)//2 - k)
               * factorial((n - m_abs)//2 - k))
        Rnm += (num / den) * r**(n - 2*k)
    return Rnm


def _radial_poly_deriv(n, m, r):
    m_abs = abs(m)
    dRnm  = np.zeros_like(r, dtype=float)
    if (n - m_abs) % 2 != 0:
        return dRnm
    for k in range((n - m_abs) // 2 + 1):
        power = n - 2*k - 1
        if power < 0:
            continue
        num = (-1)**k * factorial(n - k)
        den = (factorial(k)
               * factorial((n + m_abs)//2 - k)
               * factorial((n - m_abs)//2 - k))
        dRnm += (num / den) * (n - 2*k) * r**power
    return dRnm


def _zernike_gradient_cartesian(n, m, x, y):
    r      = np.sqrt(x**2 + y**2)
    theta  = np.arctan2(y, x)
    r_safe = np.where(r == 0, 1e-30, r)

    R  = _radial_poly(n, m, r)
    dR = _radial_poly_deriv(n, m, r)

    dr_dx  =  x / r_safe
    dr_dy  =  y / r_safe
    dt_dx  = -y / r_safe**2
    dt_dy  =  x / r_safe**2

    m_abs = abs(m)

    if m > 0:
        dZ_dr = dR * np.cos(m * theta)
        dZ_dt = -m * R * np.sin(m * theta)
    elif m < 0:
        dZ_dr = dR * np.sin(m_abs * theta)
        dZ_dt =  m_abs * R * np.cos(m_abs * theta)
    else:
        dZ_dr = dR
        dZ_dt = np.zeros_like(r)

    dZdx = dZ_dr * dr_dx + dZ_dt * dt_dx
    dZdy = dZ_dr * dr_dy + dZ_dt * dt_dy
    return dZdx, dZdy


def _spider_mask(x, y, width, pupil_radius, n_arms=4):
    half = width / 2.0 / pupil_radius
    keep = np.ones(len(x), dtype=bool)
    for angle_deg in np.linspace(0, 180, n_arms // 2, endpoint=False):
        angle = np.radians(angle_deg)
        keep &= ~(np.abs(x * np.sin(angle) - y * np.cos(angle)) <= half)
    return keep


def generate_e_matrix(
    telescope_diameter=8.0,
    central_obscuration=1.2,
    spider_width=0.05,
    num_lenslets_across=16,
    num_zernike=10,
    tikhonov_rcond=1e-3,
):
    """
    Compute the E reconstruction matrix and return it as a 1-D numpy array.

    Returns
    -------
    e_flat : np.ndarray, shape (num_zernike * 2*M,)
        Row-major flattened E matrix (N x 2M).
    """
    pupil_radius = telescope_diameter / 2.0
    obscuration_ratio = central_obscuration / telescope_diameter

    # Pupil grid and aperture mask
    lenslet_coords_1d = np.linspace(-1.0, 1.0, num_lenslets_across, endpoint=True)
    lx, ly   = np.meshgrid(lenslet_coords_1d, lenslet_coords_1d)
    lx_flat  = lx.ravel()
    ly_flat  = ly.ravel()
    lr       = np.sqrt(lx_flat**2 + ly_flat**2)

    in_outer  = lr <= 1.0
    out_inner = lr >= obscuration_ratio

    if spider_width > 0.0:
        not_spider = _spider_mask(lx_flat, ly_flat, spider_width, pupil_radius)
    else:
        not_spider = np.ones(len(lx_flat), dtype=bool)

    valid_mask = in_outer & out_inner & not_spider
    sx_coords  = lx_flat[valid_mask]
    sy_coords  = ly_flat[valid_mask]
    M          = int(valid_mask.sum())

    # Noll Zernike modes
    noll_indices = list(range(1, num_zernike + 1))
    modes_nm     = [_noll_to_nm(j) for j in noll_indices]

    # Build W matrix (2M x N)
    W = np.zeros((2 * M, num_zernike), dtype=float)
    for i, (n, m) in enumerate(modes_nm):
        dZdx, dZdy = _zernike_gradient_cartesian(n, m, sx_coords, sy_coords)
        W[0::2, i] = dZdx
        W[1::2, i] = dZdy

    # E = Tikhonov-regularised pseudo-inverse of W
    U, singular_vals, Vt = np.linalg.svd(W, full_matrices=False)
    sigma_max = singular_vals[0]
    alpha     = tikhonov_rcond * sigma_max
    sigma_reg = singular_vals / (singular_vals**2 + alpha**2)
    E         = (Vt.T * sigma_reg) @ U.T   # shape (N, 2M)

    return E.ravel()

if __name__ == "__main__":
    matrix = generate_e_matrix()
    print("Range: ", matrix.min(), " : ", matrix.max())