"""
Shack-Hartmann WFS — E Matrix (Reconstruction Matrix) Generator
  N : number of Zernike modes
  M : number of valid sub-apertures
  W : gradient matrix,       shape (2M x N)
  E : reconstruction matrix, shape (N  × 2M)   E = (WᵀW)⁻¹Wᵀ  [Eq. 14]
"""

import numpy as np
from math import factorial

# =============================================================================
# STEP 1 — Parameters
# =============================================================================

TELESCOPE_DIAMETER        = 8.0      # metres
CENTRAL_OBSCURATION       = 1.2      # metres
SPIDER_WIDTH              = 0.05     # metres (0.0 to disable)
NUM_LENSLETS_ACROSS       = 16
NUM_ZERNIKE               = 10
TIKHONOV_RCOND            = 1e-3     # regularisation for near-singular W
FRAC_BITS                 = 16
WORD_BITS                 = 18
HEX_OUTPUT_FILE           = "e_matrix.hex"
MIF_OUTPUT_FILE           = "e_matrix.mif"
CSV_OUTPUT_FILE           = "e_matrix.csv"

CENTRAL_OBSCURATION_RATIO = CENTRAL_OBSCURATION / TELESCOPE_DIAMETER
PUPIL_RADIUS              = TELESCOPE_DIAMETER / 2.0

# =============================================================================
# STEP 2 — Pupil grid and aperture mask
# =============================================================================


lenslet_coords_1d = np.linspace(-1.0, 1.0, NUM_LENSLETS_ACROSS, endpoint=True)
lx, ly   = np.meshgrid(lenslet_coords_1d, lenslet_coords_1d)
lx_flat  = lx.ravel()
ly_flat  = ly.ravel()
lr       = np.sqrt(lx_flat**2 + ly_flat**2)

in_outer  = lr <= 1.0
out_inner = lr >= CENTRAL_OBSCURATION_RATIO

def spider_mask(x, y, width, n_arms=4):
    half = width / 2.0 / PUPIL_RADIUS
    keep = np.ones(len(x), dtype=bool)
    for angle_deg in np.linspace(0, 180, n_arms // 2, endpoint=False):
        angle = np.radians(angle_deg)
        keep &= ~(np.abs(x * np.sin(angle) - y * np.cos(angle)) <= half)
    return keep

not_spider = spider_mask(lx_flat, ly_flat, SPIDER_WIDTH) if SPIDER_WIDTH > 0.0 \
             else np.ones(len(lx_flat), dtype=bool)

valid_mask = in_outer & out_inner & not_spider
sx_coords  = lx_flat[valid_mask]
sy_coords  = ly_flat[valid_mask]
M          = int(valid_mask.sum())

print(f"Valid sub-apertures M = {M}")

# =============================================================================
# STEP 3 — Noll Zernike index table
# =============================================================================

_NOLL_TABLE = {
    1:(0,0),  2:(1,1),   3:(1,-1),
    4:(2,0),  5:(2,-2),  6:(2,2),
    7:(3,-1), 8:(3,1),   9:(3,-3), 10:(3,3),
    11:(4,0), 12:(4,2),  13:(4,-2), 14:(4,4),  15:(4,-4),
    16:(5,1), 17:(5,-1), 18:(5,3), 19:(5,-3), 20:(5,5), 21:(5,-5),
}

def noll_to_nm(j):
    if j in _NOLL_TABLE:
        return _NOLL_TABLE[j]
    n = int(np.ceil((-1 + np.sqrt(1 + 8*(j-1))) / 2 - 0.5))
    pos = j - n * (n + 1) // 2
    m_list = []
    for ma in range(n % 2, n + 1, 2):
        m_list += [0] if ma == 0 else [-ma, ma]
    return n, m_list[pos - 1]

# Start from Noll index 2 (tip) to skip piston (index 1) and get 10 useful modes:
# 2=tip, 3=tilt, 4=defocus, 5=astig45, 6=astig0, 7=coma_y, 8=coma_x, 9=trefoil_y, 10=trefoil_x, 11=spherical
noll_indices = list(range(2, NUM_ZERNIKE + 2))
modes_nm     = [noll_to_nm(j) for j in noll_indices]

print(f"Reconstructing Noll modes: {noll_indices}")

# =============================================================================
# STEP 4 — Zernike radial polynomial and evaluation
# =============================================================================

def radial_poly(n, m, r):
    """Radial polynomial R_n^m(r)."""
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

def zernike_val(n, m, r, theta):
    R = radial_poly(n, m, r)
    if   m > 0: return R * np.cos(m * theta)
    elif m < 0: return R * np.sin(-m * theta)
    else:       return R

# =============================================================================
# STEP 5 — Build W matrix  (2M × N) of Zernike partial derivatives
#
#   Following the paper (Eq. 11–12):
#
#     s_x(r,c) = sum_k  a_k  * dZ_k/dx  evaluated at sub-aperture centre
#     s_y(r,c) = sum_k  a_k  * dZ_k/dy  evaluated at sub-aperture centre
#
#   The W matrix element W[2*i, k]   = dZ_k/dx at sub-aperture i centre
#                         W[2*i+1, k] = dZ_k/dy at sub-aperture i centre
#
#   Slope vector ordering: s = [sx_0, sy_0, sx_1, sy_1, ..., sx_{M-1}, sy_{M-1}]
# =============================================================================

def radial_poly_deriv(n, m, r):
    """Derivative of R_n^m(r) with respect to r."""
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

def zernike_gradient_cartesian(n, m, x, y):
    """
    Cartesian partial derivatives dZ/dx, dZ/dy at positions (x, y).

    Uses the chain rule:
        dZ/dx = dZ/dr * dr/dx  +  dZ/dtheta * dtheta/dx
        dZ/dy = dZ/dr * dr/dy  +  dZ/dtheta * dtheta/dy

    where:
        dr/dx =  x/r,   dr/dy =  y/r
        dtheta/dx = -y/r^2,  dtheta/dy = x/r^2
    """
    r      = np.sqrt(x**2 + y**2)
    theta  = np.arctan2(y, x)
    r_safe = np.where(r == 0, 1e-30, r)   # avoid division by zero at origin

    R  = radial_poly(n, m, r)
    dR = radial_poly_deriv(n, m, r)

    # Partial derivatives of polar coordinates w.r.t. Cartesian
    dr_dx  =  x / r_safe
    dr_dy  =  y / r_safe
    dt_dx  = -y / r_safe**2    # dtheta/dx
    dt_dy  =  x / r_safe**2    # dtheta/dy

    m_abs = abs(m)

    if m > 0:
        # Z = R(r) * cos(m*theta)
        dZ_dr = dR * np.cos(m * theta)
        dZ_dt = -m * R * np.sin(m * theta)
    elif m < 0:
        # Z = R(r) * sin(|m|*theta)
        dZ_dr = dR * np.sin(m_abs * theta)
        dZ_dt =  m_abs * R * np.cos(m_abs * theta)
    else:
        # Z = R(r)  (azimuthally symmetric)
        dZ_dr = dR
        dZ_dt = np.zeros_like(r)

    dZdx = dZ_dr * dr_dx + dZ_dt * dt_dx
    dZdy = dZ_dr * dr_dy + dZ_dt * dt_dy
    return dZdx, dZdy

def gen_E_matrix():
    r_subs     = np.sqrt(sx_coords**2 + sy_coords**2)
    theta_subs = np.arctan2(sy_coords, sx_coords)

    # Build W  — shape (2M, N)
    # Row order: [sx_0, sy_0, sx_1, sy_1, ..., sx_{M-1}, sy_{M-1}]
    W = np.zeros((2 * M, NUM_ZERNIKE), dtype=float)
    for i, (n, m) in enumerate(modes_nm):
        dZdx, dZdy = zernike_gradient_cartesian(n, m, sx_coords, sy_coords)
        W[0::2, i] = dZdx   # x-slope rows: 0, 2, 4, ...
        W[1::2, i] = dZdy   # y-slope rows: 1, 3, 5, ...

    # =============================================================================
    # STEP 6 — Compute E = (WᵀW)⁻¹ Wᵀ  (Eq. 14)
    #
    #   Plain least-squares pseudo-inverse with optional Tikhonov regularisation
    #   to handle near-singular W (fewer modes than well-conditioned SVD directions):
    #
    #     E_tikhonov = (WᵀW + α²I)⁻¹ Wᵀ
    #
    #   Implemented via the economy SVD  W = U Σ Vᵀ  which gives:
    #
    #     E = V * diag(σ_i / (σ_i² + α²)) * Uᵀ
    #
    #   When TIKHONOV_RCOND = 0 this reduces exactly to the Moore-Penrose pinv.
    # =============================================================================

    U, singular_vals, Vt = np.linalg.svd(W, full_matrices=False)
    sigma_max = singular_vals[0]
    alpha     = TIKHONOV_RCOND * sigma_max                  # regularisation level
    sigma_reg = singular_vals / (singular_vals**2 + alpha**2)   # Tikhonov filter factors
    E         = (Vt.T * sigma_reg) @ U.T                    # shape (N, 2M)

    # =============================================================================
    # STEP 7 — Fixed-point quantisation
    # =============================================================================

    scale   = 2**FRAC_BITS
    INT_MIN = -(2**(WORD_BITS - 1))
    INT_MAX =  (2**(WORD_BITS - 1)) - 1
    E_fp    = np.clip(np.round(E * scale).astype(np.int64), INT_MIN, INT_MAX)

    # =============================================================================
    # STEP 8 — Write output files
    # =============================================================================

    WORD_MASK   = (1 << WORD_BITS) - 1
    total_words = NUM_ZERNIKE * 2 * M

    with open(HEX_OUTPUT_FILE, "w") as f:
        f.write(f"// E reconstruction matrix  (Eq. 13-14, Kong et al. 2023)\n")
        f.write(f"// Shape : {NUM_ZERNIKE} modes x {2*M} slopes = {total_words} words\n")
        f.write(f"// Format: {WORD_BITS}-bit two's complement, Q1.{FRAC_BITS}\n")
        f.write(f"// Address = mode_index * {2*M} + slope_index\n")
        f.write(f"// Noll indices: {noll_indices}\n\n")
        for row in E_fp:
            for val in row:
                f.write(f"{val & WORD_MASK:05X}\n")

    with open(MIF_OUTPUT_FILE, "w") as f:
        f.write(f"WIDTH={WORD_BITS};\nDEPTH={total_words};\n\n")
        f.write(f"ADDRESS_RADIX=UNS;\nDATA_RADIX=HEX;\n\nCONTENT BEGIN\n")
        for addr, row in enumerate(E_fp):
            for col, val in enumerate(row):
                f.write(f"\t{addr * (2 * M) + col} : {val & WORD_MASK:05X};\n")
        f.write(f"END;\n")

    header_cols = [s for k in range(M) for s in (f"sx_{k}", f"sy_{k}")]
    np.savetxt(CSV_OUTPUT_FILE, E, delimiter=",",
            header=",".join(header_cols), comments="")

    # =============================================================================
    # STEP 8b — Write split hex files for ematrix_accumulator
    #
    # e_matrix_x.hex : x-slope columns only (even indices 0,2,4,...)
    # e_matrix_y.hex : y-slope columns only (odd  indices 1,3,5,...)
    #
    # address layout matches the accumulator ROM:
    #   address = mode_index * M + sub_index
    #   so row 0 of e_matrix_x = all x-slope coefficients for mode 0 (Noll 2, tip)
    #      row 1 of e_matrix_x = all x-slope coefficients for mode 1 (Noll 3, tilt) etc.
    # =============================================================================

    with open("e_matrix_x.hex", "w") as fx, open("e_matrix_y.hex", "w") as fy:
        fx.write(f"// E matrix x-slope columns only\n")
        fx.write(f"// Shape : {NUM_ZERNIKE} modes x {M} x-cols = {NUM_ZERNIKE * M} words\n")
        fx.write(f"// Format: {WORD_BITS}-bit two's complement, Q1.{FRAC_BITS}\n")
        fx.write(f"// Address = mode_index * {M} + sub_index\n")
        fx.write(f"// Noll indices: {noll_indices}\n\n")
        fy.write(f"// E matrix y-slope columns only\n")
        fy.write(f"// Shape : {NUM_ZERNIKE} modes x {M} y-cols = {NUM_ZERNIKE * M} words\n")
        fy.write(f"// Format: {WORD_BITS}-bit two's complement, Q1.{FRAC_BITS}\n")
        fy.write(f"// Address = mode_index * {M} + sub_index\n")
        fy.write(f"// Noll indices: {noll_indices}\n\n")
        for mode_idx, row in enumerate(E_fp):
            # row has 2*M values: [sx_0, sy_0, sx_1, sy_1, ..., sx_{M-1}, sy_{M-1}]
            for col_idx, val in enumerate(row):
                if col_idx % 2 == 0:
                    # even index = x slope column for subaperture col_idx//2
                    fx.write(f"{val & WORD_MASK:05X}\n")
                else:
                    # odd index = y slope column for subaperture col_idx//2
                    fy.write(f"{val & WORD_MASK:05X}\n")

    return E_fp

gen_E_matrix()
