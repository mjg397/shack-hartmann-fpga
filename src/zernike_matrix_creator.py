import numpy as np

def build_E_matrix(n_subap=16, pupil_radius=1.0):
    """
    Build Zernike gradient matrix E for a square Shack-Hartmann grid.

    Parameters:
        n_subap : int
            Number of subapertures per dimension (16 → 16x16 grid)
        pupil_radius : float
            Radius of normalized pupil (usually 1)

    Returns:
        E : ndarray of shape (2*N, 10)
            Gradient matrix mapping Zernike coeffs → slopes
        coords : ndarray of shape (N, 2)
            (x, y) coordinates of subaperture centers
    """

    # Create normalized grid in [-1, 1]
    x = np.linspace(-1, 1, n_subap)
    y = np.linspace(-1, 1, n_subap)
    xx, yy = np.meshgrid(x, y)

    coords = np.column_stack([xx.ravel(), yy.ravel()])
    N = coords.shape[0]

    # Initialize E matrix (2N rows, 10 modes)
    E = np.zeros((2 * N, 10))

    for i, (x, y) in enumerate(coords):

        # --- Zernike gradients ---

        # Mode 1: piston → zero
        dZdx = [0]
        dZdy = [0]

        # Mode 2: tilt x
        dZdx.append(1)
        dZdy.append(0)

        # Mode 3: tilt y
        dZdx.append(0)
        dZdy.append(1)

        # Mode 4: defocus
        dZdx.append(4 * x)
        dZdy.append(4 * y)

        # Mode 5: astig 45
        dZdx.append(2 * y)
        dZdy.append(2 * x)

        # Mode 6: astig 0
        dZdx.append(2 * x)
        dZdy.append(-2 * y)

        # Mode 7: coma x
        dZdx.append(6 * x**2 + 2 * y**2 - 2)
        dZdy.append(4 * x * y)

        # Mode 8: coma y
        dZdx.append(4 * x * y)
        dZdy.append(2 * x**2 + 6 * y**2 - 2)

        # Mode 9: trefoil x
        dZdx.append(3 * x**2 - 3 * y**2)
        dZdy.append(-6 * x * y)

        # Mode 10: trefoil y
        dZdx.append(6 * x * y)
        dZdy.append(3 * y**2 - 3 * x**2)

        # Fill E matrix
        E[2 * i, :]     = dZdx
        E[2 * i + 1, :] = dZdy

    return E, coords


# Example usage
E, coords = build_E_matrix(n_subap=16)

print("E shape:", E.shape)   # Expect (512, 10)
print("First few rows:\n", E[:6])
