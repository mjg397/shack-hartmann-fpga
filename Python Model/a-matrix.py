import numpy as np
from math import factorial

# ===============================
# Grid: 16 x 16 = 256 subapertures
# ===============================
grid_size = 16
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)

r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

# Keep only points inside unit circle
mask = r <= 1
r = r[mask]
theta = theta[mask]

N_POINTS = len(r)
print("Number of valid subapertures:", N_POINTS)

# ===============================
# Radial polynomial
# ===============================
def R(n, m, r):
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(r)

    Rnm = np.zeros_like(r)
    for k in range((n - m)//2 + 1):
        num = (-1)**k * factorial(n - k)
        den = (
            factorial(k) *
            factorial((n + m)//2 - k) *
            factorial((n - m)//2 - k)
        )
        Rnm += num / den * r**(n - 2*k)
    return Rnm

# ===============================
# Zernike function
# ===============================
def zernike(n, m, r, theta):
    if m > 0:
        return R(n, m, r) * np.cos(m * theta)
    elif m < 0:
        return R(n, -m, r) * np.sin(-m * theta)
    else:
        return R(n, 0, r)

# ===============================
# First 10 modes
# ===============================
modes = [
    (0, 0),
    (1, -1), (1, 1),
    (2, -2), (2, 0), (2, 2),
    (3, -3), (3, -1), (3, 1), (3, 3)
]

# ===============================
# Build matrix
# ===============================
Z = np.zeros((len(modes), N_POINTS))

for i, (n, m) in enumerate(modes):
    Z[i, :] = zernike(n, m, r, theta)

print("Z shape:", Z.shape)
