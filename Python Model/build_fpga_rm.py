import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path('../src').resolve()))
import shwfs_utils
from hcipy import Field, Wavefront

case = shwfs_utils.generate_shwfs_case(num_lenslets=16, num_zernike=10, demo_image_path=None)

zernike_modes = case["zernike_modes"]
aperture = case["aperture"]
wavelength = case["wavelength"]
magnifier = case["shwfs"].magnifier if hasattr(case["shwfs"], "magnifier") else shwfs_utils.Magnifier(5e-3 / 8.0)
shwfs = case["shwfs"]
# Need to manually propagate
def measure_fpga_slopes(mode, amp):
    phase = aperture * mode * amp * (2.0 * np.pi / wavelength)
    aber_ap = Field(aperture * np.exp(1j * np.array(phase)), aperture.grid)
    wf = Wavefront(aber_ap, wavelength)
    wf.total_power = 1
    # Integrate using a camera
    from hcipy import NoiselessDetector
    camera = NoiselessDetector(shwfs(magnifier(wf)).grid)
    camera.integrate(shwfs(magnifier(wf)), 1)
    img = camera.read_out()
    est = shwfs_utils.run_fpga_like_estimation(img, num_subapertures_side=16, subaperture_pixels=16)
    return est["slopes_xy"]

probe_amp = 0.05 * wavelength
IM_rows = []
valid_mask = case["fpga_subaperture_mask"].ravel()

for i, mode in enumerate(zernike_modes):
    s_p = measure_fpga_slopes(mode, probe_amp)
    s_m = measure_fpga_slopes(mode, -probe_amp)
    # We only care about valid subapertures. s_p has shape (256, 2)
    s_p_valid = s_p[valid_mask]
    s_m_valid = s_m[valid_mask]
    # Flatten to [x0, x1.. y0, y1..] or [x0, y0, x1, y1..]?
    # Wait, how is RM structured in FPGA?
    # e_matrix_x corresponds to X slopes, e_matrix_y to Y slopes.
    # So the slope vector is [x0, y0, x1, y1] or separated?
    # In my export script, I separated them: `col[:N_subs]` and `col[N_subs:]`.
    # So the slope vector used in inversion should be [x0, x1..., y0, y1...].
    diff_x = (s_p_valid[:, 0] - s_m_valid[:, 0]) / (2 * probe_amp)
    diff_y = (s_p_valid[:, 1] - s_m_valid[:, 1]) / (2 * probe_amp)
    IM_rows.append(np.concatenate([diff_x, diff_y]))

IM = np.array(IM_rows)
rcond = 1e-3
U, s, Vt = np.linalg.svd(IM, full_matrices=False)
s_reg = s / (s**2 + (rcond * s.max())**2)
RM = (Vt.T * s_reg) @ U.T

print("Built new RM based on FPGA pixel slopes. Shape:", RM.shape)
np.save("fpga_rm.npy", RM)
