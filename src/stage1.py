"""
stage1.py — Generate an aberrated Shack-Hartmann wavefront sensor image.

Builds a VLT-like pupil and SHWFS optical model using HCIPy, applies a known
Zernike aberration, propagates through the sensor, and returns the resulting
detector pixel array.  Also saves an aberrated VLT demo image for presentation.

The FPGA handles centroid estimation (using geometric lenslet centres as
references) and wavefront reconstruction (RM computed elsewhere).
"""

from hcipy import (
    make_pupil_grid,
    make_obstructed_circular_aperture,
    evaluate_supersampled,
    Wavefront,
    Field,
    SquareShackHartmannWavefrontSensorOptics,
    NoiselessDetector,
    Magnifier,
    make_zernike_basis,
)

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import scipy.ndimage as ndimage
from pathlib import Path


# ---------------------------------------------------------------------------
# Private helpers for demo-image generation
# ---------------------------------------------------------------------------

def _load_vlt_scene(size):
    """Load a VLT image from ./VLT_Images and resize to (size, size) grayscale."""
    import matplotlib.pyplot as plt

    image_dir = Path(__file__).resolve().parent.parent / "VLT_Images"
    for name in ("eso1322a.jpg", "eso1131a.jpg"):
        path = image_dir / name
        if path.exists():
            img = plt.imread(path)
            if img.ndim == 3:
                img = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            img = np.asarray(img, dtype=float)

            zoom_factor = min(size / img.shape[0], size / img.shape[1])
            resized = ndimage.zoom(img, zoom_factor, order=1)

            out = np.zeros((size, size), dtype=float)
            y0 = max((size - resized.shape[0]) // 2, 0)
            x0 = max((size - resized.shape[1]) // 2, 0)
            out[y0:y0 + min(resized.shape[0], size),
                x0:x0 + min(resized.shape[1], size)] = \
                resized[:min(resized.shape[0], size), :min(resized.shape[1], size)]

            out -= out.min()
            out /= out.max() + 1e-30
            print(f"Loaded VLT scene from {path}")
            return out

    raise FileNotFoundError("No VLT image found in ./VLT_Images")


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    import matplotlib.pyplot as plt

    # -- Telescope parameters ------------------------------------------------
    telescope_diameter = 8.0
    central_obscuration = 1.2
    spider_width = 0.05
    oversizing_factor = 16 / 15
    wavelength_wfs = 0.7e-6
    f_number = 50
    sh_diameter = 5e-3

    num_pupil_pixels = int(240 * oversizing_factor)  # 256
    pupil_grid_diameter = telescope_diameter * oversizing_factor

    # -- Pupil ---------------------------------------------------------------
    pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
    aperture_gen = make_obstructed_circular_aperture(
        telescope_diameter,
        central_obscuration / telescope_diameter,
        num_spiders=4,
        spider_width=spider_width,
    )
    aperture = evaluate_supersampled(aperture_gen, pupil_grid, 4)

    # -- SHWFS optics --------------------------------------------------------
    magnification = sh_diameter / telescope_diameter
    magnifier = Magnifier(magnification)
    shwfs = SquareShackHartmannWavefrontSensorOptics(
        pupil_grid.scaled(magnification),
        f_number,
        num_lenslets,
        sh_diameter,
    )

    # -- Camera (derive detector grid from a reference propagation) ----------
    wf_ref = Wavefront(aperture, wavelength_wfs)
    wf_ref.total_power = 1
    _wf_out = shwfs(magnifier(wf_ref))
    camera = NoiselessDetector(_wf_out.electric_field.grid)

    # -- Zernike aberration --------------------------------------------------
    zernike_basis = make_zernike_basis(
        num_zernike + 1,
        telescope_diameter,
        pupil_grid,
        starting_mode=1,
    )
    zernike_modes = [zernike_basis[i] for i in range(1, num_zernike + 1)]

    if true_coeffs is None:
        true_coeffs = np.zeros(num_zernike)
        true_coeffs[0] =  0.10 * wavelength_wfs
        true_coeffs[1] =  0.07 * wavelength_wfs
        true_coeffs[2] =  0.08 * wavelength_wfs
        true_coeffs[3] =  0.05 * wavelength_wfs
        true_coeffs[4] = -0.04 * wavelength_wfs
    true_coeffs = np.asarray(true_coeffs, dtype=float)

    opd_field = aperture * sum(c * m for c, m in zip(true_coeffs, zernike_modes))
    phase_map = opd_field * (2.0 * np.pi / wavelength_wfs)
    aber_ap = Field(aperture * np.exp(1j * np.asarray(phase_map)), pupil_grid)

    wf_aber = Wavefront(aber_ap, wavelength_wfs)
    wf_aber.total_power = 1

    # -- Capture aberrated SHWFS image ---------------------------------------
    camera.integrate(shwfs(magnifier(wf_aber)), 1)
    image_aber = camera.read_out()

    print(f"Generated {num_lenslets}x{num_lenslets} SHWFS image, "
          f"{len(image_aber)} pixels")

    # -- Save aberrated VLT demo image ---------------------------------------
    if demo_image_path is not None:
        ap_2d = np.asarray(aperture, dtype=float).reshape(
            num_pupil_pixels, num_pupil_pixels)
        opd_2d = np.asarray(opd_field, dtype=float).reshape(
            num_pupil_pixels, num_pupil_pixels)

        scene = _load_vlt_scene(num_pupil_pixels)
        psf_ab = _psf_from_opd(opd_2d, ap_2d, wavelength_wfs)
        img_ab = _convolve_with_psf(scene, psf_ab)
        img_ab /= img_ab.max() + 1e-30

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(scene, cmap='gray', origin='lower')
        axes[0].set_title("Reference (VLT image)")
        axes[0].axis('off')
        axes[1].imshow(img_ab, cmap='gray', origin='lower')
        axes[1].set_title("Aberrated")
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(demo_image_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Demo image saved to {demo_image_path}")

    return np.asarray(image_aber)


if __name__ == "__main__":
    image_aber = generate_aberrated_image()
    print(f"image_aber shape: {image_aber.shape}, "
          f"range: [{image_aber.min():.4e}, {image_aber.max():.4e}]")

