"""
Cached HCIPy Shack-Hartmann optical bench.

WHY THIS EXISTS
---------------
Two problems with the previous evaluation, both real:

1. **Circularity.** Synthetic frames were built with `pinv(E)` and then
   reconstructed with `E`. That measures arithmetic faithfully but says nothing
   about optical reconstruction -- the forward and inverse models were the same
   matrix. Optical claims need inputs produced by propagating a known OPD
   through the actual optical model, which is what this module does.

2. **Cost.** `generate_shwfs_case` rebuilds the whole optical model per call
   (~40 s), making a 300-wavefront campaign infeasible. The expensive objects --
   pupil grid, supersampled aperture, lenslet array, SH optics, detector,
   estimator, Zernike basis -- depend only on the CONFIGURATION, not on the
   wavefront. They are built once here and reused, leaving only the per-
   wavefront propagation.

The optical parameters are taken verbatim from `src/shwfs_utils.py`
`generate_shwfs_case`, so this bench and the repository's own model describe the
same instrument. They are an ASSUMED SIMULATION CONFIGURATION, not measured
hardware parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from functools import lru_cache

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from . import config as C


# ---------------------------------------------------------------------------
# Configuration -- verbatim from src/shwfs_utils.py generate_shwfs_case
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpticalConfig:
    telescope_diameter_m: float = 8.0
    central_obscuration_m: float = 1.2
    spider_width_m: float = 0.05
    num_spiders: int = 4
    oversizing_factor: float = 16.0 / 15.0
    wavelength_m: float = 0.7e-6
    f_number: float = 50.0
    sh_diameter_base_m: float = 5e-3
    num_lenslets: int = 16
    num_zernike: int = 10
    pupil_pixels_base: int = 240
    supersample: int = 4

    @property
    def sh_diameter_m(self) -> float:
        return self.sh_diameter_base_m * self.oversizing_factor

    @property
    def lenslet_diameter_m(self) -> float:
        return self.sh_diameter_m / self.num_lenslets

    @property
    def focal_length_m(self) -> float:
        return self.f_number * self.lenslet_diameter_m

    @property
    def num_pupil_pixels(self) -> int:
        return int(self.pupil_pixels_base * self.oversizing_factor)

    @property
    def pupil_grid_diameter_m(self) -> float:
        return self.telescope_diameter_m * self.oversizing_factor

    @property
    def magnification(self) -> float:
        return self.sh_diameter_base_m / self.telescope_diameter_m

    @property
    def detector_pixel_pitch_m(self) -> float:
        return self.sh_diameter_m / self.num_pupil_pixels

    def hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


class Bench:
    """Built once per configuration; propagates many wavefronts."""

    def __init__(self, cfg: OpticalConfig | None = None):
        from hcipy import (
            make_pupil_grid, make_obstructed_circular_aperture,
            evaluate_supersampled, Wavefront, CartesianGrid, SeparatedCoords,
            MicroLensArray, ShackHartmannWavefrontSensorOptics,
            ShackHartmannWavefrontSensorEstimator, NoiselessDetector,
            Magnifier, make_zernike_basis,
        )
        self.cfg = cfg or OpticalConfig()
        c = self.cfg

        self.pupil_grid = make_pupil_grid(c.num_pupil_pixels, c.pupil_grid_diameter_m)
        aperture_gen = make_obstructed_circular_aperture(
            c.telescope_diameter_m,
            c.central_obscuration_m / c.telescope_diameter_m,
            num_spiders=c.num_spiders, spider_width=c.spider_width_m)
        self.aperture = evaluate_supersampled(aperture_gen, self.pupil_grid,
                                              c.supersample)

        self.magnifier = Magnifier(c.magnification)

        # Centred lenslet grid -- HCIPy's square helper is asymmetric.
        half = c.sh_diameter_m / 2 - c.lenslet_diameter_m / 2
        centres = np.linspace(-half, half, c.num_lenslets)
        mla_grid = CartesianGrid(SeparatedCoords((centres, centres)))
        scaled = self.pupil_grid.scaled(c.magnification)
        self.mla = MicroLensArray(scaled, mla_grid, c.focal_length_m)
        self.shwfs = ShackHartmannWavefrontSensorOptics(scaled, self.mla)
        self.estimator = ShackHartmannWavefrontSensorEstimator(
            mla_grid, self.mla.mla_index)

        # Flat-wavefront (illuminated) reference -- NOT a dark frame.
        wf = Wavefront(self.aperture, c.wavelength_m)
        wf.total_power = 1
        out = self.shwfs(self.magnifier(wf))
        self.camera = NoiselessDetector(out.electric_field.grid)
        self.camera.integrate(out, 1)
        self.reference_image = self.camera.read_out()

        # Zernike basis, piston excluded (index 0 is piston).
        self.zernike_basis = make_zernike_basis(
            c.num_zernike + 1, c.telescope_diameter_m, self.pupil_grid)
        self.modes = [np.asarray(self.zernike_basis[i + 1])
                      for i in range(c.num_zernike)]
        self._Wavefront = Wavefront

    # -- pupil weighting ----------------------------------------------------
    @property
    def pupil_mask(self) -> np.ndarray:
        """Boolean mask of illuminated pupil pixels."""
        a = np.asarray(self.aperture)
        return a > 0.5

    def opd_rms(self, opd: np.ndarray, remove_piston: bool = True) -> float:
        """
        RMS of an OPD over the ACTUAL pupil, aperture-weighted.

        Not the Euclidean norm of the coefficient vector: on an obscured,
        spidered pupil the Zernike modes are not orthonormal, so those differ.
        """
        w = np.asarray(self.aperture, dtype=float)
        o = np.asarray(opd, dtype=float)
        tw = w.sum()
        if tw <= 0:
            return float("nan")
        if remove_piston:
            o = o - (w * o).sum() / tw
        return float(np.sqrt((w * o ** 2).sum() / tw))

    def opd_inner(self, a: np.ndarray, b: np.ndarray,
                  remove_piston: bool = True) -> float:
        """Aperture-weighted pupil inner product, with the RMS piston convention."""
        w = np.asarray(self.aperture, dtype=float)
        x = np.asarray(a, dtype=float)
        y = np.asarray(b, dtype=float)
        tw = w.sum()
        if tw <= 0:
            return float("nan")
        if remove_piston:
            x = x - (w * x).sum() / tw
            y = y - (w * y).sum() / tw
        return float((w * x * y).sum() / tw)

    def opd_from_coeffs(self, coeffs) -> np.ndarray:
        """Sum the (piston-excluded) Zernike modes weighted by coefficients."""
        c = np.asarray(coeffs, dtype=float)
        out = np.zeros_like(self.modes[0], dtype=float)
        for a, m in zip(c, self.modes):
            out = out + a * m
        return out

    # -- propagation --------------------------------------------------------
    def propagate(self, opd: np.ndarray) -> np.ndarray:
        """
        Propagate an OPD through the SH optics and read out the detector.

        Returns the raw (unquantised, unnormalised) intensity image. Digitisation
        is applied separately so the detector model is explicit and auditable.
        """
        c = self.cfg
        phase = 2 * np.pi * np.asarray(opd, dtype=float) / c.wavelength_m
        from hcipy import Field
        ef = self.aperture * np.exp(1j * Field(phase, self.pupil_grid))
        wf = self._Wavefront(Field(ef, self.pupil_grid), c.wavelength_m)
        wf.total_power = 1
        out = self.shwfs(self.magnifier(wf))
        self.camera.integrate(out, 1)
        return np.asarray(self.camera.read_out())

    def propagate_coeffs(self, coeffs) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: coefficients -> (intensity image, true OPD)."""
        opd = self.opd_from_coeffs(coeffs)
        return self.propagate(opd), opd


@lru_cache(maxsize=4)
def get_bench(cfg_hash: str | None = None) -> Bench:
    """Cached bench for the default configuration."""
    return Bench(OpticalConfig())


def config_record() -> dict:
    """Machine-readable record of the assumed simulation configuration."""
    c = OpticalConfig()
    d = asdict(c)
    d.update({
        "derived_sh_diameter_m": c.sh_diameter_m,
        "derived_lenslet_diameter_m": c.lenslet_diameter_m,
        "derived_focal_length_m": c.focal_length_m,
        "derived_num_pupil_pixels": c.num_pupil_pixels,
        "derived_detector_pixel_pitch_m": c.detector_pixel_pitch_m,
        "derived_magnification": c.magnification,
        "config_hash": c.hash(),
        "provenance": "verbatim from src/shwfs_utils.py generate_shwfs_case",
        "status": "ASSUMED SIMULATION CONFIGURATION -- not measured hardware",
    })
    return d
