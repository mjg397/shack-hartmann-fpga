"""
Explicit detector model: optical intensity -> 8-bit ADU.

Every stage is stated rather than implied, because the previous evaluation
normalised each frame independently to 0-255, which silently removed the very
signal-dependence a noise sweep is meant to measure.

Chain
    normalised intensity  (HCIPy, total power 1)
      x total_photons                  -> photons per pixel
      x quantum_efficiency             -> photoelectrons
      + background_e                   -> + background
      Poisson                          -> shot noise
      + N(0, read_noise_e)             -> read noise
      clip at full_well_e              -> saturation
      / gain_e_per_adu  + offset_adu   -> ADU
      round, clip 0..255               -> 8-bit, as the DUT consumes

IMPORTANT: gain is FIXED within a sweep. Changing gain per level would rescale
the signal and hide the effect being measured. If one gain cannot span a desired
photon range without clipping, use separately labelled gain configurations.

These are ASSUMED detector parameters for a simulation study. The repository
defines no real detector, so none of this is a measured characterisation of
any device, and results must not be described as such.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    total_photons: float = 3.0e6       # over the whole illuminated pupil
    quantum_efficiency: float = 0.8
    gain_e_per_adu: float = 20.0
    read_noise_e: float = 3.0
    background_e: float = 0.0
    offset_adu: float = 0.0
    full_well_e: float = 20000.0
    adc_bits: int = 8
    shot_noise: bool = True

    @property
    def adc_max(self) -> int:
        return (1 << self.adc_bits) - 1


@dataclass
class DetectorResult:
    adu: np.ndarray
    electrons: np.ndarray
    saturated_px: int
    clipped_adu_px: int
    total_electrons: float


def expose(intensity: np.ndarray, cfg: DetectorConfig,
           rng: np.random.Generator | None = None) -> DetectorResult:
    """Run one exposure. `intensity` is the HCIPy image (total power 1)."""
    inten = np.asarray(intensity, dtype=np.float64)
    tot = inten.sum()
    norm = inten / tot if tot > 0 else inten

    photons = norm * cfg.total_photons
    e = photons * cfg.quantum_efficiency + cfg.background_e

    if cfg.shot_noise:
        if rng is None:
            rng = np.random.default_rng()
        e = rng.poisson(np.clip(e, 0, None)).astype(np.float64)
    if cfg.read_noise_e > 0:
        if rng is None:
            rng = np.random.default_rng()
        e = e + rng.normal(0.0, cfg.read_noise_e, e.shape)

    saturated = int((e > cfg.full_well_e).sum())
    e = np.clip(e, 0.0, cfg.full_well_e)

    adu_f = e / cfg.gain_e_per_adu + cfg.offset_adu
    adu = np.rint(adu_f)
    clipped = int((adu > cfg.adc_max).sum() + (adu < 0).sum())
    adu = np.clip(adu, 0, cfg.adc_max).astype(np.int64)

    return DetectorResult(adu=adu, electrons=e, saturated_px=saturated,
                          clipped_adu_px=clipped, total_electrons=float(e.sum()))


def signal_per_lenslet_e(cfg: DetectorConfig, n_valid: int = 168) -> float:
    """Mean detected electrons per illuminated lenslet -- the sweep's x-axis."""
    return cfg.total_photons * cfg.quantum_efficiency / max(n_valid, 1)


def config_record(cfg: DetectorConfig) -> dict:
    d = asdict(cfg)
    d.update({
        "signal_e_per_lenslet": signal_per_lenslet_e(cfg),
        "adc_max": cfg.adc_max,
        "status": "ASSUMED detector parameters for simulation -- not a measured device",
        "note": "gain is held FIXED within any sweep; frames are NOT individually normalised",
    })
    return d
