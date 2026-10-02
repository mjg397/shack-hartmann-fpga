"""Noise couplings for paired numerical-model comparisons.

These helpers do not change the detector model. They construct a joint draw
for exactly two expected-electron images while preserving each image's
Poisson-plus-Gaussian marginal. Keeping the coupling pair-local prevents an
unrelated numerical setting from changing the paired draw.
"""

from __future__ import annotations

import numpy as np

from .detector import DetectorConfig


def _digitize(electrons: np.ndarray, cfg: DetectorConfig) -> tuple[np.ndarray, dict]:
    saturated = int(np.count_nonzero(electrons > cfg.full_well_e))
    bounded = np.clip(electrons, 0.0, cfg.full_well_e)
    unbounded = np.rint(bounded / cfg.gain_e_per_adu + cfg.offset_adu)
    clipped = int(np.count_nonzero((unbounded < 0) | (unbounded > cfg.adc_max)))
    frame = np.clip(unbounded, 0, cfg.adc_max).astype(np.int64)
    return frame, {"saturated_pixels": saturated, "adc_clipped_pixels": clipped}


def pairwise_poisson_gaussian_frames(
        expected_electrons: dict[str, np.ndarray], cfg: DetectorConfig,
        seed_components: list[int] | tuple[int, ...]) -> tuple[dict[str, np.ndarray], dict]:
    """Draw a pair with exact marginals and pair-local positive coupling.

    For each pixel, ``N_i = N_common + N_excess_i`` with independent Poisson
    means ``min(lambda_0, lambda_1)`` and ``lambda_i - min(...)``. Therefore
    each setting has exactly ``Poisson(lambda_i)`` marginal shot noise. The
    same Gaussian read-noise field is added to both settings, retaining the
    required marginal on each. Names are sorted before streams are assigned,
    so dictionary insertion order cannot change a draw.
    """
    if len(expected_electrons) != 2:
        raise ValueError("pairwise coupling requires exactly two named expectations")
    names = sorted(expected_electrons)
    lambdas = {name: np.asarray(expected_electrons[name], dtype=float) for name in names}
    if lambdas[names[0]].shape != lambdas[names[1]].shape:
        raise ValueError("paired expectations must have identical shapes")
    if any(np.any(~np.isfinite(value)) or np.any(value < 0)
           for value in lambdas.values()):
        raise ValueError("expected electrons must be finite and non-negative")

    minimum = np.minimum(lambdas[names[0]], lambdas[names[1]])
    children = np.random.SeedSequence(seed_components).spawn(2 + len(names))
    common_shot = np.random.default_rng(children[0]).poisson(minimum).astype(np.int64)
    if cfg.read_noise_e > 0:
        common_read = np.random.default_rng(children[1]).normal(
            0.0, cfg.read_noise_e, minimum.shape)
    else:
        common_read = np.zeros_like(minimum)

    frames = {}
    per_setting = {}
    for position, name in enumerate(names):
        excess_mean = np.clip(lambdas[name] - minimum, 0.0, None)
        excess = np.random.default_rng(children[2 + position]).poisson(
            excess_mean).astype(np.int64)
        frame, diagnostics = _digitize(common_shot + excess + common_read, cfg)
        frames[name] = frame
        per_setting[name] = {
            **diagnostics,
            "expected_electrons_sum": float(lambdas[name].sum()),
            "excess_poisson_mean_sum": float(excess_mean.sum()),
        }

    return frames, {
        "names_in_stable_stream_order": names,
        "common_poisson_mean_sum": float(minimum.sum()),
        "per_setting": per_setting,
        "seed_components": list(seed_components),
        "marginal_proof": (
            "independent Poisson(min(lambda)) + Poisson(lambda_i-min(lambda)) "
            "is Poisson(lambda_i); the shared read field is N(0, sigma^2) for each image"
        ),
        "identical_expectation_property": (
            "when both lambdas are equal, both excess means are zero and the digitized frames match"
        ),
        "pair_local_property": "only the two supplied expectations enter the joint draw",
    }

