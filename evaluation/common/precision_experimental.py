"""Bit-exact arithmetic experiments that coexist with the frozen baseline.

Nothing in this module changes :mod:`evaluation.common.bitaccurate`.  It models
alternative reciprocal/centroid paths while retaining the candidate reference
ROM, E-matrix integers, slope format, MAC widths, and output slice.
"""

from __future__ import annotations

import numpy as np

from . import bitaccurate as BA
from . import config as C


def divide_unsigned(numerator: int, denominator: int, rounding: str) -> int:
    """Unsigned integer division with an explicit synthesizable rounding rule."""
    if denominator <= 0:
        return 0
    if rounding == "truncate":
        return numerator // denominator
    if rounding == "nearest_half_up":
        return (numerator + denominator // 2) // denominator
    raise ValueError(f"unsupported division rounding rule: {rounding}")


def shift_unsigned(value: int, right_shift: int, rounding: str) -> int:
    """Apply a signed binary-point displacement to a nonnegative integer."""
    if right_shift <= 0:
        return value << (-right_shift)
    if rounding == "truncate":
        return value >> right_shift
    if rounding == "nearest_half_up":
        return (value + (1 << (right_shift - 1))) >> right_shift
    raise ValueError(f"unsupported shift rounding rule: {rounding}")


def exact_centroid_q18(moment: np.ndarray, intensity: np.ndarray,
                       rounding: str = "nearest_half_up") -> np.ndarray:
    """Exact integer quotient ``round(moment / intensity * 2^18)``."""
    return np.asarray([
        divide_unsigned(int(m) << C.CENTROID_FRAC, int(d), rounding)
        for m, d in zip(np.asarray(moment), np.asarray(intensity))
    ], dtype=np.int64)


def direct_reciprocal_centroid(moment: np.ndarray, intensity: np.ndarray,
                               frac_bits: int, reciprocal_rounding: str,
                               product_rounding: str) -> dict:
    """Direct ``2^F/d`` reciprocal followed by a Q18 centroid conversion."""
    reciprocals = []
    centroids = []
    for m, d in zip(np.asarray(moment), np.asarray(intensity)):
        d_i = int(d)
        if d_i == 0:
            reciprocals.append(0)
            centroids.append(0)
            continue
        r = divide_unsigned(1 << frac_bits, d_i, reciprocal_rounding)
        reciprocals.append(r)
        centroids.append(shift_unsigned(
            int(m) * r, frac_bits - C.CENTROID_FRAC, product_rounding))
    return {
        "reciprocal": np.asarray(reciprocals, dtype=np.int64),
        "centroid": np.asarray(centroids, dtype=np.int64),
        "exponent": np.zeros(len(centroids), dtype=np.int64),
    }


def normalized_reciprocal_centroid(moment: np.ndarray, intensity: np.ndarray,
                                   mantissa_frac_bits: int = 17,
                                   reciprocal_rounding: str = "nearest_half_up",
                                   product_rounding: str = "nearest_half_up") -> dict:
    """Normalized reciprocal mantissa plus exponent using one 19x18 multiply.

    For ``d = a * 2^k`` with ``1 <= a < 2``, the stored unsigned mantissa is
    ``round((1/a) * 2^F)``.  For F=17 it occupies 18 bits.  The full product is
    shifted only after multiplication to produce a Q18 centroid.
    """
    reciprocals = []
    exponents = []
    centroids = []
    for m, d in zip(np.asarray(moment), np.asarray(intensity)):
        d_i = int(d)
        if d_i == 0:
            reciprocals.append(0)
            exponents.append(0)
            centroids.append(0)
            continue
        exponent = d_i.bit_length() - 1
        numerator = 1 << (mantissa_frac_bits + exponent)
        mantissa = divide_unsigned(numerator, d_i, reciprocal_rounding)
        right_shift = mantissa_frac_bits + exponent - C.CENTROID_FRAC
        centroid = shift_unsigned(int(m) * mantissa, right_shift,
                                  product_rounding)
        reciprocals.append(mantissa)
        exponents.append(exponent)
        centroids.append(centroid)
    return {
        "reciprocal": np.asarray(reciprocals, dtype=np.int64),
        "centroid": np.asarray(centroids, dtype=np.int64),
        "exponent": np.asarray(exponents, dtype=np.int64),
    }


def newton_reciprocal_centroid(moment: np.ndarray, intensity: np.ndarray,
                               baseline_q18: np.ndarray, frac_bits: int,
                               iteration_rounding: str,
                               product_rounding: str) -> dict:
    """One hardware-shaped Newton step lifted from the existing Q18 estimate.

    ``x0`` is the existing Q18 LUT word shifted exactly into QF.  Both Newton
    products are evaluated as integers.  The retained x1 slice and the later
    centroid slice use the requested explicit rounding rule.
    """
    if frac_bits < C.CENTROID_FRAC:
        raise ValueError("Newton working precision cannot be below Q18")
    refined = []
    centroids = []
    for m, d, seed in zip(np.asarray(moment), np.asarray(intensity),
                          np.asarray(baseline_q18)):
        d_i = int(d)
        if d_i == 0:
            refined.append(0)
            centroids.append(0)
            continue
        x0 = int(seed) << (frac_bits - C.CENTROID_FRAC)
        dx0_qf = d_i * x0
        correction_qf = (2 << frac_bits) - dx0_qf
        x1_q2f = x0 * correction_qf
        x1 = shift_unsigned(x1_q2f, frac_bits, iteration_rounding)
        centroid = shift_unsigned(int(m) * x1,
                                  frac_bits - C.CENTROID_FRAC,
                                  product_rounding)
        refined.append(x1)
        centroids.append(centroid)
    return {
        "reciprocal": np.asarray(refined, dtype=np.int64),
        "centroid": np.asarray(centroids, dtype=np.int64),
        "exponent": np.zeros(len(centroids), dtype=np.int64),
    }


def reconstruct_integer(x_slope: np.ndarray, y_slope: np.ndarray,
                        ex: np.ndarray, ey: np.ndarray,
                        *, retain_output_fraction: bool = False) -> tuple[np.ndarray, dict]:
    """Candidate-ROM MAC with the frozen baseline widths and wrap semantics."""
    xs = np.asarray(x_slope, dtype=object)
    ys = np.asarray(y_slope, dtype=object)
    output = np.zeros(C.NUM_MODES, dtype=np.int64)
    exact_output = np.zeros(C.NUM_MODES, dtype=float)
    accumulator_wraps = 0
    output_wraps = 0
    max_abs_accumulator = 0
    for mode in range(C.NUM_MODES):
        acc = 0
        for subap in range(C.NUM_SUBAPS_VALID):
            product_x = BA.to_signed(BA.truncate(
                int(ex[mode, subap]) * int(xs[subap]), C.PRODUCT_BITS),
                C.PRODUCT_BITS)
            product_y = BA.to_signed(BA.truncate(
                int(ey[mode, subap]) * int(ys[subap]), C.PRODUCT_BITS),
                C.PRODUCT_BITS)
            unwrapped = acc + int(product_x) + int(product_y)
            if not -(1 << (C.ACC_BITS - 1)) <= unwrapped < (1 << (C.ACC_BITS - 1)):
                accumulator_wraps += 1
            acc = int(BA.to_signed(BA.truncate(unwrapped, C.ACC_BITS), C.ACC_BITS))
            max_abs_accumulator = max(max_abs_accumulator, abs(acc))
        unsliced = acc >> C.ZERNIKE_SLICE_LOW
        if not -(1 << (C.ZERNIKE_BITS - 1)) <= unsliced < (1 << (C.ZERNIKE_BITS - 1)):
            output_wraps += 1
        output[mode] = BA.to_signed(BA.truncate(unsliced, C.ZERNIKE_BITS),
                                    C.ZERNIKE_BITS)
        exact_output[mode] = acc / float(1 << C.ZERNIKE_SLICE_LOW)
    diagnostics = {
        "accumulator_wrap_count": accumulator_wraps,
        "output_field_wrap_count": output_wraps,
        "max_abs_accumulator": max_abs_accumulator,
    }
    return (exact_output if retain_output_fraction else output), diagnostics


def pipeline_from_centroids(x_centroid_q18: np.ndarray,
                            y_centroid_q18: np.ndarray,
                            roms: dict, *, retain_output_fraction: bool = False) -> dict:
    """Run the unchanged slope and E-matrix stages from alternative centroids."""
    x_slope = BA.slope(x_centroid_q18, roms["ref_x"])
    y_slope = BA.slope(y_centroid_q18, roms["ref_y"])
    valid = roms["valid"]
    zernike, diagnostics = reconstruct_integer(
        x_slope[valid], y_slope[valid], roms["ex"], roms["ey"],
        retain_output_fraction=retain_output_fraction)
    return {
        "x_centroid": np.asarray(x_centroid_q18, dtype=np.int64),
        "y_centroid": np.asarray(y_centroid_q18, dtype=np.int64),
        "x_slope": x_slope,
        "y_slope": y_slope,
        "x_slope_valid": x_slope[valid],
        "y_slope_valid": y_slope[valid],
        "zernike_raw": zernike,
        "diagnostics": diagnostics,
    }


def evaluate_variant(frame: np.ndarray, roms: dict, variant: dict) -> dict:
    """Run one named reciprocal/centroid variant on an integer detector frame."""
    s_i, x_i, y_i = BA.accumulate(frame)
    kind = variant["kind"]
    if kind == "baseline":
        r_i = np.where(s_i == 0, 0x7FFFF, roms["reciprocal"][s_i])
        rec_port = BA.truncate(r_i, C.RECIP_PORT_BITS)
        x_trace = {"reciprocal": rec_port, "centroid": BA.centroid(x_i, rec_port),
                   "exponent": np.zeros(len(s_i), dtype=np.int64)}
        y_trace = {"reciprocal": rec_port, "centroid": BA.centroid(y_i, rec_port),
                   "exponent": np.zeros(len(s_i), dtype=np.int64)}
    elif kind == "direct":
        kwargs = {
            "frac_bits": variant["frac_bits"],
            "reciprocal_rounding": variant["reciprocal_rounding"],
            "product_rounding": variant["product_rounding"],
        }
        x_trace = direct_reciprocal_centroid(x_i, s_i, **kwargs)
        y_trace = direct_reciprocal_centroid(y_i, s_i, **kwargs)
    elif kind == "normalized":
        kwargs = {
            "mantissa_frac_bits": variant["frac_bits"],
            "reciprocal_rounding": variant["reciprocal_rounding"],
            "product_rounding": variant["product_rounding"],
        }
        x_trace = normalized_reciprocal_centroid(x_i, s_i, **kwargs)
        y_trace = normalized_reciprocal_centroid(y_i, s_i, **kwargs)
    elif kind == "newton":
        seed = np.where(s_i == 0, 0, roms["reciprocal"][s_i])
        kwargs = {
            "baseline_q18": seed,
            "frac_bits": variant["frac_bits"],
            "iteration_rounding": variant["iteration_rounding"],
            "product_rounding": variant["product_rounding"],
        }
        x_trace = newton_reciprocal_centroid(x_i, s_i, **kwargs)
        y_trace = newton_reciprocal_centroid(y_i, s_i, **kwargs)
    elif kind == "exact":
        rounding = variant.get("product_rounding", "nearest_half_up")
        x_trace = {"reciprocal": np.zeros_like(s_i),
                   "centroid": exact_centroid_q18(x_i, s_i, rounding),
                   "exponent": np.zeros_like(s_i)}
        y_trace = {"reciprocal": np.zeros_like(s_i),
                   "centroid": exact_centroid_q18(y_i, s_i, rounding),
                   "exponent": np.zeros_like(s_i)}
    else:
        raise ValueError(f"unknown experimental arithmetic kind: {kind}")
    pipeline = pipeline_from_centroids(
        x_trace["centroid"], y_trace["centroid"], roms,
        retain_output_fraction=variant.get("retain_output_fraction", False))
    pipeline.update({
        "sI": s_i, "xI": x_i, "yI": y_i,
        "x_reciprocal": x_trace["reciprocal"],
        "reciprocal_exponent": x_trace["exponent"],
        "invalid_centroid_count": int(np.count_nonzero(s_i[roms["valid"]] == 0)),
    })
    return pipeline


BASELINE = {
    "name": "baseline_q18", "kind": "baseline",
    "reciprocal_format": "unsigned Q18, 19-bit ROM narrowed to 18-bit multiplier port",
    "product_format": "19x18 -> 37-bit exact product; low 24 bits retained",
    "rounding": "reciprocal nearest-half-up; no post-product rounding",
}

NORMALIZED_Q17_RNE = {
    "name": "normalized_q17_rne", "kind": "normalized", "frac_bits": 17,
    "reciprocal_rounding": "nearest_half_up",
    "product_rounding": "nearest_half_up",
    "reciprocal_format": "unsigned normalized Q17 mantissa (18 bits) plus 4-bit exponent",
    "product_format": "19x18 -> 37-bit exact product; exponent shift after multiply",
    "rounding": "mantissa and shifted product nearest-half-up",
}

