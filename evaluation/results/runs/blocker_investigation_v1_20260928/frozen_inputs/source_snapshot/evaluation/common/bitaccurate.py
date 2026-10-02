"""
Bit-accurate software model of the Gen B Zynq SHWFS pipeline.

Every operation below mirrors the RTL in sim/rtl/ exactly,
including the truncations and the one known width mismatch. It is intended to
reproduce RTL outputs with ZERO error; the RTL-vs-model regression (section 2A)
asserts exact equality, so any divergence is a real discrepancy rather than
expected quantization.

Traced signal path:

  Thresholding_CoG_intensity_accumulator
      thresholded = (pixel < 5) ? 0 : pixel
      sI, xI, yI accumulated over each 16x16 tile in raster order

  intensity_reciprocal_q18 / reciprocal_u16_q18
      rI = (sI == 0) ? 0x7FFFF : LUT[sI]          19 bits, Q18

  slope_calculation.unsigned_mult
      rec_port = rI[17:0]                          <-- bit 18 pruned by the port
      centroid = (xI[18:0] * rec_port)[23:0]       24 bits, Q18

  slope_calculation stage 2
      raw   = sext26({1'b0, centroid}) - sext26(ref)
      slope = raw[24:0]                            25 bits signed

  ematrix_accumulator
      prod  = sext43(E[m,s]) * slope               43 bits signed
      acc  += sext52(prod_x) + sext52(prod_y)      52 bits signed
      out   = acc[42:18]                           25 bits signed
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config as C
from .roms import REQUIRED_ROMS, read_hex_words


# ---------------------------------------------------------------------------
# Two's-complement helpers
# ---------------------------------------------------------------------------
def truncate(value: int | np.ndarray, bits: int):
    """Keep the low `bits` bits, as Verilog part-select [bits-1:0] does."""
    return value & ((1 << bits) - 1)


def to_signed(value: int | np.ndarray, bits: int):
    """Reinterpret an unsigned bit pattern as two's-complement signed."""
    mask = (1 << bits) - 1
    value = value & mask
    sign = 1 << (bits - 1)
    return (value ^ sign) - sign


# ---------------------------------------------------------------------------
# ROM loading
# ---------------------------------------------------------------------------
def _read_rom(name: str) -> list[int]:
    return read_hex_words(C.RTL_DIR / name, REQUIRED_ROMS[name])


@lru_cache(maxsize=1)
def reciprocal_lut() -> np.ndarray:
    """65536-entry Q18 reciprocal LUT from reciprocal_u16_q18.mem."""
    return np.asarray(_read_rom("reciprocal_u16_q18.mem"), dtype=np.int64)


@lru_cache(maxsize=1)
def slopes_ref() -> tuple[np.ndarray, np.ndarray]:
    """Per-subaperture reference slopes, signed 25-bit, Q18."""
    x = np.array(_read_rom("slopes_ref_x.mem"), dtype=np.int64)
    y = np.array(_read_rom("slopes_ref_y.mem"), dtype=np.int64)
    return to_signed(x, C.SLOPE_BITS), to_signed(y, C.SLOPE_BITS)


@lru_cache(maxsize=1)
def ematrix() -> tuple[np.ndarray, np.ndarray]:
    """
    E matrix as Q1.16 signed 18-bit, shaped (NUM_MODES, NUM_SUBAPS_VALID).

    ematrix_accumulator.v:91  e_rom_x[m * NUM_SUBS + s] = E[m, 2*s]
    """
    ex = np.array(_read_rom("e_matrix_x.mem"), dtype=np.int64)
    ey = np.array(_read_rom("e_matrix_y.mem"), dtype=np.int64)
    n = C.NUM_MODES * C.NUM_SUBAPS_VALID
    ex = ex.reshape(C.NUM_MODES, C.NUM_SUBAPS_VALID)
    ey = ey.reshape(C.NUM_MODES, C.NUM_SUBAPS_VALID)
    return to_signed(ex, C.EMATRIX_BITS), to_signed(ey, C.EMATRIX_BITS)


@lru_cache(maxsize=1)
def valid_subaps() -> np.ndarray:
    """
    Indices of in-pupil tiles, raster order.

    slope_calculation.v:237 indexes subap_bitmap[completed_subap_addr], a plain
    Verilog bit-select, so bit N (LSB-indexed) gates raster tile N.
    """
    bm = _read_rom("subaperture_bitmap.mem")[0]
    indices = np.array(
        [i for i in range(C.NUM_SUBAPS_GRID) if (bm >> i) & 1], dtype=np.int64
    )
    if indices.size != C.NUM_SUBAPS_VALID:
        raise ValueError(
            f"subaperture bitmap contains {indices.size} valid tiles; "
            f"RTL expects {C.NUM_SUBAPS_VALID}")
    return indices


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def accumulate(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thresholded centre-of-gravity accumulation over every 16x16 tile.

    Returns (sI, xI, yI), each length NUM_SUBAPS_GRID in raster order.
    """
    raw = np.asarray(image)
    if raw.size != C.FRAME_H * C.FRAME_W:
        raise ValueError(f"expected {C.FRAME_H*C.FRAME_W} pixels, got {raw.size}")
    if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
        raise ValueError("pixels must be finite numeric values")
    if np.any(raw != np.rint(raw)) or np.any(raw < 0) or np.any(raw > 255):
        raise ValueError("pixels must be 8-bit integers in [0, 255]")
    img = raw.astype(np.int64).reshape(C.FRAME_H, C.FRAME_W)
    tiles = img.reshape(
        C.NUM_LENSLETS, C.SUBAP_PIXELS, C.NUM_LENSLETS, C.SUBAP_PIXELS
    ).transpose(0, 2, 1, 3).reshape(-1, C.SUBAP_PIXELS, C.SUBAP_PIXELS)

    tiles = np.where(tiles < C.PIXEL_THRESHOLD, 0, tiles)

    xi = np.arange(C.SUBAP_PIXELS, dtype=np.int64)[None, None, :]
    yi = np.arange(C.SUBAP_PIXELS, dtype=np.int64)[None, :, None]

    sI = tiles.sum(axis=(1, 2))
    xI = (tiles * xi).sum(axis=(1, 2))
    yI = (tiles * yi).sum(axis=(1, 2))
    return sI, xI, yI


def reciprocal(sI: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Q18 reciprocal.

    Returns (rI_19bit, rec_port_18bit). The second value is what
    slope_calculation actually receives -- bit 18 is pruned by the 18-bit port,
    a known width mismatch reachable only when sI <= 1.
    """
    sI = np.asarray(sI, dtype=np.int64)
    if np.any(sI < 0) or np.any(sI > 0xFFFF):
        raise OverflowError("intensity sum is outside the RTL's 16-bit address range")
    rI = np.where(sI == 0, 0x7FFFF, reciprocal_lut()[sI])
    return rI, truncate(rI, C.RECIP_PORT_BITS)


def centroid(xI: np.ndarray, rec_port: np.ndarray) -> np.ndarray:
    """unsigned_mult: (a[18:0] * b[17:0])[23:0], giving Q18 in 24 bits."""
    a = truncate(np.asarray(xI, dtype=np.int64), C.XI_BITS)
    product = a * np.asarray(rec_port, dtype=np.int64)
    return truncate(product, C.CENTROID_BITS)


def slope(centroid_q18: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    stage 2: sext26({1'b0, centroid}) - sext26(ref), then truncate to 25 bits.
    """
    cen_s = np.asarray(centroid_q18, dtype=np.int64)     # already non-negative
    raw = cen_s - np.asarray(reference, dtype=np.int64)
    return to_signed(truncate(raw, C.SLOPE_BITS), C.SLOPE_BITS)


def reconstruct(x_slope: np.ndarray, y_slope: np.ndarray,
                return_diagnostics: bool = False):
    """
    E-matrix MAC over NUM_SUBAPS_VALID subapertures.

    Products are 43-bit signed, the accumulator 52-bit signed, and the output
    is acc[42:18] read back as 25-bit signed.
    """
    ex, ey = ematrix()
    xs = np.asarray(x_slope, dtype=object)   # Python ints: exact, no overflow
    ys = np.asarray(y_slope, dtype=object)

    out = np.zeros(C.NUM_MODES, dtype=np.int64)
    acc_wrap_count = 0
    output_wrap_count = 0
    max_abs_acc = 0
    for m in range(C.NUM_MODES):
        acc = 0
        for s in range(C.NUM_SUBAPS_VALID):
            prod_x = to_signed(truncate(int(ex[m, s]) * int(xs[s]), C.PRODUCT_BITS),
                               C.PRODUCT_BITS)
            prod_y = to_signed(truncate(int(ey[m, s]) * int(ys[s]), C.PRODUCT_BITS),
                               C.PRODUCT_BITS)
            unwrapped = acc + prod_x + prod_y
            if not (-(1 << (C.ACC_BITS - 1)) <= unwrapped < (1 << (C.ACC_BITS - 1))):
                acc_wrap_count += 1
            acc = to_signed(truncate(unwrapped, C.ACC_BITS), C.ACC_BITS)
            max_abs_acc = max(max_abs_acc, abs(int(acc)))
        unsliced = acc >> C.ZERNIKE_SLICE_LOW
        if not (-(1 << (C.ZERNIKE_BITS - 1)) <= unsliced
                < (1 << (C.ZERNIKE_BITS - 1))):
            output_wrap_count += 1
        field = truncate(acc >> C.ZERNIKE_SLICE_LOW, C.ZERNIKE_BITS)
        out[m] = to_signed(field, C.ZERNIKE_BITS)
    if return_diagnostics:
        return out, {
            "accumulator_wrap_count": acc_wrap_count,
            "output_field_wrap_count": output_wrap_count,
            "max_abs_accumulator": max_abs_acc,
        }
    return out


def run_pipeline(image: np.ndarray) -> dict:
    """
    Full bit-accurate pipeline for one frame.

    Returns raw fixed-point values at every stage, plus the decoded
    coefficients in metres.
    """
    sI, xI, yI = accumulate(image)
    rI, rec_port = reciprocal(sI)

    cx = centroid(xI, rec_port)
    cy = centroid(yI, rec_port)

    ref_x, ref_y = slopes_ref()
    sx_all = slope(cx, ref_x[: C.NUM_SUBAPS_GRID])
    sy_all = slope(cy, ref_y[: C.NUM_SUBAPS_GRID])

    idx = valid_subaps()
    zern, recon_diag = reconstruct(sx_all[idx], sy_all[idx], return_diagnostics=True)

    return {
        "sI": sI, "xI": xI, "yI": yI,
        "rI": rI, "rec_port": rec_port,
        "x_centroid": cx, "y_centroid": cy,
        "x_slope": sx_all, "y_slope": sy_all,
        "valid_idx": idx,
        "x_slope_valid": sx_all[idx], "y_slope_valid": sy_all[idx],
        "zernike_raw": zern,
        "zernike_m": zern.astype(np.float64) * C.METERS_PER_LSB,
        "diagnostics": {
            "invalid_centroid_count": int(np.count_nonzero(sI[idx] == 0)),
            "intensity_overflow_count": int(np.count_nonzero(sI > C.SI_MAX)),
            "x_moment_overflow_count": int(np.count_nonzero(xI > C.XI_MAX)),
            "y_moment_overflow_count": int(np.count_nonzero(yI > C.XI_MAX)),
            **recon_diag,
        },
    }


# ---------------------------------------------------------------------------
# Floating-point reference (same algorithm, no quantisation)
# ---------------------------------------------------------------------------
def run_float_reference(image: np.ndarray) -> dict:
    """
    Identical algorithm in float64: thresholded CoG, exact reciprocal, exact
    reference subtraction, exact E-matrix MAC.

    This isolates QUANTISATION error. It is not the optical golden model --
    that is HCIPy's run_hcipy_estimation, used separately.
    """
    sI, xI, yI = accumulate(image)
    sI_f = sI.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        cx = np.where(sI_f > 0, xI / sI_f, 0.0)      # pixels
        cy = np.where(sI_f > 0, yI / sI_f, 0.0)

    ref_x, ref_y = slopes_ref()
    scale = float(1 << C.CENTROID_FRAC)
    sx = cx - ref_x[: C.NUM_SUBAPS_GRID] / scale
    sy = cy - ref_y[: C.NUM_SUBAPS_GRID] / scale

    idx = valid_subaps()
    ex, ey = ematrix()
    e_scale = float(1 << C.EMATRIX_FRAC)
    zern = (ex / e_scale) @ sx[idx] + (ey / e_scale) @ sy[idx]

    return {
        "x_centroid_px": cx, "y_centroid_px": cy,
        "x_slope_px": sx, "y_slope_px": sy,
        "valid_idx": idx,
        "zernike": zern,
        # Same decode path as the RTL: Q18 slopes * Q16 E -> acc, then 2^-43.
        "zernike_m": zern * (1 << C.CENTROID_FRAC) * (1 << C.EMATRIX_FRAC)
                     * (2.0 ** -C.ZERNIKE_SLICE_LOW) * C.METERS_PER_LSB,
    }
