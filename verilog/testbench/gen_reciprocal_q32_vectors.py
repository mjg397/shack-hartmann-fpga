#!/usr/bin/env python3
"""
Reference model for reciprocal_u16_q32: simulates the exact fixed-point
hardware arithmetic and generates golden test vectors.

Output: vectors/reciprocal_golden_q32.memh
  One 48-bit word per input v in [0, 65535], hex-formatted as 12 digits:
    bits[47:32] = v_u16   (input)
    bits[31: 0] = expected reciprocal_q32 (output)

Usage: python3 gen_reciprocal_q32_vectors.py
"""

import os

# ---------------------------------------------------------------------------
# 8-bit seed LUT (Q0.16), matches RECIP_SEED_LUT_Q0_16 in the Verilog.
# Index 0 corresponds to a_q1_15 in [0x8000, 0x8080), i.e. values near 1.0.
# ---------------------------------------------------------------------------
SEED_LUT_Q0_16 = [
    0xFF80, 0xFE82, 0xFD86, 0xFC8C, 0xFB94, 0xFA9E, 0xF9A9, 0xF8B7,
    0xF7C6, 0xF6D7, 0xF5EA, 0xF4FF, 0xF415, 0xF32D, 0xF247, 0xF163,
    0xF080, 0xEF9F, 0xEEBF, 0xEDE1, 0xED05, 0xEC2A, 0xEB51, 0xEA7A,
    0xE9A4, 0xE8CF, 0xE7FC, 0xE72B, 0xE65B, 0xE58C, 0xE4BF, 0xE3F4,
    0xE329, 0xE260, 0xE199, 0xE0D3, 0xE00E, 0xDF4B, 0xDE88, 0xDDC8,
    0xDD08, 0xDC4A, 0xDB8D, 0xDAD1, 0xDA17, 0xD95E, 0xD8A6, 0xD7EF,
    0xD73A, 0xD685, 0xD5D2, 0xD520, 0xD46F, 0xD3BF, 0xD311, 0xD263,
    0xD1B7, 0xD10C, 0xD062, 0xCFB9, 0xCF11, 0xCE6A, 0xCDC4, 0xCD1F,
    0xCC7B, 0xCBD8, 0xCB36, 0xCA96, 0xC9F6, 0xC957, 0xC8B9, 0xC81C,
    0xC780, 0xC6E5, 0xC64B, 0xC5B2, 0xC51A, 0xC482, 0xC3EC, 0xC357,
    0xC2C2, 0xC22E, 0xC19B, 0xC109, 0xC078, 0xBFE8, 0xBF59, 0xBECA,
    0xBE3C, 0xBDAF, 0xBD23, 0xBC98, 0xBC0D, 0xBB83, 0xBAFB, 0xBA72,
    0xB9EB, 0xB964, 0xB8DE, 0xB859, 0xB7D5, 0xB751, 0xB6CE, 0xB64C,
    0xB5CB, 0xB54A, 0xB4CA, 0xB44B, 0xB3CC, 0xB34E, 0xB2D1, 0xB254,
    0xB1D8, 0xB15D, 0xB0E3, 0xB069, 0xAFF0, 0xAF77, 0xAEFF, 0xAE88,
    0xAE11, 0xAD9B, 0xAD26, 0xACB1, 0xAC3D, 0xABC9, 0xAB56, 0xAAE4,
    0xAA72, 0xAA01, 0xA990, 0xA920, 0xA8B1, 0xA842, 0xA7D3, 0xA766,
    0xA6F8, 0xA68C, 0xA620, 0xA5B4, 0xA549, 0xA4DF, 0xA475, 0xA40C,
    0xA3A3, 0xA33A, 0xA2D3, 0xA26B, 0xA204, 0xA19E, 0xA138, 0xA0D3,
    0xA06E, 0xA00A, 0x9FA6, 0x9F43, 0x9EE0, 0x9E7E, 0x9E1C, 0x9DBA,
    0x9D59, 0x9CF9, 0x9C99, 0x9C39, 0x9BDA, 0x9B7C, 0x9B1D, 0x9AC0,
    0x9A62, 0x9A05, 0x99A9, 0x994D, 0x98F1, 0x9896, 0x983B, 0x97E1,
    0x9787, 0x972E, 0x96D5, 0x967C, 0x9624, 0x95CC, 0x9574, 0x951D,
    0x94C7, 0x9470, 0x941B, 0x93C5, 0x9370, 0x931B, 0x92C7, 0x9273,
    0x921F, 0x91CC, 0x9179, 0x9127, 0x90D5, 0x9083, 0x9032, 0x8FE1,
    0x8F90, 0x8F40, 0x8EF0, 0x8EA0, 0x8E51, 0x8E02, 0x8DB3, 0x8D65,
    0x8D17, 0x8CC9, 0x8C7C, 0x8C2F, 0x8BE2, 0x8B96, 0x8B4A, 0x8AFF,
    0x8AB3, 0x8A68, 0x8A1E, 0x89D3, 0x8989, 0x8940, 0x88F6, 0x88AD,
    0x8864, 0x881C, 0x87D3, 0x878C, 0x8744, 0x86FD, 0x86B6, 0x866F,
    0x8628, 0x85E2, 0x859C, 0x8557, 0x8511, 0x84CC, 0x8488, 0x8443,
    0x83FF, 0x83BB, 0x8377, 0x8334, 0x82F1, 0x82AE, 0x826B, 0x8229,
    0x81E7, 0x81A5, 0x8164, 0x8123, 0x80E2, 0x80A1, 0x8060, 0x8020,
]

MASK32 = 0xFFFFFFFF
MASK33 = 0x1FFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
MASK65 = 0x1FFFFFFFFFFFFFFFF


def clz16(x):
    """Count leading zeros of a 16-bit value."""
    x &= 0xFFFF
    if x == 0:
        return 16
    n = 0
    while (x & 0x8000) == 0:
        x <<= 1
        n += 1
    return n


def newton_step_q32(a_q1_31, x0_u32):
    """
    Simulate newton_step_q32 exactly.
      a_q1_31: 32-bit Q1.31  (normalized input in [1, 2))
      x0_u32  : 32-bit        (Q0.32 seed estimate)
    Returns (x1_q0_32, saturated) both as Python ints.
    """
    # ax = (a * x0) >> 32  ->  Q1.31
    ax_mul  = (a_q1_31 & MASK32) * (x0_u32 & MASK32)          # 64-bit
    ax_q1_31 = (ax_mul >> 32) & MASK32
    ax_q1_31_ext = ax_q1_31  # 33-bit value (leading 0)

    # Clamp ax to 2.0 in Q1.31 = 0x1_00000000
    TWO_Q1_31 = 0x100000000
    ax_clamped = min(ax_q1_31_ext, TWO_Q1_31)

    # two_minus_ax in Q1.31 (33-bit)
    two_minus_ax = (TWO_Q1_31 - ax_clamped) & MASK33

    # prod = x0 * (2 - ax)  ->  65-bit
    prod_mul = (x0_u32 & MASK32) * two_minus_ax  # fits in 65 bits

    # Round-to-nearest: add 2^30, then >> 31
    bias = 1 << 30
    x1_ext = ((prod_mul + bias) >> 31) & 0x3FFFFFFFF  # 34-bit result

    saturated = x1_ext > MASK32
    x1_q0_32  = MASK32 if saturated else (x1_ext & MASK32)
    return x1_q0_32, saturated


def reciprocal_u16_q32(v):
    """
    Full reference model for reciprocal_u16_q32.
    Returns (result_q32, divide_by_zero, saturated).
    """
    v &= 0xFFFF

    if v == 0:
        return 0xFFFFFFFF, True, True

    # Normalize to Q1.15 in [1, 2)
    shift_left = clz16(v)
    a_q1_15    = (v << shift_left) & 0xFFFF

    # Zero-extend to Q1.31
    a_q1_31 = (a_q1_15 << 16) & MASK32

    # LUT lookup: lut_idx = a_q1_15[14:7]
    lut_idx    = (a_q1_15 >> 7) & 0xFF
    seed_q0_16 = SEED_LUT_Q0_16[lut_idx]

    # Zero-extend seed to Q0.32
    x0_u32 = (seed_q0_16 << 16) & MASK32

    # Two Newton refinement steps
    x1_q0_32, _ = newton_step_q32(a_q1_31, x0_u32)
    x2_q0_32, _ = newton_step_q32(a_q1_31, x1_q0_32)

    # De-normalize: shift right by msb_index = 15 - shift_left
    msb_index = 15 - shift_left

    if msb_index == 0:
        out_ext = x2_q0_32  # 33-bit value (leading 0)
    else:
        bias   = (1 << (msb_index - 1)) & MASK32
        numer  = (x2_q0_32 + bias) & MASK33
        out_ext = numer >> msb_index

    out_sat = out_ext > MASK32
    out_q32 = MASK32 if out_sat else (out_ext & MASK32)
    return out_q32, False, out_sat


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "vectors")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reciprocal_golden_q32.memh")

    # Directed test cases matching the Q16 testbench
    directed = [0, 1, 2, 3, 7, 255, 256, 257, 1023, 43692, 65535]
    print("Directed test case expected values:")
    print(f"  {'v':>6}  {'q32':>10}  {'q32_hex':>12}  {'div0':>4}  {'sat':>4}")
    for v in directed:
        q32, div0, sat = reciprocal_u16_q32(v)
        print(f"  {v:>6}  {q32:>10}  0x{q32:08X}  {int(div0):>4}  {int(sat):>4}")

    print(f"\nGenerating {out_path} ...")
    with open(out_path, "w") as f:
        for v in range(65536):
            q32, _, _ = reciprocal_u16_q32(v)
            # Pack as 48-bit word: upper 16 bits = v, lower 32 bits = q32
            word = ((v & 0xFFFF) << 32) | (q32 & MASK32)
            f.write(f"{word:012X}\n")

    print(f"Done. {65536} vectors written.")


if __name__ == "__main__":
    main()
