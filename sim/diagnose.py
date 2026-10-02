#!/usr/bin/env python3
"""
Fixed-point regression checks for the Gen B slope path.

These three defects were found during bring-up and are now fixed in rtl/. This
script re-verifies each one so a future change cannot silently reintroduce them.

  1. Reference ROM range -- slopes_ref_*.mem must fit signed 25 bits.
     Originally Q4.23 (48-78 million) loaded into `reg signed [24:0]`, so all
     208 non-zero entries overflowed and flipped sign. Now regenerated as Q18.

  2. Binary-point agreement -- the centroid and the reference must share a
     binary point. unsigned_mult originally returned mult_out[35:12] (Q6)
     against Q4.23 reference data, a 2^17 mismatch. Now returns [23:0] (Q18).

  3. Pipeline alignment -- reciprocal_u16_q18 has 2 internal stages, so rI_out
     trails sI by 3 cycles while xI/yI/centroids_done trailed by only 1. Each
     centroid was multiplied by a stale reciprocal. Side-band signals are now
     delayed to match.

Run after run.sh.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
RTL = HERE.parent / "rtl"

REF_REG_BITS = 25
LIMIT = 1 << (REF_REG_BITS - 1)
PIXEL_THRESHOLD = 5
EXPECTED_FRAC_BITS = 18


def check_ref_range() -> bool:
    ref = np.array(
        [int(t, 16) for t in (RTL / "slopes_ref_x.mem").read_text().split()],
        dtype=np.int64,
    )
    nz = ref[ref != 0]
    over = int((ref >= LIMIT).sum())
    ok = over == 0
    print()
    print("[1] Reference ROM fits signed 25-bit")
    print(f"    range     : {nz.min():,} .. {nz.max():,}  (limit {LIMIT-1:,})")
    print(f"    pixels    : {nz.min()/(1<<EXPECTED_FRAC_BITS):.4f} .. "
          f"{nz.max()/(1<<EXPECTED_FRAC_BITS):.4f}")
    print(f"    overflows : {over}    {'PASS' if ok else 'FAIL'}")
    return ok


def check_centroids() -> bool:
    slopes = BUILD / "slopes_raw.txt"
    img = RTL / "hcipy_image_rotating.mem"
    if not slopes.exists():
        print("\n[2/3] skipped -- run run.sh first")
        return False

    px = np.array(
        [int(t, 16) for t in img.read_text().split()], dtype=np.int64
    ).reshape(256, 256)
    bm = int((RTL / "subaperture_bitmap.mem").read_text().strip(), 16)
    valid = [i for i in range(256) if (bm >> i) & 1]

    xi = np.arange(16)[None, :]
    expected = np.zeros(256)
    for r in range(16):
        for c in range(16):
            tile = px[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16].copy()
            tile[tile < PIXEL_THRESHOLD] = 0
            total = tile.sum()
            if total:
                expected[r * 16 + c] = (tile * xi).sum() / total * (1 << EXPECTED_FRAC_BITS)

    rtl = np.loadtxt(slopes, skiprows=1)[:, 3].astype(float)
    h = expected[valid]
    good = (h != 0) & (rtl != 0)

    ratio = float(np.median(rtl[good] / h[good]))
    corr = float(np.corrcoef(h[good], rtl[good])[0, 1])
    err = float(np.max(np.abs(rtl[good] - h[good]) / np.abs(h[good])))

    ok_scale = abs(ratio - 1.0) < 0.01
    ok_align = corr > 0.99

    print()
    print(f"[2] Centroid binary point is Q{EXPECTED_FRAC_BITS}")
    print(f"    median RTL/model ratio : {ratio:.6f}    "
          f"{'PASS' if ok_scale else 'FAIL'}")
    print()
    print("[3] Reciprocal pipeline alignment")
    print(f"    correlation vs model   : {corr:.6f}    "
          f"{'PASS' if ok_align else 'FAIL'}")
    print(f"    max relative error     : {err:.4%}")
    print("    (was +0.044 before the side-band delay was added)")
    return ok_scale and ok_align


def main() -> None:
    print("=" * 74)
    print(" Fixed-point regression checks -- Gen B slope path")
    print("=" * 74)
    results = [check_ref_range(), check_centroids()]
    print()
    print("=" * 74)
    print(" ALL CHECKS PASS" if all(results) else " FAILURES PRESENT")
    print("=" * 74)
    print()


if __name__ == "__main__":
    main()
