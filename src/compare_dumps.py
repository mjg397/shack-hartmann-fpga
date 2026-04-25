#!/usr/bin/env python3
import os
import sys

DUMP_DIR = "debug_dumps"

PAIRS = [
    ("py_sent_e_matrix", "c_recv_e_matrix", 4),
    ("py_sent_coeffs", "c_recv_coeffs", 1),
    ("c_sent_centroids", "py_recv_centroids", 4),
    ("c_sent_slopes", "py_recv_slopes", 4),
    ("c_sent_zernike", "py_recv_zernike", 4),
]


def read_bytes(base):
    path = os.path.join(DUMP_DIR, f"{base}.bin")
    with open(path, "rb") as f:
        return f.read()


def words_from_bytes(data, elem_size):
    if elem_size <= 0 or len(data) % elem_size != 0:
        raise ValueError(f"Invalid payload size {len(data)} for elem_size={elem_size}")
    return [int.from_bytes(data[i:i + elem_size], "little", signed=False) for i in range(0, len(data), elem_size)]


def compare_pair(a_name, b_name, elem_size):
    missing = []
    a_path = os.path.join(DUMP_DIR, f"{a_name}.bin")
    b_path = os.path.join(DUMP_DIR, f"{b_name}.bin")
    if not os.path.exists(a_path):
        missing.append(a_path)
    if not os.path.exists(b_path):
        missing.append(b_path)
    if missing:
        print(f"[MISSING] {a_name} vs {b_name}")
        for m in missing:
            print(f"  - {m}")
        return False

    a_raw = read_bytes(a_name)
    b_raw = read_bytes(b_name)

    try:
        a = words_from_bytes(a_raw, elem_size)
        b = words_from_bytes(b_raw, elem_size)
    except ValueError as e:
        print(f"[ERROR] {a_name} vs {b_name}: {e}")
        return False

    ok = True
    if len(a) != len(b):
        print(f"[FAIL] {a_name} vs {b_name}: length mismatch {len(a)} != {len(b)}")
        ok = False

    mismatches = []
    for i, (av, bv) in enumerate(zip(a, b)):
        if av != bv:
            mismatches.append((i, av, bv))
            if len(mismatches) >= 8:
                break

    if mismatches:
        print(f"[FAIL] {a_name} vs {b_name}: found mismatches")
        hex_width = elem_size * 2
        for idx, av, bv in mismatches:
            print(f"  idx {idx}: A=0x{av:0{hex_width}X} B=0x{bv:0{hex_width}X}")
        ok = False
    elif ok:
        print(f"[PASS] {a_name} vs {b_name}: {len(a)} elements match")

    a_sum = sum(a) & 0xFFFFFFFF
    b_sum = sum(b) & 0xFFFFFFFF
    a_xor = 0
    for v in a:
        a_xor ^= v
    b_xor = 0
    for v in b:
        b_xor ^= v

    print(
        f"  checksums: A(sum32=0x{a_sum:08X}, xor=0x{a_xor:08X}) "
        f"B(sum32=0x{b_sum:08X}, xor=0x{b_xor:08X})"
    )

    return ok


def main():
    print(f"Comparing dump pairs in {DUMP_DIR}")
    all_ok = True
    for a_name, b_name, elem_size in PAIRS:
        pair_ok = compare_pair(a_name, b_name, elem_size)
        all_ok = all_ok and pair_ok

    if all_ok:
        print("\nALL DATASET COMPARISONS PASSED")
        return 0

    print("\nONE OR MORE DATASET COMPARISONS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
