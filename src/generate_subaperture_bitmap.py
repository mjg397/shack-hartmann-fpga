#!/usr/bin/env python3
"""Generate a row-major subaperture-inside-pupil bitmap for Verilog readmemh."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def subaperture_inside_pupil(row: int, col: int, grid_size: int, radius: float) -> bool:
    """Return True when the full unit square lies inside the inscribed pupil."""
    center = grid_size / 2.0
    corners = (
        (col, row),
        (col + 1, row),
        (col, row + 1),
        (col + 1, row + 1),
    )
    return all(math.hypot(x - center, y - center) <= radius for x, y in corners)


def build_bitmap(grid_size: int, radius: float) -> int:
    """Pack row-major subaperture membership into a single integer.

    Bit index i = grid_size * row + col, so bit 0 corresponds to row 0, col 0.
    """
    bitmap = 0
    for row in range(grid_size):
        for col in range(grid_size):
            bit_index = grid_size * row + col
            if subaperture_inside_pupil(row, col, grid_size, radius):
                bitmap |= 1 << bit_index
    return bitmap


def write_readmemh_hex(path: Path, bitmap: int, width_bits: int) -> None:
    hex_digits = (width_bits + 3) // 4
    path.write_text(f"{bitmap:0{hex_digits}X}\n", encoding="ascii")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a row-major subaperture bitmap and write it as one hex word "
            "for Verilog readmemh."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("full_pipeline_sim/data/subapeture_bitmap.hex"),
        help="Output .hex path. Default: %(default)s",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=16,
        help="Number of subapertures across one dimension. Default: %(default)s",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=7.20,
        help="Pupil radius in subaperture widths. Default: %(default)s",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    width_bits = args.grid_size * args.grid_size
    bitmap = build_bitmap(args.grid_size, args.radius)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_readmemh_hex(args.output, bitmap, width_bits)

    num_valid = bitmap.bit_count()
    print(f"Wrote {width_bits}-bit bitmap to {args.output}")
    print(f"Valid subapertures: {num_valid}")
    print(f"Hex word: {bitmap:0{(width_bits + 3) // 4}X}")
    print("Bit index mapping: bit i = grid_size * row + col")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())