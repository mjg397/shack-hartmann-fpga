#!/usr/bin/env python3
"""
Pack the 256x256 8-bit SHWFS image into 32-bit words for the streaming emulator's BRAM.

streaming_emulator.v addresses BRAM by word:

    pixel_index = (row << 8) + col
    word_addr   = pixel_index[15:2]
    byte_sel    = pixel_index[1:0]

and selects the byte as:

    byte_sel 0 -> bram_rddata[7:0]
    byte_sel 1 -> bram_rddata[15:8]
    byte_sel 2 -> bram_rddata[23:16]
    byte_sel 3 -> bram_rddata[31:24]

so four consecutive pixels pack little-endian into one word.

Input : one 8-bit hex value per line, 65536 lines (row-major).
Output: one 32-bit hex word per line, 16384 lines.
"""

import sys
from pathlib import Path

HSIZE = 256
VSIZE = 256
NUM_PIXELS = HSIZE * VSIZE
NUM_WORDS = NUM_PIXELS // 4

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent / "rtl" / "hcipy_image_rotating.mem"
DEFAULT_DST = HERE / "build" / "image_packed.mem"


def read_pixels(path: Path) -> list[int]:
    """Read whitespace-separated 8-bit hex values."""
    tokens = path.read_text().split()
    if len(tokens) != NUM_PIXELS:
        raise SystemExit(
            f"{path}: expected {NUM_PIXELS} pixel values, found {len(tokens)}"
        )
    pixels = [int(tok, 16) for tok in tokens]
    bad = [p for p in pixels if not 0 <= p <= 0xFF]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} value(s) outside 0x00..0xFF")
    return pixels


def pack(pixels: list[int]) -> list[int]:
    """Four pixels per 32-bit word, little-endian to match byte_sel."""
    return [
        pixels[i]
        | (pixels[i + 1] << 8)
        | (pixels[i + 2] << 16)
        | (pixels[i + 3] << 24)
        for i in range(0, NUM_PIXELS, 4)
    ]


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DST

    pixels = read_pixels(src)
    words = pack(pixels)
    assert len(words) == NUM_WORDS

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(f"{w:08X}\n" for w in words))

    # Round-trip check: unpack and compare against the source.
    unpacked = [(w >> (8 * s)) & 0xFF for w in words for s in range(4)]
    if unpacked != pixels:
        raise SystemExit("pack/unpack round-trip failed -- byte order is wrong")

    nonzero = sum(1 for p in pixels if p)
    print(f"packed {len(pixels)} pixels -> {len(words)} words -> {dst}")
    print(f"  round-trip verified; {nonzero} non-zero pixels, max={max(pixels)}")


if __name__ == "__main__":
    main()
