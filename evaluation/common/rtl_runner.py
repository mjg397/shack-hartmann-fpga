"""
Drive the existing Icarus Verilog simulation with arbitrary frames.

Reuses sim/ as-is: the same tb_zynq_top / tb_full_pipeline
and the same RTL sources listed in run.sh. Nothing is duplicated -- this only
supplies frames and parses results.

Note on cocotb: the repository's rtl/verification/test_*.py files are
print-only scaffolds, not cocotb tests, and cocotb is not installed. The
working simulation infrastructure is the Icarus flow, so that is what is reused.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import config as C
from .roms import stage_rom_set

# Gen B chain, matching sim/run.sh
RTL_SOURCES = [
    "shwfs_processing.v",
    "Thresholding_CoG_intensity_accumulator.v",
    "intensity_reciprocal_q18.v",
    "reciprocal_u16_q18.v",
    "slope_calculation.v",
    "ematrix_accumulator.v",
    "streaming_emulator.v",
]
TB_SOURCES = ["bram_model.v", "tb_zynq_top.v", "tb_full_pipeline.v"]

_LAT_RE = re.compile(r"that pixel -> done\s+:\s+(\d+) cy")
_PIX_RE = re.compile(r"pixels streamed\s+:\s+(\d+)")
_MIS_RE = re.compile(r"pixel mismatches\s+:\s+(\d+)")
_SUB_RE = re.compile(r"subapertures valid\s+:\s+(\d+)")
_FINALPIX_RE = re.compile(r"final contributing pixel was #(\d+)")
_UNKNOWN_SLOPE_RE = re.compile(r"unknown valid slope values\s+:\s+(\d+)")
_UNKNOWN_COEFF_RE = re.compile(r"unknown valid coefficients\s+:\s+(\d+)")
_EXTRA_RE = {
    "first_pixel_cycle": re.compile(r"first input pixel cycle\s+:\s+(-?\d+)"),
    "last_pixel_cycle": re.compile(r"last input pixel cycle\s+:\s+(-?\d+)"),
    "first_subap_cycle": re.compile(r"first subap_valid cycle\s+:\s+(-?\d+)"),
    "done_cycle": re.compile(r"done cycle\s+:\s+(-?\d+)"),
    "max_back_to_back_run": re.compile(r"longest back-to-back pixel run\s+:\s+(-?\d+)"),
    "stall_events": re.compile(r"active-region stall events\s+:\s+(-?\d+)"),
    "subap_to_done_cycles": re.compile(r"last subap_valid\s+-> done\s+:\s+(\d+) cy"),
    "stage_accum_cycle": re.compile(r"accumulator completion cycle\s+:\s+(-?\d+)"),
    "stage_recip_cycle": re.compile(r"reciprocal completion cycle\s+:\s+(-?\d+)"),
    "stage_slope_cycle": re.compile(r"slope \(subap_valid\) cycle\s+:\s+(-?\d+)"),
    "stage_done_cycle": re.compile(r"zernike \(done\) cycle\s+:\s+(-?\d+)"),
    "final_tile_pixel_cycle": re.compile(r"arrived at cycle\s+:\s+(-?\d+)"),
    "true_latency_cycles": re.compile(r"arrival -> done\s+:\s+(\d+) cy"),
    "accumulator_depth_cycles": re.compile(r"accumulator internal depth\s+:\s+(-?\d+)"),
    # Same run, three anchor points. The paper must state which it uses.
    "latency_from_s1_cycles": re.compile(r"anchor s1 -> done\s+:\s+(-?\d+)"),
    "latency_from_accum_cycles": re.compile(r"anchor accum -> done\s+:\s+(-?\d+)"),
}


def final_tile_pixel_index() -> int:
    """
    Raster index of the pixel that completes the last in-pupil subaperture.

    Derived from the pupil bitmap, so it tracks the design rather than being
    hard-coded.
    """
    from . import bitaccurate as _BA
    last = int(_BA.valid_subaps()[-1])
    r, c = divmod(last, C.NUM_LENSLETS)
    return (r * C.SUBAP_PIXELS + C.SUBAP_PIXELS - 1) * C.FRAME_W \
        + (c * C.SUBAP_PIXELS + C.SUBAP_PIXELS - 1)


class RTLSimulator:
    """Compiles the simulation once, then runs many frames through it."""

    def __init__(self, workdir: Path | None = None, rom_dir: Path | None = None,
                 timeout_seconds: float = 30.0):
        self._tmp = None
        if workdir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="shwfs_eval_")
            workdir = Path(self._tmp.name)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / "waves").mkdir(exist_ok=True)
        self.vvp = self.workdir / "sim.vvp"
        self.rom_dir = Path(rom_dir or C.RTL_DIR)
        self.timeout_seconds = float(timeout_seconds)
        self._build()

    def _build(self) -> None:
        source_paths = [C.RTL_DIR / s for s in RTL_SOURCES]
        source_paths += [C.SIM_DIR / s for s in TB_SOURCES]
        missing = [str(p) for p in source_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("required simulation source(s) missing: " + ", ".join(missing))
        sources = [str(p.resolve()) for p in source_paths]
        cmd = ["iverilog", "-g2012", "-o", str(self.vvp),
               "-s", "tb_full_pipeline",
               f"-Ptb_full_pipeline.FINAL_TILE_PIXEL={final_tile_pixel_index()}",
               ] + sources
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=self.timeout_seconds)
        if proc.returncode != 0:
            raise RuntimeError(
                f"iverilog failed with exit {proc.returncode}:\n{proc.stdout}{proc.stderr}")
        self.rom_manifest = stage_rom_set(self.workdir, self.rom_dir)

    @staticmethod
    def pack_image(pixels: np.ndarray) -> list[int]:
        """4 pixels per 32-bit word, little-endian (matches byte_sel)."""
        p = np.asarray(pixels, dtype=np.int64).ravel()
        return [int(p[i] | (p[i + 1] << 8) | (p[i + 2] << 16) | (p[i + 3] << 24))
                for i in range(0, p.size, 4)]

    def run_frame(self, pixels: np.ndarray) -> dict:
        """Stream one frame and return RTL outputs plus latency measurements."""
        p = np.asarray(pixels, dtype=np.int64).ravel()
        if p.size != C.FRAME_W * C.FRAME_H:
            raise ValueError(f"expected {C.FRAME_W*C.FRAME_H} pixels, got {p.size}")

        if np.any(p < 0) or np.any(p > 255):
            raise ValueError("pixels must be 8-bit integers in [0, 255]")
        (self.workdir / "hcipy_image_rotating.mem").write_text(
            "".join(f"{int(v):02X}\n" for v in p))
        (self.workdir / "image_packed.mem").write_text(
            "".join(f"{w:08X}\n" for w in self.pack_image(p)))

        for stale in ("zernikes_raw.txt", "slopes_raw.txt"):
            (self.workdir / stale).unlink(missing_ok=True)
        try:
            proc = subprocess.run(["vvp", str(self.vvp)], cwd=self.workdir,
                                  capture_output=True, text=True,
                                  timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"RTL simulation exceeded {self.timeout_seconds:g} s") from exc
        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"vvp failed with exit {proc.returncode}:\n{out}")
        if "[TIMEOUT]" in out:
            raise RuntimeError(f"RTL testbench timeout:\n{out}")

        def _grab(rx, label):
            m = rx.search(out)
            if not m:
                raise RuntimeError(f"RTL output is missing {label!r}:\n{out}")
            return int(m.group(1))

        zern_path = self.workdir / "zernikes_raw.txt"
        if not zern_path.is_file():
            raise RuntimeError("RTL produced no zernikes_raw.txt output")
        zern_lines = zern_path.read_text().splitlines()
        if len(zern_lines) != C.NUM_MODES + 1 or zern_lines[0].strip() != "raw_fpga_coeff":
            raise RuntimeError(
                f"malformed coefficient output: expected header plus {C.NUM_MODES} records")
        try:
            zern = np.asarray([int(v.strip(), 10) for v in zern_lines[1:]], dtype=np.int64)
        except ValueError as exc:
            raise RuntimeError("unknown or non-numeric valid coefficient in RTL output") from exc

        slope_path = self.workdir / "slopes_raw.txt"
        if not slope_path.is_file():
            raise RuntimeError("RTL produced no slopes_raw.txt output")
        slope_lines = slope_path.read_text().splitlines()
        if len(slope_lines) != C.NUM_SUBAPS_VALID + 1:
            raise RuntimeError(
                f"malformed slope output: expected {C.NUM_SUBAPS_VALID} records, "
                f"found {max(0, len(slope_lines)-1)}")
        try:
            slopes = np.asarray([[int(x, 10) for x in line.split()]
                                 for line in slope_lines[1:]], dtype=np.int64)
        except ValueError as exc:
            raise RuntimeError("unknown or non-numeric valid slope in RTL output") from exc
        if slopes.shape != (C.NUM_SUBAPS_VALID, 5):
            raise RuntimeError(f"malformed slope output shape: {slopes.shape}")
        if not np.array_equal(slopes[:, 0], np.arange(C.NUM_SUBAPS_VALID)):
            raise RuntimeError("missing, duplicate, or out-of-order slope records")

        latency = _grab(_LAT_RE, "latency")
        extra = {k: _grab(rx, k) for k, rx in _EXTRA_RE.items()}
        pixels_streamed = _grab(_PIX_RE, "pixels streamed")
        pixel_mismatches = _grab(_MIS_RE, "pixel mismatch count")
        subaps_valid = _grab(_SUB_RE, "valid subaperture count")
        unknown_slopes = _grab(_UNKNOWN_SLOPE_RE, "unknown valid slope count")
        unknown_coefficients = _grab(_UNKNOWN_COEFF_RE, "unknown valid coefficient count")
        # The testbench intentionally stops when reconstruction is valid; the
        # last in-pupil tile precedes the end of the raster, so a complete
        # 65,536-pixel stream is neither expected nor required. It must have
        # consumed the final contributing pixel and no more than one frame.
        min_pixels = final_tile_pixel_index() + 1
        max_pixels = C.FRAME_W * C.FRAME_H
        if not (min_pixels <= pixels_streamed <= max_pixels):
            raise RuntimeError(
                f"RTL streamed {pixels_streamed} pixels; expected {min_pixels}..{max_pixels} "
                "before the valid result")
        if pixel_mismatches:
            raise RuntimeError(f"RTL pixel stream had {pixel_mismatches} mismatches")
        if subaps_valid != C.NUM_SUBAPS_VALID:
            raise RuntimeError(f"RTL emitted {subaps_valid} valid subapertures, expected {C.NUM_SUBAPS_VALID}")
        if unknown_slopes or unknown_coefficients:
            raise RuntimeError(
                f"RTL emitted unknowns at valid events: slopes={unknown_slopes}, "
                f"coefficients={unknown_coefficients}")
        return {
            **extra,
            "zernike_raw": zern,
            "slopes": slopes,
            "latency_cycles": latency,
            "latency_ns": latency * C.CLOCK_NS if latency >= 0 else float("nan"),
            "pixels_streamed": pixels_streamed,
            "pixel_mismatches": pixel_mismatches,
            "subaps_valid": subaps_valid,
            "unknown_valid_slopes": unknown_slopes,
            "unknown_valid_coefficients": unknown_coefficients,
            "final_pixel_index": _grab(_FINALPIX_RE, "final contributing pixel index"),
            "rom_manifest": self.rom_manifest,
            "stdout": out,
        }

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
