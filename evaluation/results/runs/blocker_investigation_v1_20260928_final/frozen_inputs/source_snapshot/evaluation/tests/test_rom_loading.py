from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.common import config as C
from evaluation.common.roms import (
    REQUIRED_ROMS,
    ROMValidationError,
    validate_rom_set,
)
from evaluation.common.rtl_runner import RTLSimulator


class ROMLoadingTests(unittest.TestCase):
    def _copy_roms(self, destination: Path) -> None:
        for name in REQUIRED_ROMS:
            shutil.copyfile(C.RTL_DIR / name, destination / name)

    def test_authoritative_roms_have_exact_declared_shapes(self) -> None:
        manifest = validate_rom_set()
        self.assertEqual(set(manifest), set(REQUIRED_ROMS))
        for name, spec in REQUIRED_ROMS.items():
            self.assertEqual(manifest[name]["words"], spec.words)

    def test_missing_required_rom_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._copy_roms(root)
            (root / "e_matrix_y.mem").unlink()
            with self.assertRaisesRegex(ROMValidationError, "required ROM is missing"):
                validate_rom_set(root)

    def test_incomplete_required_rom_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._copy_roms(root)
            path = root / "slopes_ref_x.mem"
            path.write_text("\n".join(path.read_text().split()[:-1]) + "\n")
            with self.assertRaisesRegex(ROMValidationError, "expected 256 hex words"):
                validate_rom_set(root)

    def test_runner_is_independent_of_calling_working_directory(self) -> None:
        frame = np.zeros(C.FRAME_W * C.FRAME_H, dtype=np.uint8)
        original = Path.cwd()
        outputs = []
        try:
            for calling_dir in (C.REPO_ROOT, Path(tempfile.gettempdir())):
                os.chdir(calling_dir)
                with RTLSimulator() as sim:
                    result = sim.run_frame(frame)
                    outputs.append(result["zernike_raw"])
                    self.assertEqual(result["unknown_valid_slopes"], 0)
                    self.assertEqual(result["unknown_valid_coefficients"], 0)
        finally:
            os.chdir(original)
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_streamer_preserves_nonzero_first_pixel_of_each_row(self) -> None:
        # Regression for an 8-bit shift-expression bug in the row-prime BRAM
        # address. It was normally masked because detector column zero is dark.
        frame = np.zeros((C.FRAME_H, C.FRAME_W), dtype=np.uint8)
        frame[:, 0] = (np.arange(C.FRAME_H, dtype=np.uint16) % 250 + 5).astype(np.uint8)
        with RTLSimulator() as sim:
            result = sim.run_frame(frame.ravel())
        self.assertEqual(result["pixel_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
