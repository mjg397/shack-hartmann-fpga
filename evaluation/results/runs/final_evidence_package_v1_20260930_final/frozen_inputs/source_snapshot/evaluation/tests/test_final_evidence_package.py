"""Focused regressions for the final evidence-package metric decomposition."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.common import config as C
from evaluation.common import optics as O
from evaluation.common.roms import ROMSpec, read_hex_words
from evaluation.optical.final_evidence_package import write_mem
from evaluation.optical.resolved_model_candidate import coefficient_metric, metric_rms


class FinalEvidenceMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = O.Bench(O.OpticalConfig(supersample=20))
        cls.metric = coefficient_metric(cls.bench)

    def test_gram_metric_matches_direct_pupil_wavefront(self):
        values = np.asarray([0.7, -0.2, 0.9, 0.1, -0.4,
                             0.3, 0.5, -0.8, 0.6, -0.1]) * 100e-9
        direct = self.bench.opd_rms(self.bench.opd_from_coeffs(values))
        self.assertAlmostEqual(metric_rms(values, self.metric), direct, places=20)

    def test_gram_component_squares_and_cross_terms_close(self):
        components = [np.arange(1, 11) * 1e-10,
                      np.arange(10, 0, -1) * -2e-11,
                      np.asarray([(-1) ** i for i in range(10)]) * 3e-11]
        total = sum(components, np.zeros(C.NUM_MODES))
        expanded = sum(float(value @ self.metric @ value) for value in components)
        for first in range(len(components)):
            for second in range(first + 1, len(components)):
                expanded += float(2 * components[first] @ self.metric @ components[second])
        self.assertAlmostEqual(float(total @ self.metric @ total), expanded, places=30)

    def test_signed_rom_writer_uses_declared_sign_extension(self):
        values = np.asarray([0, 1, -1, (1 << 17) - 1, -(1 << 17)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.mem"
            write_mem(path, values, encoded_bits=20)
            loaded = read_hex_words(path, ROMSpec(5, 18, True, 20))
        signed = np.asarray([value - (1 << 18) if value & (1 << 17) else value
                             for value in loaded])
        np.testing.assert_array_equal(signed, values)


if __name__ == "__main__":
    unittest.main()
