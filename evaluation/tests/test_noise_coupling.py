import unittest

import numpy as np

from evaluation.common.detector import DetectorConfig
from evaluation.common.noise_coupling import pairwise_poisson_gaussian_frames


class PairwiseNoiseCouplingTests(unittest.TestCase):
    def setUp(self):
        self.detector = DetectorConfig(
            total_photons=1.0, quantum_efficiency=1.0,
            gain_e_per_adu=2.0, read_noise_e=3.0,
            full_well_e=1e9, adc_bits=16,
        )

    def test_identical_expectations_produce_identical_frames(self):
        expected = np.linspace(0.0, 2000.0, 256).reshape(16, 16)
        frames, diagnostics = pairwise_poisson_gaussian_frames(
            {"a": expected, "b": expected.copy()}, self.detector, [10, 20, 30])
        np.testing.assert_array_equal(frames["a"], frames["b"])
        self.assertEqual(diagnostics["per_setting"]["a"]["excess_poisson_mean_sum"], 0.0)
        self.assertEqual(diagnostics["per_setting"]["b"]["excess_poisson_mean_sum"], 0.0)

    def test_dictionary_order_does_not_change_pair(self):
        first = np.arange(256, dtype=float).reshape(16, 16) + 50
        second = np.flipud(first) + 25
        forward, _ = pairwise_poisson_gaussian_frames(
            {"coarse": first, "fine": second}, self.detector, [4, 5, 6])
        reverse, _ = pairwise_poisson_gaussian_frames(
            {"fine": second, "coarse": first}, self.detector, [4, 5, 6])
        np.testing.assert_array_equal(forward["coarse"], reverse["coarse"])
        np.testing.assert_array_equal(forward["fine"], reverse["fine"])

    def test_pair_is_repeatable_without_comparison_set_state(self):
        first = np.full((16, 16), 100.0)
        second = first.copy(); second[:, 8:] += 10.0
        reference, _ = pairwise_poisson_gaussian_frames(
            {"coarse": first, "fine": second}, self.detector, [7, 8, 9])
        # An unrelated draw cannot alter a repeated pair because the API has no
        # comparison-set state and all streams derive from pair-local inputs.
        pairwise_poisson_gaussian_frames(
            {"other_a": first * 0.5, "other_b": second * 1.5},
            self.detector, [70, 80, 90])
        repeated, _ = pairwise_poisson_gaussian_frames(
            {"coarse": first, "fine": second}, self.detector, [7, 8, 9])
        np.testing.assert_array_equal(reference["coarse"], repeated["coarse"])
        np.testing.assert_array_equal(reference["fine"], repeated["fine"])


if __name__ == "__main__":
    unittest.main()

