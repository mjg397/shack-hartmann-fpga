import unittest

import numpy as np

from evaluation.common import precision_experimental as P


class PrecisionExperimentalTests(unittest.TestCase):
    def test_exact_integer_centroid_rounding(self):
        moment = np.array([0, 15, 37, 1000], dtype=np.int64)
        intensity = np.array([0, 2, 5, 137], dtype=np.int64)
        got = P.exact_centroid_q18(moment, intensity)
        expected = np.array([
            0,
            (15 * (1 << 18) + 1) // 2,
            (37 * (1 << 18) + 2) // 5,
            (1000 * (1 << 18) + 68) // 137,
        ])
        np.testing.assert_array_equal(got, expected)

    def test_normalized_q17_fits_eighteen_bits_for_all_nonzero_u16(self):
        intensity = np.arange(1, 1 << 16, dtype=np.int64)
        trace = P.normalized_reciprocal_centroid(
            np.zeros_like(intensity), intensity)
        self.assertGreaterEqual(int(trace["reciprocal"].min()), 1 << 16)
        self.assertLessEqual(int(trace["reciprocal"].max()), 1 << 17)
        self.assertLess(int(trace["reciprocal"].max()), 1 << 18)
        self.assertGreaterEqual(int(trace["exponent"].min()), 0)
        self.assertLessEqual(int(trace["exponent"].max()), 15)

    def test_normalized_path_reduces_centroid_error(self):
        intensity = np.arange(59, 65536, 97, dtype=np.int64)
        moment = intensity * 7 + intensity // 3
        exact = P.exact_centroid_q18(moment, intensity)
        baseline = P.direct_reciprocal_centroid(
            moment, intensity, 18, "nearest_half_up", "truncate")["centroid"]
        normalized = P.normalized_reciprocal_centroid(
            moment, intensity, 17, "nearest_half_up", "nearest_half_up")["centroid"]
        baseline_error = np.sqrt(np.mean((baseline - exact).astype(float) ** 2))
        normalized_error = np.sqrt(np.mean((normalized - exact).astype(float) ** 2))
        self.assertLess(normalized_error, baseline_error / 100)

    def test_newton_is_integer_and_improves_seed(self):
        intensity = np.arange(59, 2000, 13, dtype=np.int64)
        moment = intensity * 6 + intensity // 4
        seed = np.asarray([
            P.divide_unsigned(1 << 18, int(d), "nearest_half_up")
            for d in intensity
        ], dtype=np.int64)
        exact = P.exact_centroid_q18(moment, intensity)
        baseline = moment * seed
        refined = P.newton_reciprocal_centroid(
            moment, intensity, seed, 25,
            "nearest_half_up", "nearest_half_up")["centroid"]
        self.assertTrue(np.issubdtype(refined.dtype, np.integer))
        self.assertLess(np.max(np.abs(refined - exact)),
                        np.max(np.abs(baseline - exact)))


if __name__ == "__main__":
    unittest.main()
