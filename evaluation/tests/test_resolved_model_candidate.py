"""Focused regressions for the resolved-model candidate statistics."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.optical.resolved_model_candidate import (
    independent_cross_batch_squared,
    metric_rms,
)


class CrossBatchEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.metric = np.asarray([[2.0, 0.25], [0.25, 1.0]])

    def test_identical_batch_means_recover_squared_metric_norm(self):
        vector = np.asarray([3.0, -2.0])
        batches = np.tile(vector, (4, 1))
        observed = independent_cross_batch_squared(batches, self.metric)
        expected = metric_rms(vector, self.metric) ** 2
        self.assertAlmostEqual(observed, expected, places=14)

    def test_independent_zero_mean_noise_is_not_forced_positive(self):
        batches = np.asarray([[1.0, 0.0], [-1.0, 0.0],
                              [0.0, 1.0], [0.0, -1.0]])
        observed = independent_cross_batch_squared(batches, np.eye(2))
        self.assertEqual(observed, -1.0)

    def test_frozen_rule_requires_four_batches(self):
        with self.assertRaisesRegex(ValueError, "exactly four"):
            independent_cross_batch_squared(np.zeros((3, 2)), self.metric)


if __name__ == "__main__":
    unittest.main()
