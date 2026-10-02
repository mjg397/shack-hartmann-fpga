"""Focused checks for the two explicitly supported flat-reference conventions."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.common import bitaccurate as BA
from evaluation.common import config as C
from evaluation.optical.regen_reference import reference_centroids_q18


class ReferenceConventionTests(unittest.TestCase):
    def test_rtl_reference_matches_pipeline_centroid_stage(self):
        image = np.zeros((C.FRAME_H, C.FRAME_W), dtype=np.uint8)
        # Deliberately asymmetric nontrivial tiles exercise reciprocal rounding.
        image[2::16, 3::16] = 73
        image[11::16, 12::16] = 29
        expected = BA.run_pipeline(image.ravel())
        qx, qy = reference_centroids_q18(image, "rtl")
        np.testing.assert_array_equal(qx, expected["x_centroid"])
        np.testing.assert_array_equal(qy, expected["y_centroid"])

    def test_exact_and_rtl_conventions_are_not_conflated(self):
        image = np.zeros((C.FRAME_H, C.FRAME_W), dtype=np.uint8)
        image[2::16, 3::16] = 73
        image[11::16, 12::16] = 29
        exact_x, exact_y = reference_centroids_q18(image, "exact")
        rtl_x, rtl_y = reference_centroids_q18(image, "rtl")
        self.assertGreater(np.count_nonzero(exact_x - rtl_x), 0)
        self.assertGreater(np.count_nonzero(exact_y - rtl_y), 0)

    def test_unknown_convention_is_rejected(self):
        with self.assertRaises(ValueError):
            reference_centroids_q18(
                np.zeros((C.FRAME_H, C.FRAME_W), dtype=np.uint8), "implicit")


if __name__ == "__main__":
    unittest.main()
