from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_lp = Path(__file__).resolve().parents[1]
if str(_lp) not in sys.path:
    sys.path.insert(0, str(_lp))

from notebook_runs import pairwise_metric_matrix, utility_similarity_from_vectors


class TestPairwiseMetricMatrix(unittest.TestCase):
    def test_extracts_square_matrix(self) -> None:
        data = {
            "probe_mode": "cross_role",
            "pairwise_role_metrics": {
                "roles": ["a", "b"],
                "by_layer": {
                    "0": {
                        "a": {"a": {"r2": 0.5}, "b": {"r2": 0.1}},
                        "b": {"a": {"r2": 0.2}, "b": {"r2": 0.6}},
                    }
                },
            },
        }
        roles, mat = pairwise_metric_matrix(data, 0, "r2")
        self.assertEqual(roles, ["a", "b"])
        np.testing.assert_array_almost_equal(mat, [[0.5, 0.1], [0.2, 0.6]])


class TestUtilitySimilarity(unittest.TestCase):
    def test_correlation_identity_diag_one(self) -> None:
        v = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
        s = utility_similarity_from_vectors(v, "correlation")
        self.assertAlmostEqual(float(s[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(s[1, 1]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
