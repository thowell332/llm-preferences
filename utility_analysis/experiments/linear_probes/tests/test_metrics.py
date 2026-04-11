from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_lp = Path(__file__).resolve().parents[1]
if str(_lp) not in sys.path:
    sys.path.insert(0, str(_lp))

from lp.metrics import pairwise_preference_accuracy, r2_score, spearmanr


class TestPairwisePreference(unittest.TestCase):
    def test_perfect_ranking(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(pairwise_preference_accuracy(y, y), 1.0)

    def test_reversed_fails(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([3.0, 2.0, 1.0])
        self.assertAlmostEqual(pairwise_preference_accuracy(y, p), 0.0)

    def test_ties_excluded(self) -> None:
        y = np.array([1.0, 1.0, 2.0])
        p = np.array([0.0, 1.0, 2.0])
        # (0,1) dt=0 skipped; (0,2) and (1,2): true prefers 2 over tied 1s; pred agrees on both
        self.assertAlmostEqual(pairwise_preference_accuracy(y, p), 1.0)

    def test_pred_tie_wrong(self) -> None:
        y = np.array([0.0, 1.0])
        p = np.array([0.5, 0.5])
        self.assertAlmostEqual(pairwise_preference_accuracy(y, p), 0.0)


class TestR2NotCorrelation(unittest.TestCase):
    def test_r2_vs_pearson(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        p = y + 1.0  # perfect shift: correlation 1, R2 can differ from 1
        r2 = r2_score(y, p)
        self.assertLess(r2, 0.99)  # worse than predicting mean of y for shifted pred


if __name__ == "__main__":
    unittest.main()
