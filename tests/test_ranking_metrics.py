"""Unit tests for src.core.metrics: average_precision_at_k, map_at_k, ndcg_at_k."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_V4_ROOT = Path(__file__).resolve().parent.parent
if str(_V4_ROOT) not in sys.path:
    sys.path.insert(0, str(_V4_ROOT))

from src.core.metrics import average_precision_at_k, map_at_k, ndcg_at_k


# ---------------------------------------------------------------------------
# average_precision_at_k
# ---------------------------------------------------------------------------

class TestAveragePrecisionAtK:
    """Tests for average_precision_at_k (single query)."""

    def test_perfect_ranking_returns_one(self):
        """All gt items retrieved in the leading ranks -> AP 1.0."""
        gt = np.arange(5, dtype=np.int64)
        retrieved = np.arange(5, dtype=np.int64)
        assert average_precision_at_k(retrieved, gt, k=5) == pytest.approx(1.0)

    def test_no_hits_returns_zero(self):
        """No gt item in the top-K -> AP 0.0."""
        gt = np.array([0, 1, 2], dtype=np.int64)
        retrieved = np.array([7, 8, 9], dtype=np.int64)
        assert average_precision_at_k(retrieved, gt, k=3) == pytest.approx(0.0)

    def test_known_hand_computed_case(self):
        """retrieved=[0,5,1,6,2], gt={0,1,2,3}: hits at ranks 1, 3, 5."""
        gt = np.array([0, 1, 2, 3], dtype=np.int64)
        retrieved = np.array([0, 5, 1, 6, 2], dtype=np.int64)
        expected = (1.0 / 1 + 2.0 / 3 + 3.0 / 5) / 4  # normalizer min(|gt|, k) = 4
        assert average_precision_at_k(retrieved, gt, k=5) == pytest.approx(expected)

    def test_ranking_order_matters(self):
        """The same hit count scores higher when the hits come earlier."""
        gt = np.array([0, 1], dtype=np.int64)
        early = average_precision_at_k(np.array([0, 1, 8, 9]), gt, k=4)
        late = average_precision_at_k(np.array([8, 9, 0, 1]), gt, k=4)
        assert early > late

    def test_truncated_k_normalizes_by_k(self):
        """|gt| = 100 but k = 10: a perfect top-10 prefix still scores 1.0."""
        gt = np.arange(100, dtype=np.int64)
        retrieved = np.arange(100, dtype=np.int64)
        assert average_precision_at_k(retrieved, gt, k=10) == pytest.approx(1.0)

    def test_invalid_ids_are_not_hits(self):
        """Retrieved -1 placeholders never count as hits."""
        gt = np.array([5, -1, -1], dtype=np.int64)
        retrieved = np.array([-1, -1, -1], dtype=np.int64)
        assert average_precision_at_k(retrieved, gt, k=3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# map_at_k
# ---------------------------------------------------------------------------

class TestMapAtK:
    """Tests for map_at_k (mean over queries)."""

    def test_perfect_ranking_returns_one(self):
        gt = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        retrieved = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        assert map_at_k(retrieved, gt, k=3) == pytest.approx(1.0)

    def test_no_hits_returns_zero(self):
        gt = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        retrieved = np.array([[6, 7, 8], [6, 7, 8]], dtype=np.int64)
        assert map_at_k(retrieved, gt, k=3) == pytest.approx(0.0)

    def test_mean_over_queries(self):
        """One perfect query and one empty query -> mAP 0.5."""
        gt = np.array([[0, 1], [2, 3]], dtype=np.int64)
        retrieved = np.array([[0, 1], [8, 9]], dtype=np.int64)
        assert map_at_k(retrieved, gt, k=2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

class TestNdcgAtK:
    """Tests for ndcg_at_k."""

    def test_perfect_ranking_returns_one(self):
        gt = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        retrieved = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        assert ndcg_at_k(retrieved, gt, k=5) == pytest.approx(1.0)

    def test_no_hits_returns_zero(self):
        gt = np.array([[0, 1, 2]], dtype=np.int64)
        retrieved = np.array([[7, 8, 9]], dtype=np.int64)
        assert ndcg_at_k(retrieved, gt, k=3) == pytest.approx(0.0)

    def test_known_hand_computed_case(self):
        """retrieved=[0,5,1,6,2], gt={0,1,2,3}: hits at ranks 1, 3, 5."""
        gt = np.array([[0, 1, 2, 3]], dtype=np.int64)
        retrieved = np.array([[0, 5, 1, 6, 2]], dtype=np.int64)
        dcg = 1.0 / np.log2(2) + 1.0 / np.log2(4) + 1.0 / np.log2(6)
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, 5))  # min(|gt|, k) = 4
        assert ndcg_at_k(retrieved, gt, k=5) == pytest.approx(dcg / idcg)

    def test_permuted_perfect_ranking_still_one(self):
        """All gt items retrieved but out of ideal order still scores 1.0.

        With binary gains every gt item carries the same gain, so any ordering
        that fills the first |gt| ranks with gt items is ideal.
        """
        gt = np.array([[0, 1, 2]], dtype=np.int64)
        retrieved = np.array([[2, 0, 1]], dtype=np.int64)
        assert ndcg_at_k(retrieved, gt, k=3) == pytest.approx(1.0)

    def test_late_hits_score_lower(self):
        gt = np.array([[0, 1]], dtype=np.int64)
        early = ndcg_at_k(np.array([[0, 1, 8, 9]]), gt, k=4)
        late = ndcg_at_k(np.array([[8, 9, 0, 1]]), gt, k=4)
        assert early > late

    def test_truncated_k_normalizes_by_k(self):
        """|gt| = 100 but k = 10: a perfect top-10 prefix still scores 1.0."""
        gt = np.arange(100, dtype=np.int64).reshape(1, -1)
        retrieved = np.arange(100, dtype=np.int64).reshape(1, -1)
        assert ndcg_at_k(retrieved, gt, k=10) == pytest.approx(1.0)
