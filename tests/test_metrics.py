"""Unit tests for src.metrics: recall_at_k, nn_recall_at_k, compute_ground_truth."""

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

from src.core.metrics import recall_at_k, nn_recall_at_k, compute_ground_truth


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    """Tests for recall_at_k."""

    def test_perfect_recall_returns_one(self):
        """Perfect retrieval: retrieved == ground truth -> recall 1.0."""
        gt = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        retrieved = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        assert recall_at_k(retrieved, gt, k=5) == pytest.approx(1.0)

    def test_zero_overlap_returns_zero(self):
        """No shared IDs between retrieved and gt -> recall 0.0."""
        gt = np.array([[0, 1, 2]], dtype=np.int64)
        retrieved = np.array([[3, 4, 5]], dtype=np.int64)
        assert recall_at_k(retrieved, gt, k=3) == pytest.approx(0.0)

    def test_partial_overlap_correct_fraction(self):
        """Exactly 1 of 4 gt items retrieved -> recall 0.25."""
        gt = np.array([[0, 1, 2, 3]], dtype=np.int64)
        retrieved = np.array([[0, 4, 5, 6]], dtype=np.int64)
        assert recall_at_k(retrieved, gt, k=4) == pytest.approx(0.25)

    def test_invalid_ids_excluded(self):
        """IDs of -1 are treated as invalid and excluded from both sets."""
        # retrieved has -1 placeholders; gt has only 2 valid items
        gt = np.array([[0, 1, -1, -1]], dtype=np.int64)
        retrieved = np.array([[0, 1, -1, -1]], dtype=np.int64)
        # Both sets: {0, 1} -> perfect recall
        assert recall_at_k(retrieved, gt, k=4) == pytest.approx(1.0)

    def test_invalid_ids_do_not_inflate_recall(self):
        """A retrieved -1 should never count as a hit."""
        gt = np.array([[5, -1, -1]], dtype=np.int64)
        retrieved = np.array([[-1, -1, -1]], dtype=np.int64)
        # gt_set = {5}, topk_set = {} -> recall 0.0
        assert recall_at_k(retrieved, gt, k=3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# nn_recall_at_k
# ---------------------------------------------------------------------------

class TestNNRecallAtK:
    """Tests for nn_recall_at_k."""

    def test_true_nn_found_returns_one(self):
        """True 1-NN is at position 0 of retrieved -> recall 1.0."""
        gt = np.array([[7, 2, 3]], dtype=np.int64)
        retrieved = np.array([[7, 99, 100]], dtype=np.int64)
        assert nn_recall_at_k(retrieved, gt, k=3) == pytest.approx(1.0)

    def test_true_nn_not_found_returns_zero(self):
        """True 1-NN absent from top-K -> recall 0.0."""
        gt = np.array([[7, 2, 3]], dtype=np.int64)
        retrieved = np.array([[0, 1, 2]], dtype=np.int64)
        assert nn_recall_at_k(retrieved, gt, k=3) == pytest.approx(0.0)

    def test_true_nn_at_boundary(self):
        """True 1-NN at exactly position k-1 should count as a hit."""
        gt = np.array([[42, 0, 1]], dtype=np.int64)
        retrieved = np.array([[0, 1, 42]], dtype=np.int64)
        assert nn_recall_at_k(retrieved, gt, k=3) == pytest.approx(1.0)

    def test_mixed_queries(self):
        """Half of queries have their 1-NN in top-K -> recall 0.5."""
        gt = np.array([[10, 99], [20, 99]], dtype=np.int64)
        retrieved = np.array([[10, 5], [0, 1]], dtype=np.int64)
        assert nn_recall_at_k(retrieved, gt, k=2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_ground_truth
# ---------------------------------------------------------------------------

class TestComputeGroundTruth:
    """Tests for compute_ground_truth."""

    def test_nearest_neighbor_is_self(self):
        """Each vector's nearest neighbor should be itself (top-1)."""
        rng = np.random.default_rng(0)
        db = rng.standard_normal((20, 8)).astype(np.float32)
        gt = compute_ground_truth(db, db, top_k=1)
        expected = np.arange(20, dtype=np.int64).reshape(-1, 1)
        np.testing.assert_array_equal(gt, expected)

    def test_output_shape(self):
        """Output shape should be (Q, top_k)."""
        rng = np.random.default_rng(1)
        db = rng.standard_normal((30, 16)).astype(np.float32)
        q = rng.standard_normal((5, 16)).astype(np.float32)
        gt = compute_ground_truth(q, db, top_k=10)
        assert gt.shape == (5, 10)

    def test_known_nearest_neighbor(self):
        """Injected duplicate: query equals db[3], so top-1 must be 3."""
        rng = np.random.default_rng(2)
        db = rng.standard_normal((10, 4)).astype(np.float32)
        query = db[3:4].copy()
        gt = compute_ground_truth(query, db, top_k=1)
        assert int(gt[0, 0]) == 3
