"""Unit tests for src.utils: l2_normalize, synthetic_dataset,
compute_modality_means, apply_meanshift."""

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

from src.utils import (
    l2_normalize,
    synthetic_dataset,
    compute_modality_means,
    apply_meanshift,
)


# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------

class TestL2Normalize:
    """Tests for l2_normalize."""

    def test_output_rows_have_unit_norm(self):
        """All rows of the output should have L2 norm == 1."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((50, 32)).astype(np.float32)
        out = l2_normalize(x)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_zero_vector_no_division_by_zero(self):
        """Zero vector should not raise; result should be finite (all zeros)."""
        x = np.zeros((3, 4), dtype=np.float32)
        out = l2_normalize(x)
        assert np.all(np.isfinite(out))

    def test_output_dtype_is_float32(self):
        """Output should be float32 regardless of input dtype."""
        x = np.random.default_rng(1).standard_normal((10, 8))  # float64
        out = l2_normalize(x.astype(np.float64))
        assert out.dtype == np.float32

    def test_already_normalized_unchanged(self):
        """Already-unit-norm rows should pass through unchanged."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal((20, 16)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        out = l2_normalize(x)
        np.testing.assert_allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# synthetic_dataset
# ---------------------------------------------------------------------------

class TestSyntheticDataset:
    """Tests for synthetic_dataset."""

    def test_returns_correct_shapes(self):
        """Should return two arrays each of shape (n, d)."""
        n, d = 100, 32
        t, v = synthetic_dataset(n=n, d=d, seed=0)
        assert t.shape == (n, d)
        assert v.shape == (n, d)

    def test_deterministic_with_same_seed(self):
        """Two calls with the same seed must return identical arrays."""
        t1, v1 = synthetic_dataset(n=50, d=16, seed=7)
        t2, v2 = synthetic_dataset(n=50, d=16, seed=7)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(v1, v2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different arrays."""
        t1, _ = synthetic_dataset(n=50, d=16, seed=1)
        t2, _ = synthetic_dataset(n=50, d=16, seed=2)
        assert not np.allclose(t1, t2)

    def test_output_dtype_is_float32(self):
        """Both returned arrays should be float32."""
        t, v = synthetic_dataset(n=20, d=8, seed=0)
        assert t.dtype == np.float32
        assert v.dtype == np.float32


# ---------------------------------------------------------------------------
# compute_modality_means
# ---------------------------------------------------------------------------

class TestComputeModalityMeans:
    """Tests for compute_modality_means."""

    def test_correct_mean_computation(self):
        """Mean of each modality should match np.mean(emb, axis=0)."""
        rng = np.random.default_rng(0)
        img = rng.standard_normal((30, 8)).astype(np.float32)
        txt = rng.standard_normal((40, 8)).astype(np.float32)
        means = compute_modality_means({"image": img, "text": txt})
        np.testing.assert_allclose(means["image"], img.mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(means["text"], txt.mean(axis=0), atol=1e-5)

    def test_output_dtype_is_float32(self):
        """Returned means should be float32."""
        rng = np.random.default_rng(1)
        emb = rng.standard_normal((10, 4)).astype(np.float32)
        means = compute_modality_means({"mod": emb})
        assert means["mod"].dtype == np.float32

    def test_single_vector_mean_equals_itself(self):
        """Mean of a single-row embedding equals that row."""
        emb = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        means = compute_modality_means({"m": emb})
        np.testing.assert_allclose(means["m"], emb[0], atol=1e-6)


# ---------------------------------------------------------------------------
# apply_meanshift
# ---------------------------------------------------------------------------

class TestApplyMeanshift:
    """Tests for apply_meanshift."""

    def test_correct_shift_direction(self):
        """Result should equal query - mean(query_mod) + mean(db_mod)."""
        rng = np.random.default_rng(0)
        q = rng.standard_normal((10, 8)).astype(np.float32)
        img_mean = rng.standard_normal(8).astype(np.float32)
        txt_mean = rng.standard_normal(8).astype(np.float32)
        db_means = {"image": img_mean, "text": txt_mean}

        shifted = apply_meanshift(q, db_means, query_modality="text", db_modality="image")
        expected = (q - txt_mean + img_mean).astype(np.float32)
        np.testing.assert_allclose(shifted, expected, atol=1e-5)

    def test_same_modality_shift_is_identity(self):
        """When query and db share the same modality, shift is zero -> no change."""
        rng = np.random.default_rng(1)
        q = rng.standard_normal((5, 4)).astype(np.float32)
        mean = rng.standard_normal(4).astype(np.float32)
        db_means = {"mod": mean}

        shifted = apply_meanshift(q, db_means, query_modality="mod", db_modality="mod")
        np.testing.assert_allclose(shifted, q, atol=1e-5)

    def test_output_dtype_is_float32(self):
        """Output should be float32."""
        q = np.ones((3, 4), dtype=np.float32)
        db_means = {"a": np.zeros(4, dtype=np.float32), "b": np.ones(4, dtype=np.float32)}
        shifted = apply_meanshift(q, db_means, query_modality="a", db_modality="b")
        assert shifted.dtype == np.float32
