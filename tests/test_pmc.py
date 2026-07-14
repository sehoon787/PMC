"""Unit tests for PMC (Per-Modality Centroid) correction."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Path setup: add v4 root so "src.pmc" resolves to v4's src
# ---------------------------------------------------------------------------
_V4_ROOT = Path(__file__).resolve().parent.parent
if str(_V4_ROOT) not in sys.path:
    sys.path.insert(0, str(_V4_ROOT))

from src.core.pmc import (
    compute_gap,
    shift_db_vectors,
    shift_query_vectors,
    build_pmc_rabitq_index,
    search_pmc,
    _l2_normalize,
)


# ---------------------------------------------------------------------------
# Self-contained helpers (avoid importing pareto_calibration's src package
# which would collide with v4's src package in sys.modules).
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids, gt_ids, k):
    """Compute Recall@K averaged over all queries."""
    assert retrieved_ids.ndim == 2
    assert len(retrieved_ids) == len(gt_ids)
    total = 0.0
    q = len(retrieved_ids)
    for i in range(q):
        topk_set = set(int(x) for x in retrieved_ids[i, :k] if x >= 0)
        if gt_ids.ndim == 1:
            gt_set = {int(gt_ids[i])}
        elif hasattr(gt_ids[i], '__iter__'):
            gt_set = set(int(x) for x in gt_ids[i] if x >= 0)
        else:
            gt_set = {int(gt_ids[i])}
        if len(gt_set) == 0:
            continue
        total += len(topk_set & gt_set) / len(gt_set)
    return total / q if q > 0 else 0.0


def compute_ground_truth(queries, db, top_k=100):
    """Brute force exact NN using faiss."""
    import faiss
    d = db.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(db.astype(np.float32))
    _, indices = index.search(queries.astype(np.float32), top_k)
    return indices


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeGap:
    """test_compute_gap: Gap direction is correct."""

    def test_gap_direction(self, synth_data):
        gap = compute_gap(synth_data["image_emb"], synth_data["text_emb"])
        assert gap.shape == (synth_data["d"],)
        assert gap.dtype == np.float32

        # Gap should point roughly from image mean toward text mean
        image_mean = synth_data["image_emb"].mean(axis=0)
        text_mean = synth_data["text_emb"].mean(axis=0)
        expected = text_mean - image_mean
        cos_sim = np.dot(gap, expected) / (np.linalg.norm(gap) * np.linalg.norm(expected) + 1e-12)
        assert cos_sim > 0.99, f"Gap direction incorrect: cos_sim={cos_sim:.4f}"

    def test_gap_zero_for_same_modality(self, synth_data):
        gap = compute_gap(synth_data["image_emb"], synth_data["image_emb"])
        assert np.allclose(gap, 0.0, atol=1e-6)


class TestShiftRoundtrip:
    """test_shift_roundtrip: shift_db + shift_query with alpha=0.5
    brings means to the same point."""

    def test_means_converge_at_alpha_half(self, synth_data):
        db = synth_data["image_emb"]
        q = synth_data["text_emb"]
        gap = compute_gap(db, q)

        db_shifted = shift_db_vectors(db, gap, alpha=0.5)
        q_shifted = shift_query_vectors(q, gap, alpha=0.5)

        db_mean = db_shifted.mean(axis=0)
        q_mean = q_shifted.mean(axis=0)

        db_mean_n = db_mean / (np.linalg.norm(db_mean) + 1e-12)
        q_mean_n = q_mean / (np.linalg.norm(q_mean) + 1e-12)

        cos_sim = np.dot(db_mean_n, q_mean_n)
        assert cos_sim > 0.95, f"Shifted means not aligned: cos_sim={cos_sim:.4f}"

    def test_shifted_vectors_are_unit_norm(self, synth_data):
        db = synth_data["image_emb"]
        q = synth_data["text_emb"]
        gap = compute_gap(db, q)

        db_shifted = shift_db_vectors(db, gap, alpha=0.5)
        q_shifted = shift_query_vectors(q, gap, alpha=0.5)

        db_norms = np.linalg.norm(db_shifted, axis=1)
        q_norms = np.linalg.norm(q_shifted, axis=1)

        np.testing.assert_allclose(db_norms, 1.0, atol=1e-5)
        np.testing.assert_allclose(q_norms, 1.0, atol=1e-5)


class TestPMCImprovesCrossModal:
    """test_pmc_improves_cross_modal: On synthetic data with injected modality
    gap, PMC recall >= vanilla recall."""

    def test_pmc_recall_ge_vanilla(self, synth_data):
        import faiss

        db = synth_data["image_emb"]
        q = synth_data["text_emb"]
        d = synth_data["d"]

        # Ground truth: exact NN on ORIGINAL vectors
        gt = compute_ground_truth(q, db, top_k=100)

        # Vanilla RaBitQ (direct faiss, no PC wrapper needed)
        nlist = 8
        quantizer = faiss.IndexFlatL2(d)
        vanilla_raw = faiss.IndexIVFRaBitQFastScan(quantizer, d, nlist, 0)
        vanilla_raw.cp.seed = 42
        vanilla_raw.cp.min_points_per_centroid = 1
        vanilla_raw.train(db)
        vanilla_raw.add(db)
        vanilla_raw.nprobe = nlist
        _, vanilla_ids = vanilla_raw.search(q, 100)
        vanilla_r10 = recall_at_k(vanilla_ids, gt, k=10)

        # PMC RaBitQ
        pmc_idx, gap = build_pmc_rabitq_index(
            db, q, alpha=0.5, nlist=nlist, seed=42, use_fastscan=True,
        )
        _, pmc_ids = search_pmc(pmc_idx, q, gap, alpha=0.5, top_k=100, nprobe=nlist)
        pmc_r10 = recall_at_k(pmc_ids, gt, k=10)

        print(f"  vanilla R@10={vanilla_r10:.4f}  PMC R@10={pmc_r10:.4f}")
        # PMC should be at least as good (allow small noise margin)
        assert pmc_r10 >= vanilla_r10 - 0.02, (
            f"PMC recall ({pmc_r10:.4f}) significantly worse than vanilla ({vanilla_r10:.4f})"
        )


class TestPMCIdentityForSameModality:
    """test_pmc_identity_for_same_modality: When query_mod == db_mod,
    PMC has no effect."""

    def test_no_effect_same_modality(self, synth_data):
        db = synth_data["image_emb"]
        gap = compute_gap(db, db)
        assert np.allclose(gap, 0.0, atol=1e-6)

        db_shifted = shift_db_vectors(db, gap, alpha=0.5)
        np.testing.assert_allclose(db_shifted, db, atol=1e-5)


class TestAlphaBounds:
    """test_alpha_bounds: alpha=0 gives vanilla, alpha=1 fully aligns."""

    def test_alpha_zero_is_vanilla(self, synth_data):
        db = synth_data["image_emb"]
        q = synth_data["text_emb"]
        gap = compute_gap(db, q)

        # alpha=0 => db_shifted = db + 0*gap = db (after renorm, same as original)
        db_shifted = shift_db_vectors(db, gap, alpha=0.0)
        np.testing.assert_allclose(db_shifted, _l2_normalize(db), atol=1e-5)

        # alpha=0 => q_shifted = q - 1*gap
        q_shifted = shift_query_vectors(q, gap, alpha=0.0)
        q_manual = _l2_normalize(q - gap[np.newaxis, :])
        np.testing.assert_allclose(q_shifted, q_manual, atol=1e-5)

    def test_alpha_one_fully_aligns_db(self, synth_data):
        db = synth_data["image_emb"]
        q = synth_data["text_emb"]
        gap = compute_gap(db, q)

        # alpha=1 => db_shifted = db + gap (fully aligned to query center)
        db_shifted = shift_db_vectors(db, gap, alpha=1.0)
        db_manual = _l2_normalize(db + gap[np.newaxis, :])
        np.testing.assert_allclose(db_shifted, db_manual, atol=1e-5)

        # alpha=1 => q_shifted = q - 0*gap = q (queries unchanged)
        q_shifted = shift_query_vectors(q, gap, alpha=1.0)
        np.testing.assert_allclose(q_shifted, _l2_normalize(q), atol=1e-5)
