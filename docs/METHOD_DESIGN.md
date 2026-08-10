# PMC: Per-Modality Centroid Correction

## Problem

IVF-based quantized indexes learn centroids and compressed representations from the database distribution. When queries come from a different modality (e.g., text->image), the modality gap shifts the query distribution away from those construction-time statistics.

For IVFPQ this appears as IVF routing and residual/codebook mismatch; for OPQ the learned rotation remains tied to the original database distribution; for RaBitQ the problem is sharpest because fixed sign() codes can flip under small centroid displacement. Standard meanshift (query-side only) can therefore hurt because the index was still built on the uncorrected database distribution.

## Core Idea

Correct the modality gap BEFORE index construction by shifting both DB and query vectors toward a common centroid.

## Algorithm

```
gap = mean(query_emb) - mean(db_emb)

# At index build time:
db_shifted = L2_normalize(db_emb + alpha * gap)
index = BuildIndex(db_shifted)  # RaBitQ, IVFPQ, or OPQ

# At query time:
q_shifted = L2_normalize(query_emb - (1 - alpha) * gap)
results = index.search(q_shifted)
```

### Alpha parameter

- alpha = 0.0: No DB shift, full query shift (equivalent to vanilla meanshift)
- alpha = 0.5: Both meet at midpoint (default)
- alpha = 1.0: Full DB shift, no query shift
- alpha = 0.5 is the symmetric choice; we sweep [0.25, 0.50, 0.75, 1.00]

### Why L2 normalization after shift?

The evaluated embeddings and indexes assume unit-norm vectors. After adding/subtracting the gap vector, norms change. Re-normalization is mandatory.

## Ground Truth

Ground truth is always computed on ORIGINAL (unshifted) L2-normalized vectors. The PMC correction only changes the index construction and search procedure, not the definition of "true nearest neighbor."

## Evaluation Grid

| Method | Description | alpha |
|--------|-------------|-------|
| vanilla_rabitq | Standard RaBitQ | N/A |
| vanilla_rabitq_meanshift | RaBitQ + query-side meanshift | N/A |
| pmc_0.25 | PMC alpha=0.25 | 0.25 |
| pmc_0.50 | PMC alpha=0.50 | 0.50 |
| pmc_0.75 | PMC alpha=0.75 | 0.75 |
| pmc_1.00 | PMC alpha=1.00 | 1.00 |
| ivfpq_meanshift_64B | IVFPQ+meanshift @64B/vec | N/A |

## Success Criterion

Gate 1: PMC best-alpha R@10 >= vanilla index R@10 + 0.03 on text->image (aligned with ABOUT.md)
