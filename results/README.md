# results/ — committed measurement record

Canonical source of truth for every number in the paper (see
`docs/PAPER_RESULT_PROVENANCE.md` for the element-by-element map).
All CSVs are committed; scripts regenerate them deterministically.

| Folder | Contents | Written by |
|---|---|---|
| `tables/` | Per-table reproduce outputs (`tab1/2/5/6…_reproduced`, `rerank_deployable_reproduced`, aggregator cross-checks, K'-sweep ablation artifacts). One file ↔ one printed table (or repo-only ablation). | `scripts/tables/reproduce_*.py` |
| `figures/` | Figure-panel data: α-sweep (Fig 3a), selective PMC (Fig 3b), QPS Pareto (Fig 3c), and the Fig 3 reproduce output. | `scripts/figures/*` |
| `sources/` | Shared raw measurements that back multiple elements — e.g. `nprobe_sweep_pivot.csv` feeds Tables 2 **and** 3, `signbit_original_gt.csv` feeds Tables 1 **and** 4, the LAION-400M runs feed Table 2 and §4.3. Never edit by hand. | `scripts/builders/*` (FAISS-bound, offline) |
| `diagnostics/` | Verification and mechanism-control derivations: pool-coverage / probe-budget diagnoses, prose-claim audits, sign-bit verification, mechanism control CSVs. | `scripts/analysis/*` |
| `legacy/` | Superseded measurements nothing reads (see `legacy/README.md`). | — (historical) |

Reproduce loop (CPU-only, reads committed CSVs):

```bash
python3 scripts/tables/reproduce_tab2_main.py     # etc. — full list in docs/PAPER_RESULT_PROVENANCE.md
```

A `# REPRODUCE-MISMATCH` line means a cell drifted from its committed
expectation. Deterministic metrics must match exactly; QPS is tolerance-only.
