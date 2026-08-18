# scripts/ — one script per paper element

Folder mirrors `results/`: a script in `tables/` writes to `results/tables/`,
and so on. The full element → script → CSV map lives in
`docs/PAPER_RESULT_PROVENANCE.md`.

| Folder | Role | Needs |
|---|---|---|
| `tables/` | `reproduce_tab*.py` — re-derive every printed table cell from the committed CSVs in `results/sources/` and write `results/tables/*_reproduced*.csv`. Print `# REPRODUCE-MISMATCH` on drift. | committed CSVs only (CPU, no FAISS) |
| `figures/` | Reproduce/emit the data behind Figures 1 and 3 (t-SNE, α-sweep, selective PMC, QPS Pareto). | mostly committed CSVs; `reproduce_figure1_tsne.py` needs `data/features/` |
| `builders/` | FAISS-bound emitters that produced the committed measurements (`emit_*`), the LAION ground-truth builder, and feature-bound rebuilds (`reproduce_gap_energy.py`, LAION / AudioCaps / Clotho re-evals). Run offline; never required for the cached reproduce loop. | `data/features/`, `laion_dir` (`config/paths.yaml`; `PMC_LAION_DIR` / `LAION400M_DIR` env) |
| `analysis/` | Verification lane: mechanism/control derivations, calibration and sign-bit verifiers, pool-coverage / probe-budget diagnoses → `results/diagnostics/`. | committed CSVs (a few FAISS-bound, marked in the provenance doc) |
| `data_prep/` | Raw dataset → `data/features/` caches (downloads + frozen-encoder extraction). | network + encoders |

Every script locates the repo root by scanning parent directories, so they run
from any CWD: `python3 scripts/tables/reproduce_tab2_main.py`.
