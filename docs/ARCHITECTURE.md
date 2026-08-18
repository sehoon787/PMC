# Architecture

## Status

`final/` is the clean standalone reproduction package for the CIKM 2026 short paper.
It contains only the code, scripts, results, and paper source needed to reproduce all
reported numbers. Historical research scripts and compatibility layers from
`current/pmc_crossmodal/` are not present here.

## Project Layout

```text
final/
├── config/
│   └── paths.yaml
├── data/
│   └── features → symlink to external feature cache
├── docs/
├── paper/
│   ├── main.tex
│   ├── refs.bib
│   ├── sections/
│   │   ├── abstract.tex
│   │   ├── 01_introduction.tex
│   │   ├── 01b_related.tex
│   │   ├── 02_method.tex
│   │   ├── 03_experiments.tex
│   │   └── 04_conclusion.tex
│   └── figures/
│       ├── fig_gap_vs_gain.pdf
│       ├── fig_analysis_bcd.pdf
│       └── fig_combined_1x4.pdf
├── results/                       # committed measurement record (see results/README.md)
│   ├── tables/       (7 CSVs)    # per-table reproduce outputs
│   ├── figures/      (5 CSVs)    # figure-panel data
│   ├── sources/     (24 CSVs)    # shared raw measurements (never edited by hand)
│   ├── diagnostics/  (9 CSVs)    # verification / mechanism-control derivations
│   └── legacy/       (7 CSVs)    # superseded, unread (results/legacy/README.md)
├── scripts/                       # folder mirrors results/ (see scripts/README.md)
│   ├── tables/                    # reproduce_tab*.py — re-derive every printed table cell
│   ├── figures/                   # figure-data reproduce/emit scripts
│   ├── builders/                  # FAISS/feature-bound emitters (emit_*, LAION GT, gap_energy)
│   ├── analysis/                  # verifiers + mechanism/control derivations
│   └── data_prep/                 # dataset download + feature extraction (needs raw data)
├── src/
│   ├── core/
│   │   ├── pmc.py
│   │   ├── metrics.py
│   │   └── index_wrappers.py
│   ├── datasets/
│   │   ├── mscoco.py
│   │   ├── flickr30k.py
│   │   ├── audiocaps.py
│   │   └── clotho.py
│   ├── features/
│   │   ├── loader.py
│   │   ├── cache.py
│   │   └── jobs.py
│   ├── fixtures/
│   │   └── synthetic.py
│   ├── io/
│   │   └── bigann.py
│   └── utils.py
├── tests/
│   ├── test_pmc.py
│   ├── test_metrics.py
│   ├── test_ranking_metrics.py
│   └── test_utils.py
└── requirements.txt
```

## Source Packages

| Package | Responsibility |
|---|---|
| `src/core/` | PMC transforms (`pmc.py`), recall and ranking metric functions (`metrics.py`: recall@k, mAP@k, nDCG@k), and ANN index wrappers (`index_wrappers.py`). |
| `src/datasets/` | MSCOCO, Flickr30K, AudioCaps, and Clotho loaders and download helpers. |
| `src/features/` | Feature cache format (`cache.py`), feature loader (`loader.py`), and extraction job orchestration (`jobs.py`). |
| `src/fixtures/` | Synthetic test fixtures for unit tests (no external datasets required). |
| `src/io/` | Binary vector file readers (BIGANN format). |
| `src/utils.py` | Shared utility functions. |

## Script Families

All reproduction scripts live in `final/scripts/`. Each script corresponds to one or more paper elements:

| Script | Paper Element |
|---|---|
| `tables/reproduce_tab1_signbit_methods.py` | Table 1 (`tab:signbit_methods`) — BQ methods on MSCOCO and AudioCaps |
| `tables/reproduce_tab2_main.py` | Table 2 (`tab:mainresults`) — main PMC results, No-reranking columns |
| `tables/reproduce_tab2_rerank.py` | Table 2 (`tab:mainresults`) — main PMC results, With-reranking columns (K'=400) |
| `tables/reproduce_ablation_rerank.py` | LAION-400M K'-sweep reranking ablation (repo-only) |
| `analysis/verify_signbit_analysis.py` | Table 4 (`tab:mechanism`) — sign-bit Flip% and J@100 metrics |
| `analysis/verify_calibration.py` | Table 4 (`tab:mechanism`) — calibration cosine (cos@25); backs the calibration prose |
| `tables/reproduce_tab5_mech_extra.py` | Table 5 (`tab:mech_extra`) — component ablation and IVF-RaBitQ controls (filename keeps legacy `tab3` prefix) |
| `analysis/reproduce_mechanism_controls.py` | Source CSVs for Tables 3–4 (bit-flip, exact control, component ablation, calibration sensitivity) |
| `analysis/reproduce_mechanism_additional_controls.py` | Table 5 additional IVF-RaBitQ controls |
| `analysis/reproduce_gapcal_comparison.py` | Centroid-alignment strategy comparison (validates the DB-side build-time choice) |
| `tables/reproduce_tab6_multibit.py` | Table 6 (`tab:multibit`) — multi-bit generality aggregator (filename keeps legacy `tab4` prefix) |
| `data_prep/*.py` | Dataset download and feature extraction: AudioCaps/Clotho download, CLIP and ImageBind extraction. Produces `data/features/`, which every FAISS-bound script consumes. |
| `builders/emit_map_ndcg.py` | mAP/nDCG ranking-quality sweep across the six small-corpus rows; FAISS-bound, requires `data/features/`; exploratory, not tied to a paper element |
| `tables/reproduce_table3_pq_sweep.py` | IVFPQ/OPQ alpha sweep (produces the PQ CSVs feeding Table 6 and Fig. 3a) |
| `builders/reproduce_audiocaps.py` | Table 2 AudioCaps rows |
| `builders/reproduce_clotho.py` | Table 2 Clotho rows |
| `builders/reproduce_laion400m.py` | Table 2 LAION-400M large-scale row |
| `builders/reproduce_gap_energy.py` | Method-section gap-energy concentration claim |
| `figures/reproduce_fig3_analysis_bcd.py` | `fig:analysis-bcd` — alpha sweep, selective PMC, and QPS Pareto panels |
| `figures/reproduce_figure_c.py` | Analysis source for selective PMC curve (`selective_pmc_rabitq.csv`) |
| `figures/reproduce_qps_pareto.py` | Analysis source for QPS Pareto curve (`pmc_qps_pareto_clip_mscoco_seed42.csv`) |
| `paper/figures/fig3_analysis.py` | Renders split figure assets (`fig_gap_vs_gain`, `fig_analysis_bcd`) and legacy `fig_combined_1x4` |

## Runtime Paths

Default paths are controlled by `config/paths.yaml`. Local machine overrides can be
placed in `config/paths.local.yaml` or provided through environment variables such as
`PMC_RESULTS_DIR` and `PMC_FEATURES_DIR`.

This is the required path mechanism for Mac-local replay and external drives.
Hard-coded absolute paths should not be introduced.

## Verification Contract

The architecture is considered valid only when:

- `python -m py_compile` passes across `src/`, `scripts/`, and `tests/`.
- `pytest -q tests/` passes (synthetic fixtures, no GPU or external data required).
- Canonical reproduce scripts write the same CSV schemas and row counts as committed
  result files in `results/`.
- Deterministic recall/count metrics are exact or within documented tolerance.
- QPS/latency values are treated as volatile and compared by tolerance only.
