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
├── results/              (52 CSVs)
├── scripts/
│   ├── analysis/
│   │   ├── verify_signbit_analysis.py
│   │   └── verify_calibration.py
│   ├── reproduce_tab1_signbit_methods.py
│   ├── reproduce_tab2_main.py
│   ├── reproduce_tab2_rerank.py
│   ├── reproduce_tab3_mech_extra.py
│   ├── reproduce_tab4_multibit.py
│   ├── reproduce_map_ndcg.py
│   ├── reproduce_table3_pq_sweep.py
│   ├── reproduce_ablation_rerank.py
│   ├── reproduce_mechanism_controls.py
│   ├── reproduce_mechanism_additional_controls.py
│   ├── reproduce_gapcal_comparison.py
│   ├── reproduce_gap_energy.py
│   ├── reproduce_audiocaps.py
│   ├── reproduce_clotho.py
│   ├── reproduce_laion400m.py
│   ├── reproduce_fig3_analysis_bcd.py
│   ├── reproduce_figure_c.py
│   ├── reproduce_qps_pareto.py
│   └── generate_figure.py
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
| `reproduce_tab1_signbit_methods.py` | Table 1 (`tab:signbit_methods`) — BQ methods on MSCOCO and AudioCaps |
| `reproduce_tab2_main.py` | Table 2 (`tab:mainresults`) — main PMC results, No-reranking columns |
| `reproduce_tab2_rerank.py` | Table 2 (`tab:mainresults`) — main PMC results, With-reranking columns (K'=400) |
| `reproduce_ablation_rerank.py` | LAION-400M K'-sweep reranking ablation (repo-only) |
| `analysis/verify_signbit_analysis.py` | Table 3 (`tab:mechanism`) — sign-bit Flip% and J@100 metrics |
| `analysis/verify_calibration.py` | Table 3 (`tab:mechanism`) — calibration cosine (cos@25); backs the calibration prose |
| `reproduce_tab3_mech_extra.py` | Table 4 (`tab:mech_extra`) — component ablation and IVF-RaBitQ controls (filename keeps legacy `tab3` prefix) |
| `reproduce_mechanism_controls.py` | Source CSVs for Tables 3–4 (bit-flip, exact control, component ablation, calibration sensitivity) |
| `reproduce_mechanism_additional_controls.py` | Table 4 additional IVF-RaBitQ controls |
| `reproduce_gapcal_comparison.py` | Centroid-alignment strategy comparison (validates the DB-side build-time choice) |
| `reproduce_tab4_multibit.py` | Table 5 (`tab:multibit`) — multi-bit generality aggregator (filename keeps legacy `tab4` prefix) |
| `reproduce_map_ndcg.py` | mAP/nDCG ranking-quality sweep across the six small-corpus rows; exploratory, not tied to a paper element |
| `reproduce_table3_pq_sweep.py` | IVFPQ/OPQ alpha sweep (produces the PQ CSVs feeding Table 5 and Fig. 3a) |
| `reproduce_audiocaps.py` | Table 2 AudioCaps rows |
| `reproduce_clotho.py` | Table 2 Clotho rows |
| `reproduce_laion400m.py` | Table 2 LAION-400M large-scale row |
| `reproduce_gap_energy.py` | Method-section gap-energy concentration claim |
| `reproduce_fig3_analysis_bcd.py` | `fig:analysis-bcd` — alpha sweep, selective PMC, and QPS Pareto panels |
| `reproduce_figure_c.py` | Analysis source for selective PMC curve (`selective_pmc_rabitq.csv`) |
| `reproduce_qps_pareto.py` | Analysis source for QPS Pareto curve (`pmc_qps_pareto_clip_mscoco_seed42.csv`) |
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
