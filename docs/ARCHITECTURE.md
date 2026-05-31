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
├── results/              (13 CSVs)
├── scripts/
│   ├── reproduce_table1_aggregator.py
│   ├── reproduce_table2_signbit.py
│   ├── reproduce_table3_pq_sweep.py
│   ├── reproduce_table3_multibit.py
│   ├── reproduce_audiocaps.py
│   ├── reproduce_clotho.py
│   ├── reproduce_laion400m.py
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
│   └── test_utils.py
└── requirements.txt
```

## Source Packages

| Package | Responsibility |
|---|---|
| `src/core/` | PMC transforms (`pmc.py`), recall metric functions (`metrics.py`), and ANN index wrappers (`index_wrappers.py`). |
| `src/datasets/` | MSCOCO, Flickr30K, AudioCaps, and Clotho loaders and download helpers. |
| `src/features/` | Feature cache format (`cache.py`), feature loader (`loader.py`), and extraction job orchestration (`jobs.py`). |
| `src/fixtures/` | Synthetic test fixtures for unit tests (no external datasets required). |
| `src/io/` | Binary vector file readers (BIGANN format). |
| `src/utils.py` | Shared utility functions. |

## Script Families

All reproduction scripts live in `final/scripts/`. Each script corresponds to one or more paper elements:

| Script | Paper Element |
|---|---|
| `reproduce_table1_aggregator.py` | Table 1 — main PMC results (IVFRaBitQFastScan across all encoder–dataset pairs) |
| `reproduce_table2_signbit.py` | Table 2 — sign-bit methods on MSCOCO and AudioCaps |
| `reproduce_table3_pq_sweep.py` | Table 3 — IVFPQ/OPQ alpha sweep (produces PQ CSVs) |
| `reproduce_table3_multibit.py` | Table 3 — multi-bit generality aggregator (reads PQ CSVs) |
| `reproduce_audiocaps.py` | Table 1 AudioCaps rows |
| `reproduce_clotho.py` | Table 1 Clotho rows |
| `reproduce_laion400m.py` | Table 1 LAION-400M large-scale row |
| `reproduce_figure_c.py` | Analysis source for selective PMC curve (`selective_pmc_rabitq.csv`) |
| `reproduce_qps_pareto.py` | Analysis source for QPS Pareto curve (`pmc_qps_pareto_clip_mscoco_seed42.csv`) |
| `paper/figures/fig_combined_1x4.py` | Renders split figure assets (`fig_gap_vs_gain`, `fig_analysis_bcd`) and legacy `fig_combined_1x4` |

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
