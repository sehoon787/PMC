# Architecture

## Status

This repository is the clean standalone reproduction package for the CIKM 2026 short
paper *PMC: Build-Time Per-Modality Centroid Correction for Cross-Modal Binary-Quantized
Retrieval*. The repository root **is** the package: it contains only the code, scripts,
results, and paper source needed to reproduce all reported numbers. Historical research
scripts and compatibility layers from the development tree are not included.

## Project Layout

```text
PMC/
├── config/
│   └── paths.yaml
├── data/
│   └── features/           # external feature cache (not committed)
├── docs/
│   ├── ARCHITECTURE.md
│   └── METHOD_DESIGN.md
├── paper/
│   ├── main.tex            # acmart sigconf, anonymous=true
│   ├── main.pdf
│   ├── refs.bib
│   ├── sections/
│   │   ├── abstract.tex
│   │   ├── 01_introduction.tex
│   │   ├── 01b_related.tex
│   │   ├── 02_method.tex
│   │   ├── 03_experiments.tex
│   │   └── 04_conclusion.tex
│   └── figures/            # figure source (.py) and outputs (.pdf, .png)
├── results/                # paper-critical CSV outputs (32 files, committed)
├── scripts/                # reproduction scripts (one per paper element)
├── src/
│   ├── core/
│   │   ├── pmc.py
│   │   ├── metrics.py
│   │   └── index_wrappers.py
│   ├── datasets/
│   │   ├── mscoco.py
│   │   ├── flickr30k.py
│   │   ├── audiocaps.py
│   │   ├── clotho.py
│   │   ├── downloads.py
│   │   └── items.py
│   ├── encoders/
│   │   ├── clip.py
│   │   ├── imagebind.py
│   │   ├── clap.py
│   │   └── fake.py
│   ├── experiments/
│   │   ├── baselines.py
│   │   ├── pq.py
│   │   ├── opq_ablation.py
│   │   ├── paired_recall_eval.py
│   │   └── sweeps.py
│   ├── features/
│   │   ├── loader.py
│   │   ├── cache.py
│   │   └── jobs.py
│   ├── fixtures/
│   │   └── synthetic.py
│   ├── io/
│   │   └── bigann.py
│   ├── runtime/
│   │   └── config.py
│   └── utils.py
├── tests/
│   ├── conftest.py
│   ├── test_pmc.py
│   ├── test_metrics.py
│   └── test_utils.py
└── requirements.txt
```

## Source Packages

| Package | Responsibility |
|---|---|
| `src/core/` | PMC transforms (`pmc.py`), recall metric functions (`metrics.py`), and ANN index wrappers (`index_wrappers.py`). |
| `src/datasets/` | MSCOCO, Flickr30K, AudioCaps, and Clotho loaders, item records, and download helpers. |
| `src/encoders/` | CLIP, ImageBind, and CLAP feature encoders, plus a `fake` encoder for tests. |
| `src/experiments/` | Experiment drivers: baselines, IVFPQ/OPQ ablations, paired recall evaluation, and alpha sweeps. |
| `src/features/` | Feature cache format (`cache.py`), feature loader (`loader.py`), and extraction job orchestration (`jobs.py`). |
| `src/fixtures/` | Synthetic test fixtures for unit tests (no external datasets required). |
| `src/io/` | Binary vector file readers (BIGANN format) for LAION-400M scale data. |
| `src/runtime/` | Runtime/path configuration resolution. |
| `src/utils.py` | Shared utility functions. |

## Script Families

All reproduction scripts live in `scripts/`. The canonical script-to-paper-element
mapping is maintained in the README "Reproduction" table; the current paper has five
tables (Table 1: sign-bit methods, Table 2: main PMC results, Table 3: mechanism checks,
Table 4: additional IVF-RaBitQ controls, Table 5: multi-bit generality).

| Script | Paper Element |
|---|---|
| `reproduce_table1_signbit.py` | Table 1 — sign-bit methods (R@100, Vanilla/PMC) |
| `reproduce_table2_main_aggregator.py` | Table 2 — main PMC results (IVF-RaBitQFastScan) |
| `reproduce_laion400m.py` | Table 2 — LAION-400M large-scale row (`n_list=80K`, `n_probe=256`) |
| `reproduce_table3_pq_sweep.py` | PMC + PQ alpha sweep (feeds Table 5; Fig. 3a) |
| `reproduce_table5_multibit.py` | Table 5 — multi-bit IVFPQ/OPQ generality |
| `reproduce_mechanism_controls.py` | Tables 3-4 — bit-flip/J@100, exact control, component ablation, calibration sensitivity |
| `reproduce_mechanism_additional_controls.py` | Table 4 — additional IVF-RaBitQ controls |
| `reproduce_tab2_rerank.py` | Table 2 — "With reranking (K'=400)" column group (§4.3) |
| `reproduce_ablation_rerank.py` | §4.3 rerank K'-sweep / LAION-400M deployable-recall ablation |
| `reproduce_audiocaps.py` | AudioCaps audio retrieval (R@1) |
| `reproduce_clotho.py` | Clotho audio retrieval (R@1) |
| `reproduce_figure_c.py` | Selective PMC analysis curve (`selective_pmc_rabitq.csv`) |
| `reproduce_qps_pareto.py` | QPS vs Recall Pareto curve |
| `generate_figure.py` | Combined figure assets for the paper |

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
