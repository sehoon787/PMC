# Paper Result Provenance

Maps every result in the CIKM 2026 short paper to the script that reproduces it
and the committed CSV(s) that hold the numbers. **Committed CSVs in `results/`
(and the `final/results/` mirror) are the canonical source of truth.** Each
`reproduce_*` script reads only committed CSVs (no FAISS), re-derives the printed
values, and writes a `*_reproduced*.csv` for inspection. A `# REPRODUCE-MISMATCH`
line on any run means a cell drifted from its committed expectation.

> **Table numbering note.** The paper has **6 tables**, numbered in order of first
> citation. Main results are a **single combined table** `tab:mainresults` (in
> `sections/01b_related.tex`) presenting two operating points side by side: **No
> reranking** (production point $n_\mathrm{probe}{\approx}n_\mathrm{list}/4$) and
> **With reranking** ($n_\mathrm{probe}{\approx}n_\mathrm{list}$, $K'{=}400$,
> recall-tuned first stage). Each cell shows Vanilla / MeanShift / PMC; bold = best
> per cell. **Table 3** (`tab:r10`) reports R@10 at Table 2's no-reranking operating
> points. Reproduce-script and CSV filenames match their printed table number.

---

## Per-Section Code Map

What each result-bearing section claims and the exact scripts that verify it.
`reproduce_*.py` read committed CSVs (no FAISS) and re-derive the printed values;
`emit_*.py` are the FAISS-bound builders that produce those CSVs from raw features
(run offline). Paths are relative to `current/pmc_crossmodal/`.

### §3 Method
PMC corrects the modality gap before index construction; the ratio condition
predicts when sign-bit corruption is severe (top 10% of gap dimensions carry ≈90%
of ‖g‖²). Verified via:
* `scripts/reproduce_gap_energy.py`

### §4.1 Setup
Describes datasets (MSCOCO, Flickr30K, AudioCaps, Clotho, LAION-400M), index
parameters (IVFRaBitQFastScan, IVFPQ/OPQ, nlist, nprobe), and LAION-400M exact-IP
ground-truth construction. Configuration prose; no per-number reproduce script.

### §4.2 Main Results
PMC improves RaBitQ R@100 across nearly all configs (the No-reranking columns of
`tab:mainresults`) and improves/matches R@100 in all 16 BQ configs (Table 1,
`tab:signbit_methods`), including LAION-400M at scale. Sign-bit mechanism prose
(formerly §3.3) is also reported here: flip 14–18%, J@100 0.54–0.80, gap direction
stable from n=25 (cosine ≥ 0.95). Verified via:
* `scripts/reproduce_tab2_main.py`            — No-reranking columns of `tab:mainresults`
* `scripts/reproduce_tab1_signbit_methods.py` — Table 1 (`tab:signbit_methods`)
* `scripts/analysis/verify_signbit_analysis.py` — §4.2 sign-bit prose metrics → `results/signbit_analysis_verified.csv`
* `scripts/emit_signbit_metrics.py`           — upstream FAISS builder

### §4.3 Robustness under Exact Reranking
The With-reranking ($K'{=}400$) columns of `tab:mainresults`: R@100
Vanilla/MeanShift/PMC after exact top-$K'$ rerank with a **recall-tuned first
stage** ($n_\mathrm{probe}{\approx}n_\mathrm{list}$, ~90% of cells probed on
5K–31K corpora), across dataset×encoder rows. PMC dominates every cell at every
depth $K'\in\{100,200,400,500\}$. LAION forward shown at np256 (0.32% probed) for
scale; **LAION reverse np256 is shown in the table and Vanilla overtakes PMC
(.158 vs .140)** — a scan-depth effect: small corpora rerank near-exhaustively
(n_probe≈n_list) while LAION scans only 0.32% of its lists (n_probe=256), where
PMC's flatter candidate pool covers fewer true positives than Vanilla. Verified via:
* `scripts/reproduce_tab2_rerank.py`         — With-reranking columns of `tab:mainresults` / §4.3
* `scripts/reproduce_ablation_rerank.py`     — §4.3 LAION-400M K'-sweep ablation (repo-only)
* `scripts/analysis/29_diagnose_pool_coverage.py` — pool-coverage diagnostic across all Table-2 configs + LAION proxy rows → `results/pool_coverage_diagnostic_seed42.csv` (FAISS-bound; cov_gap<0 only for LAION reverse at probe_budget≈0.0032)
* `scripts/analysis/30_probe_budget_sweep.py` — decisive single-variable sweep on MSCOCO/CLIP fixed index (nlist=320), sweeping n_probe from probe_budget 0.0031→0.25 → `results/probe_budget_sweep_mscoco_seed42.csv` (FAISS-bound; cov_gap flips − to + as budget increases, confirming probe budget not modality concentration drives the reversal)

### §4.4 Ablation
Table 5 (`tab:mech_extra`): component ablation + IVF-RaBitQ controls. Calibration
PROSE: cosine ≈ 0.986; R@100 flat within .001 up to n_calib=400. DB-only build-time
correction outperforms Q-only and both-sides. Verified via:
* `scripts/reproduce_tab5_mech_extra.py`          — Table 5 (`tab:mech_extra`)
* `scripts/analysis/verify_calibration.py`        — verifies §4.4 calibration prose → `results/signbit_analysis_verified.csv`
* `scripts/emit_calibration_metrics.py`           — upstream FAISS builder
* `scripts/reproduce_gapcal_comparison.py`        — six centroid-alignment strategies; validates DB-only build-time choice

### §4.5 Multi-Bit Generality
Table 6 (`tab:multibit`): IVFPQ/OPQ R@100 Vanilla/PMC direction-averaged across
encoders and datasets. Verified via:
* `scripts/reproduce_tab6_multibit.py`

### Figures
`fig:modality-gap` (Fig 1) is data-driven: `paper/figures/fig1_tsne.py`, wrapped by
`scripts/reproduce_figure1_tsne.py`, t-SNEs ImageBind embeddings and therefore needs
`data/features/`. `fig:pmc-overview` (Fig 2) is schematic; no reproduce script.
`fig:analysis-bcd` (Fig 3) panels: α-sweep, selective PMC,
QPS Pareto. Verified via:
* `scripts/reproduce_fig3_analysis_bcd.py`
* `scripts/emit_fig_alpha_sweep.py`
* `paper/figures/fig3_analysis.py`

---

## Current paper table inventory

| Printed | Label | Appears in | Reproduce script | Source CSV(s) | Reproduced CSV |
|---|---|---|---|---|---|
| **Table 1** | `tab:signbit_methods` | `sections/01b_related.tex` | `scripts/reproduce_tab1_signbit_methods.py` | `results/signbit_original_gt.csv` | `results/tab1_signbit_methods_reproduced.csv` |
| **Table 2** | `tab:mainresults` | `sections/01b_related.tex` | `scripts/reproduce_tab2_main.py` (No-reranking columns) + `scripts/reproduce_tab2_rerank.py` (With-reranking columns, + `scripts/reproduce_ablation_rerank.py` for LAION K'-sweep) | No-reranking: `results/multiseed_rabitq_summary.csv`, `results/pmc_eval_clip-l_flickr30k_full_seed42.csv`, `results/pmc_clotho_r1_seed42.csv`, `results/pmc_laion400m_nlist80k_seed42.csv`, `results/pmc_laion400m_reverse_nlist80k_seed42.csv`; With-reranking: `results/rerank_subset_sqrtN_v2_seed42.csv`, `results/pmc_laion400m_rerank_nlist80k_seed42.csv`, `results/pmc_laion400m_reverse_rerank_nlist80k_seed42.csv` | `results/tab2_main_reproduced.csv` (No-reranking, incl. the `q_r10_*` / `db_r10_*` columns); `results/rerank_deployable_reproduced.csv` (With-reranking) |
| **Table 3** | `tab:r10` | `sections/03_experiments.tex` | `scripts/reproduce_tab2_main.py` (the `q_r10_*` / `db_r10_*` columns) | same sources as Table 2's No-reranking columns | `results/tab2_main_reproduced.csv` |
| **Table 4** | `tab:mechanism` | `sections/03_experiments.tex` | (no dedicated `reproduce_tab*.py`) `scripts/analysis/verify_signbit_analysis.py` + `scripts/analysis/verify_calibration.py` | `results/signbit_analysis_verified.csv`; BinaryFlat R@100 from `results/signbit_original_gt.csv` | `results/signbit_analysis_verified.csv` |
| **Table 5** | `tab:mech_extra` | `sections/03_experiments.tex` | `scripts/reproduce_tab5_mech_extra.py` | `results/mechanism_additional_controls.csv` (component ablation + IVF-RaBitQ controls); `results/mech_extra_calibration.csv` (calibration, **CSV retained**, panel removed from table) | `results/tab5_mech_extra_reproduced.csv` |
| **Table 6** | `tab:multibit` | `sections/03_experiments.tex` | `scripts/reproduce_tab6_multibit.py` | `results/rerank_multibit_seed42.csv`; OPQ CLIP/MSCOCO from `results/pmc_opq_multiseed_clip_mscoco.csv` | `results/tab6_multibit_reproduced__np16_k0.csv` (**per-direction**, referenced by caption "per-direction in repo") |

### Table 1 — `tab:signbit_methods`
BQ methods (PureBinary, BinaryIVF, RotatedBinary/BBQ-style, RaBitQ, PMC-RaBitQ),
R@100 Vanilla/PMC, on MSCOCO (CLIP-B/32) and AudioCaps (ImageBind). All rows use
`original_exact_ip` ground truth uniformly.

### Table 2 — `tab:mainresults`
Single combined table presenting two operating points side by side.
**No-reranking columns** use the production operating point
($n_\mathrm{probe}{\approx}n_\mathrm{list}/4$, no reranking), showing
Vanilla/MeanShift/PMC R@100 across MSCOCO (CLIP-B/32, CLIP-L/14, ImageBind),
Flickr30K-31K (CLIP-L/14), Clotho (ImageBind), AudioCaps (ImageBind), and
LAION-400M (CLIP-B/32). **With-reranking columns** use a recall-tuned first stage
($n_\mathrm{probe}{\approx}n_\mathrm{list}$, ~90% of cells probed on 5K–31K
corpora) with exact reranking at $K'{=}400$, showing the same Vanilla/MeanShift/PMC
breakdown. PMC dominates every cell at every depth $K'\in\{100,200,400,500\}$
across the small-to-mid corpora. LAION reverse np256 is shown in the With-reranking
columns and Vanilla overtakes PMC (.158 vs .140) — a scan-depth effect (probe_budget
≈0.0032 vs ≈0.25 for small corpora). Reproduced by `reproduce_tab2_main.py`
(No-reranking) and `reproduce_tab2_rerank.py` (With-reranking).

### Table 4 — `tab:mechanism`
Sign-bit mechanism / control checks at $\alpha{=}1$: per-direction sign-bit Flip%,
top-100 Jaccard $J@100$ vs. exact-IP GT, BinaryFlat R@100 (Vanilla→PMC), and
calibration cosine at $n_\mathrm{calib}{=}25$ (cos@25), on MSCOCO (CLIP-B/32) and
AudioCaps (ImageBind). Numbers come from `results/signbit_analysis_verified.csv`
(BinaryFlat R@100 from `results/signbit_original_gt.csv`), verified by
`scripts/analysis/verify_signbit_analysis.py` and `scripts/analysis/verify_calibration.py`.

### Table 5 — `tab:mech_extra`
MSCOCO/CLIP-B/32 checks: **(b) component ablation** and **(c) IVF-RaBitQ
controls** (random, shuffled, sign-flipped, un-normalized; Rand/Shuf are
mean±std over 5 seeds). The script's `print_latex_rows` emits only parts (b)+(c).
The **calibration-sensitivity panel (n=25/100/400) was removed from the table**,
but `results/mech_extra_calibration.csv` and its verification are retained because
the numbers back inline prose (see below).

### Table 6 — `tab:multibit`
PMC generality on IVFPQ and OPQ across encoders, R@100 Vanilla/PMC, **mean over
both retrieval directions**. Each printed cell is the direction-average of the two
per-direction rows held at full precision in `results/rerank_multibit_seed42.csv`
(OPQ CLIP/MSCOCO from `results/pmc_opq_multiseed_clip_mscoco.csv`); the committed
per-direction breakdown is `results/tab6_multibit_reproduced__np16_k0.csv`. The
reproduce script prints `# AVG-LATEX-NOTE` lines for any averaged cell that differs
from the printed table cell.

### Table 3 — `tab:r10`
R@10 at Table 2's no-reranking operating points, Vanilla/MeanShift/PMC. Same sources,
same rows and same round-half-up convention as Table 2: `scripts/reproduce_tab2_main.py`
emits the `q_r10_*` / `db_r10_*` columns of `results/tab2_main_reproduced.csv` alongside
the R@100 ones.

---

## Prose-backing results (data without a table)

The calibration-sensitivity panel was dropped from Table 5 (`tab:mech_extra`), so
those numbers survive only in body prose; the core flip%/Jaccard/cosine metrics are
now shown in Table 4 (`tab:mechanism`). Their CSVs are kept verifiable.

| Prose claim (location) | Verifier script | Source CSV(s) |
|---|---|---|
| "PMC induces 14–18% sign-bit flips with J@100 of 0.54–0.80; gap direction stable from n=25 (cosine ≥0.95)" — §4.2 Main Results (`03_experiments.tex`) | `scripts/analysis/verify_signbit_analysis.py` (verifies the metrics → `results/signbit_analysis_verified.csv`) | built by `scripts/emit_signbit_metrics.py` from `results/mechanism_bitflip.csv`, `mechanism_exact_control.csv`, `mechanism_calibration_sensitivity.csv`; BinaryFlat van/pmc from `results/signbit_original_gt.csv` |
| "Calibration stable from n_calib=25 (cosine ≈0.986; R@100 flat within .001 up to n_calib=400)" — §4.4 Ablation (`03_experiments.tex`) | `scripts/analysis/verify_calibration.py` (calibration block) | `results/mech_extra_calibration.csv` (built by `scripts/emit_calibration_metrics.py` from `results/mechanism_calibration_sensitivity.csv`) |

---

## Figures

| Asset | Reproduce / render script | Source CSV(s) | Reproduced CSV |
|---|---|---|---|
| `fig:analysis-bcd` panels (a) alpha sweep, (b) selective PMC, (c) QPS Pareto | data: `scripts/figures/rederive_fig3_alpha_sweep.py` (+ verifier `scripts/reproduce_fig3_analysis_bcd.py`); render: `paper/figures/fig3_analysis.py` | `results/fig_alpha_sweep_rabitq.csv`, `results/selective_pmc_rabitq.csv`, `results/pmc_qps_pareto_clip_mscoco_seed42.csv` | `results/fig3_analysis_bcd_reproduced.csv` |
| Gap-energy concentration (§Method "top 10% carry ≈90% of \|g\|²") | `scripts/reproduce_gap_energy.py` | `results/gap_energy_all_datasets.csv` | — |

---

## Reproduce everything (cached, no FAISS / no GPU)

From `current/pmc_crossmodal/`:

```bash
python3 scripts/reproduce_tab1_signbit_methods.py         # Table 1
python3 scripts/reproduce_tab2_main.py                    # Table 2 No-reranking columns
python3 scripts/reproduce_tab2_rerank.py                  # Table 2 With-reranking columns / §4.3
python3 scripts/reproduce_tab5_mech_extra.py              # Table 5 (tab:mech_extra) + calibration prose
python3 scripts/reproduce_tab6_multibit.py                # Table 6 (tab:multibit)
python3 scripts/reproduce_ablation_rerank.py              # §4.3 LAION K'-sweep ablation (repo-only)
python3 scripts/analysis/verify_signbit_analysis.py       # §4.2 sign-bit prose metrics
python3 scripts/analysis/verify_calibration.py            # §4.4 calibration prose
python3 scripts/reproduce_fig3_analysis_bcd.py            # Figure analysis panels
python3 scripts/reproduce_gap_energy.py                   # §3 Method gap-energy claim
```

Each script exits non-zero / prints `# REPRODUCE-MISMATCH` if a committed CSV no
longer backs the paper value. Large-dataset regeneration from raw embeddings
(Flickr30K-31K, LAION-400M, Yandex) requires external data and is out of scope for
the cached path; the `emit_*.py` builders listed above are the FAISS-bound entry
points for it.

## Upstream of every result

`scripts/data_prep/` turns raw datasets into the `data/features/` caches that the
`emit_*.py` builders read: AudioCaps and Clotho download end to end, MSCOCO and
Flickr30K need their images obtained separately. Re-running this stage does not
reproduce the committed CSVs bit for bit -- AudioCaps clip availability drifts --
which is why those CSVs, not the raw data, are the canonical source of truth.

## Not a paper element

* `scripts/emit_map_ndcg.py` — mAP@k / nDCG@k sweep (FAISS-bound; requires `data/features/`) over the six small-corpus rows, scored
  against exact-IP ground truth at the Table-2 operating points → `results/map_ndcg_seed42.csv`.
  Exploratory only: no figure or table in the paper reports these metrics. Note that Table 2 scores
  AudioCaps against caption--clip pairings rather than exact-IP, so those two rows are not directly
  comparable with that table; the other ten reproduce its R@100 within 0.005.
