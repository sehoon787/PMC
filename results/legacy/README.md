# Legacy result CSVs

Historical measurement outputs kept for the record. **Nothing in the paper,
the reproduce scripts, or the README reads these files** — they predate the
final experimental protocol and are preserved per the project convention of
keeping negative/superseded results on record.

| File | Origin | Superseded by |
|---|---|---|
| `pmc_audiocaps_r1_single_seed42.csv` | Early AudioCaps R@1 re-eval under a since-renamed output convention (`_single` suffix) | `pmc_audiocaps_r1_seed42.csv` naming in `scripts/builders/reproduce_audiocaps.py` (regenerable) |
| `pmc_laion400m_reverse_rerank_nlist80k_np512_1024_seed42.csv` | LAION-400M reverse-direction rerank probe sweep at n_probe = 512/1024 | Final operating point n_probe = 256 (`pmc_laion400m_reverse_rerank_nlist80k_seed42.csv`) |
| `pmc_pq_audiocaps_seed42.csv` | Per-dataset IVFPQ/OPQ runs from the old `scripts/pq/` experiment lane | `rerank_multibit_seed42.csv` + `pmc_opq_multiseed_clip_mscoco.csv` (Table 6 sources) |
| `pmc_pq_clipl_mscoco_seed42.csv` | 〃 | 〃 |
| `pmc_pq_clotho_seed42.csv` | 〃 | 〃 |
| `pmc_pq_flickr30k_clipl_seed42.csv` | 〃 | 〃 |
| `pmc_pq_imagebind_mscoco_seed42.csv` | 〃 | 〃 |
