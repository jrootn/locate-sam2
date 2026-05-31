# Benchmark summaries

JSON summaries for the numbers reported in the paper and root README. Per-sample `*_records.json` files are not stored in git; regenerate them with the scripts listed in the root README.

| File | Description |
|------|-------------|
| `refcoco_val_table.json` | RefCOCO validation, all methods |
| `refcoco_plus_table.json` | RefCOCO+ validation |
| `refcocog_table.json` | RefCOCO-g validation (Google split) |
| `test_splits/` | RefCOCO and RefCOCO+ testA and testB (hybrid and DINO-Base) |
| `analysis/` | Hybrid vs fast and negative-prompt probe statistics |
| `ablation_table.json` | Adapter ablations (200 references, seed 42) |
| `dino_threshold_sweep.json` | Grounding DINO-Tiny threshold sweep |
| `benchmark_subset_200.json` | Initial 200-reference comparison |

Paper update notes: [`../research_paper/UPDATE_GUIDE.md`](../research_paper/UPDATE_GUIDE.md).
