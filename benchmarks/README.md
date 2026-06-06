# Benchmark summaries

JSON summaries for the numbers reported in the paper and root README. Per-sample `*_records.json` files are not stored in git; regenerate them with the scripts listed in the root README.

| File | Description |
|------|-------------|
| `refcoco_val_table.json` | RefCOCO validation, all methods |
| `refcoco_plus_table.json` | RefCOCO+ validation |
| `refcocog_table.json` | RefCOCO-g validation (Google split) |
| `test_splits/` | RefCOCO and RefCOCO+ testA and testB (hybrid and DINO-Base) |
| `analysis/box_iou_stratification.json` | Box-IoU bins vs mask quality across generated record files |
| `analysis/hybrid_disagreement_miou.json` | Fast vs hybrid mIoU on agree/disagree subsets (RefCOCO val, 500 refs, seed 42) |
| `analysis/generation_mode_study.json` | LocateAnything fast / hybrid / slow mode study (RefCOCO val, 200 refs, seed 42) |
| `analysis/hybrid_fallback_stats.json` | Hybrid fallback trigger statistics from an earlier 200-reference probe |
| `analysis/hallucination_probe.json` | Negative-prompt sanity check |
| `ablation_table.json` | Adapter ablations (200 references, seed 42) |
| `dino_threshold_sweep.json` | Grounding DINO-Tiny threshold sweep |
| `benchmark_subset_200.json` | Initial 200-reference comparison |
