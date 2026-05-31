# Published benchmark summaries

Small JSON artifacts for **reproducing paper numbers** without cloning multi-GB eval logs.

Full per-sample `*_records.json` files stay local under `outputs/` (gitignored). Re-run eval with:

```bash
bash scripts/run_missing_experiments.sh   # val splits
bash scripts/run_test_splits.sh           # testA / testB
```

## Files

| Path | Contents |
|------|----------|
| `refcoco_val_table.json` | RefCOCO val — all methods |
| `refcoco_plus_table.json` | RefCOCO+ val |
| `refcocog_table.json` | RefCOCO-g val (Google split) |
| `test_splits/` | testA/testB summaries (hybrid + DINO-Base) |
| `analysis/` | Hybrid fallback + hallucination probe stats |
| `ablation_table.json` | 200-ref adapter ablations |
| `dino_threshold_sweep.json` | DINO-Tiny threshold sanity check |
| `benchmark_subset_200.json` | Early 200-ref comparison |

Numbers match `research_paper/UPDATE_GUIDE.md`.
