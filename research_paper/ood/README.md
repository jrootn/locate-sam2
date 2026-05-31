# Out-of-domain stress test (v1 protocol)

**Canonical location:** [`experiments/ood/`](../../experiments/ood/) — prompts, images, and batch scripts.

This folder kept for paper references (`ood_protocol.tex`). Do not duplicate images here.

## Quick start

```bash
# 1. Add images to experiments/ood/images/{domain}/
# 2. Edit experiments/ood/prompts.csv
python scripts/run_ood_batch.py
cp outputs/ood/results_template.csv outputs/ood/results.csv
# 3. Fill human scores, then:
python scripts/aggregate_ood_scores.py
```

See [`experiments/ood/README.md`](../../experiments/ood/README.md) for full protocol.
