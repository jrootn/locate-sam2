# Locate-SAM2 experiments

## Status: numeric benchmarks complete (2026-05-31)

See [`RESULTS.md`](RESULTS.md) for all tables. OOD still needs your images.

| Experiment | Status |
|------------|--------|
| RefCOCO val (full) | Done |
| RefCOCO+ val (all methods) | Done |
| RefCOCO-g val (all methods) | Done |
| Ablations + DINO sweep | Done |
| Qual export (18 cases) | Done — `outputs/full_val/figures/` |
| Paper qual (4 cases + grids) | Done — `research_paper/figures/` |
| OOD stress test | **Needs images** — `experiments/ood/` |

## Pull results from VM (if re-run)

```bash
gcloud compute scp --recurse locateanythingsam-vm:~/locateanythingsam/outputs/refcocog outputs/ \
  --zone=us-east4-c --project=kaagle-nemotron-lora-20260326
```

## Commands

```bash
bash scripts/run_missing_experiments.sh   # skip-if-exists full suite
python scripts/build_experiment_tables.py
python scripts/run_ood_batch.py           # after OOD images added
```
