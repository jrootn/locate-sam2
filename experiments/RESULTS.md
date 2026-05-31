# Benchmark results (complete)

All numeric RefCOCO-family evals finished **2026-05-31**. JSON sources under `outputs/` (gitignored).

## RefCOCO val (n=3,811)

| Method | mIoU | P@0.5 | ms |
|--------|-----:|------:|---:|
| DINO-Tiny + SAM2 | 0.441 | 48.6% | 146 |
| DINO-Base + SAM2 | 0.717 | 81.7% | 164 |
| Locate-SAM2 fast | 0.769 | 87.5% | 190 |
| **Locate-SAM2 hybrid** | **0.772** | **88.1%** | 195 |
| GT-box + SAM2 oracle | 0.836 | 95.2% | 72 |

Source: `outputs/full_val/full_val_table.json`

## RefCOCO+ val (n=3,805)

| Method | mIoU | P@0.5 | ms |
|--------|-----:|------:|---:|
| DINO-Tiny + SAM2 | 0.440 | 47.7% | 146 |
| DINO-Base + SAM2 | 0.612 | 69.4% | 164 |
| Locate-SAM2 fast | 0.709 | 80.3% | 188 |
| **Locate-SAM2 hybrid** | **0.717** | **81.6%** | 198 |
| GT-box + SAM2 oracle | 0.836 | 95.2% | 72 |

Source: `outputs/refcoco_plus/refcoco_plus_table.json`

## RefCOCO-g val (n=5,000, Google split)

| Method | mIoU | P@0.5 | ms |
|--------|-----:|------:|---:|
| DINO-Tiny + SAM2 | 0.503 | 55.7% | 148 |
| DINO-Base + SAM2 | 0.666 | 75.4% | 166 |
| Locate-SAM2 fast | 0.741 | 84.0% | 199 |
| **Locate-SAM2 hybrid** | **0.746** | **85.0%** | 207 |
| GT-box + SAM2 oracle | 0.815 | 93.1% | 72 |

Source: `outputs/refcocog/refcocog_table.json`

## Qualitative assets

| Asset | Count | Git-tracked? | Path |
|-------|------:|:---:|------|
| Paper qual panels | 4 cases × 3 images | Yes | `research_paper/figures/{win,fail,both}_*/` |
| Comparison grids | 2 PNGs | Yes | `research_paper/figures/comparison_*.png` |
| Full qual export | 18 cases | No (gitignored) | `outputs/full_val/figures/` |
| README preview | 1 PNG | Yes | `docs/assets/comparison_wins.png` |

No extra inference needed for paper/git qual — panels were exported during full-val runs.

## OOD stress test

**Not run** — requires license-safe images in `experiments/ood/images/`. Protocol: `experiments/ood/README.md`.

## VM

Benchmark suite log: `outputs/experiments/missing_experiments.log`
