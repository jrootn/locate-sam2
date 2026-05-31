# Locate-SAM2: Fast Zero-Shot Text-Guided Segmentation with Parallel Box Grounding

**Technical report (v1)**

## Abstract

We present Locate-SAM2, a reproducible zero-shot text-to-mask pipeline that studies whether LocateAnything-3B can serve as the grounding front end in a Grounded-SAM-style system with SAM 2.1. A lightweight Prompt-to-Mask Adapter converts grounding outputs into SAM2 box prompts with optional cropping and mask reranking. On RefCOCO, RefCOCO+, and RefCOCO-g validation, Locate-SAM2 hybrid reaches **0.772**, **0.717**, and **0.746** mIoU. Against DINO-Base + SAM2 under the same adapter, the gains are **+5.5**, **+10.5**, and **+8.0** mIoU. On RefCOCO testA/testB, hybrid reaches **0.807/0.730** mIoU versus **0.761/0.661** for DINO-Base; on RefCOCO+ testA/testB, it reaches **0.766/0.650** versus **0.708/0.517**. GT-box + SAM2 oracles reach **0.836**, **0.836**, and **0.815** mIoU, so hybrid recovers **85.8-92.3%** of the oracle result depending on split. The contribution is the integration, matched evaluation, adapter ablation, and oracle diagnostic, not a new foundation model or a replacement for LocateAnything or SAM2.

## 1. Introduction

**Grounded SAM** (Ren et al., 2024) assembles open-world models by pairing Grounding DINO with SAM for text-prompted detection and segmentation. **Grounding DINO** (Liu et al., 2023) is a language-conditioned open-set detector; **SAM 2 / SAM 2.1** (Ravi et al., 2024) provides promptable high-quality masks from boxes, points, or masks. **LocateAnything** (NVIDIA, 2025) introduces parallel box decoding (PBD), treating boxes as atomic geometric units for faster, more coherent VLM grounding.

We ask: *Can LocateAnything improve the practical speed/quality tradeoff of modular text-to-mask systems without training or fusion?*

We do **not** introduce a new segmentation model and do **not** claim to lead supervised RES benchmarks. Trained referring-expression segmentation models use a different regime from this frozen modular pipeline. Our contribution is a **reproducible systems study** with matched baselines and oracle analysis.

## 2. Related Work

- **Grounded SAM / Grounded SAM 2:** Modular DINO + SAM pipelines for open-vocabulary segmentation; our work follows the same recipe but swaps the grounder.
- **Grounding DINO:** Standard open-vocabulary detector baseline; we compare Tiny and Base variants.
- **SAM 2.1:** Segmenter backend; oracle GT-box evaluation isolates its contribution.
- **LocateAnything:** Parallel-box VLM grounder; we evaluate it as a drop-in Grounded-SAM front-end.

## 3. Method

```text
Image + expression → LocateAnything (boxes) → Prompt-to-Mask Adapter → SAM2.1 → mask
```

**Prompt-to-Mask Adapter:** candidate boxes → optional crop → SAM2 box prompt → mask reranking.

## 4. Experiments

**Dataset:** RefCOCO val (3,811 refs), RefCOCO+ val (3,805 refs), and RefCOCO-g val (5,000 refs, Google split). COCO train2014 images; GT from split annotations.

**Metrics:** mIoU, overall IoU (cIoU), P@0.5–P@0.9, box IoU, latency, peak VRAM.

**Baselines:** GroundingDINO-Tiny + SAM2, GroundingDINO-Base + SAM2, GT-box + SAM2 oracle.

DINO-Tiny thresholds were swept on a 200-ref subset (12 combinations); default 0.25/0.25 was optimal.

### 4.1 RefCOCO val (3,811 refs)

| Method | mIoU | cIoU | P@0.5 | P@0.7 | Box IoU | Latency |
|---|---:|---:|---:|---:|---:|---:|
| DINO-Tiny + SAM2 | 0.441 | 0.354 | 48.6% | 43.3% | 0.506 | 146 ms |
| DINO-Base + SAM2 | 0.717 | 0.657 | 81.7% | 74.4% | 0.802 | 164 ms |
| Locate-SAM2 fast | 0.769 | 0.745 | 87.5% | 79.2% | 0.847 | 190 ms |
| **Locate-SAM2 hybrid** | **0.772** | **0.753** | **88.1%** | **80.0%** | **0.851** | 195 ms |
| GT-box + SAM2 oracle | 0.836 | 0.830 | 95.2% | 88.5% | 1.000 | 72 ms |

Locate-SAM2 beats DINO-Base by **+5.5 mIoU** and reaches **92.3%** of the GT-box oracle.

### 4.2 RefCOCO+ val (3,805 refs)

| Method | mIoU | P@0.5 | Box IoU | Latency |
|---|---:|---:|---:|---:|
| DINO-Tiny + SAM2 | 0.440 | 47.7% | 0.502 | 146 ms |
| DINO-Base + SAM2 | 0.612 | 69.4% | 0.691 | 164 ms |
| Locate-SAM2 fast | 0.709 | 80.3% | 0.784 | 188 ms |
| **Locate-SAM2 hybrid** | **0.717** | **81.6%** | **0.795** | 198 ms |
| GT-box + SAM2 oracle | 0.836 | 95.2% | 1.000 | 72 ms |

### 4.3 RefCOCO-g val (5,000 refs)

| Method | mIoU | cIoU | P@0.5 | P@0.7 | Box IoU | Latency |
|---|---:|---:|---:|---:|---:|---:|
| DINO-Tiny + SAM2 | 0.503 | 0.414 | 55.7% | 49.1% | 0.588 | 148 ms |
| DINO-Base + SAM2 | 0.666 | 0.589 | 75.4% | 67.5% | 0.760 | 166 ms |
| Locate-SAM2 fast | 0.741 | 0.719 | 84.0% | 74.3% | 0.835 | 199 ms |
| **Locate-SAM2 hybrid** | **0.746** | **0.725** | **85.0%** | **75.2%** | **0.839** | 207 ms |
| GT-box + SAM2 oracle | 0.815 | 0.815 | 93.1% | 84.6% | 1.000 | 72 ms |

### 4.4 RefCOCO / RefCOCO+ test splits

| Split | Method | mIoU | P@0.5 |
|---|---|---:|---:|
| RefCOCO testA | DINO-Base + SAM2 | 0.761 | 87.4% |
| RefCOCO testA | **Locate-SAM2 hybrid** | **0.807** | **93.1%** |
| RefCOCO testB | DINO-Base + SAM2 | 0.661 | 73.5% |
| RefCOCO testB | **Locate-SAM2 hybrid** | **0.730** | **81.5%** |
| RefCOCO+ testA | DINO-Base + SAM2 | 0.708 | 81.3% |
| RefCOCO+ testA | **Locate-SAM2 hybrid** | **0.766** | **88.2%** |
| RefCOCO+ testB | DINO-Base + SAM2 | 0.517 | 56.8% |
| RefCOCO+ testB | **Locate-SAM2 hybrid** | **0.650** | **72.4%** |

### 4.5 Ablations (200-ref subset)

Point-only SAM prompting: **0.509 mIoU** vs **~0.77** for box prompts. Box-based prompting is essential.

### 4.6 Failure analysis

Failures are mostly wrong instance, spatial/ordinal phrases, and ambiguity, not SAM boundary errors.

## 5. Conclusion

Locate-SAM2 demonstrates that modern parallel-box VLM grounding can improve modular zero-shot text-to-mask systems over matched Grounding DINO + SAM2 baselines across the three standard RefCOCO-family validation splits, with higher latency and memory as the main systems cost.

## 6. License & Usage

- **LocateAnything-3B:** NVIDIA non-commercial research license; academic/arXiv use OK; **not for commercial deployment**.
- **SAM 2.1:** Apache 2.0.
- **Grounding DINO:** Apache 2.0.
- **This codebase:** MIT (research/evaluation tooling only).

## References

1. Grounded SAM: arXiv:2401.14159  
2. Grounding DINO: arXiv:2303.05499  
3. SAM 2: arXiv:2408.00714  
4. LocateAnything: arXiv:2605.27365  
