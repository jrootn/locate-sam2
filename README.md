# Locate-SAM2

**Fast zero-shot text-guided segmentation** with [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) + [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-large).

> Grounded-SAM-style pipeline, but with parallel-box VLM grounding instead of Grounding DINO.

```text
Text + Image  →  LocateAnything  →  Prompt-to-Mask Adapter  →  SAM2  →  masks
```

**Research question:** Can a parallel-box VLM grounder improve the speed/quality tradeoff of Grounded-SAM-style zero-shot segmentation?

---

## Quick start

```bash
pip install -e ".[demo]"
# Models (~10 GB): bash scripts/download_models.sh
# DINO baseline (~700 MB): bash scripts/download_baseline.sh
```

```python
from locate_sam2 import segment

masks = segment("image.jpg", "red car on the left")
```

CLI:

```bash
locate-sam2 segment image.jpg "person holding umbrella" -o outputs/demo.png
```

Gradio demo:

```bash
python scripts/demo_gradio.py
```

---

## Method: Prompt-to-Mask Adapter

The adapter is the small method layer beyond model chaining:

1. Get candidate boxes/points from LocateAnything (or DINO baseline)
2. Optionally crop around each box for SAM2
3. Prompt SAM2 with `box`, `box+point`, or `point`
4. Rank masks by SAM confidence when multiple candidates exist
5. Return final mask(s)

Configure via `configs/default.yaml` or CLI flags:

| Knob | Options | Default |
|---|---|---|
| `prompt_mode` | box, box_point, point | box |
| `crop_mode` | full, crop | crop |
| `rerank` | top1, best_score, largest_box | best_score |
| `generation_mode` | fast, hybrid, slow | hybrid |

---

## Results (full RefCOCO val, 3,811 refs)

| Method | mIoU | cIoU | P@0.5 | P@0.7 | Box IoU | Latency |
|---|---:|---:|---:|---:|---:|---:|
| DINO-Tiny + SAM2 | 0.441 | 0.354 | 48.6% | 43.3% | 0.506 | 146 ms |
| DINO-Base + SAM2 | 0.717 | 0.657 | 81.7% | 74.4% | 0.802 | 164 ms |
| Locate-SAM2 fast | 0.769 | 0.745 | 87.5% | 79.2% | 0.847 | 190 ms |
| **Locate-SAM2 hybrid** | **0.772** | **0.753** | **88.1%** | **80.0%** | **0.851** | 195 ms |
| GT-box + SAM2 oracle | 0.836 | 0.830 | 95.2% | 88.5% | 1.000 | 72 ms |

**+33.1 mIoU vs DINO-Tiny · +5.5 mIoU vs DINO-Base · 92.3% of GT-box oracle**

### RefCOCO+ val (3,805 refs)

| Method | mIoU | P@0.5 |
|---|---:|---:|
| DINO-Tiny + SAM2 | 0.440 | 47.7% |
| DINO-Base + SAM2 | 0.612 | 69.4% |
| **Locate-SAM2 hybrid** | **0.717** | **81.6%** |

Full tables: `outputs/full_val/full_val_table.json` · Paper figures: `outputs/full_val/paper_figures/`

### 200-ref debug subset (seed=42)

Same trend on a verified subset (`subset_hash=399cff78deba693e`): hybrid **0.774** vs DINO-Tiny **0.426** mIoU.

### Reproduce

```bash
# Full RefCOCO val suite (hybrid, fast, DINO-Tiny, GT-oracle)
bash scripts/run_full_eval_suite.sh

# 200-subset quick benchmark
python scripts/run_benchmark.py --subset-size 200

# Ablations
python scripts/run_ablation.py --subset-size 200

# DINO threshold sanity check
python scripts/run_dino_threshold_sweep.py
```

Metrics: mIoU, overall IoU (cIoU), P@0.5–P@0.9, box IoU, latency split, peak VRAM.

**Eval verified:** `python scripts/validate_eval.py` · Run history: `outputs/experiments/EXPERIMENT_LOG.md`

---

## Project layout

```text
locate_sam2/
  adapter.py      # Prompt-to-Mask Adapter
  locate.py       # LocateAnything grounder
  dino.py         # Grounding DINO-Tiny baseline
  segment.py      # SAM2 segmenter
  pipeline.py     # LocateSam2Pipeline
  api.py          # segment() public API
  eval/           # RefCOCO metrics + scripts
scripts/
  run_benchmark.py
  run_ablation.py
  run_demo.py
  demo_gradio.py
configs/default.yaml
```

---

## GCP VM (optional)

Same setup as before — see `scripts/bootstrap_vm.sh`, `scripts/setup_gpu.sh`.

```bash
# Eval + auto-stop VM
nohup bash scripts/run_eval_and_stop.sh 200 > ~/run_eval.log 2>&1 &
```

**Blackwell G4:** PyTorch **cu128**, driver **R580 GRID**, `transformers==4.57.1`.

---

## License & usage

| Component | License | Notes |
|---|---|---|
| **LocateAnything-3B** | NVIDIA non-commercial | Academic / arXiv OK — **not for commercial use** |
| **SAM 2.1** | Apache 2.0 | |
| **Grounding DINO** | Apache 2.0 | |
| **This repo** | MIT | Research & evaluation tooling |

See `PAPER.md` for full technical report, related work, and honest positioning.

---

## v1 status: **COMPLETE**

Core RefCOCO / RefCOCO+ / **RefCOCO-g** benchmarks complete. See `experiments/RESULTS.md` and `outputs/experiments/EXPERIMENTS_COMPLETE.md`.

Paper figures (git-tracked): `research_paper/figures/` · README preview: `docs/assets/comparison_wins.png`

---

## Paper claim (v1)

> On RefCOCO, RefCOCO+, and RefCOCO-g val, Locate-SAM2 hybrid beats matched DINO-Tiny + SAM2 and DINO-Base + SAM2 baselines under the same adapter, with gains of +5.5 to +10.5 mIoU over DINO-Base and 85.8 to 92.3% of GT-box oracle performance.

We do **not** claim SOTA or best zero-shot segmentation. See `PAPER.md` for draft technical report.

---

## References

- [Grounded SAM](https://arxiv.org/abs/2401.14159) — DINO + SAM modular pipeline
- [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [LocateAnything](https://arxiv.org/abs/2605.27365) — parallel box decoding
- [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-large)

---

## Repository contents

| Included in git | Not included (local / download) |
|-----------------|----------------------------------|
| `locate_sam2/` + `scripts/` | Model weights (`models/`) |
| `configs/`, `benchmarks/` summaries | COCO / RefCOCO data (`data/`) |
| `research_paper/` LaTeX + qual figures | Full eval logs (`outputs/*_records.json`) |
| OOD protocol templates | `.venv/`, VM logs |

**Public repo:** https://github.com/jrootn/locate-sam2

---

## License

LocateAnything: NVIDIA non-commercial (academic/arXiv OK). SAM 2.1: Apache 2.0.
