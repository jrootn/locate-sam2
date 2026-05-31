# Experiments: what is done vs. what is still missing

This document is the checklist for **evaluation work** (not LaTeX polish). Numbers in the paper should match the JSON files listed under “Done.”

---

## Done (no new images or runs required)

These are **finished** in the repo and already cited in `main.tex`.

| Experiment | What it measures | Size | Source file(s) |
|------------|------------------|------|----------------|
| **RefCOCO val — full** | mIoU, P@0.5 for all methods | 3,811 refs | `outputs/full_val/full_val_table.json` |
| Locate-SAM2 hybrid | Main system result | ↑ | `outputs/full_val/locate_sam2_hybrid_full_summary.json` |
| Locate-SAM2 fast | Latency-oriented variant | ↑ | `outputs/full_val/locate_sam2_fast_full_summary.json` |
| DINO-Tiny + SAM2 | Weak modular baseline | ↑ | `outputs/full_val/dino_tiny_full_summary.json` |
| DINO-Base + SAM2 | Strong modular baseline | ↑ | `outputs/full_val/dino_swint_full_summary.json` |
| **GT-box oracle** | Upper bound if grounding were perfect | ↑ | `outputs/full_val/gt_oracle_full_summary.json` |
| **RefCOCO+ val** | Harder split (no appearance-only) | 3,805 refs | `outputs/refcoco_plus/*_summary.json` |
| **Adapter ablations** | Point vs box, thresholds, etc. | 200 refs, seed 42 | `outputs/ablations/ablation_table.json` |
| **DINO threshold sweep** | Tiny grounder sensitivity | 200 refs | `outputs/dino_sweep/dino_threshold_sweep.json` |
| **Qualitative export** | Win / fail / both cases | 18 cases | `outputs/full_val/figures/` (+ `metadata.json` each) |
| **Paper qual panels** | 4 cases in PDF | 4 dirs | `research_paper/figures/{win_ref5466,win_ref2764,fail_ref2885,both_ref3281}/` |

**Artifacts already produced for qual (per case):**

- `image_raw.jpg` — input crop/frame  
- `dino_overlay.png` — DINO-Tiny box + SAM mask overlay  
- `ours_overlay.png` — Locate-SAM2 hybrid overlay  
- `metadata.json` — ref id, expression, mIoU both sides  

**Core claim you can make today:** On RefCOCO / RefCOCO+, hybrid Locate-SAM2 beats both DINO baselines; oracle gap shows grounding is the bottleneck. That is enough for a **v1 systems paper** once OOD is either scored or clearly labeled “in progress.”

---

## Missing experiment A — Out-of-domain stress test (largest gap)

**Why it matters:** RefCOCO is natural photos with COCO objects. The paper’s “zero-shot / modular” story is weaker until you show (or honestly limit) behavior on **UI, robotics, aerial, microscopy, documents**.

**Status:** Protocol and empty tables are in the PDF (`ood_protocol.tex`). **No real images, no runs, no human scores.**

### What you need to collect

| Item | Quantity | Details |
|------|----------|---------|
| **Images** | **50–150 total** (recommended **10–30 per domain × 5 domains**) | License-safe for paper + supplement |
| **Domains** | 5 folders | See below |
| **Prompts** | **1 per image** | Short natural language, **not** COCO category names |
| **Methods** | **2 per (image, prompt)** | `locate_sam2_hybrid` and `dino_tiny` (same adapter settings as Table 1) |
| **Outputs** | **3 files minimum per method run** | Same layout as full_val qual (see below) |
| **Human scores** | 1 row per (image, method) | CSV checklist → Table `tab:ood-results` |

### Domain folders (create under `experiments/ood/images/`)

```text
ood/images/ui/           # app screenshots, dashboards, buttons, icons
ood/images/robotics/     # tabletop, gripper, objects on bench
ood/images/aerial/       # satellite / drone, small structures, roads
ood/images/microscopy/   # lab slides, defects, regions
ood/images/documents/    # PDFs, diagrams, charts, OCR-like text regions
```

**Image requirements:**

- **Format:** PNG or JPG, typical resolution 512–2048 px (match what your pipeline expects).  
- **Content:** One clear referent per prompt; avoid images you cannot publish.  
- **Naming:** Stable ids, e.g. `ui/screen01.png`, `robotics/bottle_gripper_03.jpg`.  
- **No GT masks** — this protocol is **qualitative only** (human Y/N, not mIoU).

### Prompt requirements (one per image)

Examples (already in `ood/prompts.csv` as templates — **replace** `example_*.png` paths):

| Domain | Good prompt examples |
|--------|----------------------|
| UI | “blue submit button”, “search bar at the top” |
| Robotics | “transparent bottle near the gripper”, “red block left of the arm” |
| Aerial | “small white building next to the road”, “parking lot lower right” |
| Microscopy | “dark circular region near the edge”, “bright scratch in the center” |
| Documents | “title text block at the top”, “legend box bottom right” |

**Bad prompts for this test:** single COCO words (“dog”, “car”) on non-COCO domains — that does not stress domain shift.

### What to run (per image)

From repo root, for **each** row in `prompts.csv`:

1. **Locate-SAM2 hybrid** — same config as full val hybrid.  
2. **DINO-Tiny + SAM2** — same adapter as main baseline table.

Save at least:

```text
ood/runs/{domain}/{id}_ours.png      # or overlay + optional raw
ood/runs/{domain}/{id}_dino.png
```

Ideal (mirrors `outputs/full_val/figures/`):

```text
{case}/image_raw.jpg
{case}/ours_overlay.png
{case}/dino_overlay.png
{case}/metadata.json   # prompt, domain, optional notes
```

CLI sketch (adjust to your installed entrypoint):

```bash
locate-sam2 segment ood/images/ui/screen01.png "blue submit button" -o ood/runs/ui/screen01_ours.png
# DINO-Tiny path via eval config / grounder=grounding_dino_tiny
```

There is a batch script: `python scripts/run_ood_batch.py` (reads `experiments/ood/prompts.csv`).

### Human scoring (fill `ood/results.csv`)

Copy `ood/results_template.csv` → `ood/results.csv`. For each **(domain, image, method)** row:

| Column | Meaning |
|--------|---------|
| `prompt_ok` | Criterion 1: model “understood” the phrase (reasonable attempt) |
| `instance_ok` | Criterion 2: correct object instance |
| `box_ok` | Criterion 3: box good enough to segment |
| `mask_usable` | Criterion 4: final mask usable for the task |
| `failure_tag` | If any N: `spatial`, `attribute`, `small-object`, `domain-shift`, `ocr`, `clutter`, `no-detection` |

Aggregate **% mask_usable = Y** per domain × method → paste into `ood_protocol.tex` Table `tab:ood-results`.

Optional for paper: copy 1–2 best/worst OOD overlays into `research_paper/figures/ood_{domain}/` and add a qual row in `qual_figures.tex`.

### Do not claim until this is done

- “Works on any domain” or “strong zero-shot generalization” beyond natural photos.  
- Filled OOD table with real percentages.

---

## Missing experiment B — Optional benchmarks (numeric, not OOD)

These **strengthen** the paper but are **not required** for the main RefCOCO story.

| Experiment | Why run it | Command / action | What you get |
|------------|------------|------------------|--------------|
| **RefCOCO+ GT oracle** | Same “grounding bottleneck” argument on RefCOCO+ | `bash scripts/run_missing_experiments.sh` or `run_eval.py --dataset refcoco+ --grounder gt_oracle` | New row in RefCOCO+ table |
| **RefCOCO-g val** | Third standard split (Google partition) | `bash scripts/run_missing_experiments.sh` | `outputs/refcocog/refcocog_table.json` |
| **Locate-SAM2 fast on RefCOCO+** | Symmetry with RefCOCO table | included in `run_missing_experiments.sh` | Extra row |
| **More qual cases** | Cover failure modes (spatial, OCR, etc.) | `export_qualitative.py` (or existing export) from full val | Pick 2–4 more dirs under `research_paper/figures/` |

**Artifacts:** Only JSON summaries + optional PNG overlays — **no new domain images**, uses existing RefCOCO-family downloads.

---

## Missing experiment C — Paper / submission (not eval)

| Item | What you need |
|------|----------------|
| Author block | Real affiliation + email in `authors.tex` |
| Acknowledgments | Funding / compute sources (if any) |
| Supervised RES table | **Citations only** — no new runs |
| Supplementary PDF | Extra qual grid from `outputs/full_val/figures/` |
| OOD inter-rater | Second reviewer on subset of OOD CSV (optional) |

---

## Quick decision guide

| Goal | Minimum work |
|------|----------------|
| **Internal / lab draft** | Done eval + 4 qual cases (current state) |
| **arXiv v1 systems** | Fix author info + proofread; label OOD “in progress” OR score 50+ OOD images |
| **Strong generalization claim** | **Must complete Experiment A** with documented images + `results.csv` |
| **Venue expecting 3 splits** | Add RefCOCO-g (Experiment B) |

---

## File map

| Path | Role |
|------|------|
| `outputs/full_val/full_val_table.json` | Main RefCOCO numbers |
| `outputs/refcoco_plus/*.json` | RefCOCO+ numbers |
| `outputs/ablations/ablation_table.json` | Ablations |
| `experiments/ood/prompts.csv` | OOD image list + prompts |
| `outputs/ood/results_template.csv` | Human score sheet (generated by run_ood_batch) |
| `experiments/README.md` | Experiment status + commands |
| `research_paper/GAPS.md` | Submission checklist (includes non-eval items) |
