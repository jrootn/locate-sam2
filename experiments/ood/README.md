# Out-of-domain stress test

Qualitative evaluation **outside** RefCOCO — no mIoU, human checklist only.

## Layout

```text
experiments/ood/
  prompts.csv          # one row per image (edit before running)
  images/
    ui/
    robotics/
    aerial/
    microscopy/
    documents/

outputs/ood/
  runs/{domain}/{case_id}/   # image_raw.jpg, *_overlay.png, metadata.json
  results_template.csv       # auto-generated after batch run
  results.csv                # copy template here and fill scores
  ood_summary.json           # from aggregate_ood_scores.py
```

## 1. Collect images

**50–150 total** recommended (10–30 per domain). Requirements:

- PNG or JPG, roughly 512–2048 px
- One clear referent per prompt; license-safe for publication
- Stable names, e.g. `ui/screen01.png`, `robotics/bottle_gripper_03.jpg`

## 2. Edit prompts.csv

Columns: `domain`, `image_id`, `image_file`, `prompt`, `notes`

`image_file` is relative to `experiments/ood/images/`. Example:

```csv
domain,image_id,image_file,prompt,notes
ui,screen01,ui/screen01.png,blue submit button,
robotics,bottle_01,robotics/bottle_gripper_01.jpg,transparent bottle near the gripper,
```

Use natural language — **not** bare COCO category names on non-photo domains.

## 3. Run both methods

From repo root (GPU recommended):

```bash
python scripts/run_ood_batch.py
# optional: --skip-existing to resume
```

Runs **Locate-SAM2 hybrid** and **DINO-Tiny + SAM2** with the same adapter settings as RefCOCO Table 1 (`box` prompt, `crop`, `best_score` rerank).

## 4. Human scoring

```bash
cp outputs/ood/results_template.csv outputs/ood/results.csv
# fill Y/N columns for each (image, method) row
python scripts/aggregate_ood_scores.py
```

| Column | Meaning |
|--------|---------|
| `prompt_ok` | Model understood the phrase |
| `instance_ok` | Correct object instance |
| `box_ok` | Box good enough to segment |
| `mask_usable` | Final mask usable for the task |
| `failure_tag` | If any N: `spatial`, `attribute`, `small-object`, `domain-shift`, `ocr`, `clutter`, `no-detection` |

## 5. What you can claim

- **Before scoring:** label OOD as in progress; do not claim cross-domain generalization.
- **After 50+ scored images:** report `% mask_usable` per domain × method from `ood_summary.json`.
