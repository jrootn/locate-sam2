# Paper figures

Qualitative panels for self-contained PDF builds (tracked in git).

## Win / baseline comparison (existing)

| Directory | Mode | ref_id |
|-----------|------|--------|
| `win_ref5466/` | Win vs DINO-Tiny | 5466 |
| `win_ref2764/` | Spatial win | 2764 |
| `fail_ref2885/` | Wrong instance (ours fails) | 2885 |
| `both_ref3281/` | Both succeed | 3281 |

## Failure taxonomy (auto-exported)

Labeled dirs: `fail_{mode}_ref{id}/` where mode is:

| Folder prefix | Paper label |
|---------------|-------------|
| `fail_wrong_instance_*` | Failure (i): wrong object instance |
| `fail_spatial_*` | Failure (ii): spatial / ordinal language |
| `fail_attribute_*` | Failure (iii): attribute ambiguity |
| `fail_rare_or_long_*` | Failure (iv): rare or long expression |

Each contains: `image_raw.jpg`, `ours_overlay.png`, `dino_overlay.png`, `query.txt`, `metadata.json`.

Regenerate on VM:
```bash
python scripts/export_paper_figures.py \
  --ours-records outputs/full_val/locate_sam2_hybrid_full_records.json \
  --dino-records outputs/full_val/dino_tiny_full_records.json \
  --per-mode 2
```

## Hallucination / negative-prompt probe

`hallucination_probe/` — unrelated prompts on COCO images; see `metadata.json` and `outputs/analysis/hallucination_probe.json`.

Regenerate:
```bash
python scripts/probe_hallucination.py --n-images 8
```

## Comparison grids

- `comparison_wins.png`
- `comparison_failures.png`
