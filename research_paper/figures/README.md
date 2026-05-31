# Paper figures (DINO-Tiny + hybrid)

All qualitative exports use **Grounding DINO-Tiny + SAM2** as the visual baseline and **Locate-SAM2 hybrid** as ours. Quantitative tables use DINO-Base (Swin-T) as the primary baseline.

## Fixed LaTeX cases

| Directory | ref_id | Role |
|-----------|--------|------|
| `win_ref5466/` | 5466 | Win — spatial ("right white spoon") |
| `win_ref2764/` | 2764 | Win — spatial ("right") |
| `fail_ref2885/` | 2885 | Hybrid failure — wrong person |
| `both_ref3281/` | 3281 | Both grounders succeed |

Each dir: `image_raw.jpg`, `dino_overlay.png`, `ours_overlay.png`, `query.txt`, `metadata.json`.

## Failure taxonomy (`fail_{mode}_ref{id}/`)

| Mode | Count (target ≥3) | Paper example |
|------|------------------:|---------------|
| `wrong_instance` | 3 | `fail_wrong_instance_ref20398` |
| `spatial` | 3 | `fail_spatial_ref5750` |
| `attribute` | 3 | `fail_attribute_ref18360` |
| `rare_or_long` | 3 | `fail_rare_or_long_ref36776` |

## Hallucination probe

`hallucination_probe/` — negative prompts on val images; see `metadata.json`.

## README composites

Built offline into `docs/assets/`:

- `comparison_wins.png`, `comparison_failures.png` — 5-column grid (input, GT, DINO-Tiny, hybrid, oracle)
- `readme_qualitative_wins.png`, `readme_qualitative_failures.png` — 3-column taxonomy panels

## Regenerate (GPU VM)

```bash
bash scripts/run_full_figure_export.sh
```

Requires `outputs/full_val/locate_sam2_hybrid_full_records.json` and `dino_tiny_full_records.json`.
