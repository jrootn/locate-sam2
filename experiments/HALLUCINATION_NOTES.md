# Hallucination / false-positive grounding (paper notes)

LocateAnything is **prompted to always locate** (“Locate a single instance…”). Unlike Grounding DINO, it has **no native box/text confidence threshold**.

## Observed behavior

Run `scripts/probe_hallucination.py` — probes nonsense prompts on val images.
Check `outputs/analysis/hallucination_probe.json` for:
- `box_emission_rate` — how often a box is parsed from the VLM answer
- `mask_emission_rate` — how often SAM produces a mask
- `mean_sam_score_when_mask` — SAM predicted IoU for those masks

## Suggested mitigations (for paper Discussion)

1. **Empty parse = no detection** — if no `<box>` token in answer, skip SAM (already in pipeline).
2. **SAM score gate** — if `best_score` (SAM `iou_scores`) < τ (e.g. 0.5), return no mask / flag uncertain. Tunable post-hoc; not used in main Table numbers.
3. **DINO-style threshold N/A** for VLM grounder — cite contrast with DINO-Tiny sweep in paper.
4. **User-facing:** require phrase relevance check or ensemble with DINO when confidence low (future work).

## Paper figure

Use one case from `research_paper/figures/hallucination_probe/` with caption:
“Unrelated prompt still yields a box+mask (false positive); SAM score gate or empty-box rejection reduces this.”
