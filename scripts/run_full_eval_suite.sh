#!/usr/bin/env bash
# Full RefCOCO val evaluation suite (paper numbers).
# Runs: Locate hybrid/fast, DINO-Tiny, DINO-SwinT (if present), GT-oracle.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/outputs/full_val"
LOG="${HOME}/full_eval_suite.log"
SWINT_DIR="${ROOT}/data/grounding-dino-swint"

cd "${ROOT}"
source .venv/bin/activate
export PYTHONPATH="${ROOT}"

run_eval() {
  python scripts/run_eval.py --subset-size 0 --output-dir "${OUT}" "$@"
}

{
  echo "=== FULL EVAL SUITE START $(date -Is) ==="

  echo "--- A: Locate-SAM2 hybrid ---"
  run_eval --grounder locateanything --generation-mode hybrid --tag locate_sam2_hybrid_full

  echo "--- B: Locate-SAM2 fast ---"
  run_eval --grounder locateanything --generation-mode fast --tag locate_sam2_fast_full

  echo "--- C: GroundingDINO-Tiny + SAM2 ---"
  run_eval --grounder grounding_dino_tiny --tag dino_tiny_full

  if [[ -f "${SWINT_DIR}/config.json" ]]; then
    echo "--- D: GroundingDINO-SwinT + SAM2 ---"
    run_eval --grounder grounding_dino_swint --tag dino_swint_full
  else
    echo "--- D: SKIP DINO-SwinT (not downloaded) ---"
  fi

  echo "--- E: GT-box + SAM2 oracle ---"
  run_eval --grounder gt_oracle --tag gt_oracle_full

  echo "=== BUILDING COMPARISON TABLE ==="
  python - <<'PY'
import json
from pathlib import Path
out = Path("outputs/full_val")
rows = []
for p in sorted(out.glob("*_full_summary.json")):
    rows.append(json.loads(p.read_text()))
(out / "full_val_table.json").write_text(json.dumps(rows, indent=2))
for r in rows:
    print(
        f"{r['run_name']:30} mIoU={r['mean_mask_iou']:.3f} "
        f"oIoU={r['overall_iou']:.3f} P@0.5={r.get('precision_at_0.5', r.get('success_at_0.5', 0)):.1%} "
        f"boxIoU={r['mean_box_iou']:.3f} {r['mean_total_ms']:.0f}ms"
    )
PY

  echo "=== EXPORT QUALITATIVE (subset from hybrid records) ---"
  if [[ -f "${OUT}/locate_sam2_hybrid_full_records.json" ]] && [[ -f "${OUT}/dino_tiny_full_records.json" ]]; then
    python scripts/export_qualitative.py \
      --ours-records "${OUT}/locate_sam2_hybrid_full_records.json" \
      --dino-records "${OUT}/dino_tiny_full_records.json" \
      --output-dir "${OUT}/figures" \
      --max-per-group 6 || echo "figure export skipped"
  fi

  echo "=== FULL EVAL SUITE DONE $(date -Is) ==="
} | tee "${LOG}"

gcloud compute instances stop "$(hostname -s)" \
  --zone=us-east4-c \
  --project=kaagle-nemotron-lora-20260326 \
  --quiet || echo "Could not auto-stop VM"
