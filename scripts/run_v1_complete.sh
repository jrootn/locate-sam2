#!/usr/bin/env bash
# Complete v1 gaps: DINO-Base, RefCOCO+, paper figures. Run on VM after main full_val done.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/outputs/full_val"
LOG="${HOME}/v1_complete.log"
SWINT="${ROOT}/data/grounding-dino-swint"

cd "${ROOT}"
source .venv/bin/activate
export PYTHONPATH="${ROOT}"

run_eval() {
  python scripts/run_eval.py --subset-size 0 --output-dir "${OUT}" "$@"
}

{
  echo "=== V1 COMPLETE START $(date -Is) ==="

  if [[ ! -f "${SWINT}/config.json" ]]; then
    echo "--- Download Grounding DINO Base ---"
    bash scripts/download_dino_swint.sh
  fi

  if [[ -f "${OUT}/dino_swint_full_summary.json" ]]; then
    echo "--- SKIP dino_swint_full (exists) ---"
  else
    echo "--- DINO-Base + SAM2 full RefCOCO val ---"
    run_eval --grounder grounding_dino_swint --tag dino_swint_full
  fi

  echo "--- RefCOCO+ val: Locate-SAM2 hybrid ---"
  python scripts/run_eval.py --subset-size 0 --dataset refcoco+ --output-dir outputs/refcoco_plus \
    --grounder locateanything --generation-mode hybrid --tag locate_sam2_hybrid_refcoco_plus || true

  echo "--- RefCOCO+ val: DINO-Tiny ---"
  python scripts/run_eval.py --subset-size 0 --dataset refcoco+ --output-dir outputs/refcoco_plus \
    --grounder grounding_dino_tiny --tag dino_tiny_refcoco_plus || true

  if [[ -f "${SWINT}/config.json" ]]; then
    echo "--- RefCOCO+ val: DINO-Base ---"
    python scripts/run_eval.py --subset-size 0 --dataset refcoco+ --output-dir outputs/refcoco_plus \
      --grounder grounding_dino_swint --tag dino_swint_refcoco_plus || true
  fi

  echo "--- Build paper figure grids ---"
  python scripts/build_paper_figures.py \
    --output-dir "${OUT}/paper_figures" \
    --n-wins 4 --n-fails 3

  echo "--- Update full val table ---"
  python - <<'PY'
import json
from pathlib import Path
out = Path("outputs/full_val")
rows = [json.loads(p.read_text()) for p in sorted(out.glob("*_full_summary.json"))]
(out / "full_val_table.json").write_text(json.dumps(rows, indent=2))
for r in rows:
    print(f"{r['run_name']:30} mIoU={r['mean_mask_iou']:.3f} P@0.5={r.get('precision_at_0.5',0):.1%}")
plus = Path("outputs/refcoco_plus")
if plus.exists():
    print("\nRefCOCO+:")
    for p in sorted(plus.glob("*_summary.json")):
        r = json.loads(p.read_text())
        print(f"  {r['run_name']:30} mIoU={r['mean_mask_iou']:.3f} P@0.5={r.get('precision_at_0.5',0):.1%}")
PY

  echo "=== V1 COMPLETE DONE $(date -Is) ==="
} | tee "${LOG}"

gcloud compute instances stop "$(hostname -s)" \
  --zone=us-east4-c \
  --project=kaagle-nemotron-lora-20260326 \
  --quiet || true
