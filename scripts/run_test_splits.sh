#!/usr/bin/env bash
# RefCOCO / RefCOCO+ testA and testB; hybrid + DINO-Base only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${HOME}/test_splits.log"

cd "${ROOT}"
source .venv/bin/activate
export PYTHONPATH="${ROOT}"

run_one() {
  local dataset="$1"
  local split="$2"
  local grounder="$3"
  local extra=("${@:4}")
  local out="${ROOT}/outputs/test_splits"
  local tag="${grounder}_${dataset}_${split}"
  tag="${tag//+/_plus}"
  local summary="${out}/${tag}_summary.json"
  if [[ -f "${summary}" ]]; then
    echo "--- SKIP (exists): ${tag} ---"
    return 0
  fi
  echo "--- RUN: ${dataset} ${split} ${grounder} ---"
  python scripts/run_eval.py --subset-size 0 --dataset "${dataset}" --split "${split}" \
    --output-dir "${out}" --grounder "${grounder}" "${extra[@]}"
}

{
  echo "=== TEST SPLITS START $(date -Is) ==="
  mkdir -p "${ROOT}/outputs/test_splits"

  for split in testA testB; do
    run_one refcoco "${split}" locateanything --generation-mode hybrid --tag "locate_sam2_hybrid_refcoco_${split}"
    run_one refcoco "${split}" grounding_dino_swint --tag "dino_swint_refcoco_${split}"
    run_one "refcoco+" "${split}" locateanything --generation-mode hybrid --tag "locate_sam2_hybrid_refcoco_plus_${split}"
    run_one "refcoco+" "${split}" grounding_dino_swint --tag "dino_swint_refcoco_plus_${split}"
  done

  python scripts/build_experiment_tables.py --outputs "${ROOT}/outputs" 2>/dev/null || \
    python - <<'PY'
import json
from pathlib import Path
out = Path("outputs/test_splits")
rows = [json.loads(p.read_text()) for p in sorted(out.glob("*_summary.json"))]
if rows:
    (out / "test_splits_table.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{r['run_name']:40} mIoU={r['mean_mask_iou']:.3f} P@0.5={r.get('precision_at_0.5',0):.1%} n={r['subset_size']}")
PY

  echo "=== TEST SPLITS DONE $(date -Is) ==="
} | tee "${LOG}"
