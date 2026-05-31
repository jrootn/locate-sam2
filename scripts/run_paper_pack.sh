#!/usr/bin/env bash
# VM batch: test splits + failure figure export + analysis probes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${HOME}/paper_pack.log"

cd "${ROOT}"
source .venv/bin/activate
export PYTHONPATH="${ROOT}"

{
  echo "=== PAPER PACK START $(date -Is) ==="

  bash scripts/run_test_splits.sh

  OURS="${ROOT}/outputs/full_val/locate_sam2_hybrid_full_records.json"
  DINO="${ROOT}/outputs/full_val/dino_tiny_full_records.json"
  if [[ -f "${OURS}" && -f "${DINO}" ]]; then
    echo "--- Export labeled failure figures ---"
    python scripts/export_paper_figures.py \
      --ours-records "${OURS}" \
      --dino-records "${DINO}" \
      --per-mode 2
  else
    echo "--- SKIP export_paper_figures (missing records) ---"
  fi

  echo "--- Hybrid vs fast probe (500 refs) ---"
  python scripts/probe_hybrid_fallback.py --subset-size 500

  echo "--- Hallucination / negative-prompt probe ---"
  python scripts/probe_hallucination.py --n-images 8

  echo "=== PAPER PACK DONE $(date -Is) ==="
} 2>&1 | tee "${LOG}"
