#!/usr/bin/env bash
# Full RefCOCO val eval for main paper numbers, then stop VM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/outputs/full_val"
LOG="${HOME}/full_val.log"

cd "${ROOT}"
source .venv/bin/activate
export PYTHONPATH="${ROOT}"

{
  echo "=== FULL VAL START $(date -Is) ==="

  python scripts/run_eval.py \
    --subset-size 0 \
    --output-dir "${OUT}" \
    --grounder locateanything \
    --generation-mode hybrid \
    --crop-mode crop \
    --tag locate_sam2_hybrid_full

  python scripts/run_eval.py \
    --subset-size 0 \
    --output-dir "${OUT}" \
    --grounder grounding_dino_tiny \
    --crop-mode crop \
    --tag grounded_sam2_dino_tiny_full

  echo "=== FULL VAL DONE $(date -Is) ==="
  cat "${OUT}/locate_sam2_hybrid_full_summary.json"
  echo "---"
  cat "${OUT}/grounded_sam2_dino_tiny_full_summary.json"

  python scripts/generate_figures.py \
    --records "${OUT}/locate_sam2_hybrid_full_records.json" \
    --output-dir "${OUT}/figures" \
    --top 8 --bottom 8 --failures 4

  echo "=== STOPPING VM $(date -Is) ==="
} | tee "${LOG}"

gcloud compute instances stop "$(hostname -s)" \
  --zone=us-east4-c \
  --project=kaagle-nemotron-lora-20260326 \
  --quiet || echo "Could not auto-stop VM"
