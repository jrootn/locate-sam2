#!/usr/bin/env bash
# One-shot export of all paper + README qualitative assets (DINO-Tiny baseline).
# Run on GPU VM after full-val records exist. Safe to re-run (--force refreshes PNGs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source .venv/bin/activate
export PYTHONPATH="${ROOT}"

LOG="${HOME}/figure_export.log"
OURS="${ROOT}/outputs/full_val/locate_sam2_hybrid_full_records.json"
DINO="${ROOT}/outputs/full_val/dino_tiny_full_records.json"
ORACLE="${ROOT}/outputs/full_val/gt_oracle_full_records.json"
PAPER="${ROOT}/research_paper/figures"
ASSETS="${ROOT}/docs/assets"

{
  echo "=== FULL FIGURE EXPORT START $(date -Is) ==="

  for f in "${OURS}" "${DINO}"; do
    if [[ ! -f "${f}" ]]; then
      echo "ERROR: missing ${f}"
      exit 1
    fi
  done

  echo "--- [1/7] Fixed + all existing paper case dirs ---"
  python scripts/export_fixed_paper_cases.py \
    --ours-records "${OURS}" \
    --dino-records "${DINO}" \
    --paper-dir "${PAPER}" \
    --rerender-all

  echo "--- [2/7] Failure taxonomy (3 per mode, force re-export) ---"
  python scripts/export_paper_figures.py \
    --ours-records "${OURS}" \
    --dino-records "${DINO}" \
    --paper-dir "${PAPER}" \
    --per-mode 3 \
    --force

  echo "--- [3/7] Extra qual groups (outputs/full_val/figures) ---"
  python scripts/export_qualitative.py \
    --ours-records "${OURS}" \
    --dino-records "${DINO}" \
    --output-dir "${ROOT}/outputs/full_val/figures" \
    --max-per-group 8

  echo "--- [4/7] 5-column comparison grids (GT + oracle) ---"
  mkdir -p "${ASSETS}"
  python scripts/build_paper_figures.py \
    --ours-records "${OURS}" \
    --dino-records "${DINO}" \
    --oracle-records "${ORACLE}" \
    --output-dir "${ASSETS}" \
    --n-wins 4 \
    --n-fails 3

  echo "--- [5/7] Hallucination probe ---"
  python scripts/probe_hallucination.py --n-images 8

  echo "--- [6/7] Strip any matplotlib title bands from overlays ---"
  python scripts/restrip_paper_overlays.py --paper-dir "${PAPER}"

  echo "--- [7/7] README composite panels (CPU) ---"
  python scripts/build_readme_figures.py --figures-dir "${PAPER}" --output-dir "${ASSETS}"
  if [[ -f scripts/build_readme_assets.py ]]; then
    python scripts/build_readme_assets.py || true
  fi

  echo "--- Manifest ---"
  find "${PAPER}" -maxdepth 2 -name '*.png' | wc -l | xargs echo "paper PNG count:"
  find "${ASSETS}" -name '*.png' | wc -l | xargs echo "docs/assets PNG count:"
  ls -la "${PAPER}"/fail_* 2>/dev/null | head -20 || true

  echo "=== FULL FIGURE EXPORT DONE $(date -Is) ==="
} 2>&1 | tee "${LOG}"
