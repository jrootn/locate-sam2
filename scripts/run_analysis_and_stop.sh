#!/usr/bin/env bash
# Run overnight analysis jobs, then stop this GCP VM to avoid idle GPU charges.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOG="${HOME}/analysis_and_stop.log"

{
  echo "=== ANALYSIS AND STOP START $(date -Is) ==="
  source .venv/bin/activate

  if [[ ! -f scripts/run_generation_mode_study.py ]]; then
    echo "Missing scripts/run_generation_mode_study.py in ${ROOT}"
    exit 1
  fi

  python3 scripts/run_generation_mode_study.py --subset-size 200

  if [[ -d outputs/full_val ]]; then
    python3 scripts/analyze_box_iou_stratification.py --input-dir outputs/full_val \
      --output benchmarks/analysis/box_iou_stratification_full_val.json || true
  fi

  echo "=== ANALYSIS AND STOP DONE $(date -Is) ==="
} | tee -a "${LOG}"

ZONE="${GCP_ZONE:-us-east4-c}"
PROJECT="${GCP_PROJECT:-kaagle-nemotron-lora-20260326}"
NAME="$(hostname -s)"

if command -v gcloud >/dev/null 2>&1; then
  gcloud compute instances stop "${NAME}" --zone="${ZONE}" --project="${PROJECT}" --quiet \
    && echo "Stopped VM via gcloud: ${NAME}" \
    || echo "gcloud stop failed; stopping locally"
fi

sudo shutdown -h now 2>/dev/null || shutdown -h now 2>/dev/null || true
