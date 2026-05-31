#!/usr/bin/env bash
# Run RefCOCO eval then stop VM to save GPU cost. Safe to nohup and walk away.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBSET_SIZE="${1:-200}"
LOG="${HOME}/run_eval.log"

cd "${ROOT}"
source .venv/bin/activate
export PYTHONPATH="${ROOT}"

{
  echo "=== EVAL START $(date -Is) ==="
  python scripts/run_benchmark.py --subset-size "${SUBSET_SIZE}" --output-dir outputs/benchmark
  echo "=== EVAL DONE $(date -Is) ==="
  cat outputs/benchmark/benchmark_table.json
  echo "=== STOPPING VM $(date -Is) ==="
} | tee "${LOG}"

# Stop this VM from inside (requires compute.instanceStop permission on default SA)
gcloud compute instances stop "$(hostname -s)" \
  --zone=us-east4-c \
  --project=kaagle-nemotron-lora-20260326 \
  --quiet || echo "Could not auto-stop VM — stop manually to save GPU cost"
