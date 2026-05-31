#!/usr/bin/env bash
# Optional: Grounding DINO weights only (Grounded SAM = DINO + SAM pipeline, not one download).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/data/grounding-dino-tiny"
mkdir -p "${OUT}"

if ! command -v hf >/dev/null 2>&1; then
  pip install -q "huggingface_hub[cli]"
fi

echo "==> Grounding DINO tiny (~700 MB) for baseline comparison"
hf download IDEA-Research/grounding-dino-tiny --local-dir "${OUT}"
echo "Done: ${OUT}"
