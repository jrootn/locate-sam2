#!/usr/bin/env bash
# Optional: Grounding DINO SwinT/Base (~1.5 GB) for stronger baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/data/grounding-dino-swint"
mkdir -p "${OUT}"

if ! command -v hf >/dev/null 2>&1; then
  pip install -q "huggingface_hub[cli]"
fi

echo "==> Grounding DINO base/SwinT (~1.5 GB) for stronger baseline"
hf download IDEA-Research/grounding-dino-base --local-dir "${OUT}"
echo "Done: ${OUT}"
