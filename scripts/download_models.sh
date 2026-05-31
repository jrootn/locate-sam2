#!/usr/bin/env bash
# Download model weights into ./models/ (run locally or on GCP — not on git).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT}/models"
mkdir -p "${MODELS_DIR}"

echo "==> Download directory: ${MODELS_DIR}"
echo "==> Estimated download size: ~10 GB (LocateAnything + SAM 2.1)"

if ! command -v hf >/dev/null 2>&1; then
  echo "Installing huggingface_hub CLI..."
  pip install -q "huggingface_hub[cli]"
fi

echo "==> [1/2] LocateAnything-3B (~7.8 GB, not gated)"
hf download nvidia/LocateAnything-3B \
  --local-dir "${MODELS_DIR}/LocateAnything-3B"

echo "==> [2/2] SAM 2.1 Hiera Large (~0.9 GB, not gated)"
if [[ ! -f "${MODELS_DIR}/sam2.1_hiera_large.pt" ]]; then
  wget -q --show-progress \
    -O "${MODELS_DIR}/sam2.1_hiera_large.pt" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
else
  echo "    Already exists, skipping."
fi

echo ""
echo "Done. Models ready under ${MODELS_DIR}/"
du -sh "${MODELS_DIR}"/* 2>/dev/null || true

echo ""
echo "Optional — SAM 3 (gated, request access first):"
echo "  https://huggingface.co/facebook/sam3"
echo "  hf auth login"
echo "  hf download facebook/sam3.1 --local-dir ${MODELS_DIR}/sam3.1"
