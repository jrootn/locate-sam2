#!/usr/bin/env bash
# Full VM bootstrap after models/data are downloaded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash scripts/setup_env.sh
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "If nvidia-smi fails, run: bash scripts/setup_gpu.sh"
echo "For RefCOCO eval images: bash scripts/download_train2014.sh  (~13 GB)"
