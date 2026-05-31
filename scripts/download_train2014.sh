#!/usr/bin/env bash
# RefCOCO eval requires COCO train2014 images (~13 GB).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT}/data"
mkdir -p "${DATA_DIR}"

echo "==> Downloading COCO train2014 (~13 GB) for RefCOCO eval"
if [[ ! -d "${DATA_DIR}/train2014" ]]; then
  wget -q --show-progress -O "${DATA_DIR}/train2014.zip" \
    http://images.cocodataset.org/zips/train2014.zip
  unzip -q -o "${DATA_DIR}/train2014.zip" -d "${DATA_DIR}"
fi
echo "Done: ${DATA_DIR}/train2014"
