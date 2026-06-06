#!/usr/bin/env bash
# Minimal benchmark data for RefCOCO-family eval (publish-first setup).
# Total download: ~7-8 GB (val images + COCO mask annotations + RefCOCO JSONs)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT}/data"
mkdir -p "${DATA_DIR}"

echo "==> Data directory: ${DATA_DIR}"
echo "==> Estimated download: ~7-8 GB (minimal eval pack)"

download() {
  local url="$1"
  local dest="$2"
  if [[ -f "${dest}" ]]; then
    echo "    skip (exists): $(basename "${dest}")"
    return 0
  fi
  echo "    fetching: $(basename "${dest}")"
  wget -q --show-progress -O "${dest}" "${url}"
}

extract_if_needed() {
  local archive="$1"
  local marker="$2"
  if [[ -e "${marker}" ]]; then
    echo "    skip extract (exists): ${marker}"
    return 0
  fi
  echo "    extracting: $(basename "${archive}")"
  unzip -q -o "${archive}" -d "${DATA_DIR}"
}

echo "==> Installing unzip (required for COCO/RefCOCO archives)"
if ! command -v unzip >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq unzip
fi

# --- COCO 2014 val images (needed for RefCOCO val/test splits) ---
echo "==> [1/5] COCO val2014 images (~6.2 GB)"
download "http://images.cocodataset.org/zips/val2014.zip" "${DATA_DIR}/val2014.zip"
extract_if_needed "${DATA_DIR}/val2014.zip" "${DATA_DIR}/val2014"

# --- COCO instance annotations (GT masks for mIoU) ---
echo "==> [2/5] COCO 2014 annotations (~250 MB)"
download "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" \
  "${DATA_DIR}/annotations_trainval2014.zip"
extract_if_needed "${DATA_DIR}/annotations_trainval2014.zip" \
  "${DATA_DIR}/annotations/instances_val2014.json"

# --- RefCOCO / RefCOCO+ / RefCOCOg referring expression annotations ---
# UNC links are often down; use Internet Archive mirrors (see lichengunc/refer issue #14)
REF_BASE="https://web.archive.org/web"
echo "==> [3/5] RefCOCO annotations"
download "${REF_BASE}/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip" \
  "${DATA_DIR}/refcoco.zip"
download "${REF_BASE}/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip" \
  "${DATA_DIR}/refcoco+.zip"
download "${REF_BASE}/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip" \
  "${DATA_DIR}/refcocog.zip"

for name in refcoco "refcoco+" refcocog; do
  extract_if_needed "${DATA_DIR}/${name}.zip" "${DATA_DIR}/${name}"
done

# --- Optional: Grounding DINO tiny for ONE local baseline (not full Grounded SAM) ---
echo "==> [4/5] Grounding DINO tiny baseline (~700 MB, optional but recommended)"
if command -v hf >/dev/null 2>&1; then
  hf download IDEA-Research/grounding-dino-tiny \
    --local-dir "${DATA_DIR}/grounding-dino-tiny" || \
    echo "    warning: grounding-dino download failed (optional)"
else
  echo "    skip: install hf CLI first (pip install huggingface_hub[cli])"
fi

# --- refer API helper (for loading RefCOCO splits) ---
echo "==> [5/5] refer API (small, for eval scripts)"
REFER_DIR="${DATA_DIR}/refer"
if [[ ! -f "${REFER_DIR}/refer.py" ]]; then
  git clone --depth 1 https://github.com/lichengunc/refer.git "${REFER_DIR}"
fi

echo ""
echo "==> Done. Layout:"
echo "    ${DATA_DIR}/val2014/              COCO val images"
echo "    ${DATA_DIR}/annotations/          COCO instance masks (instances_val2014.json)"
echo "    ${DATA_DIR}/refcoco/              RefCOCO refs (UNC/Google splits)"
echo "    ${DATA_DIR}/refcoco+/             RefCOCO+ refs"
echo "    ${DATA_DIR}/refcocog/             RefCOCOg refs"
echo "    ${DATA_DIR}/grounding-dino-tiny/  optional baseline detector"
echo ""
du -sh "${DATA_DIR}"/* 2>/dev/null | head -20
df -h "${DATA_DIR}" | tail -1
