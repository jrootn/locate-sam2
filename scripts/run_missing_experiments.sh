#!/usr/bin/env bash
# Run numeric benchmarks still missing from the v1 suite (RefCOCO-g, RefCOCO+ extras).
# OOD is separate: add images to experiments/ood/images/, then scripts/run_ood_batch.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${HOME}/missing_experiments.log"
SWINT="${ROOT}/data/grounding-dino-swint"
REFCOCOG_REFS="${ROOT}/data/refcocog/refs(google).p"

cd "${ROOT}"
source .venv/bin/activate
export PYTHONPATH="${ROOT}"

run_eval() {
  python scripts/run_eval.py --subset-size 0 "$@"
}

maybe_eval() {
  local summary="$1"
  shift
  if [[ -f "${summary}" ]]; then
    echo "--- SKIP (exists): $(basename "${summary}") ---"
  else
    echo "--- RUN: $* ---"
    run_eval "$@"
  fi
}

{
  echo "=== MISSING EXPERIMENTS START $(date -Is) ==="

  if [[ ! -f "${REFCOCOG_REFS}" ]]; then
    echo "--- RefCOCO-g data missing; run: bash scripts/download_data.sh ---"
  else
    OUT_G="${ROOT}/outputs/refcocog"
    mkdir -p "${OUT_G}"

    maybe_eval "${OUT_G}/locate_sam2_hybrid_refcocog_summary.json" \
      --dataset refcocog --output-dir "${OUT_G}" \
      --grounder locateanything --generation-mode hybrid \
      --tag locate_sam2_hybrid_refcocog

    maybe_eval "${OUT_G}/locate_sam2_fast_refcocog_summary.json" \
      --dataset refcocog --output-dir "${OUT_G}" \
      --grounder locateanything --generation-mode fast \
      --tag locate_sam2_fast_refcocog

    maybe_eval "${OUT_G}/dino_tiny_refcocog_summary.json" \
      --dataset refcocog --output-dir "${OUT_G}" \
      --grounder grounding_dino_tiny --tag dino_tiny_refcocog

    if [[ -f "${SWINT}/config.json" ]]; then
      maybe_eval "${OUT_G}/dino_swint_refcocog_summary.json" \
        --dataset refcocog --output-dir "${OUT_G}" \
        --grounder grounding_dino_swint --tag dino_swint_refcocog
    else
      echo "--- SKIP dino_swint_refcocog (download: bash scripts/download_dino_swint.sh) ---"
    fi

    maybe_eval "${OUT_G}/gt_oracle_refcocog_summary.json" \
      --dataset refcocog --output-dir "${OUT_G}" \
      --grounder gt_oracle --tag gt_oracle_refcocog
  fi

  OUT_PLUS="${ROOT}/outputs/refcoco_plus"
  mkdir -p "${OUT_PLUS}"

  maybe_eval "${OUT_PLUS}/gt_oracle_refcoco_plus_summary.json" \
    --dataset refcoco+ --output-dir "${OUT_PLUS}" \
    --grounder gt_oracle --tag gt_oracle_refcoco_plus

  maybe_eval "${OUT_PLUS}/locate_sam2_fast_refcoco_plus_summary.json" \
    --dataset refcoco+ --output-dir "${OUT_PLUS}" \
    --grounder locateanything --generation-mode fast \
    --tag locate_sam2_fast_refcoco_plus

  echo "--- Build consolidated tables ---"
  python scripts/build_experiment_tables.py

  echo ""
  echo "OOD (manual): place images in experiments/ood/images/, edit prompts.csv, then:"
  echo "  python scripts/run_ood_batch.py"
  echo "  # score outputs/ood/results_template.csv -> results.csv"
  echo "  python scripts/aggregate_ood_scores.py"

  echo "=== MISSING EXPERIMENTS DONE $(date -Is) ==="
} | tee "${LOG}"
