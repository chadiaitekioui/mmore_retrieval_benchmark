#!/usr/bin/env bash
# MedXpertQA setup: PLOS corpus (proc_demo.db) + query JSONL (steps 2 + 3).
#
# Prerequisites: install.sh done, then source env.benchmark
#
# Usage:
#   source env.benchmark
#   bash jobs/setup_medxpertqa.sh           # PLOS-1k + 200 questions
#   bash jobs/setup_medxpertqa.sh 5000      # PLOS-5k
#   bash jobs/setup_medxpertqa.sh 1000 --pilot
#
# Env: same as build_corpus.sh (CLEAN_DB, SKIP_DOWNLOAD, COLLECTION_NAME, …)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

SIZE="1000"
PREPARE_ARGS=()
for arg in "$@"; do
  case "$arg" in
    1000|5000) SIZE="$arg" ;;
    *) PREPARE_ARGS+=("$arg") ;;
  esac
done

echo "=== MedXpertQA setup (corpus ${SIZE} + queries) ==="
bash "${ROOT}/jobs/build_corpus.sh" "$SIZE"
bash "${ROOT}/jobs/00_prepare.sh" "${PREPARE_ARGS[@]}"

echo ""
echo "✓ Ready for collect:"
echo "  data/medxpertqa_200.jsonl"
echo "  data/medxpertqa_200_mmore.jsonl"
echo "  DB_URI=${DB_URI:-${ROOT}/proc_demo.db}"
echo "  export HF_TOKEN=hf_...   # README § Prerequisites (run_C / run_C_ctrl)"
echo "  bash jobs/collect_all.sh"
