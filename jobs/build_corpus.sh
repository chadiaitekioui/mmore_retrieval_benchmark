#!/usr/bin/env bash
# PLOS download + MMORE process/postprocess/index → proc_demo.db (GPU recommended).
#
# Prerequisites:
#   bash install.sh && cd ... && source env.benchmark
#
# Usage:
#   source env.benchmark
#   bash jobs/build_corpus.sh 1000
#   bash jobs/build_corpus.sh 5000
#
# Env:
#   CLEAN_DB=1 (default) — remove DB_URI before indexing
#   SKIP_DOWNLOAD=1 / SKIP_CONVERT=1 — reuse cached PLOS data
#
# Run:ai submit example:
#   runai submit mmore-build-corpus \\
#     --image "\$IMAGE" --gpu 1 --cpu 8 --memory 32G \\
#     --pvc "\$PVC:/workspace" --working-dir /workspace/mmore_retrieval_benchmark \\
#     -e MMORE_ROOT=/workspace/mmore \\
#     -- bash jobs/build_corpus.sh 1000

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

SIZE="${1:-1000}"
MMORE_ROOT="${MMORE_ROOT:-$(cd "${ROOT}/.." && pwd)/mmore}"

export MMORE_ROOT
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export DB_URI="${DB_URI:-${ROOT}/proc_demo.db}"
export DB_NAME="${DB_NAME:-my_db}"
export COLLECTION_NAME="${COLLECTION_NAME:-my_db}"
export CLEAN_DB="${CLEAN_DB:-1}"

mkdir -p "$(dirname "$DB_URI")" "$HF_HOME"

echo "=== Corpus build (PLOS-${SIZE}) ==="
echo "  BENCH_ROOT=$ROOT"
echo "  MMORE_ROOT=$MMORE_ROOT"
echo "  HF_HOME=$HF_HOME"
echo "  DB_URI=$DB_URI (CLEAN_DB=$CLEAN_DB)"

if [[ ! -d "$MMORE_ROOT/src/mmore" ]]; then
  echo "MMORE repo not found at: $MMORE_ROOT" >&2
  exit 1
fi

bash "${ROOT}/corpus/build_index.sh" "$SIZE"
