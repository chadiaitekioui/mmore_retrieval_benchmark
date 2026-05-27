#!/usr/bin/env bash
# Clinical corpus (MedRAG) or legacy PLoS → proc_demo.db (GPU recommended for index).
#
# Prerequisites:
#   bash install.sh && cd ... && source env.benchmark
#
# Usage:
#   source env.benchmark
#   bash jobs/build_corpus.sh                    # MedRAG: StatPearls + textbooks
#   MEDRAG_PILOT=1 bash jobs/build_corpus.sh     # 5k snippets smoke test
#   CORPUS=plos bash jobs/build_corpus.sh 5000   # legacy PLoS-5k
#
# Env:
#   CORPUS=medrag|plos (default: medrag)
#   CLEAN_DB=1 (default) — remove DB_URI before indexing
#   SKIP_DOWNLOAD=1 / SKIP_CONVERT=1 — reuse cached data
#   MEDRAG_SOURCES="statpearls textbooks"
#
# Run:ai submit example:
#   runai submit mmore-build-corpus \\
#     --image "\$IMAGE" --gpu 1 --cpu 8 --memory 32G \\
#     --pvc "\$PVC:/workspace" --working-dir /workspace/mmore_retrieval_benchmark \\
#     -e MMORE_ROOT=/workspace/mmore \\
#     -- bash jobs/build_corpus.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

CORPUS="${CORPUS:-medrag}"
SIZE="${1:-}"
MMORE_ROOT="${MMORE_ROOT:-$(cd "${ROOT}/.." && pwd)/mmore}"

export MMORE_ROOT
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export DB_URI="${DB_URI:-${ROOT}/proc_demo.db}"
export DB_NAME="${DB_NAME:-my_db}"
export COLLECTION_NAME="${COLLECTION_NAME:-my_db}"
export CLEAN_DB="${CLEAN_DB:-1}"

mkdir -p "$(dirname "$DB_URI")" "$HF_HOME"

echo "=== Corpus build (${CORPUS}) ==="
echo "  BENCH_ROOT=$ROOT"
echo "  MMORE_ROOT=$MMORE_ROOT"
echo "  HF_HOME=$HF_HOME"
echo "  DB_URI=$DB_URI (CLEAN_DB=$CLEAN_DB)"

if [[ ! -d "$MMORE_ROOT/src/mmore" ]]; then
  echo "MMORE repo not found at: $MMORE_ROOT" >&2
  exit 1
fi

if [[ "$CORPUS" == "plos" ]]; then
  SIZE="${SIZE:-1000}"
  echo "  PLoS size=${SIZE}"
  bash "${ROOT}/corpus/build_index.sh" "$SIZE"
else
  if [[ -n "$SIZE" && "$SIZE" != "medrag" ]]; then
    echo "[!] Ignoring positional arg '$SIZE' for CORPUS=medrag (use MEDRAG_PILOT=1 for subset)" >&2
  fi
  bash "${ROOT}/corpus/build_index_medrag.sh"
fi
