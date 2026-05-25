#!/usr/bin/env bash
# Run PLOS corpus download + MMORE process/postprocess/index on Run:ai (GPU + PVC).
#
# Prerequisites on PVC:
#   - This benchmark repo (mmore-bench)
#   - MMORE fork at $MMORE_ROOT (branch llm-as-a-judge): pip install -e
#
# Usage (inside a Run:ai job shell, or submit wrapper):
#   export MMORE_ROOT=/workspace/mmore
#   export HF_HOME=/workspace/hf_cache
#   export DB_URI=/workspace/mmore-bench/proc_demo.db
#   bash jobs/00_build_corpus.sh 1000
#   bash jobs/00_build_corpus.sh 5000
#
# Submit example (adjust IMAGE / PVC / paths):
#   runai submit mmore-build-corpus \
#     --image "$IMAGE" --gpu 1 --cpu 8 --memory 32G \
#     --pvc "$PVC:/workspace" --working-dir /workspace/mmore-bench \
#     -e MMORE_ROOT=/workspace/mmore \
#     -e HF_HOME=/workspace/hf_cache \
#     -- bash jobs/00_build_corpus.sh 1000

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SIZE="${1:-1000}"
MMORE_ROOT="${MMORE_ROOT:-/workspace/mmore}"
WORKDIR="${WORKDIR:-$ROOT}"
HF_CACHE="${HF_HOME:-${WORKDIR}/hf_cache}"

export MMORE_ROOT
export HF_HOME="$HF_CACHE"
export DB_URI="${DB_URI:-${WORKDIR}/proc_demo.db}"
export DB_NAME="${DB_NAME:-my_db}"
export COLLECTION_NAME="${COLLECTION_NAME:-my_db}"

mkdir -p "$(dirname "$DB_URI")" "$HF_CACHE"

echo "=== Run:ai corpus build (PLOS-${SIZE}) ==="
echo "  WORKDIR=$ROOT"
echo "  MMORE_ROOT=$MMORE_ROOT"
echo "  HF_HOME=$HF_HOME"
echo "  DB_URI=$DB_URI"

bash "${ROOT}/corpus/build_index.sh" "$SIZE"
