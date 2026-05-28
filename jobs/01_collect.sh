#!/usr/bin/env bash
# Step 1 — collect chunks.json for one run via MMORE HTTP API.
#
# Prerequisite: MMORE deployed with the matching config (see README.md).
#
# Usage:
#   bash jobs/01_collect.sh run_B http://localhost:8001
#   bash jobs/01_collect.sh run_C http://localhost:8000
#
# Runs: run_A … run_F run_G

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN="${1:?Usage: $0 RUN BASE_URL   e.g. run_B http://localhost:8001}"
BASE_URL="${2:?Missing BASE_URL}"

case "$RUN" in
  run_C|run_C_ctrl|run_C_suff_*) API_TYPE="rag" ;;
  run_A|run_B|run_D|run_E|run_F|run_G) API_TYPE="retriever" ;;
  *)
    echo "Unknown run: $RUN" >&2
    exit 1
    ;;
esac

case "$RUN" in
  run_F|run_G) K=10 ;;
  run_C_suff_*) K=5 ;;  # initial retrieve k; Hit@10 computed at eval
  *) K=5 ;;
esac

QUERY_KEY="input"
RECORD_QUERY_KEY="input"
COLLECT_EXTRA=()
QUERIES="${QUERIES:-data/medxpertqa_200_mmore.jsonl}"

OUT="results/${RUN}/chunks.json"
RAW="results/${RUN}/raw_api.json"

pip install -r requirements.txt -q
mkdir -p "results/${RUN}"

echo "=== Collect $RUN (api=$API_TYPE k=$K queries=$QUERIES) → $OUT ==="

python scripts/collect_from_api.py \
  --queries "$QUERIES" \
  --out "$OUT" \
  --api-type "$API_TYPE" \
  --base-url "$BASE_URL" \
  --k "$K" \
  --query-key "$QUERY_KEY" \
  --record-query-key "$RECORD_QUERY_KEY" \
  --raw-out "$RAW" \
  "${COLLECT_EXTRA[@]}"

echo "✓ $OUT"
