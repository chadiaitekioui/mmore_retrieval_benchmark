#!/usr/bin/env bash
# Smoke test — collect run_C on 30 questions (RAG API must already be running).
#
# Usage:
#   bash jobs/test_judge.sh [BASE_URL]
#
# Default BASE_URL: http://localhost:8000

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_URL="${1:-http://localhost:8000}"
QUERIES="data/medxpertqa_30_mmore.jsonl"

if [[ ! -f "$QUERIES" ]]; then
  echo "Missing $QUERIES — run: bash jobs/00_prepare.sh --pilot" >&2
  exit 1
fi

pip install -r requirements.txt -q
mkdir -p results/run_C

echo "=== Judge smoke test (30 q) → results/run_C/chunks.json ==="

python scripts/collect_from_api.py \
  --queries "$QUERIES" \
  --out results/run_C/chunks.json \
  --api-type rag \
  --base-url "$BASE_URL" \
  --k 5 \
  --raw-out results/run_C/raw_api_smoke.json

echo "✓ Done — inspect results/run_C/chunks.json for judge fields"
