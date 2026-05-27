#!/usr/bin/env bash
# HyDE query expansion for run_H (LangChain HypotheticalDocumentEmbedder).
#
# Prerequisites:
#   source env.benchmark
#   bash jobs/00_prepare.sh
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash jobs/prepare_hyde_queries.sh
#
#   # Local HF (gated Llama):
#   export HF_TOKEN=hf_...
#   export HYDE_MODEL=meta-llama/Llama-3.1-8B-Instruct
#   pip install -r requirements-hyde.txt -r requirements-hf-annotate.txt
#   bash jobs/prepare_hyde_queries.sh
#
# Env:
#   HYDE_MODEL          default gpt-4o-mini (OpenAI)
#   OPENAI_BASE_URL     optional compatible API
#   QUERIES_IN / QUERIES_OUT  override paths

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

QUERIES_IN="${QUERIES_IN:-data/medxpertqa_200_mmore.jsonl}"
QUERIES_OUT="${QUERIES_OUT:-data/medxpertqa_200_hyde_mmore.jsonl}"

pip install -r requirements-hyde.txt -q
if [[ -n "${HYDE_MODEL:-}" ]] && [[ "$HYDE_MODEL" == */* ]]; then
  pip install -r requirements-hf-annotate.txt langchain-huggingface -q
fi

python scripts/hyde_expand_queries.py \
  --in "$QUERIES_IN" \
  --out "$QUERIES_OUT" \
  --incremental

echo "✓ HyDE queries ready: $QUERIES_OUT"
echo "  bash jobs/01_collect.sh run_H http://localhost:8001"
