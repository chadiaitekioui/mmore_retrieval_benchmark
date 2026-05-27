#!/usr/bin/env bash
# Step 2 — corpus-level ground truth (union run_A + run_B + run_C).
#
# Annotator backends (set ANNOTATOR_MODEL + credentials for one of these):
#
#   OpenAI API (default):
#     export OPENAI_API_KEY=sk-...
#     export ANNOTATOR_MODEL=gpt-4o-mini    # optional; default is gpt-4o-mini
#     # export ANNOTATOR_MODEL=gpt-4o
#     bash jobs/02_annotate.sh
#
#   OpenAI-compatible endpoint (Azure, vLLM, LiteLLM, …):
#     export OPENAI_API_KEY=...
#     export OPENAI_BASE_URL=https://your-host/v1
#     export ANNOTATOR_MODEL=<model name on that server>
#     bash jobs/02_annotate.sh
#
#   Hugging Face (local GPU, e.g. Llama-3.1-8B-Instruct):
#     export HF_TOKEN=...   # required for gated meta-llama/* weights
#     export ANNOTATOR_MODEL=meta-llama/Llama-3.1-8B-Instruct
#     bash jobs/02_annotate.sh
#
# Env:
#   ANNOTATOR_MODEL     default: gpt-4o-mini (OpenAI). HF id if it contains '/'.
#   OPENAI_API_KEY      required for OpenAI / compatible API backends
#   OPENAI_BASE_URL     optional; passed as --base-url to the Python script

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${ANNOTATOR_MODEL:-gpt-4o-mini}"
if [[ "$MODEL" == */* ]] || [[ "$MODEL" == "local" ]] || [[ "$MODEL" == "hf" ]]; then
  pip install -r requirements-hf-annotate.txt -q
fi

ARGS=(
  --chunks results/run_A/chunks.json
  --chunks results/run_B/chunks.json
  --chunks results/run_C/chunks.json
  --out data/ground_truth.json
  --model "$MODEL"
  --incremental
)
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  ARGS+=(--base-url "$OPENAI_BASE_URL")
fi

python scripts/01_annotate_ground_truth.py "${ARGS[@]}"
