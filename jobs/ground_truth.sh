#!/usr/bin/env bash
# Build ground truth labels (union of chunks from run_A, run_B, run_C).
#
# Prerequisites:
#   source env.benchmark
#   results/run_{A,B,C}/chunks.json
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash jobs/ground_truth.sh
#
# Hugging Face annotator:
#   export HF_TOKEN=hf_...
#   export ANNOTATOR_MODEL=meta-llama/Llama-3.1-8B-Instruct
#   bash jobs/ground_truth.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

exec bash "${ROOT}/jobs/02_annotate.sh" "$@"
