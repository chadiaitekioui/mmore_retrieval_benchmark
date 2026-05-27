#!/usr/bin/env bash
# Evaluation: Hit@k / MRR / NDCG for all runs + McNemar summary.
#
# Prerequisites:
#   source env.benchmark
#   data/ground_truth.json
#   results/run_*/chunks.json
#
# Usage:
#   bash jobs/evaluate.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

if [[ ! -f "${ROOT}/data/ground_truth.json" ]]; then
  echo "Missing data/ground_truth.json — run: bash jobs/ground_truth.sh" >&2
  exit 1
fi

for run in run_A run_B run_C run_C_ctrl run_D run_E run_F run_G run_H; do
  if [[ ! -f "results/${run}/chunks.json" ]]; then
    echo "Missing results/${run}/chunks.json — run collect first." >&2
    exit 1
  fi
done

echo "=== Metrics (all runs) ==="
bash "${ROOT}/jobs/03_metrics.sh"

echo "=== Compare (McNemar + summary) ==="
bash "${ROOT}/jobs/04_compare.sh"

echo "✓ Evaluation complete → results/summary.json"
