#!/usr/bin/env bash
# Step 3 — compute Hit@k, MRR, NDCG and judge stats for all runs.
#
# Usage:
#   bash jobs/03_metrics.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pip install -r requirements.txt -q

for run in run_A run_B run_C run_C_ctrl run_D run_E run_F; do
  echo "=== Metrics $run ==="
  python scripts/02_compute_metrics.py \
    --gt data/ground_truth.json \
    --chunks "results/${run}/chunks.json" \
    --out "results/${run}/metrics.json" \
    --run-name "$run"
done

echo "✓ All metrics written under results/*/metrics.json"
