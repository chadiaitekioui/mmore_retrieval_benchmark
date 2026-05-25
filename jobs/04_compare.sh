#!/usr/bin/env bash
# Step 4 — comparison table + McNemar (primary: run_B → run_C).
#
# Usage:
#   bash jobs/04_compare.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pip install -r requirements.txt -q

python scripts/03_compare_runs.py \
  --results-dir results/ \
  --baseline run_A \
  --judge-pair run_B,run_C

echo "✓ Summary → results/summary.json"
