#!/usr/bin/env bash
# Judge study analysis: Pareto (axis 2) + action comparison (axis 3).
#
# Prerequisites:
#   bash jobs/collect_judge_study.sh
#   bash jobs/evaluate_judge_study.sh
#
# Usage:
#   bash jobs/judge_study.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pip install -r requirements.txt -q 2>/dev/null || true

python scripts/judge_study.py \
  --results-dir "${ROOT}/results" \
  --out-dir "${ROOT}/results/judge_study"

echo "✓ results/judge_study/judge_study.json"
echo "✓ results/judge_study/pareto_quality_vs_cost.png"
echo "✓ results/judge_study/axis3_action_comparison.png"
