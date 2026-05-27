#!/usr/bin/env bash
# Judge calibration analysis: threshold curve + judge score vs exam Hit correlation.
#
# Prerequisites:
#   results/run_C/chunks.json
#   data/ground_truth.json
#
# Optional (full curve with observed corrective + Hit@10):
#   bash jobs/collect_judge_calibration.sh
#   bash jobs/evaluate_judge_calibration.sh
#
# Usage:
#   bash jobs/judge_calibration.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

pip install -r requirements.txt -q

python scripts/05_judge_calibration.py \
  --chunks "${ROOT}/results/run_C/chunks.json" \
  --gt "${ROOT}/data/ground_truth.json" \
  --calib-dir "${ROOT}/results" \
  --out-dir "${ROOT}/results/judge_calibration"

echo "✓ See results/judge_calibration/calibration.json and calibration_curve.png"
