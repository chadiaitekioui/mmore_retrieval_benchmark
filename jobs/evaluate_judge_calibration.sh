#!/usr/bin/env bash
# Metrics for judge calibration runs (run_C_suff_* @ Hit@10).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

GT="${ROOT}/data/ground_truth.json"
[[ -f "$GT" ]] || { echo "Missing $GT" >&2; exit 1; }

for run in run_C_suff_030 run_C_suff_050 run_C_suff_070 run_C_suff_090; do
  chunks="results/${run}/chunks.json"
  [[ -f "$chunks" ]] || { echo "Missing $chunks — run collect_judge_calibration.sh" >&2; exit 1; }
  python scripts/02_compute_metrics.py \
    --gt "$GT" \
    --chunks "$chunks" \
    --out "results/${run}/metrics.json" \
    --run-name "$run" \
    --k 10
done

echo "✓ Calibration metrics written under results/run_C_suff_*/metrics.json"
