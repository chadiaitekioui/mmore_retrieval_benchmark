#!/usr/bin/env bash
# Retrieval + downstream RAG quality metrics for judge study runs.
#
# Usage:
#   bash jobs/evaluate_judge_study.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pip install -r requirements.txt -q 2>/dev/null || true

STUDY_RUNS=(
  run_steps_0 run_steps_1 run_steps_2
  run_judge_scout
  run_force_rr run_force_aq run_force_ac
)

for run in "${STUDY_RUNS[@]}"; do
  chunks="results/${run}/chunks.json"
  [[ -f "$chunks" ]] || { echo "Missing $chunks — run collect_judge_study.sh" >&2; exit 1; }

  echo "=== Retrieval metrics $run ==="
  python scripts/02_compute_metrics.py \
    --gt data/ground_truth.json \
    --chunks "$chunks" \
    --out "results/${run}/metrics.json" \
    --run-name "$run" \
    --k 10

  max_steps=""
  case "$run" in
    run_steps_*) max_steps="${run#run_steps_}" ;;
  esac

  echo "=== RAG quality $run ==="
  extra=()
  [[ -n "$max_steps" ]] && extra+=(--max-corrective-steps "$max_steps")
  python scripts/eval_rag_quality.py \
    --questions data/medxpertqa_200.jsonl \
    --chunks "$chunks" \
    --out "results/${run}/rag_quality.json" \
    --run-name "$run" \
    "${extra[@]}"
done

echo "✓ Judge study metrics under results/run_steps_*/ and results/run_force_*/"
