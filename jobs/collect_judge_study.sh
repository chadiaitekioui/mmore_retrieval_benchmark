#!/usr/bin/env bash
# Collect judge study runs (axis 2 + axis 3).
#
# Prerequisites:
#   source env.benchmark
#   export HF_TOKEN=hf_...
#   proc_demo.db + data/medxpertqa_200_mmore.jsonl
#
# Usage:
#   bash jobs/collect_judge_study.sh
#
# Outputs:
#   results/run_steps_{0,1,2,3}/chunks.json
#   results/run_judge_scout/chunks.json
#   results/run_force_{rr,aq,ac}/chunks.json

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

: "${DB_URI:?source env.benchmark}"
: "${HF_TOKEN:?HF_TOKEN required (gated Llama)}"

RAG_PORT="${MMORE_RAG_PORT:-8000}"
STARTUP_WAIT="${MMORE_STARTUP_WAIT:-600}"
STUDY_RUNS=(
  run_steps_0 run_steps_1 run_steps_2 run_steps_3
  run_judge_scout
  run_force_rr run_force_aq run_force_ac
)

MMORE_PID=""
stop_mmore() {
  if [[ -n "${MMORE_PID}" ]] && kill -0 "${MMORE_PID}" 2>/dev/null; then
    kill "${MMORE_PID}" 2>/dev/null || true
    wait "${MMORE_PID}" 2>/dev/null || true
  fi
  MMORE_PID=""
}
trap stop_mmore EXIT

wait_url() {
  local url="$1"
  local i=0
  while ((i < STARTUP_WAIT)); do
    curl -sf "${url}" >/dev/null 2>&1 && return 0
    sleep 2
    ((i += 2)) || true
  done
  echo "Timed out waiting for ${url}" >&2
  return 1
}

pip install -r requirements.txt -q 2>/dev/null || true

for run in "${STUDY_RUNS[@]}"; do
  cfg="${ROOT}/config/rag/study/${run}.yaml"
  [[ -f "$cfg" ]] || { echo "Missing $cfg — run: python scripts/generate_judge_study_configs.py" >&2; exit 1; }
  stop_mmore
  echo "=== Start MMORE RAG (${run}) ==="
  python -m mmore rag --config-file "$cfg" &
  MMORE_PID=$!
  wait_url "http://127.0.0.1:${RAG_PORT}/health"
  bash "${ROOT}/jobs/01_collect.sh" "$run" "http://127.0.0.1:${RAG_PORT}"
  stop_mmore
done

echo "✓ Judge study collects done. Next:"
echo "  bash jobs/evaluate_judge_study.sh"
echo "  bash jobs/judge_study.sh"
