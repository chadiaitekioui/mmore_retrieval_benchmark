#!/usr/bin/env bash
# Collect run_C variants for judge sufficiency calibration (0.3 / 0.5 / 0.7 / 0.9).
#
# Prerequisites:
#   source env.benchmark
#   export HF_TOKEN=hf_...
#   proc_demo.db + data/medxpertqa_200_mmore.jsonl
#
# Usage:
#   bash jobs/collect_judge_calibration.sh
#
# Outputs:
#   results/run_C_suff_{030,050,070,090}/chunks.json

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

: "${DB_URI:?source env.benchmark}"
: "${HF_TOKEN:?HF_TOKEN required (gated Llama for run_C judge)}"

RAG_PORT="${MMORE_RAG_PORT:-8000}"
STARTUP_WAIT="${MMORE_STARTUP_WAIT:-600}"
CALIB_RUNS=(run_C_suff_030 run_C_suff_050 run_C_suff_070 run_C_suff_090)

python scripts/generate_calib_configs.py

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

pip install -r requirements.txt -q

for run in "${CALIB_RUNS[@]}"; do
  tag="${run#run_C_suff_}"
  cfg="${ROOT}/config/rag/calib/run_C_suff_${tag}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "Missing $cfg" >&2
    exit 1
  fi
  stop_mmore
  echo "=== Start MMORE RAG (${run}) ==="
  python -m mmore rag --config-file "$cfg" &
  MMORE_PID=$!
  wait_url "http://127.0.0.1:${RAG_PORT}/health"
  bash "${ROOT}/jobs/01_collect.sh" "$run" "http://127.0.0.1:${RAG_PORT}"
  stop_mmore
done

echo "✓ Calibration collects done. Next:"
echo "  bash jobs/evaluate_judge_calibration.sh"
echo "  bash jobs/judge_calibration.sh"
