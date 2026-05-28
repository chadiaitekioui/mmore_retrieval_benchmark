#!/usr/bin/env bash
# Collect benchmark runs in one terminal: start MMORE per config, HTTP collect, stop server.
#
# Prerequisites:
#   source env.benchmark
#   export HF_TOKEN=hf_...   # gated Llama for run_C / run_C_ctrl (README § Prerequisites)
#   proc_demo.db built (jobs/setup_medxpertqa.sh)
#   bash jobs/00_prepare.sh
#
# Usage:
#   source env.benchmark
#   bash jobs/collect_all.sh
#
# Env:
#   MMORE_RETRIEVER_PORT=8001   retriever API (runs A,B,D,E,F)
#   MMORE_RAG_PORT=8000         RAG + judge (runs C, C_ctrl)
#   MMORE_STARTUP_WAIT=600      seconds max wait per server start
#   RUNS="run_A run_B ..."      explicit subset override

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

: "${DB_URI:?Set DB_URI (source env.benchmark after install)}"
: "${DB_NAME:?Missing DB_NAME}"
: "${COLLECTION_NAME:?Missing COLLECTION_NAME}"

RETRIEVER_PORT="${MMORE_RETRIEVER_PORT:-8001}"
RAG_PORT="${MMORE_RAG_PORT:-8000}"
STARTUP_WAIT="${MMORE_STARTUP_WAIT:-600}"
ALL_RUNS=(run_A run_B run_C run_C_ctrl run_D run_E run_F run_G)

if [[ -n "${RUNS:-}" ]]; then
  # shellcheck disable=SC2206
  ALL_RUNS=($RUNS)
fi

MMORE_PID=""

stop_mmore() {
  if [[ -n "${MMORE_PID}" ]] && kill -0 "${MMORE_PID}" 2>/dev/null; then
    echo "  Stopping MMORE (pid ${MMORE_PID})"
    kill "${MMORE_PID}" 2>/dev/null || true
    wait "${MMORE_PID}" 2>/dev/null || true
  fi
  MMORE_PID=""
}

wait_url() {
  local url="$1"
  local i=0
  while ((i < STARTUP_WAIT)); do
    if curl -sf "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    ((i += 2)) || true
  done
  echo "Timed out waiting for ${url}" >&2
  return 1
}

config_for_run() {
  case "$1" in
    run_C) echo "${ROOT}/config/rag/run_C_api.yaml" ;;
    run_C_ctrl) echo "${ROOT}/config/rag/run_C_ctrl_api.yaml" ;;
    run_steps_*|run_judge_scout|run_force_*)
      echo "${ROOT}/config/rag/study/${1}.yaml"
      ;;
    *) echo "${ROOT}/config/retrieve/${1}.yaml" ;;
  esac
}

require_hf_token_for_rag() {
  if [[ -n "${HF_TOKEN:-}" ]] || huggingface-cli whoami &>/dev/null; then
    return 0
  fi
  echo "Missing HF_TOKEN for ${1} (gated meta-llama/Llama-3.1-8B-Instruct)." >&2
  echo "  See README.md — Prerequisites: Hugging Face (gated Llama)" >&2
  echo "  export HF_TOKEN=hf_...   # or: huggingface-cli login" >&2
  exit 1
}

start_mmore() {
  local run="$1"
  local cfg
  cfg="$(config_for_run "$run")"
  if [[ ! -f "$cfg" ]]; then
    echo "Missing config: $cfg" >&2
    exit 1
  fi

  stop_mmore

  case "$run" in
    run_C|run_C_ctrl|run_steps_*|run_judge_scout|run_force_*)
      require_hf_token_for_rag "$run"
      echo "=== Start MMORE RAG (${run}) on :${RAG_PORT} ==="
      python -m mmore rag --config-file "$cfg" &
      MMORE_PID=$!
      wait_url "http://127.0.0.1:${RAG_PORT}/health"
      COLLECT_URL="http://127.0.0.1:${RAG_PORT}"
      ;;
    *)
      echo "=== Start MMORE retrieve (${run}) on :${RETRIEVER_PORT} ==="
      python -m mmore retrieve --config-file "$cfg" --host 127.0.0.1 --port "${RETRIEVER_PORT}" &
      MMORE_PID=$!
      # Retriever API has no /health; OpenAPI is enough.
      wait_url "http://127.0.0.1:${RETRIEVER_PORT}/openapi.json"
      COLLECT_URL="http://127.0.0.1:${RETRIEVER_PORT}"
      ;;
  esac
}

trap stop_mmore EXIT

pip install -r requirements.txt -q 2>/dev/null || true

for run in "${ALL_RUNS[@]}"; do
  start_mmore "$run"
  bash "${ROOT}/jobs/01_collect.sh" "$run" "$COLLECT_URL"
  stop_mmore
done

echo "✓ All runs collected under results/"
