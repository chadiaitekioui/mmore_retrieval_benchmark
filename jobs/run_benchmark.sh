#!/usr/bin/env bash
# Full benchmark pipeline in one command.
#
# Each run needs its own MMORE deployment (different YAML). Set one base URL per
# deployment — either all at once (parallel endpoints) or one port with redeploy
# between runs (SEQUENTIAL_COLLECT=1).
#
# Usage:
#   # Seven deployments (e.g. different ports or hosts):
#   export MMORE_URL_run_A=http://127.0.0.1:8011
#   export MMORE_URL_run_B=http://127.0.0.1:8012
#   export MMORE_URL_run_C=http://127.0.0.1:8000
#   export MMORE_URL_run_C_ctrl=http://127.0.0.1:8000
#   export MMORE_URL_run_D=http://127.0.0.1:8014
#   export MMORE_URL_run_E=http://127.0.0.1:8015
#   export MMORE_URL_run_F=http://127.0.0.1:8016
#   bash jobs/run_benchmark.sh
#
#   # One retriever port — redeploy MMORE between runs when prompted:
#   export SEQUENTIAL_COLLECT=1
#   export MMORE_RETRIEVER_URL=http://127.0.0.1:8001
#   export MMORE_RAG_URL_C=http://127.0.0.1:8000
#   export MMORE_RAG_URL_C_CTRL=http://127.0.0.1:8000
#   bash jobs/run_benchmark.sh
#
# Options (env):
#   SKIP_PREPARE=1          Skip dataset step
#   SKIP_COLLECT=1          Only annotate + metrics + compare (chunks already present)
#   SKIP_ANNOTATE=1         Skip ground truth (requires data/ground_truth.json)
#   PARALLEL_COLLECT=1      Collect runs with distinct URLs concurrently
#   QUERIES=...             Override questions JSONL (default: data/medxpertqa_200_mmore.jsonl)
#   RUNAI=1                 Submit steps via runai (needs IMAGE, PVC, WORKDIR)
#
# Run:ai example:
#   RUNAI=1 IMAGE=... PVC=... WORKDIR=... \
#   MMORE_URL_run_B=http://mmore-b:8001 ... \
#   bash jobs/run_benchmark.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ALL_RUNS=(run_A run_B run_C run_C_ctrl run_D run_E run_F)
RETRIEVER_RUNS=(run_A run_B run_D run_E run_F)
RAG_RUNS=(run_C run_C_ctrl)
GT_RUNS=(run_A run_B run_C)

# --- URL resolution (per-run > RAG-specific > shared retriever) ---

url_for_run() {
  local run="$1"
  local per_run="MMORE_URL_${run}"
  if [[ -n "${!per_run:-}" ]]; then
    echo "${!per_run}"
    return 0
  fi
  case "$run" in
    run_C)
      if [[ -n "${MMORE_RAG_URL_C:-}" ]]; then echo "${MMORE_RAG_URL_C}"; return 0; fi
      if [[ -n "${MMORE_RAG_URL:-}" ]]; then echo "${MMORE_RAG_URL}"; return 0; fi
      ;;
    run_C_ctrl)
      if [[ -n "${MMORE_RAG_URL_C_CTRL:-}" ]]; then echo "${MMORE_RAG_URL_C_CTRL}"; return 0; fi
      ;;
    run_A|run_B|run_D|run_E|run_F)
      if [[ -n "${MMORE_RETRIEVER_URL:-}" ]]; then echo "${MMORE_RETRIEVER_URL}"; return 0; fi
      if [[ -n "${MMORE_API_URL:-}" ]]; then echo "${MMORE_API_URL}"; return 0; fi
      ;;
  esac
  return 1
}

config_hint_for_run() {
  local run="$1"
  case "$run" in
    run_C) echo "config/rag/run_C_api.yaml" ;;
    run_C_ctrl) echo "config/rag/run_C_ctrl_api.yaml" ;;
    *) echo "config/retrieve/${run}.yaml" ;;
  esac
}

collect_one() {
  local run="$1"
  local url="$2"
  bash "${ROOT}/jobs/01_collect.sh" "$run" "$url"
}

pause_for_redeploy() {
  local run="$1"
  local cfg
  cfg="$(config_hint_for_run "$run")"
  echo ""
  echo ">>> Redeploy MMORE with ${cfg}, ensure API is up, then press Enter to collect ${run}."
  read -r _
}

collect_run() {
  local run="$1"
  local url
  if ! url="$(url_for_run "$run")"; then
    echo "SKIP collect ${run}: set MMORE_URL_${run} or the appropriate MMORE_*_URL (see script header)." >&2
    return 0
  fi
  if [[ "${SEQUENTIAL_COLLECT:-0}" == "1" ]]; then
    pause_for_redeploy "$run"
  fi
  collect_one "$run" "$url"
}

collect_all() {
  local -a ready=()
  local run url

  for run in "${ALL_RUNS[@]}"; do
    if url="$(url_for_run "$run" 2>/dev/null || true)" && [[ -n "$url" ]]; then
      ready+=("$run")
    else
      echo "SKIP collect ${run}: no URL configured." >&2
    fi
  done

  if [[ ${#ready[@]} -eq 0 ]]; then
    echo "No collect URLs set. Export MMORE_URL_run_* or use SEQUENTIAL_COLLECT with MMORE_RETRIEVER_URL / MMORE_RAG_URL_C / MMORE_RAG_URL_C_CTRL." >&2
    exit 1
  fi

  if [[ "${SEQUENTIAL_COLLECT:-0}" == "1" ]]; then
    for run in "${ready[@]}"; do
      collect_run "$run"
    done
    return 0
  fi

  if [[ "${PARALLEL_COLLECT:-0}" == "1" && ${#ready[@]} -gt 1 ]]; then
    local -a pids=()
    for run in "${ready[@]}"; do
      url="$(url_for_run "$run")"
      collect_one "$run" "$url" &
      pids+=($!)
    done
    local ok=0
    for pid in "${pids[@]}"; do
      wait "$pid" || ok=1
    done
    return "$ok"
  fi

  for run in "${ready[@]}"; do
    url="$(url_for_run "$run")"
    collect_one "$run" "$url"
  done
}

run_local() {
  if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
    echo "=== Step 0: dataset ==="
    bash "${ROOT}/jobs/00_prepare.sh" "$@"
  fi

  if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
    echo "=== Step 1: collect (per-run MMORE URLs) ==="
    collect_all
  fi

  if [[ "${SKIP_ANNOTATE:-0}" != "1" ]]; then
    for run in "${GT_RUNS[@]}"; do
      if [[ ! -f "results/${run}/chunks.json" ]]; then
        echo "Missing results/${run}/chunks.json (needed for ground truth)." >&2
        exit 1
      fi
    done
    echo "=== Step 2: ground truth ==="
    bash "${ROOT}/jobs/02_annotate.sh"
  fi

  echo "=== Step 3: metrics ==="
  bash "${ROOT}/jobs/03_metrics.sh"

  echo "=== Step 4: compare ==="
  bash "${ROOT}/jobs/04_compare.sh"

  echo "✓ Benchmark complete → results/summary.json"
}

# --- Optional Run:ai batching (one job per run URL, then annotate + metrics) ---

run_runai() {
  local IMAGE="${IMAGE:?Set IMAGE for RUNAI=1}"
  local PVC="${PVC:?Set PVC for RUNAI=1}"
  local WORKDIR="${WORKDIR:-/workspace/mmore-retrieval-benchmark}"
  local HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
  local QUERIES="${QUERIES:-${WORKDIR}/data/medxpertqa_200_mmore.jsonl}"
  local OUTPUT_DIR="${WORKDIR}/results"
  local BASE_CMD="pip install -r requirements.txt --quiet"
  local run url per_run

  if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
    echo "=== Run:ai step 0: dataset ==="
    runai submit mmore-bench-prepare \
      --image "${IMAGE}" --cpu 2 --memory 8G \
      --pvc "${PVC}:/workspace" --working-dir "${WORKDIR}" \
      -- bash -c "${BASE_CMD} && python scripts/00_prepare_dataset.py --collection-name \${COLLECTION_NAME:-my_docs}"
    runai wait mmore-bench-prepare --timeout 600
  fi

  if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
    echo "=== Run:ai step 1: collect ==="
    for run in "${ALL_RUNS[@]}"; do
      if ! url="$(url_for_run "$run" 2>/dev/null || true)" || [[ -z "$url" ]]; then
        echo "SKIP Run:ai collect ${run}: no URL" >&2
        continue
      fi
      local api_type k
      case "$run" in
        run_C|run_C_ctrl) api_type="rag" ;;
        *) api_type="retriever" ;;
      esac
      case "$run" in run_F) k=10 ;; *) k=5 ;; esac
      runai submit "mmore-bench-${run}" \
        --image "${IMAGE}" --cpu 2 --memory 4G \
        --pvc "${PVC}:/workspace" --working-dir "${WORKDIR}" \
        -e MMORE_API_URL="${url}" \
        -- bash -c "
          ${BASE_CMD} &&
          python scripts/collect_from_api.py \
            --queries ${QUERIES} \
            --out ${OUTPUT_DIR}/${run}/chunks.json \
            --api-type ${api_type} \
            --base-url ${url} \
            --k ${k} \
            --raw-out ${OUTPUT_DIR}/${run}/raw_api.json
        "
    done
    for run in "${GT_RUNS[@]}"; do
      runai wait "mmore-bench-${run}" --timeout 7200
    done
    for run in run_C_ctrl run_D run_E run_F; do
      if url="$(url_for_run "$run" 2>/dev/null || true)" && [[ -n "$url" ]]; then
        runai wait "mmore-bench-${run}" --timeout 7200 || true
      fi
    done
  fi

  if [[ "${SKIP_ANNOTATE:-0}" != "1" ]]; then
    echo "=== Run:ai step 2: ground truth ==="
    runai submit mmore-bench-annotate \
      --image "${IMAGE}" --cpu 2 --memory 8G \
      --pvc "${PVC}:/workspace" --working-dir "${WORKDIR}" \
      -- bash -c "
        ${BASE_CMD} &&
        python scripts/01_annotate_ground_truth.py \
          --chunks ${OUTPUT_DIR}/run_A/chunks.json \
          --chunks ${OUTPUT_DIR}/run_B/chunks.json \
          --chunks ${OUTPUT_DIR}/run_C/chunks.json \
          --out data/ground_truth.json \
          --model \${ANNOTATOR_MODEL:-gpt-4o-mini} \
          --incremental
      "
    runai wait mmore-bench-annotate --timeout 7200
  fi

  echo "=== Run:ai steps 3–4: metrics + compare ==="
  runai submit mmore-bench-metrics \
    --image "${IMAGE}" --cpu 2 --memory 8G \
    --pvc "${PVC}:/workspace" --working-dir "${WORKDIR}" \
    -- bash -c "
      ${BASE_CMD} &&
      for run in run_A run_B run_C run_C_ctrl run_D run_E run_F; do
        python scripts/02_compute_metrics.py \
          --gt data/ground_truth.json \
          --chunks ${OUTPUT_DIR}/\${run}/chunks.json \
          --out ${OUTPUT_DIR}/\${run}/metrics.json \
          --run-name \${run}
      done &&
      python scripts/03_compare_runs.py --results-dir ${OUTPUT_DIR} --judge-pair run_B,run_C
    "
  runai wait mmore-bench-metrics --timeout 1200
  echo "✓ Done → ${OUTPUT_DIR}/summary.json"
}

if [[ "${RUNAI:-0}" == "1" ]]; then
  run_runai "$@"
else
  run_local "$@"
fi
