#!/usr/bin/env bash
# MedXpertQA setup: clinical corpus (MedRAG → proc_demo.db) + query JSONL.
#
# Prerequisites: install.sh done, then source env.benchmark
#
# Usage:
#   source env.benchmark
#   bash jobs/setup_medxpertqa.sh              # MedRAG (StatPearls + textbooks)
#   MEDRAG_PILOT=1 bash jobs/setup_medxpertqa.sh --pilot
#   CORPUS=plos bash jobs/setup_medxpertqa.sh 5000   # legacy PLoS-5k
#
# Env: same as build_corpus.sh (CLEAN_DB, SKIP_DOWNLOAD, MEDRAG_PILOT, …)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.benchmark" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env.benchmark"
fi

CORPUS="${CORPUS:-medrag}"
SIZE=""
PREPARE_ARGS=()
for arg in "$@"; do
  case "$arg" in
    1000|5000)
      if [[ "$CORPUS" == "plos" ]]; then
        SIZE="$arg"
      else
        echo "[!] PLoS size '$arg' ignored (CORPUS=medrag). Use CORPUS=plos for PLoS." >&2
      fi
      ;;
    *) PREPARE_ARGS+=("$arg") ;;
  esac
done

if [[ "$CORPUS" == "plos" ]]; then
  echo "=== MedXpertQA setup (PLoS-${SIZE:-1000} + queries) ==="
  bash "${ROOT}/jobs/build_corpus.sh" "${SIZE:-1000}"
else
  echo "=== MedXpertQA setup (MedRAG clinical corpus + queries) ==="
  bash "${ROOT}/jobs/build_corpus.sh"
fi

bash "${ROOT}/jobs/00_prepare.sh" "${PREPARE_ARGS[@]}"

echo ""
echo "✓ Ready for collect:"
echo "  data/medxpertqa_200.jsonl"
echo "  data/medxpertqa_200_mmore.jsonl"
echo "  DB_URI=${DB_URI:-${ROOT}/proc_demo.db}"
