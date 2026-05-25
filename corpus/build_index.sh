#!/usr/bin/env bash
# Build proc_demo.db from PLOS articles: download → txt → process → postprocess → index.
#
# Prerequisites:
#   - MMORE installed: pip install -e "${MMORE_ROOT}"
#   - GPU recommended for index embeddings
#
# Usage:
#   export MMORE_ROOT=/path/to/mmore   # fork, branch llm-as-a-judge
#   bash corpus/build_index.sh 1000    # PLOS-1k
#   bash corpus/build_index.sh 5000    # PLOS-5k
#
# Env (defaults shown):
#   DB_URI          → ${BENCH_ROOT}/proc_demo.db
#   DB_NAME         → my_db
#   COLLECTION_NAME → my_db
#   HF_HOME         → Hugging Face cache (embeddings)
#   SKIP_DOWNLOAD=1 SKIP_CONVERT=1     → reuse cached JSON / txt

set -euo pipefail

SIZE="${1:?Usage: $0 1000|5000}"
if [[ "$SIZE" != "1000" && "$SIZE" != "5000" ]]; then
  echo "SIZE must be 1000 or 5000" >&2
  exit 1
fi

BENCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS_ROOT="${BENCH_ROOT}/corpus"
MMORE_ROOT="${MMORE_ROOT:-}"

if [[ -z "$MMORE_ROOT" || ! -d "$MMORE_ROOT/src/mmore" ]]; then
  echo "Set MMORE_ROOT to your mmore fork (pip install -e \$MMORE_ROOT)." >&2
  exit 1
fi

if [[ "$SIZE" == "1000" ]]; then
  LABEL="plos_1k"
else
  LABEL="plos_5k"
fi

PLOS_JSON="${CORPUS_ROOT}/data/plos_${SIZE}.json"
MMORE_INPUT="${CORPUS_MMORE_INPUT:-${CORPUS_ROOT}/mmore_input/${LABEL}}"
PROCESS_OUT="${CORPUS_PROCESS_OUT:-${CORPUS_ROOT}/work/${LABEL}/process}"
MERGED_JSONL="${PROCESS_OUT}/merged/merged_results.jsonl"
PP_OUT="${CORPUS_PP_OUT:-${CORPUS_ROOT}/work/${LABEL}/postprocess/results.jsonl}"

DB_URI="${DB_URI:-${BENCH_ROOT}/proc_demo.db}"
DB_NAME="${DB_NAME:-my_db}"
COLLECTION_NAME="${COLLECTION_NAME:-my_db}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HOME DB_URI DB_NAME COLLECTION_NAME
export CORPUS_MMORE_INPUT="$MMORE_INPUT"
export CORPUS_PROCESS_OUT="$PROCESS_OUT"
export CORPUS_PP_OUT="$PP_OUT"

CONFIG_DIR="${CORPUS_ROOT}/config"
mkdir -p "${CORPUS_ROOT}/data" "${MMORE_INPUT}" "${PROCESS_OUT}" "$(dirname "$PP_OUT")"

echo "=== PLOS corpus build (${LABEL}) ==="
echo "  MMORE_ROOT=${MMORE_ROOT}"
echo "  DB_URI=${DB_URI}  DB_NAME=${DB_NAME}  COLLECTION_NAME=${COLLECTION_NAME}"

pip install -r "${CORPUS_ROOT}/requirements.txt" -q
pip install -e "${MMORE_ROOT}" -q

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  echo "=== Download PLOS (${SIZE}) ==="
  python "${CORPUS_ROOT}/download_plos.py" --size "$SIZE" --output "$PLOS_JSON"
fi

if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
  echo "=== Convert to MMORE .txt input ==="
  python "${CORPUS_ROOT}/convert_plos_to_mmore.py" \
    --input "$PLOS_JSON" \
    --output-dir "$MMORE_INPUT"
fi

n_txt="$(find "$MMORE_INPUT" -maxdepth 1 -name '*.txt' | wc -l | tr -d ' ')"
if [[ "$n_txt" -eq 0 ]]; then
  echo "No .txt files in ${MMORE_INPUT}" >&2
  exit 1
fi
echo "  ${n_txt} text files ready"

echo "=== MMORE process ==="
python -m mmore process --config-file "${CONFIG_DIR}/process_plos.yaml"

if [[ ! -f "$MERGED_JSONL" ]]; then
  echo "Missing ${MERGED_JSONL} after process" >&2
  exit 1
fi

echo "=== MMORE postprocess ==="
python -m mmore postprocess \
  --config-file "${CONFIG_DIR}/postprocess_plos.yaml" \
  --input-data "$MERGED_JSONL"

if [[ ! -f "$PP_OUT" ]]; then
  echo "Missing ${PP_OUT} after postprocess" >&2
  exit 1
fi

echo "=== MMORE index ==="
python -m mmore index \
  --config-file "${CONFIG_DIR}/index_plos.yaml" \
  --documents-path "$PP_OUT" \
  --collection-name "$COLLECTION_NAME"

echo ""
echo "✓ Done — index ready:"
echo "    DB_URI=${DB_URI}"
echo "    DB_NAME=${DB_NAME}"
echo "    COLLECTION_NAME=${COLLECTION_NAME}"
echo ""
echo "Before running the benchmark, export:"
echo "  export DB_URI=${DB_URI}"
echo "  export DB_NAME=${DB_NAME}"
echo "  export COLLECTION_NAME=${COLLECTION_NAME}"
echo "  export HF_HOME=${HF_HOME}"
