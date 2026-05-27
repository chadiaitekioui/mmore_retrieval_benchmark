#!/usr/bin/env bash
# Build proc_demo.db from MedRAG clinical corpora (StatPearls + textbooks).
#
# Pre-chunked MedRAG snippets → index JSONL → mmore index (BGE-small + SPLADE).
# Skips MMORE process/postprocess (chunks are already at MedRAG granularity).
#
# Usage:
#   export MMORE_ROOT=/path/to/mmore
#   bash corpus/build_index_medrag.sh
#   MEDRAG_PILOT=1 bash corpus/build_index_medrag.sh   # 5k snippets only
#
# Env:
#   SKIP_DOWNLOAD=1 / SKIP_CONVERT=1 — reuse cached HF data / JSONL
#   MEDRAG_SOURCES="statpearls textbooks" — passed to download_medrag.py
#   MEDRAG_PILOT=1 — cap snippets at 5000 (convert_medrag_to_index.py)
#   CLEAN_DB=1 — remove DB_URI before indexing (default in jobs/build_corpus.sh)

set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS_ROOT="${BENCH_ROOT}/corpus"
MMORE_ROOT="${MMORE_ROOT:-$(cd "${BENCH_ROOT}/.." && pwd)/mmore}"

if [[ ! -d "$MMORE_ROOT/src/mmore" ]]; then
  echo "MMORE repo not found at: ${MMORE_ROOT}" >&2
  exit 1
fi

LABEL="medrag"
MEDRAG_DATA="${CORPUS_DATA_DIR:-${CORPUS_ROOT}/data/medrag}"
PP_OUT="${CORPUS_PP_OUT:-${CORPUS_ROOT}/work/${LABEL}/postprocess/results.jsonl}"

DB_URI="${DB_URI:-${BENCH_ROOT}/proc_demo.db}"
DB_NAME="${DB_NAME:-my_db}"
COLLECTION_NAME="${COLLECTION_NAME:-my_db}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HOME DB_URI DB_NAME COLLECTION_NAME CORPUS_PP_OUT="$PP_OUT"

CONFIG_DIR="${CORPUS_ROOT}/config"
mkdir -p "${MEDRAG_DATA}" "$(dirname "$PP_OUT")"

echo "=== MedRAG corpus build (${LABEL}) ==="
echo "  MMORE_ROOT=${MMORE_ROOT}"
echo "  DB_URI=${DB_URI}  COLLECTION_NAME=${COLLECTION_NAME}"
echo "  MEDRAG_DATA=${MEDRAG_DATA}"

pip install -r "${CORPUS_ROOT}/requirements.txt" -q
pip install -e "${MMORE_ROOT}[index]" -q

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  echo "=== Download MedRAG corpora ==="
  # shellcheck disable=SC2206
  SOURCES=(${MEDRAG_SOURCES:-statpearls textbooks})
  python "${CORPUS_ROOT}/download_medrag.py" \
    --sources "${SOURCES[@]}" \
    --output-dir "$MEDRAG_DATA"
fi

if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
  echo "=== Convert snippets → index JSONL ==="
  CONVERT_ARGS=(--input-dir "$MEDRAG_DATA" --output "$PP_OUT")
  if [[ "${MEDRAG_PILOT:-0}" == "1" ]]; then
    CONVERT_ARGS+=(--max-snippets 5000)
  fi
  python "${CORPUS_ROOT}/convert_medrag_to_index.py" "${CONVERT_ARGS[@]}"
fi

if [[ ! -f "$PP_OUT" ]]; then
  echo "Missing ${PP_OUT} after convert" >&2
  exit 1
fi

n_lines="$(wc -l < "$PP_OUT" | tr -d ' ')"
echo "  ${n_lines} snippets ready for indexing"

if [[ "${CLEAN_DB:-1}" == "1" && -f "$DB_URI" ]]; then
  echo "=== Remove existing DB (CLEAN_DB=1): ${DB_URI} ==="
  rm -f "$DB_URI"
fi

echo "=== MMORE index (BGE-small + SPLADE) ==="
python -m mmore index \
  --config-file "${CONFIG_DIR}/index_medrag.yaml" \
  --documents-path "$PP_OUT" \
  --collection-name "$COLLECTION_NAME"

echo ""
echo "✓ Done — clinical index ready:"
echo "    DB_URI=${DB_URI}"
echo "    COLLECTION_NAME=${COLLECTION_NAME}"
echo ""
echo "Re-run ground truth after corpus change:"
echo "  bash jobs/ground_truth.sh"
