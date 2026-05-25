#!/usr/bin/env bash
# Step 0 — prepare MedXpertQA dataset for MMORE.
#
# Usage:
#   bash jobs/00_prepare.sh [--pilot]
#
# Env:
#   COLLECTION_NAME  (default: my_docs)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pip install -r requirements.txt -q

python scripts/00_prepare_dataset.py \
  --collection-name "${COLLECTION_NAME:-my_docs}" \
  "$@"
