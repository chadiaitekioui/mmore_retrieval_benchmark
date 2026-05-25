"""
Normalize API / retriever JSON into benchmark chunks.json.

Accepts:
  - chunks.json-like entries (query + chunks[])
  - retriever local format (context = [{page_content, metadata}, ...])
  - retriever API format (documents = [{fileId, chunkId, content, similarity}, ...])
  - custom payloads from your external repo if fields match bench_io.normalize_chunk

Usage:
    python scripts/04_convert_rag_output.py \\
        --in results/run_B/raw_retriever.json \\
        --out results/run_B/chunks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_io import chunks_from_entry, entry_from_api_payload, query_from_entry  # noqa: E402


def convert_entry(entry: dict) -> dict:
    query = query_from_entry(entry)
    if entry.get("response") is not None:
        out = entry_from_api_payload(query, entry["response"])
    else:
        out = entry_from_api_payload(query, entry)
    if "chunks" not in out:
        out["chunks"] = chunks_from_entry(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.inp) as f:
        data = json.load(f)

    converted = [convert_entry(e) for e in data]
    n_empty = sum(1 for c in converted if not chunks_from_entry(c))
    if n_empty:
        print(
            f"[!] {n_empty}/{len(converted)} entries without chunks — "
            "use collect_from_api.py (--api-type retriever) or "
            "include chunks/documents in your API response."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    print(f"✓ {len(converted)} entries → {out_path}")


if __name__ == "__main__":
    main()
