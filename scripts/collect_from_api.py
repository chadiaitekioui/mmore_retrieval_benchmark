"""
Collect benchmark chunks.json by calling a deployed MMORE API (no change to run_rag.py).

Supports:
  - retriever API: POST {base_url}/v1/retrieve  (returns ranked documents + scores)
  - rag API:       POST {base_url}{rag_endpoint} (judge metrics; chunks only if your
                   deployment returns them — standard MMORE RAG API does not)

Usage (retriever — runs without judge: A, B, D, E, F):
  python scripts/collect_from_api.py \\
    --queries data/medxpertqa_200_mmore.jsonl \\
    --out results/run_B/chunks.json \\
    --api-type retriever \\
    --base-url http://localhost:8001 \\
    --k 5

Usage (RAG — runs with judge: C; requires chunks in response or use retriever + wrapper):
  python scripts/collect_from_api.py \\
    --queries data/medxpertqa_200_mmore.jsonl \\
    --out results/run_C/chunks.json \\
    --api-type rag \\
    --base-url http://localhost:8000 \\
    --rag-endpoint /rag

If your external repo returns custom JSON, save raw files and run:
  python scripts/04_convert_rag_output.py --in raw.json --out results/run_C/chunks.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_io import chunks_from_entry, entry_from_api_payload  # noqa: E402

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


def load_queries(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def post_retrieve(
    client: "httpx.Client",
    base_url: str,
    query: str,
    k: int,
    file_ids: list[str],
    min_similarity: float,
) -> list[dict]:
    url = f"{base_url.rstrip('/')}/v1/retrieve"
    body = {
        "query": query,
        "fileIds": file_ids,
        "maxMatches": k,
        "minSimilarity": min_similarity,
    }
    resp = client.post(url, json=body, timeout=600.0)
    resp.raise_for_status()
    return resp.json()


def post_rag(
    client: "httpx.Client",
    base_url: str,
    endpoint: str,
    query: str,
    collection_name: str,
) -> dict:
    url = f"{base_url.rstrip('/')}{endpoint}"
    body = {"input": {"input": query, "collection_name": collection_name}}
    resp = client.post(url, json=body, timeout=600.0)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Collect chunks.json from MMORE HTTP API")
    parser.add_argument("--queries", required=True, help="JSONL with input (+ collection_name for RAG)")
    parser.add_argument("--out", required=True, help="Output chunks.json path")
    parser.add_argument(
        "--api-type",
        choices=("retriever", "rag"),
        default="retriever",
        help="retriever: /v1/retrieve (chunks guaranteed). rag: judge fields, chunks if API provides them",
    )
    parser.add_argument("--base-url", required=True, help="e.g. http://localhost:8001")
    parser.add_argument("--rag-endpoint", default="/rag")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--file-ids", default="", help="Comma-separated fileIds for retriever filter")
    parser.add_argument("--min-similarity", type=float, default=-1.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Pause between requests (seconds)")
    parser.add_argument("--raw-out", default="", help="Optional path to save raw API responses (JSON)")
    args = parser.parse_args()

    if httpx is None:
        raise SystemExit("pip install httpx")

    file_ids = [x.strip() for x in args.file_ids.split(",") if x.strip()]
    queries = load_queries(Path(args.queries))
    results: list[dict] = []
    raw_log: list[dict] = []

    with httpx.Client() as client:
        for i, row in enumerate(queries):
            query = row.get("input") or row.get("query") or ""
            if not query:
                print(f"  [!] line {i}: missing query/input")
                continue

            try:
                if args.api_type == "retriever":
                    payload = post_retrieve(
                        client,
                        args.base_url,
                        query,
                        args.k,
                        file_ids,
                        args.min_similarity,
                    )
                else:
                    collection = row.get("collection_name", "my_docs")
                    payload = post_rag(
                        client,
                        args.base_url,
                        args.rag_endpoint,
                        query,
                        collection,
                    )
            except httpx.HTTPError as e:
                print(f"  [!] {query[:50]}… : {e}")
                results.append({"query": query, "chunks": [], "error": str(e)})
                continue

            entry = entry_from_api_payload(query, payload)
            results.append(entry)
            raw_log.append({"query": query, "response": payload})

            n_chunks = len(chunks_from_entry(entry))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(queries)} ({n_chunks} chunks last)")
            if args.sleep:
                time.sleep(args.sleep)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_empty = sum(1 for r in results if not chunks_from_entry(r))
    print(f"\n✓ {len(results)} entries → {out_path}")
    if n_empty:
        print(
            f"[!] {n_empty} entries without chunks — "
            f"with standard RAG API use --api-type retriever or "
            f"export chunks from your deployment wrapper."
        )

    if args.raw_out:
        raw_path = Path(args.raw_out)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w") as f:
            json.dump(raw_log, f, indent=2, ensure_ascii=False)
        print(f"✓ Raw responses → {raw_path}")


if __name__ == "__main__":
    main()
