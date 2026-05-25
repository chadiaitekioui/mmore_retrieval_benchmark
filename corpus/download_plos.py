#!/usr/bin/env python3
"""
Download open-access PLOS articles via the public Search API (api.plos.org).

Reproducible selection: doc_type:full, sorted by id ascending, paginated.

Outputs JSON (not committed):
  corpus/data/plos_1000.json
  corpus/data/plos_5000.json

Usage:
  python corpus/download_plos.py --size 1000
  python corpus/download_plos.py --size 5000
  python corpus/download_plos.py --size 1000 --output corpus/data/plos_1000.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

PLOS_SEARCH = "https://api.plos.org/search"
DEFAULT_FIELDS = "id,title,abstract,body,author,publication_date,journal"
ROWS_PER_PAGE = 1000
DEFAULT_QUERY = "doc_type:full"


def fetch_page(
    client: httpx.Client,
    query: str,
    start: int,
    rows: int,
    fields: str,
) -> dict[str, Any]:
    params = {
        "q": query,
        "fl": fields,
        "start": start,
        "rows": rows,
        "sort": "id asc",
        "wt": "json",
    }
    resp = client.get(PLOS_SEARCH, params=params, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


def normalize_doc(raw: dict[str, Any]) -> dict[str, Any]:
    body = raw.get("body") or ""
    if isinstance(body, list):
        body = "\n\n".join(str(x) for x in body if x)
    abstract = raw.get("abstract") or ""
    if isinstance(abstract, list):
        abstract = "\n\n".join(str(x) for x in abstract if x)
    title = raw.get("title") or ""
    if isinstance(title, list):
        title = title[0] if title else ""

    return {
        "id": raw.get("id", ""),
        "title": str(title).strip(),
        "abstract": str(abstract).strip(),
        "body": str(body).strip(),
        "author": raw.get("author"),
        "publication_date": raw.get("publication_date"),
        "journal": raw.get("journal"),
    }


def download(
    size: int,
    query: str,
    fields: str,
    sleep_s: float,
) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    start = 0

    with httpx.Client(follow_redirects=True) as client:
        while len(articles) < size:
            rows = min(ROWS_PER_PAGE, size - len(articles))
            payload = fetch_page(client, query, start, rows, fields)
            response = payload.get("response") or {}
            docs = response.get("docs") or []
            if not docs:
                break
            for raw in docs:
                articles.append(normalize_doc(raw))
                if len(articles) >= size:
                    break
            start += len(docs)
            print(f"  fetched {len(articles)}/{size} (start={start})")
            if len(docs) < rows:
                break
            if sleep_s:
                time.sleep(sleep_s)

    return articles[:size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PLOS articles (Search API)")
    parser.add_argument(
        "--size",
        type=int,
        choices=(1000, 5000),
        required=True,
        help="Number of articles (PLOS-1k or PLOS-5k)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON path (default: corpus/data/plos_{size}.json)",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Solr query")
    parser.add_argument("--fields", default=DEFAULT_FIELDS, help="Solr fl= fields")
    parser.add_argument("--sleep", type=float, default=0.2, help="Pause between pages")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out = Path(args.output) if args.output else root / "data" / f"plos_{args.size}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.size} PLOS articles (q={args.query!r})…")
    articles = download(args.size, args.query, args.fields, args.sleep)

    meta = {
        "source": "plos_search_api",
        "api": PLOS_SEARCH,
        "query": args.query,
        "fields": args.fields,
        "sort": "id asc",
        "requested": args.size,
        "count": len(articles),
    }
    payload = {"meta": meta, "articles": articles}

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✓ {len(articles)} articles → {out}")
    if len(articles) < args.size:
        print(f"[!] Only {len(articles)} articles returned (API may have fewer matches).")


if __name__ == "__main__":
    main()
