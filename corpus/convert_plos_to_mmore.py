#!/usr/bin/env python3
"""
Convert PLOS JSON (from download_plos.py) into plain .txt files for MMORE process.

MMORE process input: a directory of files (.txt supported by TXTProcessor).
Pipeline: folder of .txt → mmore process → merged JSONL → postprocess → index.

Usage:
  python corpus/convert_plos_to_mmore.py --input corpus/data/plos_1000.json
  python corpus/convert_plos_to_mmore.py --input corpus/data/plos_5000.json \\
      --output-dir corpus/mmore_input/plos_5k
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def safe_filename(article_id: str) -> str:
    """PLOS id is a DOI, e.g. 10.1371/journal.pone.0123456."""
    s = article_id.strip()
    s = re.sub(r"[^\w.\-]+", "_", s.replace("/", "_"))
    return s[:200] or "unknown"


def article_text(article: dict) -> str:
    title = (article.get("title") or "").strip()
    abstract = (article.get("abstract") or "").strip()
    body = (article.get("body") or "").strip()
    parts = []
    if title:
        parts.append(f"# {title}\n")
    if abstract:
        parts.append(f"## Abstract\n{abstract}\n")
    if body:
        parts.append(f"## Body\n{body}\n")
    elif abstract:
        parts.append("(Full text body not available from API; using abstract only.)\n")
    text = "\n".join(parts).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PLOS JSON to .txt files for mmore process"
    )
    parser.add_argument("--input", required=True, help="plos_1000.json or plos_5000.json")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output folder (default: corpus/mmore_input/plos_1k or plos_5k)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Skip articles shorter than this after normalization",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    with inp.open(encoding="utf-8") as f:
        payload = json.load(f)

    articles = payload.get("articles") or payload
    if not isinstance(articles, list):
        raise SystemExit(f"{inp}: expected 'articles' list or JSON array")

    root = Path(__file__).resolve().parent
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        n = len(articles)
        label = "plos_5k" if n >= 4000 else "plos_1k"
        if "5000" in inp.name:
            label = "plos_5k"
        elif "1000" in inp.name:
            label = "plos_1k"
        out_dir = root / "mmore_input" / label

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    written = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for article in articles:
            aid = article.get("id") or ""
            text = article_text(article)
            if len(text) < args.min_chars:
                skipped += 1
                continue
            fname = safe_filename(aid) + ".txt"
            path = out_dir / fname
            path.write_text(text, encoding="utf-8")
            manifest.write(
                json.dumps(
                    {
                        "id": aid,
                        "file": fname,
                        "chars": len(text),
                        "title": article.get("title", "")[:200],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"✓ {written} .txt files → {out_dir}")
    print(f"✓ manifest → {manifest_path}")
    if skipped:
        print(f"[!] skipped {skipped} articles (< {args.min_chars} chars)")


if __name__ == "__main__":
    main()
