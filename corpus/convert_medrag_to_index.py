#!/usr/bin/env python3
"""
Convert MedRAG pre-chunked snippets to MMORE index JSONL (MultimodalSample rows).

MedRAG snippet fields: id, title, content, contents (title + content for BM25).

Pre-chunked snippets skip MMORE process/postprocess; index with the same BGE + SPLADE
config as the PLoS pipeline (corpus/config/index_medrag.yaml).

Usage:
  python corpus/convert_medrag_to_index.py
  python corpus/convert_medrag_to_index.py --input-dir corpus/data/medrag \\
      --output corpus/work/medrag/postprocess/results.jsonl
  python corpus/convert_medrag_to_index.py --max-snippets 5000   # pilot subset
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def snippet_text(row: dict) -> str:
    """Retrieval text: prefer MedRAG 'contents', else title + content."""
    contents = (row.get("contents") or "").strip()
    if contents:
        return contents
    title = (row.get("title") or "").strip()
    content = (row.get("content") or "").strip()
    if title and content:
        return f"{title}. {content}" if not content.startswith(title) else content
    return title or content


def infer_source(path: Path, row: dict) -> str:
    name = path.name.lower()
    if "statpearls" in name:
        return "statpearls"
    if "chunk" in str(path.parent).lower() or path.suffix == ".jsonl":
        return "textbook"
    title = (row.get("title") or "").lower()
    if "statpearls" in title or "stat pearl" in title:
        return "statpearls"
    return "textbook"


def document_id_from_snippet(snippet_id: str, title: str) -> str:
    """Parent doc id for Milvus document_id (fileId in retrieve API)."""
    sid = snippet_id.strip()
    if "_" in sid:
        base, tail = sid.rsplit("_", 1)
        if tail.isdigit():
            return base
    safe = re.sub(r"[^\w.\-]+", "_", title.strip())[:120]
    return safe or sid[:120]


def iter_snippet_files(input_dir: Path) -> list[Path]:
    manifest = input_dir / "manifest.json"
    paths: list[Path] = []
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        comps = meta.get("components") or {}
        for key in ("textbooks", "statpearls"):
            val = comps.get(key)
            if isinstance(val, list):
                paths.extend(Path(p) for p in val)
            elif isinstance(val, str):
                paths.append(Path(val))
    if not paths:
        paths = sorted(input_dir.rglob("*.jsonl"))
    return [p for p in paths if p.name != "manifest.jsonl"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MedRAG JSONL snippets to MMORE index JSONL"
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Directory with manifest.json and/or *.jsonl (default: corpus/data/medrag)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSONL (default: corpus/work/medrag/postprocess/results.jsonl)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Skip snippets shorter than this after normalization",
    )
    parser.add_argument(
        "--max-snippets",
        type=int,
        default=0,
        help="Cap total snippets (0 = no limit; useful for pilots)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir) if args.input_dir else root / "data" / "medrag"
    output = (
        Path(args.output)
        if args.output
        else root / "work" / "medrag" / "postprocess" / "results.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    sources = iter_snippet_files(input_dir)
    if not sources:
        raise SystemExit(f"No JSONL inputs under {input_dir} (run download_medrag.py first)")

    written = 0
    skipped = 0
    counts: dict[str, int] = {}

    with output.open("w", encoding="utf-8") as out_f:
        for src_path in sources:
            source_kind = "statpearls" if "statpearls" in src_path.name.lower() else "textbook"
            with src_path.open(encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    text = snippet_text(row)
                    if len(text) < args.min_chars:
                        skipped += 1
                        continue

                    sid = str(row.get("id") or f"{source_kind}_{written}")
                    doc_id = document_id_from_snippet(sid, str(row.get("title") or ""))
                    kind = infer_source(src_path, row)
                    counts[kind] = counts.get(kind, 0) + 1

                    # MMORE index reads text + metadata; id is derived from text hash at load time.
                    sample = {
                        "text": text,
                        "modalities": [],
                        "metadata": {
                            "file_path": str(src_path.name),
                            "processor_type": "medrag",
                            "medrag_id": sid,
                            "medrag_source": kind,
                            "title": (row.get("title") or "")[:500],
                            "document_id": doc_id,
                        },
                    }
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1

                    if args.max_snippets and written >= args.max_snippets:
                        break
            if args.max_snippets and written >= args.max_snippets:
                break

    print(f"✓ {written} snippets → {output}")
    for kind, n in sorted(counts.items()):
        print(f"    {kind}: {n}")
    if skipped:
        print(f"[!] skipped {skipped} snippets (< {args.min_chars} chars)")


if __name__ == "__main__":
    main()
