#!/usr/bin/env python3
"""
Download MedRAG clinical corpora for the MMORE retrieval benchmark.

Sources (pre-chunked snippets, MedRAG format: id, title, content, contents):
  - textbooks — MedRAG/textbooks on Hugging Face (18 USMLE textbooks, ~126k snippets)
  - statpearls — community mirror MilyaShams/MedRAG_statpearls (~301k snippets)

Official StatPearls chunks are not redistributed on HF; use the mirror above or chunk
raw NCBI Bookshelf data with MedRAG (see corpus/README.md).

Usage:
  python corpus/download_medrag.py
  python corpus/download_medrag.py --sources textbooks
  python corpus/download_medrag.py --sources statpearls textbooks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEXTBOOKS_REPO = "MedRAG/textbooks"
TEXTBOOKS_CHUNK_DIR = "chunk"
STATPEARLS_DATASET = "MilyaShams/MedRAG_statpearls"


def _download_textbooks(out_dir: Path) -> list[Path]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_tree(TEXTBOOKS_REPO, repo_type="dataset", recursive=True)
    paths: list[Path] = []
    for entry in files:
        path = getattr(entry, "path", None) or entry.get("path", "")
        if not path.startswith(f"{TEXTBOOKS_CHUNK_DIR}/") or not path.endswith(".jsonl"):
            continue
        local = hf_hub_download(
            repo_id=TEXTBOOKS_REPO,
            filename=path,
            repo_type="dataset",
            local_dir=str(out_dir / "hf_textbooks"),
        )
        paths.append(Path(local))
    if not paths:
        raise SystemExit(f"No textbook JSONL files found under {TEXTBOOKS_REPO}/{TEXTBOOKS_CHUNK_DIR}/")
    return paths


def _download_statpearls(out_dir: Path) -> Path:
    from datasets import load_dataset

    print(f"Loading StatPearls snippets from {STATPEARLS_DATASET}…")
    ds = load_dataset(STATPEARLS_DATASET, split="train")
    out_path = out_dir / "statpearls.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MedRAG corpora (HF)")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["statpearls", "textbooks"],
        choices=("statpearls", "textbooks"),
        help="Corpus slices to download (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: corpus/data/medrag)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "medrag"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "source": "medrag",
        "components": {},
    }

    if "textbooks" in args.sources:
        print("=== Textbooks (MedRAG/textbooks) ===")
        tb_paths = _download_textbooks(out_dir)
        manifest["components"]["textbooks"] = [str(p) for p in tb_paths]
        print(f"✓ {len(tb_paths)} textbook JSONL files")

    if "statpearls" in args.sources:
        print("=== StatPearls (HF mirror) ===")
        sp_path = _download_statpearls(out_dir)
        manifest["components"]["statpearls"] = str(sp_path)
        print(f"✓ {sp_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"✓ manifest → {manifest_path}")


if __name__ == "__main__":
    main()
