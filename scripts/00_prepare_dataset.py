"""
Step 0 — Prepare MedXpertQA subset (200 questions, seed 42, text track).

Outputs:
  data/medxpertqa_200.jsonl       — questions + MCQ answers (annotation)
  data/medxpertqa_200_mmore.jsonl — input + collection_name (MMORE API)
  data/medxpertqa_30*.jsonl       — pilot subset (optional, --pilot)
"""

import argparse
import json
import random
from pathlib import Path

SEED = 42
N_FULL = 200
TRACK = "text"
OUT_QUESTIONS = Path("data/medxpertqa_200.jsonl")
OUT_MMORE = Path("data/medxpertqa_200_mmore.jsonl")
OUT_PILOT_QUESTIONS = Path("data/medxpertqa_30.jsonl")
OUT_PILOT_MMORE = Path("data/medxpertqa_30_mmore.jsonl")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_options(options) -> list[str]:
    if isinstance(options, dict):
        return [options[k] for k in sorted(options.keys())]
    return list(options)


def normalize_answer_idx(item: dict, n_options: int) -> int:
    answer_idx = item.get("answer_idx", item.get("answer", item.get("label", 0)))
    if isinstance(answer_idx, str):
        if len(answer_idx) == 1 and answer_idx.isalpha():
            return ord(answer_idx.upper()) - ord("A")
        return int(answer_idx)
    return int(answer_idx)


def build_rows(sample: list[dict], collection_name: str) -> tuple[list[dict], list[dict]]:
    questions_rows = []
    mmore_rows = []
    for item in sample:
        options = normalize_options(item.get("options", []))
        answer_idx = normalize_answer_idx(item, len(options))
        answer_text = options[answer_idx] if options else ""
        query = item["question"]
        questions_rows.append(
            {
                "query": query,
                "options": options,
                "answer_idx": answer_idx,
                "answer_text": answer_text,
            }
        )
        mmore_rows.append(
            {
                "input": query,
                "collection_name": collection_name,
            }
        )
    return questions_rows, mmore_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-name",
        default="my_db",
        help="Milvus collection name passed to MMORE",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Also write 30-question pilot files (seed 42, first 30 of the 200 sample)",
    )
    parser.add_argument("-n", type=int, default=N_FULL, help="Number of questions")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets")

    print("Loading MedXpertQA (Text config)…")
    ds = load_dataset("TsinghuaC3I/MedXpertQA", "Text", split="test")
    text_items = list(ds)
    print(f"  {len(text_items)} text-track questions available")

    random.seed(SEED)
    sample = random.sample(text_items, min(args.n, len(text_items)))

    q_rows, m_rows = build_rows(sample, args.collection_name)
    write_jsonl(OUT_QUESTIONS, q_rows)
    write_jsonl(OUT_MMORE, m_rows)
    print(f"✓ {len(q_rows)} questions → {OUT_QUESTIONS}")
    print(f"✓ {len(m_rows)} MMORE lines → {OUT_MMORE}")

    if args.pilot:
        pilot_q, pilot_m = build_rows(sample[:30], args.collection_name)
        write_jsonl(OUT_PILOT_QUESTIONS, pilot_q)
        write_jsonl(OUT_PILOT_MMORE, pilot_m)
        print(f"✓ Pilot 30 questions → {OUT_PILOT_QUESTIONS}")


if __name__ == "__main__":
    main()
