"""
Step 1 — Corpus-level ground truth (labels per chunk_id, union across runs).

Usage:
    python scripts/01_annotate_ground_truth.py \\
        --chunks results/run_A/chunks.json \\
        --chunks results/run_B/chunks.json \\
        --chunks results/run_C/chunks.json \\
        --out data/ground_truth.json \\
        --incremental
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_io import (  # noqa: E402
    GT_VERSION,
    build_chunks_text,
    chunk_id,
    load_ground_truth,
    load_questions_jsonl,
    union_chunks_by_query,
)

SYSTEM = (
    "You are a medical retrieval annotator. "
    "Your only job is to decide whether each text chunk contains information "
    "that directly supports the correct answer to a medical question. "
    "Be strict: a chunk is relevant only if it explicitly supports the answer, "
    "not just if it is vaguely related to the topic."
)

USER_TMPL = """\
Question: {question}
Correct answer: {answer}

For each chunk below, answer true if the chunk contains information that \
directly justifies the correct answer, false otherwise.

{chunks}

Respond with a JSON array of booleans only, one per chunk, in the same order, \
e.g. [true, false, true]. No explanation, no markdown, just the array."""


def max_tokens_for_chunks(n_chunks: int) -> int:
    """Enough tokens for `[false, ...]` with one bool per chunk."""
    return max(256, n_chunks * 12 + 32)


def parse_bool_array(raw: str, expected_len: int | None = None) -> list[bool]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = text.find("[")
    if start < 0:
        raise ValueError(f"No JSON array in model output: {raw[:200]!r}")
    end = text.rfind("]") + 1
    if end > start:
        try:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, list):
                return [bool(x) for x in parsed]
        except json.JSONDecodeError:
            pass

    import re

    tokens = re.findall(r"\b(true|false)\b", text[start:].lower())
    if not tokens:
        raise ValueError(f"No JSON array in model output: {raw[:200]!r}")
    vals = [t == "true" for t in tokens]
    if expected_len is not None:
        if len(vals) < expected_len:
            raise ValueError(
                f"Truncated bool array ({len(vals)}/{expected_len}): {raw[:200]!r}"
            )
        vals = vals[:expected_len]
    return vals


def annotate_openai(client, model: str, question: str, answer: str, chunks: list[dict]) -> list[bool]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER_TMPL.format(
                    question=question,
                    answer=answer,
                    chunks=build_chunks_text(chunks),
                ),
            },
        ],
        max_tokens=max_tokens_for_chunks(len(chunks)),
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    return parse_bool_array(raw, expected_len=len(chunks))


DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def resolve_hf_model_id(model: str) -> str:
    if model in ("local", "hf"):
        return DEFAULT_HF_MODEL
    return model


def use_hf_backend(model: str, backend: str | None) -> bool:
    if backend == "hf":
        return True
    if backend == "openai":
        return False
    return model in ("local", "hf") or "/" in model


def load_hf_annotator(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()
    return model, tokenizer


def annotate_hf(
    model,
    tokenizer,
    question: str,
    answer: str,
    chunks: list[dict],
) -> list[bool]:
    import torch

    user = USER_TMPL.format(
        question=question,
        answer=answer,
        chunks=build_chunks_text(chunks),
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    max_new = max_tokens_for_chunks(len(chunks))
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_bool_array(text, expected_len=len(chunks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", action="append", required=True)
    parser.add_argument("--questions", default="data/medxpertqa_200.jsonl")
    parser.add_argument("--out", default="data/ground_truth.json")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name (e.g. gpt-4o-mini, gpt-4o), Hugging Face id "
        "(e.g. meta-llama/Llama-3.1-8B-Instruct), or 'local' for the default HF model",
    )
    parser.add_argument(
        "--backend",
        choices=("openai", "hf", "auto"),
        default="auto",
        help="auto: HF if model contains '/' or is 'local'; else OpenAI API",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL (default: env OPENAI_BASE_URL if set)",
    )
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()
    if not args.base_url:
        args.base_url = os.environ.get("OPENAI_BASE_URL") or None

    questions = load_questions_jsonl(Path(args.questions))
    chunk_paths = [Path(p) for p in args.chunks]
    union = union_chunks_by_query(chunk_paths)
    print(f"✓ {len(questions)} questions, union {len(union)} queries from {len(chunk_paths)} runs")

    existing: dict[str, dict] = {}
    if args.incremental and Path(args.out).exists():
        existing = load_ground_truth(Path(args.out))

    backend = args.backend
    if backend == "auto":
        backend = "hf" if use_hf_backend(args.model, None) else "openai"

    annotator_note = ""
    if backend == "hf":
        hf_model_id = resolve_hf_model_id(args.model)
        print(f"✓ Hugging Face annotator: {hf_model_id} (set HF_TOKEN if the repo is gated)")
        hf_model, hf_tokenizer = load_hf_annotator(hf_model_id)
        annotator_note = f"HF model {hf_model_id}"
        annotate_fn = lambda q, a, c: annotate_hf(hf_model, hf_tokenizer, q, a, c)
    else:
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()
        where = args.base_url or "https://api.openai.com/v1"
        annotator_note = f"OpenAI-compatible {args.model} @ {where}"
        annotate_fn = lambda q, a, c: annotate_openai(client, args.model, q, a, c)

    gt_questions: list[dict] = []
    errors = 0
    n_new_labels = 0

    for i, (query, chunks) in enumerate(sorted(union.items(), key=lambda x: x[0])):
        q_meta = questions.get(query)
        if not q_meta:
            errors += 1
            continue

        answer = q_meta.get("answer_text", "")
        labels_map: dict[str, bool] = dict(
            existing.get(query, {}).get("labels_by_chunk_id", {})
        )
        to_annotate = [c for c in chunks if chunk_id(c) not in labels_map]

        if to_annotate:
            try:
                new_labels = annotate_fn(query, answer, to_annotate)
                if len(new_labels) < len(to_annotate):
                    new_labels += [False] * (len(to_annotate) - len(new_labels))
                for ch, lbl in zip(to_annotate, new_labels):
                    labels_map[chunk_id(ch)] = bool(lbl)
                    n_new_labels += 1
            except Exception as e:
                print(f"  [!] {query[:40]}… : {e}")
                errors += 1
                # Leave chunk_ids unlabeled so --incremental can retry after a fix.
            time.sleep(args.sleep)

        gt_questions.append(
            {
                "query": query,
                "answer_text": answer,
                "answer_idx": q_meta.get("answer_idx"),
                "labels_by_chunk_id": labels_map,
                "n_unique_chunks": len(labels_map),
                "n_relevant": sum(labels_map.values()),
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(union)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "version": GT_VERSION,
                "annotator_model": annotator_note or args.model,
                "annotator_note": "Answer-aware labels per (query, chunk_id).",
                "source_runs": [str(p) for p in chunk_paths],
                "questions": gt_questions,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    n_hit = sum(1 for q in gt_questions if q["n_relevant"] > 0)
    print(f"\n✓ → {out_path} | questions with ≥1 relevant chunk: {n_hit}/{len(gt_questions)}")
    print(f"  New labels: {n_new_labels} | errors: {errors}")


if __name__ == "__main__":
    main()
