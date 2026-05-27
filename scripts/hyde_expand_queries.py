"""
Expand MedXpertQA queries with LangChain HyDE (HypotheticalDocumentEmbedder).

Writes JSONL rows with:
  - input: original vignette (used for ground-truth lookup)
  - hyde_input: LLM-generated hypothetical passage (sent to MMORE retrieve)
  - collection_name: unchanged from source

Usage:
  export OPENAI_API_KEY=sk-...
  python scripts/hyde_expand_queries.py \\
      --in data/medxpertqa_200_mmore.jsonl \\
      --out data/medxpertqa_200_hyde_mmore.jsonl

  # Local HF LLM (gated models need HF_TOKEN):
  export HYDE_MODEL=meta-llama/Llama-3.1-8B-Instruct
  python scripts/hyde_expand_queries.py --in ... --out ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from langchain_classic.chains import HypotheticalDocumentEmbedder
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

MEDICAL_HYDE_TEMPLATE = """Please write a medical textbook passage that would help answer the clinical question below.
Include relevant pathophysiology, diagnosis, and management details where appropriate.
Question: {QUESTION}
Passage:"""

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class _StubEmbeddings(Embeddings):
    """Minimal embeddings stub — HyDE passage generation does not use vectors here."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_hyde_model(explicit: str | None) -> str:
    return explicit or os.environ.get("HYDE_MODEL") or DEFAULT_OPENAI_MODEL


def use_hf_llm(model: str) -> bool:
    return model in ("local", "hf") or "/" in model


def build_llm(model: str) -> BaseLanguageModel:
    if use_hf_llm(model):
        model_id = DEFAULT_HF_MODEL if model in ("local", "hf") else model
        import torch
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        gen = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto",
            ),
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=False,
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen))
    from langchain_openai import ChatOpenAI

    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs: dict = {"model": model, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def build_hyde_embedder(llm: BaseLanguageModel) -> HypotheticalDocumentEmbedder:
    medical_prompt = PromptTemplate(
        template=MEDICAL_HYDE_TEMPLATE,
        input_variables=["QUESTION"],
    )
    # Real dense model is optional; MMORE embeds hyde_input server-side.
    base_embeddings: Embeddings = _StubEmbeddings()
    return HypotheticalDocumentEmbedder.from_llm(
        llm,
        base_embeddings,
        custom_prompt=medical_prompt,
    )


def hypothetical_passage(hyde: HypotheticalDocumentEmbedder, question: str) -> str:
    """Generate hypothetical document text (LangChain HyDE llm_chain)."""
    var_name = hyde.input_keys[0]
    result = hyde.llm_chain.invoke({var_name: question})
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        key = hyde.output_keys[0] if hyde.output_keys else next(iter(result))
        return str(result.get(key, result)).strip()
    return str(result).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="HyDE-expand MMORE query JSONL (run_H)")
    parser.add_argument("--in", dest="in_path", default="data/medxpertqa_200_mmore.jsonl")
    parser.add_argument("--out", dest="out_path", default="data/medxpertqa_200_hyde_mmore.jsonl")
    parser.add_argument("--model", default=None, help="OpenAI model or HF id (default: HYDE_MODEL / gpt-4o-mini)")
    parser.add_argument("--incremental", action="store_true", help="Skip rows that already have hyde_input")
    parser.add_argument("--sleep", type=float, default=0.0, help="Pause between LLM calls (seconds)")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    model_name = resolve_hyde_model(args.model)

    if not use_hf_llm(model_name) and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "Set OPENAI_API_KEY for HyDE (gpt-4o-mini), or HYDE_MODEL=<hf-model-id> for local HF."
        )

    rows_in = load_jsonl(in_path)
    existing: dict[str, dict] = {}
    if args.incremental and out_path.exists():
        for row in load_jsonl(out_path):
            key = row.get("input") or row.get("query") or ""
            if key:
                existing[key] = row

    print(f"HyDE model: {model_name}")
    llm = build_llm(model_name)
    hyde = build_hyde_embedder(llm)

    out_rows: list[dict] = []
    for i, row in enumerate(rows_in):
        original = row.get("input") or row.get("query") or ""
        if not original:
            print(f"  [!] line {i}: missing input")
            continue
        if args.incremental and original in existing and existing[original].get("hyde_input"):
            out_rows.append(existing[original])
            continue

        try:
            passage = hypothetical_passage(hyde, original)
        except Exception as e:
            print(f"  [!] HyDE failed for query {i}: {e}")
            raise

        out_row = {
            **row,
            "input": original,
            "hyde_input": passage,
        }
        out_rows.append(out_row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows_in)}")
        if args.sleep:
            time.sleep(args.sleep)

    write_jsonl(out_path, out_rows)
    print(f"✓ {len(out_rows)} HyDE queries → {out_path}")


if __name__ == "__main__":
    main()
