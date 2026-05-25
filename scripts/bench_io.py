"""
Shared I/O helpers for the MMORE retrieval benchmark.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Default k per run (run_F uses 10 at retrieval)
RUN_K: dict[str, int] = {
    "run_A": 5,
    "run_B": 5,
    "run_C": 5,
    "run_C_ctrl": 5,
    "run_D": 5,
    "run_E": 5,
    "run_F": 10,
}

GT_VERSION = 2


def content_hash(text: str, n: int = 400) -> str:
    """Stable content fingerprint (unlike built-in hash())."""
    return hashlib.sha256(text[:n].encode("utf-8", errors="replace")).hexdigest()[:16]


def chunk_text(chunk: dict) -> str:
    return (
        chunk.get("content")
        or chunk.get("text")
        or chunk.get("page_content")
        or ""
    )


def chunk_id(chunk: dict) -> str:
    """Stable id: Milvus id in metadata, else explicit id fields, else content hash."""
    meta = chunk.get("metadata") or {}
    for key in ("id", "chunk_id", "doc_id"):
        val = chunk.get(key) or meta.get(key)
        if val is not None:
            return str(val)
    text = chunk_text(chunk)
    return f"hash:{content_hash(text)}"


def normalize_retriever_api_doc(raw: dict) -> dict:
    """Map MMORE POST /v1/retrieve document to benchmark chunk dict."""
    file_id = raw.get("fileId", "")
    chunk_part = raw.get("chunkId")
    if file_id and chunk_part:
        stable_id = f"{file_id}+{chunk_part}"
    else:
        stable_id = str(file_id or "")
    meta = dict(raw.get("metadata") or {})
    if stable_id:
        meta["id"] = stable_id
    return normalize_chunk(
        {
            "id": stable_id,
            "content": raw.get("content", ""),
            "similarity_score": raw.get("similarity"),
            "rerank_score": raw.get("rerank_score"),
            "metadata": meta,
        }
    )


def normalize_chunk(raw: dict) -> dict:
    """Normalize MMORE retriever/RAG chunk dict to benchmark schema."""
    if "fileId" in raw:
        return normalize_retriever_api_doc(raw)

    meta = dict(raw.get("metadata") or {})
    text = chunk_text(raw)
    if not text and meta:
        text = str(meta.get("text", ""))

    sim = raw.get("similarity_score")
    if sim is None:
        sim = raw.get("score")
    if sim is None:
        sim = meta.get("similarity")

    rerank = raw.get("rerank_score")
    if rerank is None:
        rerank = meta.get("rerank_score")

    cid = chunk_id(raw if raw.get("metadata") is None else {**raw, "metadata": meta})

    out = {
        "id": cid,
        "page_content": text,
        "content": text,
        "similarity_score": float(sim) if sim is not None else None,
        "rerank_score": float(rerank) if rerank is not None else None,
        "metadata": meta,
    }
    if meta.get("rank") is not None:
        out["rank"] = meta["rank"]
    return out


def query_from_entry(entry: dict) -> str:
    return (
        entry.get("query")
        or entry.get("question")
        or entry.get("input")
        or ""
    ).strip()


def chunks_from_entry(entry: dict) -> list[dict]:
    raw = entry.get("chunks") or entry.get("retrieved_chunks")
    if raw:
        return [normalize_chunk(c) for c in raw]

    ctx = entry.get("context")
    if isinstance(ctx, list):
        return [normalize_chunk(c) for c in ctx]

    # Raw retriever API response saved as {"documents": [...]} per query
    docs = entry.get("documents")
    if isinstance(docs, list):
        return [normalize_chunk(d) for d in docs]

    return []


def entry_from_api_payload(query: str, payload: dict) -> dict:
    """Build one benchmark result row from an API JSON body."""
    entry: dict[str, Any] = {"query": query}

    if isinstance(payload, list):
        entry["documents"] = payload
        entry["chunks"] = [normalize_chunk(d) for d in payload]
    else:
        entry.update(payload)
        entry["chunks"] = chunks_from_entry(entry)
        for key in (
            "retrieval_metrics",
            "judge_decision",
            "judge_actions",
            "retrieval_corrections",
            "input",
            "answer",
            "context",
        ):
            if key in payload:
                entry[key] = payload[key]
        if payload.get("input") and not entry.get("query"):
            entry["query"] = payload["input"]

    return entry


def load_retriever_json(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array")
    return data


def union_chunks_by_query(paths: list[Path]) -> dict[str, list[dict]]:
    """Merge chunks per query across runs (dedupe by chunk_id, preserve order)."""
    per_query: dict[str, dict[str, dict]] = {}

    for path in paths:
        for entry in load_retriever_json(path):
            q = query_from_entry(entry)
            if not q:
                continue
            bucket = per_query.setdefault(q, {})
            for ch in chunks_from_entry(entry):
                cid = chunk_id(ch)
                if cid not in bucket:
                    bucket[cid] = ch

    return {q: list(chunks.values()) for q, chunks in per_query.items()}


def build_chunks_text(chunks: list[dict], max_chars: int = 600) -> str:
    lines = []
    for i, c in enumerate(chunks):
        content = chunk_text(c)
        lines.append(f"[{i}] {content[:max_chars]}")
    return "\n---\n".join(lines)


def load_questions_jsonl(path: Path) -> dict[str, dict]:
    questions: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = item.get("query") or item.get("input") or ""
            if key:
                questions[key] = item
    return questions


def align_labels_for_run(
    labels_by_chunk_id: dict[str, bool],
    run_chunks: list[dict],
) -> tuple[list[bool], int]:
    """
    Map corpus-level GT to this run's chunk order.
    Returns (labels, n_unlabeled) for chunks missing from GT.
    """
    labels: list[bool] = []
    n_unlabeled = 0
    for ch in run_chunks:
        cid = chunk_id(ch)
        if cid in labels_by_chunk_id:
            labels.append(bool(labels_by_chunk_id[cid]))
        else:
            labels.append(False)
            n_unlabeled += 1
    return labels, n_unlabeled


def extract_judge_metrics(entry: dict) -> dict[str, Any]:
    rm = entry.get("retrieval_metrics") or {}
    actions = entry.get("judge_actions") or []
    decision = entry.get("judge_decision") or entry.get("decision")

    ctx_score = entry.get("context_relevance_score")
    if ctx_score is None and rm:
        ctx_score = rm.get("context_relevance_score")

    sufficient = entry.get("sufficient")
    if sufficient is None and rm:
        sufficient = bool(rm.get("thresholds_met"))

    return {
        "context_relevance_score": ctx_score,
        "sufficient": sufficient,
        "decision": decision,
        "mean_similarity": rm.get("mean_similarity") or entry.get("mean_similarity"),
        "mean_rerank_score": rm.get("mean_rerank_score") or entry.get("mean_rerank_score"),
        "max_rerank_score": rm.get("max_rerank_score") or entry.get("max_rerank_score"),
        "corrective_steps": len(actions) if actions else entry.get("corrective_steps", 0),
    }


def load_ground_truth(path: Path) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and data.get("version") == GT_VERSION:
        return {q["query"]: q for q in data["questions"]}

    # Legacy v1: list with chunk_relevance aligned to run_A chunks
    if isinstance(data, list):
        out: dict[str, dict] = {}
        for item in data:
            q = item["query"]
            chunks = [normalize_chunk(c) for c in item.get("chunks", [])]
            labels = item.get("chunk_relevance", [])
            labels_map = {
                chunk_id(c): bool(lbl)
                for c, lbl in zip(chunks, labels)
            }
            out[q] = {
                "query": q,
                "answer_text": item.get("answer_text", ""),
                "answer_idx": item.get("answer_idx"),
                "labels_by_chunk_id": labels_map,
            }
        return out

    raise ValueError(f"Unsupported ground truth format: {path}")


def k_for_run(run_name: str, override: int | None = None) -> int:
    if override is not None:
        return override
    return RUN_K.get(run_name, 5)
