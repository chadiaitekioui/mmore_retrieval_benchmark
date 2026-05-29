"""
Evaluate downstream RAG answer quality on collected chunks.json (MCQ accuracy + LLM cost).

Usage:
    python scripts/eval_rag_quality.py \\
        --questions data/medxpertqa_200.jsonl \\
        --chunks results/run_steps_1/chunks.json \\
        --out results/run_steps_1/rag_quality.json \\
        --run-name run_steps_1 \\
        --max-corrective-steps 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_io import (  # noqa: E402
    extract_judge_metrics,
    hit_max_corrective_steps,
    llm_calls_from_entry,
    load_questions_jsonl,
    load_retriever_json,
    parse_mcq_choice,
    query_from_entry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/medxpertqa_200.jsonl")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-corrective-steps", type=int, default=None)
    args = parser.parse_args()

    run_name = args.run_name or Path(args.chunks).parent.name
    max_steps = args.max_corrective_steps
    if max_steps is None and run_name.startswith("run_steps_"):
        max_steps = int(run_name.rsplit("_", 1)[-1])

    qmap = load_questions_jsonl(Path(args.questions))
    rows = load_retriever_json(Path(args.chunks))

    n = 0
    n_parsed = 0
    n_correct = 0
    n_api_cost = 0
    judge_calls = 0
    total_calls = 0
    hit_max = 0
    per_question: list[dict] = []

    for entry in rows:
        query = query_from_entry(entry)
        if not query or query not in qmap:
            continue
        gold_idx = int(qmap[query]["answer_idx"])
        answer_text = entry.get("answer") or ""
        pred_idx = parse_mcq_choice(answer_text)
        correct = pred_idx == gold_idx if pred_idx is not None else False

        costs = llm_calls_from_entry(entry)
        jm = extract_judge_metrics(entry)
        at_max = hit_max_corrective_steps(entry)
        if at_max:
            hit_max += 1

        n += 1
        if costs.get("from_api"):
            n_api_cost += 1
        if pred_idx is not None:
            n_parsed += 1
        if correct:
            n_correct += 1
        judge_calls += costs["judge_llm_calls"]
        total_calls += costs["total_llm_calls"]

        per_question.append(
            {
                "query": query,
                "gold_idx": gold_idx,
                "pred_idx": pred_idx,
                "mcq_correct": int(correct),
                "parse_ok": pred_idx is not None,
                "llm_calls": costs,
                "hit_max_steps": at_max,
                "judge": jm,
            }
        )

    if n == 0:
        raise ValueError("No questions matched")

    metrics = {
        "run": run_name,
        "n": n,
        "mcq_accuracy": round(n_correct / n, 4),
        "mcq_parse_rate": round(n_parsed / n, 4),
        "mean_judge_llm_calls": round(judge_calls / n, 4),
        "mean_total_llm_calls": round(total_calls / n, 4),
        "non_convergence_rate": round(hit_max / n, 4),
        "max_corrective_steps": max_steps,
        "llm_calls_from_api_rate": round(n_api_cost / n, 4),
        # Legacy keys for judge_study.py backward compat
        "mean_judge_llm_calls_est": round(judge_calls / n, 4),
        "mean_total_llm_calls_est": round(total_calls / n, 4),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"summary": metrics, "per_question": per_question}, f, indent=2)

    src = "API" if n_api_cost == n else f"API {n_api_cost}/{n}"
    print(
        f"  {run_name}: MCQ acc={metrics['mcq_accuracy']:.3f}  "
        f"cost={metrics['mean_total_llm_calls']:.2f} LLM/q ({src})  "
        f"non_conv={metrics['non_convergence_rate']:.1%}"
    )
    print(f"✓ {out_path}")


if __name__ == "__main__":
    main()
