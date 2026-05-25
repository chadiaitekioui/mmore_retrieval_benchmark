"""
Step 2 — Retrieval metrics for one run (corpus-level GT by chunk_id).

Usage:
    python scripts/02_compute_metrics.py \\
        --gt data/ground_truth.json \\
        --chunks results/run_C/chunks.json \\
        --out results/run_C/metrics.json \\
        --run-name run_C
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_io import (  # noqa: E402
    align_labels_for_run,
    chunk_id,
    chunks_from_entry,
    extract_judge_metrics,
    k_for_run,
    load_ground_truth,
    load_retriever_json,
    query_from_entry,
)


def hit_at_k(labels: list[bool], k: int) -> float:
    return float(any(labels[:k]))


def mrr_at_k(labels: list[bool], k: int) -> float:
    for rank, rel in enumerate(labels[:k], 1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels: list[bool], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, rel in enumerate(labels[:k], 1) if rel)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(sum(labels), k) + 1))
    return dcg / idcg if idcg else 0.0


def precision_at_k(labels: list[bool], k: int) -> float:
    return sum(labels[:k]) / k


def recall_at_k(labels: list[bool], k: int) -> float:
    total = sum(labels)
    return sum(labels[:k]) / total if total else 0.0


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(center - margin, 4), round(center + margin, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--k", type=int, default=None, help="Override k (default: from run name)")
    parser.add_argument(
        "--reference-run",
        default="run_B",
        help="Run used for stratification (B hit@k = 0 vs ≥1)",
    )
    args = parser.parse_args()

    run_name = args.run_name or Path(args.chunks).parent.name
    k = k_for_run(run_name, args.k)
    gt_map = load_ground_truth(Path(args.gt))
    run_data = load_retriever_json(Path(args.chunks))

    hits, mrrs, ndcgs, precs, recs = [], [], [], [], []
    per_question: list[dict] = []
    judge_scores, sufficient_flags, corrective_flags = [], [], []
    sim_means, rerank_means = [], []
    n_skipped = 0
    n_unlabeled_total = 0

    for entry in run_data:
        query = query_from_entry(entry)
        run_chunks = chunks_from_entry(entry)
        if not query or query not in gt_map:
            n_skipped += 1
            continue

        gt_item = gt_map[query]
        labels_map = gt_item["labels_by_chunk_id"]
        labels, n_unlabeled = align_labels_for_run(labels_map, run_chunks)
        n_unlabeled_total += n_unlabeled

        hit = int(hit_at_k(labels, k))
        hits.append(hit)
        mrrs.append(mrr_at_k(labels, k))
        ndcgs.append(ndcg_at_k(labels, k))
        precs.append(precision_at_k(labels, k))
        recs.append(recall_at_k(labels, k))

        jm = extract_judge_metrics(entry)
        if jm["context_relevance_score"] is not None:
            judge_scores.append(jm["context_relevance_score"])
        if jm["sufficient"] is not None:
            sufficient_flags.append(int(bool(jm["sufficient"])))
        corrective_flags.append(int(jm["corrective_steps"] > 0))

        chunks_k = run_chunks[:k]
        sims = [c.get("similarity_score") for c in chunks_k if c.get("similarity_score") is not None]
        rerank = [c.get("rerank_score") for c in chunks_k if c.get("rerank_score") is not None]
        if sims:
            sim_means.append(sum(sims) / len(sims))
        if rerank:
            rerank_means.append(sum(rerank) / len(rerank))

        per_question.append(
            {
                "query": query,
                "hit": hit,
                "n_unlabeled_in_top_k": sum(
                    1
                    for ch, lbl in zip(run_chunks[:k], labels[:k])
                    if ch and chunk_id(ch) not in labels_map
                ),
                "judge": jm,
            }
        )

    n = len(hits)
    if n == 0:
        raise ValueError("No questions matched — check query/input and ground_truth.json")

    p_hit = sum(hits) / n
    ci = wilson_ci(p_hit, n)

    metrics = {
        "run": run_name,
        "n": n,
        "k": k,
        "n_skipped": n_skipped,
        "n_unlabeled_chunks": n_unlabeled_total,
        f"hit_at_{k}": round(p_hit, 4),
        f"hit_at_{k}_ci_95": ci,
        f"mrr_at_{k}": round(sum(mrrs) / n, 4),
        f"ndcg_at_{k}": round(sum(ndcgs) / n, 4),
        f"precision_at_{k}": round(sum(precs) / n, 4),
        f"recall_at_{k}": round(sum(recs) / n, 4),
        "mean_similarity": round(sum(sim_means) / len(sim_means), 4) if sim_means else None,
        "mean_rerank_score": round(sum(rerank_means) / len(rerank_means), 4) if rerank_means else None,
        "judge_score_mean": round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None,
        "sufficient_rate": round(sum(sufficient_flags) / len(sufficient_flags), 4)
        if sufficient_flags
        else None,
        "corrective_rate": round(sum(corrective_flags) / n, 4),
        "judge_hit_correlation_note": (
            "Secondary only; annotator and judge may share model family."
        ),
    }

    if judge_scores and len(judge_scores) == n:
        r, p = stats.pointbiserialr(judge_scores, hits)
        metrics["judge_hit_correlation"] = {"r": round(r, 4), "p": round(p, 4)}

    # Stratification vs reference run B (if per_question_hits for B exist)
    ref_hits_path = Path(args.chunks).parent.parent / args.reference_run / "per_question_hits.json"
    stratification = None
    if ref_hits_path.exists() and run_name != args.reference_run:
        with ref_hits_path.open() as f:
            ref = {r["query"]: r["hit"] for r in json.load(f)}
        strat_miss, strat_hit = [], []
        for pq in per_question:
            q, h = pq["query"], pq["hit"]
            if q not in ref:
                continue
            (strat_miss if ref[q] == 0 else strat_hit).append(h)
        if strat_miss or strat_hit:
            stratification = {
                f"{args.reference_run}_miss_hit_rate": round(sum(strat_miss) / len(strat_miss), 4)
                if strat_miss
                else None,
                f"{args.reference_run}_hit_hit_rate": round(sum(strat_hit) / len(strat_hit), 4)
                if strat_hit
                else None,
                "n_miss": len(strat_miss),
                "n_already_hit": len(strat_hit),
            }
            metrics["stratification"] = stratification

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    hits_path = out_path.parent / "per_question_hits.json"
    with hits_path.open("w") as f:
        json.dump(per_question, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"  Run {run_name} | n={n} | k={k} | unlabeled chunks={n_unlabeled_total}")
    print(f"  Hit@{k} = {p_hit:.3f}  CI {ci}")
    print(f"  MRR@{k} = {metrics[f'mrr_at_{k}']:.3f}  NDCG@{k} = {metrics[f'ndcg_at_{k}']:.3f}")
    if stratification:
        print(f"  Stratification vs {args.reference_run}: {stratification}")
    print(f"✓ {out_path}")
    print(f"✓ {hits_path}")


if __name__ == "__main__":
    main()
