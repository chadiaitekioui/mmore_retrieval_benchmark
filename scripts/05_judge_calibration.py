"""
Judge calibration — sufficiency threshold sweep + judge score vs exam Hit correlation.

1) Threshold grid (0.3 / 0.5 / 0.7 / 0.9 → min_context_relevance 3–9 on 1–10 scale):
   - Offline: simulated threshold-only corrective rate from run_C first-pass metrics.
   - Observed: corrective_rate + Hit@10 from collected run_C_suff_* runs (if present).

2) Correlation: context_relevance_score (judge) vs binary Hit@k on GT-aligned queries,
   including the recoverable subset (≥1 exam-relevant chunk in GT union).

Usage:
  python scripts/05_judge_calibration.py \\
      --chunks results/run_C/chunks.json \\
      --gt data/ground_truth.json \\
      --out-dir results/judge_calibration

  # After calibration collects:
  python scripts/05_judge_calibration.py ... --calib-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_io import (  # noqa: E402
    align_labels_for_run,
    chunks_from_entry,
    extract_judge_metrics,
    hit_at_k,
    k_for_run,
    load_ground_truth,
    load_retriever_json,
    query_from_entry,
)

try:
    from scipy import stats
except ImportError:
    stats = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

SUFFICIENCY_LEVELS = (0.3, 0.5, 0.7, 0.9)

# Default run_C metric thresholds (other than context relevance).
BASE_THRESHOLDS = {
    "min_mean_similarity": 0.5,
    "min_max_rerank_score": 1.0,
    "min_mean_rerank_score": 0.5,
    "min_num_docs": 3,
}


_THRESHOLD_CHECKS = {
    "min_mean_similarity": "mean_similarity",
    "min_max_similarity": "max_similarity",
    "min_num_docs": "num_docs",
    "min_max_rerank_score": "max_rerank_score",
    "min_mean_rerank_score": "mean_rerank_score",
    "min_context_relevance": "context_relevance_score",
}


def metrics_meet_thresholds(metrics: dict[str, float], thresholds: dict[str, float]) -> bool:
    """Mirror mmore.rag.judge.metrics_meet_thresholds (offline replay)."""
    for key, metric_key in _THRESHOLD_CHECKS.items():
        if key not in thresholds:
            continue
        if metrics.get(metric_key, 0.0) < thresholds[key]:
            return False
    return True


def first_pass_metrics(entry: dict) -> dict[str, float]:
    """Metrics at the first judge evaluation (before corrective retrieval)."""
    corrections = entry.get("retrieval_corrections") or []
    ctx = extract_judge_metrics(entry).get("context_relevance_score")

    if corrections:
        before = corrections[0].get("before") or {}
        metrics = {k: float(v) for k, v in before.items() if v is not None}
    else:
        rm = entry.get("retrieval_metrics") or {}
        metrics = {
            k: float(rm[k])
            for k in (
                "num_docs",
                "mean_similarity",
                "max_similarity",
                "mean_rerank_score",
                "max_rerank_score",
                "has_rerank_scores",
            )
            if k in rm and rm[k] is not None
        }
        if ctx is not None:
            metrics["context_relevance_score"] = float(ctx)
        elif rm.get("context_relevance_score") is not None:
            metrics["context_relevance_score"] = float(rm["context_relevance_score"])

    if ctx is not None:
        metrics["context_relevance_score"] = float(ctx)
    return metrics


def thresholds_for_sufficiency(sufficiency: float) -> dict[str, float]:
    return {
        **BASE_THRESHOLDS,
        "min_context_relevance": round(sufficiency * 10.0, 1),
    }


def simulated_corrective_rate(
    entries: list[dict],
    sufficiency: float,
) -> dict[str, Any]:
    """Threshold-only: would first-pass metrics fail at this sufficiency?"""
    thresholds = thresholds_for_sufficiency(sufficiency)
    flags = []
    for entry in entries:
        metrics = first_pass_metrics(entry)
        if not metrics:
            continue
        flags.append(not metrics_meet_thresholds(metrics, thresholds))
    n = len(flags)
    rate = sum(flags) / n if n else 0.0
    return {
        "sufficiency": sufficiency,
        "min_context_relevance": thresholds["min_context_relevance"],
        "n": n,
        "simulated_threshold_corrective_rate": round(rate, 4),
    }


def correlation_report(
    scores: list[float],
    hits: list[int],
    label: str,
) -> dict[str, Any]:
    n = len(scores)
    if n < 3:
        return {"label": label, "n": n, "note": "too few pairs"}
    if stats is None:
        return {"label": label, "n": n, "note": "scipy not installed"}

    r_pb, p_pb = stats.pointbiserialr(scores, hits)
    rho, p_sp = stats.spearmanr(scores, hits)
    return {
        "label": label,
        "n": n,
        "mean_judge_score": round(sum(scores) / n, 3),
        "hit_rate": round(sum(hits) / n, 4),
        "point_biserial_r": round(float(r_pb), 4),
        "point_biserial_p": round(float(p_pb), 4),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p_sp), 4),
    }


def per_question_hits(
    entries: list[dict],
    gt_map: dict,
    k: int,
) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        query = query_from_entry(entry)
        if not query or query not in gt_map:
            continue
        chunks = chunks_from_entry(entry)
        labels_map = gt_map[query]["labels_by_chunk_id"]
        labels, _ = align_labels_for_run(labels_map, chunks)
        jm = extract_judge_metrics(entry)
        score = jm.get("context_relevance_score")
        hit_k = int(hit_at_k(labels, k))
        hit10 = int(hit_at_k(labels, 10))
        recoverable = any(labels_map.values())
        rows.append(
            {
                "query": query,
                "hit_at_k": hit_k,
                "hit_at_10": hit10,
                "recoverable": recoverable,
                "judge_score": score,
                "corrective": int(jm.get("corrective_steps", 0) > 0),
                "sufficient": jm.get("sufficient"),
            }
        )
    return rows


def load_calib_run_metrics(calib_dir: Path) -> list[dict[str, Any]]:
    """Load metrics.json from run_C_suff_* directories."""
    points = []
    for metrics_path in sorted(calib_dir.glob("run_C_suff_*/metrics.json")):
        with metrics_path.open() as f:
            m = json.load(f)
        run = m.get("run") or metrics_path.parent.name
        tag = run.replace("run_C_suff_", "")
        suff = int(tag) / 100.0 if tag.isdigit() else None
        points.append(
            {
                "run": run,
                "sufficiency": suff,
                "min_context_relevance": thresholds_for_sufficiency(suff)["min_context_relevance"]
                if suff is not None
                else None,
                "corrective_rate": m.get("corrective_rate"),
                "hit_at_10": m.get("hit_at_10"),
                "hit_at_5": m.get("hit_at_5"),
                "n": m.get("n"),
            }
        )
    return sorted(points, key=lambda x: x.get("sufficiency") or 0)


def plot_calibration_curve(
    sweep: list[dict],
    observed: list[dict],
    out_path: Path,
) -> None:
    if plt is None:
        print("[!] matplotlib not installed — skip plot")
        return

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x_sim = [p["min_context_relevance"] for p in sweep]
    y_sim = [p["simulated_threshold_corrective_rate"] * 100 for p in sweep]
    ax1.plot(
        x_sim,
        y_sim,
        "o--",
        color="#555",
        label="Simulated threshold-only corrective % (run_C first pass)",
    )

    if observed:
        x_obs = [p["min_context_relevance"] for p in observed if p.get("hit_at_10") is not None]
        y_corr = [p["corrective_rate"] * 100 for p in observed if p.get("hit_at_10") is not None]
        y_hit = [p["hit_at_10"] * 100 for p in observed if p.get("hit_at_10") is not None]
        ax1.plot(x_obs, y_corr, "s-", color="#c44", label="Observed corrective % (full judge)")
        ax2 = ax1.twinx()
        ax2.plot(x_obs, y_hit, "^-", color="#26a", label="Hit@10 %")
        ax2.set_ylabel("Hit@10 (%)")
        ax2.set_ylim(0, max(100, max(y_hit) * 1.15 if y_hit else 100))
        ax2.legend(loc="lower right")

    ax1.set_xlabel("min_context_relevance (1–10 scale; sufficiency × 10)")
    ax1.set_ylabel("Corrective rate (%)")
    ax1.set_title("Judge calibration: sufficiency threshold vs corrective rate / Hit@10")
    ax1.set_xticks([3, 5, 7, 9])
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Plot → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge threshold calibration + GT correlation")
    parser.add_argument("--chunks", default="results/run_C/chunks.json")
    parser.add_argument("--gt", default="data/ground_truth.json")
    parser.add_argument("--out-dir", default="results/judge_calibration")
    parser.add_argument(
        "--calib-dir",
        default="results",
        help="Parent dir containing run_C_suff_*/metrics.json (after calibration collect)",
    )
    parser.add_argument("--eval-k", type=int, default=10, help="Primary Hit@k for correlation")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit(f"Missing {chunks_path} — collect run_C first.")

    entries = load_retriever_json(chunks_path)
    gt_map = load_ground_truth(Path(args.gt))
    pq = per_question_hits(entries, gt_map, k=args.eval_k)

    scores_all = [r["judge_score"] for r in pq if r["judge_score"] is not None]
    hits_all = [r[f"hit_at_{args.eval_k}"] for r in pq if r["judge_score"] is not None]

    recoverable = [r for r in pq if r["recoverable"] and r["judge_score"] is not None]
    scores_rec = [r["judge_score"] for r in recoverable]
    hits_rec = [r[f"hit_at_{args.eval_k}"] for r in recoverable]

    hits10_rec = [r for r in recoverable if r["hit_at_10"] == 1]
    scores_hit10 = [r["judge_score"] for r in hits10_rec]
    hits_hit10 = [1] * len(hits10_rec)  # trivial — report descriptive stats instead

    sweep = [simulated_corrective_rate(entries, s) for s in SUFFICIENCY_LEVELS]

    observed = load_calib_run_metrics(Path(args.calib_dir))

    # Merge observed into sweep rows when available
    obs_by_suff = {p["sufficiency"]: p for p in observed if p.get("sufficiency") is not None}
    for row in sweep:
        obs = obs_by_suff.get(row["sufficiency"])
        if obs:
            row["observed_corrective_rate"] = obs.get("corrective_rate")
            row["hit_at_10"] = obs.get("hit_at_10")

    correlations = [
        correlation_report(scores_all, hits_all, f"all_queries_hit@{args.eval_k}"),
        correlation_report(scores_rec, hits_rec, f"recoverable_hit@{args.eval_k}"),
    ]
    if hits10_rec:
        correlations.append(
            {
                "label": "recoverable_with_hit@10",
                "n": len(hits10_rec),
                "mean_judge_score": round(sum(scores_hit10) / len(scores_hit10), 3),
                "note": "subset with exam Hit@10=1; correlation vs binary hit is degenerate",
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "sufficiency_levels": list(SUFFICIENCY_LEVELS),
        "min_context_relevance_mapping": "min_context_relevance = sufficiency × 10 (1–10 judge score scale)",
        "threshold_sweep": sweep,
        "calibration_collects": observed,
        "correlations": correlations,
        "n_evaluated": len(pq),
        "n_recoverable": len(recoverable),
        "n_recoverable_hit_at_10": len(hits10_rec),
        "interpretation": [
            "High simulated corrective rate at low thresholds means metric gates fire often.",
            "Divergence between simulated (threshold-only) and observed (full LLM judge) curves "
            "isolates LLM-driven vs metric-driven corrections.",
            "Low point-biserial / Spearman correlation between judge score and exam Hit@k "
            "suggests the judge scores semantic plausibility, not exam-grounded relevance.",
        ],
    }

    with (out_dir / "calibration.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with (out_dir / "per_question_judge_gt.json").open("w") as f:
        json.dump(pq, f, indent=2, ensure_ascii=False)

    plot_calibration_curve(sweep, observed, out_dir / "calibration_curve.png")

    print(f"\n{'─'*55}")
    print("  Judge calibration")
    print(f"{'─'*55}")
    print(f"  Evaluated queries: {len(pq)}  |  Recoverable (≥1 GT relevant chunk): {len(recoverable)}")
    print(f"  Recoverable with Hit@10=1: {len(hits10_rec)}")
    print("\n  Threshold sweep (simulated threshold-only corrective rate):")
    for row in sweep:
        line = (
            f"    suff={row['sufficiency']:.1f}  min_ctx={row['min_context_relevance']:.1f}  "
            f"sim_corrective={row['simulated_threshold_corrective_rate']:.1%}"
        )
        if row.get("observed_corrective_rate") is not None:
            line += f"  observed={row['observed_corrective_rate']:.1%}"
        if row.get("hit_at_10") is not None:
            line += f"  Hit@10={row['hit_at_10']:.1%}"
        print(line)
    print("\n  Judge score vs exam Hit correlation:")
    for c in correlations:
        if c.get("point_biserial_r") is not None:
            print(
                f"    {c['label']}: n={c['n']}  r_pb={c['point_biserial_r']:.3f} (p={c['point_biserial_p']:.4f})  "
                f"ρ={c['spearman_rho']:.3f}  mean_judge={c['mean_judge_score']:.2f}  hit_rate={c['hit_rate']:.1%}"
            )
        else:
            print(f"    {c['label']}: {c}")
    print(f"\n✓ {out_dir / 'calibration.json'}")
    if not observed:
        print(
            "\n[!] No run_C_suff_* metrics found — for observed corrective + Hit@10 curve:\n"
            "    bash jobs/collect_judge_calibration.sh && bash jobs/evaluate_judge_calibration.sh"
        )


if __name__ == "__main__":
    main()
