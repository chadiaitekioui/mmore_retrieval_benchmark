"""
Step 3 — Comparison table + McNemar (primary: run_B vs run_C for judge).

Usage:
    python scripts/03_compare_runs.py --results-dir results/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_io import RUN_K, k_for_run  # noqa: E402


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(center - margin, 4), round(center + margin, 4)


def mcnemar_test(hits_a: list[int], hits_b: list[int]) -> dict:
    b = sum(1 for a, bv in zip(hits_a, hits_b) if a == 1 and bv == 0)
    c = sum(1 for a, bv in zip(hits_a, hits_b) if a == 0 and bv == 1)
    if b + c == 0:
        return {"b": 0, "c": 0, "delta_pp": 0.0, "or": None, "p": 1.0}
    from scipy.stats import chi2 as chi2_dist

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = chi2_dist.sf(chi2, df=1)
    n = len(hits_a)
    return {
        "b": b,
        "c": c,
        "delta_pp": round((c - b) / n * 100, 2),
        "or": round(c / b, 3) if b > 0 else float("inf"),
        "p": round(p, 4),
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [None] * n
    running_min = 1.0
    for rank, (orig_idx, p) in enumerate(reversed(indexed)):
        adjusted = p * (rank + 1)
        running_min = min(running_min, adjusted)
        corrected[orig_idx] = min(running_min, 1.0)
    return corrected


def load_run_metrics(results_dir: Path) -> list[dict]:
    runs = []
    for metrics_file in sorted(results_dir.glob("*/metrics.json")):
        with metrics_file.open() as f:
            m = json.load(f)
        m["run"] = m.get("run") or metrics_file.parent.name
        runs.append(m)
    return runs


def load_per_question_hits(results_dir: Path, run_name: str) -> list[int]:
    path = results_dir / run_name / "per_question_hits.json"
    if not path.exists():
        return []
    with path.open() as f:
        rows = json.load(f)
    return [int(r["hit"]) for r in rows]


def hit_key(metrics: dict) -> tuple[int, float]:
    k = metrics.get("k", 5)
    return k, metrics.get(f"hit_at_{k}", 0.0)


def print_table(runs: list[dict]):
    col_run = max(max(len(r["run"]) for r in runs), 12)
    header = (
        f"{'Run':<{col_run}} | {'k':>3} | {'Hit@k':>8} | {'95% CI':>15} | "
        f"{'MRR':>6} | {'NDCG':>6} | {'Judge':>6} | {'Suff%':>6} | {'Corr%':>6}"
    )
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for r in runs:
        k = r.get("k", 5)
        ci = r.get(f"hit_at_{k}_ci_95", ("—", "—"))
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if isinstance(ci, (list, tuple)) else "—"
        judge = f"{r['judge_score_mean']:.1f}" if r.get("judge_score_mean") else "—"
        suff = f"{r['sufficient_rate']:.1%}" if r.get("sufficient_rate") is not None else "—"
        corr = f"{r['corrective_rate']:.1%}" if r.get("corrective_rate") is not None else "—"
        print(
            f"{r['run']:<{col_run}} | {k:>3} | "
            f"{r.get(f'hit_at_{k}', 0):>8.3f} | {ci_str:>15} | "
            f"{r.get(f'mrr_at_{k}', 0):>6.3f} | {r.get(f'ndcg_at_{k}', 0):>6.3f} | "
            f"{judge:>6} | {suff:>6} | {corr:>6}"
        )
    print("─" * len(header))


def print_mcnemar_block(title: str, comparisons: list[dict]):
    if not comparisons:
        return
    p_vals = [c["p"] for c in comparisons]
    for c, p_holm in zip(comparisons, holm_bonferroni(p_vals)):
        c["p_holm"] = round(p_holm, 4)

    print(f"\n── {title} ──")
    print(
        f"  {'Comparison':<28} | {'(b,c)':>9} | {'Δpp':>6} | "
        f"{'p':>7} | {'p_holm':>7} | {'sig':>4}"
    )
    print("  " + "─" * 72)
    for c in comparisons:
        sig = "✓" if c["p_holm"] < 0.05 else "✗"
        print(
            f"  {c['comparison']:<28} | ({c['b']:>3},{c['c']:>3}) | "
            f"{c['delta_pp']:>+6.2f} | {c['p']:>7.4f} | {c['p_holm']:>7.4f} | {sig:>4}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--baseline", default="run_A")
    parser.add_argument(
        "--judge-pair",
        default="run_B,run_C",
        help="Primary paired comparison for judge utility (McNemar)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_run_metrics(results_dir)
    if not runs:
        raise SystemExit(f"No metrics.json found in {results_dir}")

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RETRIEVAL — {len(runs)} runs")
    print(f"{'='*60}")
    print_table(runs)

    # McNemar vs baseline
    baseline_hits = load_per_question_hits(results_dir, args.baseline)
    baseline_comparisons: list[dict] = []
    if baseline_hits:
        for run_name in sorted(RUN_K.keys()):
            if run_name == args.baseline:
                continue
            run_hits = load_per_question_hits(results_dir, run_name)
            if not run_hits or len(run_hits) != len(baseline_hits):
                continue
            k_a = k_for_run(args.baseline)
            k_b = k_for_run(run_name)
            if k_a != k_b:
                continue
            res = mcnemar_test(baseline_hits, run_hits)
            res["comparison"] = f"{args.baseline} → {run_name}"
            baseline_comparisons.append(res)
    print_mcnemar_block(f"McNemar vs baseline '{args.baseline}'", baseline_comparisons)

    # Primary judge comparison: B vs C
    judge_comparisons: list[dict] = []
    pair = [p.strip() for p in args.judge_pair.split(",")]
    if len(pair) == 2:
        run_a, run_b = pair
        hits_a = load_per_question_hits(results_dir, run_a)
        hits_b = load_per_question_hits(results_dir, run_b)
        if hits_a and hits_b and len(hits_a) == len(hits_b):
            res = mcnemar_test(hits_a, hits_b)
            res["comparison"] = f"{run_a} → {run_b} (judge)"
            judge_comparisons.append(res)
            # Optional: C vs C_ctrl
            ctrl_hits = load_per_question_hits(results_dir, "run_C_ctrl")
            if ctrl_hits and len(ctrl_hits) == len(hits_b):
                res2 = mcnemar_test(ctrl_hits, hits_b)
                res2["comparison"] = "run_C_ctrl → run_C"
                judge_comparisons.append(res2)
    print_mcnemar_block("McNemar — judge effect (reranker held constant)", judge_comparisons)

    summary = {
        "runs": runs,
        "mcnemar_baseline": baseline_comparisons,
        "mcnemar_judge": judge_comparisons,
        "methodology_notes": [
            "GT v2: labels per (query, chunk_id) from union of source runs.",
            "Primary judge test: run_B vs run_C (McNemar).",
            "run_F evaluated at k=10; others at k=5.",
            "Chunks collected via HTTP API or 04_convert (no run_rag.py changes).",
        ],
    }
    out = results_dir / "summary.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary → {out}")


if __name__ == "__main__":
    main()
