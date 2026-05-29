"""
Judge study analysis — axis 2 (max_steps Pareto) + axis 3 (forced action comparison).

Usage:
    python scripts/judge_study.py --results-dir results --out-dir results/judge_study
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_io import load_retriever_json, trigger_queries_from_entries  # noqa: E402

STUDY_RUNS_AXIS2 = tuple(f"run_steps_{n}" for n in range(3))
STUDY_RUNS_AXIS3 = ("run_force_rr", "run_force_aq", "run_force_ac")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore


def load_rag_quality(results_dir: Path, run: str) -> dict[str, Any] | None:
    path = results_dir / run / "rag_quality.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f).get("summary")


def plot_pareto(axis2: list[dict], out_path: Path) -> None:
    if plt is None or not axis2:
        return
    xs = [
        p.get("mean_total_llm_calls") or p.get("mean_total_llm_calls_est", 0)
        for p in axis2
    ]
    ys = [p["mcq_accuracy"] for p in axis2]
    labels = [f"steps={p['max_corrective_steps']}" for p in axis2]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=80, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.plot(xs, ys, "--", alpha=0.4, color="gray")
    ax.set_xlabel("Mean LLM calls per query (answer + judge)")
    ax.set_ylabel("MCQ accuracy (answer relevance proxy)")
    ax.set_title("Axis 2 — Quality vs cost by max_corrective_steps")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_axis3(axis3: list[dict], out_path: Path) -> None:
    if plt is None or not axis3:
        return
    names = [p["run"].replace("run_force_", "").upper() for p in axis3]
    accs = [p["mcq_accuracy"] for p in axis3]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    ax.bar(names, accs, color=colors[: len(names)])
    ax.set_ylabel("MCQ accuracy on trigger subset T")
    ax.set_title("Axis 3 — Forced corrective action comparison")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/judge_study")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    axis2: list[dict] = []
    for run in STUDY_RUNS_AXIS2:
        summary = load_rag_quality(results_dir, run)
        if summary:
            axis2.append(summary)

    scout_chunks = results_dir / "run_judge_scout" / "chunks.json"
    trigger: set[str] = set()
    if scout_chunks.exists():
        trigger = trigger_queries_from_entries(load_retriever_json(scout_chunks))

    axis3: list[dict] = []
    for run in STUDY_RUNS_AXIS3:
        rq_path = results_dir / run / "rag_quality.json"
        if not rq_path.exists() or not trigger:
            continue
        with rq_path.open() as f:
            data = json.load(f)
        subset = [pq for pq in data.get("per_question", []) if pq["query"] in trigger]
        if not subset:
            continue
        n = len(subset)
        axis3.append(
            {
                "run": run,
                "n_trigger": n,
                "mcq_accuracy": round(sum(p["mcq_correct"] for p in subset) / n, 4),
                "mean_total_llm_calls": round(
                    sum(
                        p["llm_calls"].get("total_llm_calls")
                        or p["llm_calls"].get("total_llm_calls_est", 0)
                        for p in subset
                    )
                    / n,
                    4,
                ),
                "full_corpus_mcq_accuracy": data["summary"].get("mcq_accuracy"),
            }
        )

    report = {
        "axis2_max_steps": sorted(axis2, key=lambda x: x.get("max_corrective_steps", 0)),
        "axis3_forced_actions": axis3,
        "trigger_subset": {
            "scout_run": "run_judge_scout",
            "n_queries": len(trigger),
            "definition": "exit_reason=llm_corrective in judge_steps (fallback: judge_actions non-empty)",
        },
        "notes": {
            "mcq_accuracy": "Proxy for answer relevance on MedXpertQA MCQ",
            "llm_calls": "From MMORE judge_llm_calls + 1 answer LLM when API trace present",
            "non_convergence_rate": "hit_max_corrective_steps or judge_reason=max_corrective_steps",
        },
    }

    with (out_dir / "judge_study.json").open("w") as f:
        json.dump(report, f, indent=2)

    plot_pareto(axis2, out_dir / "pareto_quality_vs_cost.png")
    plot_axis3(axis3, out_dir / "axis3_action_comparison.png")

    print("\n── Axis 2 (max_corrective_steps) ──")
    for row in report["axis2_max_steps"]:
        cost = row.get("mean_total_llm_calls") or row.get("mean_total_llm_calls_est")
        print(
            f"  steps={row.get('max_corrective_steps')}  "
            f"MCQ={row.get('mcq_accuracy', '—')}  "
            f"cost={cost}  "
            f"non_conv={row.get('non_convergence_rate', '—')}"
        )
    print(f"\n── Axis 3 (trigger subset n={len(trigger)}) ──")
    for row in axis3:
        print(f"  {row['run']}: MCQ={row['mcq_accuracy']} (n={row['n_trigger']})")
    print(f"\n✓ {out_dir / 'judge_study.json'}")


if __name__ == "__main__":
    main()
