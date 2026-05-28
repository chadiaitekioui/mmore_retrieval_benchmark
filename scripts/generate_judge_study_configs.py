"""Generate RAG configs for judge study (axis 2: max_steps, axis 3: forced actions + scout)."""

from __future__ import annotations

import argparse
import re
from copy import deepcopy
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

# Shared index thresholds (MMORE _THRESHOLD_CHECKS only — no min_context_relevance).
METRIC_THRESHOLDS = {
    "min_mean_similarity": 0.5,
    "min_max_rerank_score": 1.0,
    "min_mean_rerank_score": 0.5,
    "min_num_docs": 3,
}

JUDGE_SYSTEM_PROMPT = """\
You are a retrieval quality judge for a medical RAG system (English only).
You are called when index thresholds have failed at least once, or to re-evaluate after a corrective action.

In one step: (1) set context_relevance_score 1-10, (2) read Threshold check (PASS/FAIL),
(3) pick exactly ONE decision from Allowed actions.

Respond with a single valid JSON object only (no markdown fences, no commentary).
Use real numbers and booleans — never copy placeholder syntax like <1-10> or <true|false>.
Keep "reason" to one short sentence (max 120 chars). Do not paste chunk text into JSON strings.
Never copy the example below verbatim; adapt every field to the current question, metrics, and chunks.
decision must be exactly one of: PROCEED, ADD_QUESTIONS, ADD_CONTEXT, RE_RETRIEVE.

Hard rules (mandatory):
- If after_correction is false and Threshold check contains any FAIL: do NOT choose PROCEED.
  Pick the best corrective action from Allowed actions.
- When unsure among corrective actions, prefer RE_RETRIEVE if it is in Allowed actions.
- Choose PROCEED only if (a) every Threshold check line is PASS, or (b) after_correction is true
  AND context_relevance_score >= 8 AND chunks explicitly support the answer.
- If corrective_actions_used >= max_corrective_steps, choose PROCEED only.

Example when Threshold check has FAIL (adapt values; do not reuse this reason text):
{
  "context_relevance_score": 4,
  "sufficient": false,
  "decision": "RE_RETRIEVE",
  "reason": "Rerank score failed; widen retrieval.",
  "extra_questions": [],
  "web_query": null,
  "retrieve_params": {"input": "reformulated clinical query", "k": 10}
}

Action guidelines when retrieval is weak:
1. RE_RETRIEVE: Default corrective action — reformulate retrieve_params.input and/or raise k (e.g. 8-10).
   Use when any index threshold FAIL.
2. ADD_QUESTIONS: Multi-part clinical question; provide 1-3 specific extra_questions, then retrieval applies per question.
3. ADD_CONTEXT: Last resort only if the indexed corpus cannot contain the answer (novel drug, no indexed source).
   Set web_query. Do not use for standard clinical facts likely in the corpus.
4. PROCEED: Only when all thresholds PASS, or after a corrective action with strong chunk support (see Hard rules).
Never choose an action not listed under Allowed actions.
"""

JUDGE_USER_PROMPT = """\
Question: {query}

after_correction: {after_correction}
correction_step: {correction_step}
corrective_actions_used: {corrective_actions_used}
max_corrective_steps: {max_corrective_steps}
remaining_corrective_steps: {remaining_corrective_steps}

Retrieval metrics (numeric; context_relevance_score is yours to set in JSON):
{metrics}

Threshold check (compare each line to Configured thresholds):
{metrics_status}

Configured thresholds: {thresholds}
Allowed actions: {allowed_actions}

Retrieved chunks:
{chunks}

Return JSON only.
"""


def build_config(
    *,
    header: str,
    max_corrective_steps: int,
    skip_llm_judge: bool = False,
    allow_re_retrieve: bool = True,
    allow_add_questions: bool = True,
    allow_add_context: bool = True,
    force_corrective_action: str | None = None,
) -> dict:
    judge_block: dict = {
        "llm": {
            "llm_name": "meta-llama/Llama-3.1-8B-Instruct",
            "max_new_tokens": 512,
            "temperature": 0.01,
        },
        "skip_llm_judge": skip_llm_judge,
        "system_prompt": JUDGE_SYSTEM_PROMPT,
        "user_prompt": JUDGE_USER_PROMPT,
        "metric_thresholds": deepcopy(METRIC_THRESHOLDS),
        "max_corrective_steps": max_corrective_steps,
        "allow_add_questions": allow_add_questions,
        "allow_add_context": allow_add_context,
        "allow_re_retrieve": allow_re_retrieve,
        "max_web_results": 5,
    }
    if force_corrective_action:
        judge_block["force_corrective_action"] = force_corrective_action
    return {
        "rag": {
            "llm": {
                "llm_name": "meta-llama/Llama-3.1-8B-Instruct",
                "max_new_tokens": 1200,
            },
            "retriever": {
                "db": {"uri": "${DB_URI}", "name": "${DB_NAME}"},
                "collection_name": "${COLLECTION_NAME}",
                "hybrid_search_weight": 0.5,
                "k": 5,
                "use_web": False,
                "reranker_model_name": "BAAI/bge-reranker-base",
            },
            "judge": judge_block,
            "system_prompt": (
                "Use the following context to answer the questions.\n\nContext:\n{context}"
            ),
        },
        "mode": "api",
        "mode_args": {"endpoint": "/rag", "port": 8000, "host": "0.0.0.0"},
        "_header": header,
    }


def dump_yaml(cfg: dict, path: Path) -> None:
    header = cfg.pop("_header", "")

    def _literal_str(dumper: yaml.Dumper, data: str) -> yaml.nodes.Node:
        style = "|" if "\n" in data else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

    yaml.add_representer(str, _literal_str)
    body = yaml.dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False)
    text = (header + "\n" if header else "") + body
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if yaml is None:
        raise SystemExit("pip install pyyaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="config/rag/study")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, dict]] = [
        (
            "run_steps_0",
            build_config(
                header="# Axis 2 — max_corrective_steps=0 (judge may run, no correction applied)",
                max_corrective_steps=0,
            ),
        ),
        (
            "run_steps_1",
            build_config(
                header="# Axis 2 — max_corrective_steps=1",
                max_corrective_steps=1,
            ),
        ),
        (
            "run_steps_2",
            build_config(
                header="# Axis 2 — max_corrective_steps=2",
                max_corrective_steps=2,
            ),
        ),
        (
            "run_steps_3",
            build_config(
                header="# Axis 2 — max_corrective_steps=3",
                max_corrective_steps=3,
            ),
        ),
        (
            "run_judge_scout",
            build_config(
                header="# Axis 3 — free judge (all actions); used to build trigger subset T",
                max_corrective_steps=1,
            ),
        ),
        (
            "run_force_rr",
            build_config(
                header="# Axis 3 — forced RE_RETRIEVE (force_corrective_action)",
                max_corrective_steps=1,
                allow_re_retrieve=True,
                allow_add_questions=False,
                allow_add_context=False,
                force_corrective_action="RE_RETRIEVE",
            ),
        ),
        (
            "run_force_aq",
            build_config(
                header="# Axis 3 — forced ADD_QUESTIONS (force_corrective_action)",
                max_corrective_steps=1,
                allow_re_retrieve=False,
                allow_add_questions=True,
                allow_add_context=False,
                force_corrective_action="ADD_QUESTIONS",
            ),
        ),
        (
            "run_force_ac",
            build_config(
                header="# Axis 3 — forced ADD_CONTEXT (force_corrective_action; needs websearch)",
                max_corrective_steps=1,
                allow_re_retrieve=False,
                allow_add_questions=False,
                allow_add_context=True,
                force_corrective_action="ADD_CONTEXT",
            ),
        ),
    ]

    for name, cfg in specs:
        path = out_dir / f"{name}.yaml"
        dump_yaml(cfg, path)
        print(f"✓ {path.relative_to(root)}")


if __name__ == "__main__":
    main()
