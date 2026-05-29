## MMORE Retrieval Benchmark

Benchmark repository to evaluate MMORE retrieval quality on **MedXpertQA (200 questions)**, including the **LLM-as-a-judge** feature from [mmore PR #312](https://github.com/swiss-ai/mmore/pull/312).

Every step after install assumes you are in the benchmark tree with the environment loaded.

**Before collect:** set up [Hugging Face access](#prerequisites-hugging-face-gated-llama) for `meta-llama/Llama-3.1-8B-Instruct` see `Prerequisites.md`

---

## Install

### Clone this repo, then install

```bash
cd /workspace
git clone https://github.com/chadiaitekioui/mmore_retrieval_benchmark.git
cd mmore_retrieval_benchmark
bash install.sh # or INSTALL_HF_ANNOTATE=1 bash install.sh if you are using HF model
source env.benchmark
```

If you already had mmore installed, please update the branch version.

What your workspace folder should look like:

```text
/workspace/
  mmore/                      ← cloned by install.sh
  mmore_retrieval_benchmark/  ← you cloned this
  hf_cache/
```

## MedXpertQA data

Builds `proc_demo.db` (StatPearls + USMLE textbooks via [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG)):

```bash
bash jobs/setup_medxpertqa.sh                            # MedRAG clinical corpus (default)
# MEDRAG_PILOT=1 bash jobs/setup_medxpertqa.sh --pilot   # 5k snippets smoke test, 30 questions
# CORPUS=plos bash jobs/setup_medxpertqa.sh 5000         # legacy PLoS-5k
```

`CLEAN_DB=1` (default) recreates `proc_demo.db` on each corpus build.

```bash
export DB_URI=/scratch/users/aitekiou/dbs/medrag_v2.db
export COLLECTION_NAME=my_db_v2   # if you want to keep my_db a seperate folder
```

---

## Collect retrieval results

**Requires:** [Hugging Face token](#prerequisites-hugging-face-gated-llama)

```bash
export OPENAI_API_KEY=sk-...
# Or
export HF_TOKEN=hf_xxxxxxxx
```

```bash
bash jobs/collect_all.sh
```

`collect_all.sh`: for each run A → G: start MMORE, 200 HTTP queries, `results/<run>/chunks.json`

---

## Build ground truth

```bash
export OPENAI_API_KEY=sk-...
```

Or (local HF annotator):

```bash
export HF_TOKEN=hf_xxxxxxxx
export ANNOTATOR_MODEL=meta-llama/Llama-3.1-8B-Instruct
bash jobs/ground_truth.sh
```

Output: `data/ground_truth.json`.

---

## Evaluation

```bash
bash jobs/evaluate.sh
```

Writes `results/<run>/metrics.json` and `results/summary.json`

---

## Run comparison table


| Run          | Reranker | Judge           | hybrid | k                 | MMORE command    |
| ------------ | -------- | --------------- | ------ | ----------------- | ---------------- |
| `run_A`      | no       | no              | 0.5    | 5                 | `mmore retrieve` |
| `run_B`      | yes      | no              | 0.5    | 5                 | `mmore retrieve` |
| `run_C`      | yes      | yes             | 0.5    | 5 (+ Hit@10 eval) | `mmore rag`      |
| `run_C_ctrl` | yes      | thresholds only | 0.5    | 5 (+ Hit@10 eval) | `mmore rag`      |
| `run_D`      | yes      | no              | 0.0    | 5                 | `mmore retrieve` |
| `run_E`      | yes      | no              | 1.0    | 5                 | `mmore retrieve` |
| `run_F`      | no       | no              | 0.5    | 10                | `mmore retrieve` |
| `run_G`      | yes      | no              | 0.5    | 10                | `mmore retrieve` |


Primary judge comparison (legacy): `run_B → run_C` at Hit@5.  
Judge widening: `run_F → run_C` and `run_C_ctrl → run_C` at Hit@10.  
Rerank isolation @10: `run_F → run_G` (same k, +BGE rerank).

---

## Judge study

**Corrective steps impact** For `max_corrective_steps ∈ {0, 1, 2}`, measure final response quality (faithfulness, answer relevance) against total LLM call cost, and track how many queries exhaust the step budget without reaching `PROCEED` — the goal is to find the step count where marginal quality gain no longer justifies the added cost.

**Corrective action comparison** On queries where the judge triggers a corrective action, compare the final response quality across `RE_RETRIEVE`, `ADD_QUESTIONS`, and `ADD_CONTEXT`, segmented by query type (factual, multi-hop, ambiguous) — the goal is to identify which action delivers the most improvement depending on the failure mode.

```bash
bash jobs/collect_judge_study.sh
bash jobs/evaluate_judge_study.sh
bash jobs/judge_study.sh
```

Outputs: `results/judge_study/judge_study.json`, `pareto_quality_vs_cost.png`, `axis3_action_comparison.png`.

Metrics per study run: `results/<run>/metrics.json` (Hit@10), `rag_quality.json` (MCQ accuracy, `judge_llm_calls` from API, non-convergence via `hit_max_corrective_steps`).

---

