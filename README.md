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

## Judge calibration

The default judge uses `min_context_relevance: 7.0` (sufficiency **0.7** on the 1–10 score scale). A **99.5% corrective rate** usually means the LLM almost always triggers a corrective action.

### Threshold sweep (corrective rate vs Hit@10)

```bash
# Offline curve from existing run_C (simulated threshold-only corrective rate):
bash jobs/judge_calibration.sh

# Curve (4× run_C collect with sufficiency 0.3 / 0.5 / 0.7 / 0.9):
export HF_TOKEN=hf_...
bash jobs/collect_judge_calibration.sh
bash jobs/evaluate_judge_calibration.sh
bash jobs/judge_calibration.sh
```

Outputs: `results/judge_calibration/calibration.json`, `calibration_curve.png`, `per_question_judge_gt.json`.