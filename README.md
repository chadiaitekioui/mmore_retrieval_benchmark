# MMORE Retrieval Benchmark

Benchmark repository to evaluate MMORE retrieval quality on **MedXpertQA (200 questions)**, including the **LLM-as-a-judge** feature from [mmore PR #312](https://github.com/swiss-ai/mmore/pull/312).

## What this repo does

1. **Install** — clone this repo, then `install.sh` clones MMORE + venv.
2. **MedXpertQA data** — PLOS corpus (`proc_demo.db`) + 200 questions JSONL.
3. **Collect** — 7 retrieval configs → `results/*/chunks.json`.
4. **Annotate** — ground truth labels (`data/ground_truth.json`).
5. **Metrics / compare** — Hit@k, MRR, NDCG, McNemar → `results/summary.json`.

Every step after install assumes you are in the benchmark tree with the environment loaded.

---

## 1. Install

### Clone this repo, then install

```bash
cd /workspace
git clone https://github.com/chadiaitekioui/mmore_retrieval_benchmark.git
cd mmore_retrieval_benchmark
bash install.sh # or INSTALL_HF_ANNOTATE=1 bash install.sh if you are using HF model
source env.benchmark
```

`install.sh` clones `../mmore` (`swiss-ai/mmore`, branch `llm-as-a-judge`), creates `../hf_cache`, `.venv`, and `env.benchmark`.

Layout:

```text
/workspace/
  mmore/                      ← cloned by install.sh
  mmore_retrieval_benchmark/  ← you cloned this
  hf_cache/
```


| Variable                      | After install                                |
| ----------------------------- | -------------------------------------------- |
| `WORKDIR`                     | parent of benchmark repo (e.g. `/workspace`) |
| `BENCH_ROOT`                  | this repo                                    |
| `MMORE_ROOT`                  | `$WORKDIR/mmore`                             |
| `HF_HOME`                     | `$WORKDIR/hf_cache`                          |
| `DB_URI`                      | `$BENCH_ROOT/proc_demo.db`                   |
| `DB_NAME` / `COLLECTION_NAME` | `my_db`                                      |


---

## 2. MedXpertQA data (corpus + queries)

**Requires:** `source env.benchmark`. GPU recommended for corpus indexing.

One command (build `proc_demo.db` + `data/medxpertqa_200*.jsonl`):

```bash
bash jobs/setup_medxpertqa.sh        # PLOS-1k (default)
# bash jobs/setup_medxpertqa.sh 5000   # PLOS-5k
```

`CLEAN_DB=1` (default) recreates `proc_demo.db` on each corpus build. Details: `corpus/README.md`.

---

## 3. Collect retrieval results (7 runs)

**Requires:** MedXpertQA data ready, `source env.benchmark`.

```bash
bash jobs/collect_all.sh
```

`collect_all.sh`: for each run A→F: start MMORE, 200 HTTP queries, `results/<run>/chunks.json`, stop MMORE. Ports `8001` (retrieve) / `8000` (RAG).

---

## 4. Build ground truth

**Requires:** `source env.benchmark`; collect done for `run_A`, `run_B`, `run_C`.

```bash
export OPENAI_API_KEY=sk-...
```

Or

```bash
export HF_TOKEN=hf_...
export ANNOTATOR_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

```bash
bash jobs/ground_truth.sh
```

Output: `data/ground_truth.json`.

---

## 5. Evaluation

**Requires:** `ground_truth.json` + all 7 `results/run_*/chunks.json`.

```bash
bash jobs/evaluate.sh
```

Writes `results/<run>/metrics.json` and `results/summary.json` (McNemar: B→C @ Hit@5, F→C and C_ctrl→C @ Hit@10).

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
| `run_F`      | yes      | no              | 0.5    | 10                | `mmore retrieve` |


Primary judge comparison (legacy): `run_B → run_C` at Hit@5.  
Judge widening: `run_F → run_C` and `run_C_ctrl → run_C` at Hit@10.