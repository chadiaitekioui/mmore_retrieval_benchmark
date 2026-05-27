# MMORE Retrieval Benchmark

Benchmark repository to evaluate MMORE retrieval quality on **MedXpertQA (200 questions)**, including the **LLM-as-a-judge** feature from [mmore PR #312](https://github.com/swiss-ai/mmore/pull/312).

## What this repo does

1. Install — clone this repo, then `install.sh` clones MMORE + venv.
2. MedXpertQA data — clinical MedRAG corpus (`proc_demo.db`) + 200 questions JSONL.
3. Collect — 7 retrieval configs → `results/*/chunks.json`.
4. Annotate — ground truth labels (`data/ground_truth.json`).
5. Metrics / Compare — Hit@k, MRR, NDCG, McNemar → `results/summary.json`.

Every step after install assumes you are in the benchmark tree with the environment loaded.

**Before collect (step 3):** set up [Hugging Face access](#prerequisites-hugging-face-gated-llama) for `meta-llama/Llama-3.1-8B-Instruct` (runs `run_C`, `run_C_ctrl`).

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

`install.sh` clones `../mmore` (`swiss-ai/mmore`, branch `llm-as-a-judge`) and creates `../hf_cache`, `.venv`, and `env.benchmark`.

What your workspace folder should look like:

```text
/workspace/
  mmore/                      ← cloned by install.sh
  mmore_retrieval_benchmark/  ← you cloned this
  hf_cache/
```

## Prerequisites: Hugging Face (gated Llama)

Runs `run_C` and `run_C_ctrl` load `meta-llama/Llama-3.1-8B-Instruct` locally (RAG LLM + judge). That model is gated on Hugging Face: without account access and a token, MMORE fails with `401` / `GatedRepoError` when starting the RAG server.

The same token is needed if you annotate ground truth with the local HF model (step 4) instead of OpenAI.

### One-time setup

1. Create a [Hugging Face](https://huggingface.co/join) account.
2. Open [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), accept Meta’s license, and wait until access is granted (usually minutes; sometimes longer).
3. Create a token: [Settings → Access Tokens](https://huggingface.co/settings/tokens).

### Every session (cluster or laptop)

Export the token

```bash
export HF_TOKEN=hf_xxxxxxxx
huggingface-cli login
```

`HF_HOME` from `env.benchmark` points model weights to `$WORKDIR/hf_cache`; the token is separate and must still be set.

### Verify access

```bash
source env.benchmark
export HF_TOKEN=hf_xxxxxxxx
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('meta-llama/Llama-3.1-8B-Instruct', 'config.json'); print('OK')"
```

If this prints `OK`, you can run `jobs/collect_all.sh` through the `run_C` / `run_C_ctrl` steps.

---

## MedXpertQA data

Builds `proc_demo.db` (StatPearls + USMLE textbooks via [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG)) and `data/medxpertqa_200*.jsonl`:

```bash
bash jobs/setup_medxpertqa.sh              # MedRAG clinical corpus (default)
# MEDRAG_PILOT=1 bash jobs/setup_medxpertqa.sh --pilot   # 5k snippets smoke test, 30 questions
# CORPUS=plos bash jobs/setup_medxpertqa.sh 5000 # legacy PLoS-5k
```

See [corpus/README.md](corpus/README.md) for sources and NCBI StatPearls rebuild instructions.

`CLEAN_DB=1` (default) recreates `proc_demo.db` on each corpus build. **Re-run `jobs/ground_truth.sh` after re-indexing** (labels are corpus-specific).

---

## Collect retrieval results (7 runs)

**Requires:** [Hugging Face token](#prerequisites-hugging-face-gated-llama) (for `run_C` / `run_C_ctrl`).

```bash
bash jobs/collect_all.sh
```

`collect_all.sh`: for each run A→F: start MMORE, 200 HTTP queries, `results/<run>/chunks.json`, stop MMORE. Ports `8001` (retrieve) / `8000` (RAG).

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