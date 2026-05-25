# MMORE retrieval benchmark

Benchmark on **MedXpertQA** (200 questions, seed 42) to measure retrieval quality and whether the **LLM-as-a-judge** corrective loop helps ([mmore PR #312](https://github.com/swiss-ai/mmore/pull/312)).

This repo does not install or patch the MMORE package. You deploy MMORE separately (e.g. branch `llm-as-a-judge`) and this repo calls it over **HTTP**, then scores `chunks.json` files locally.

---

## What this repo does

1. Builds a fixed question set and MMORE-ready JSONL inputs.
2. Collects retrieved chunks per experimental run (via API).
3. Builds corpus-level ground truth (which chunks support the correct MCQ answer).
4. Computes Hit@k, MRR, NDCG, and judge-related stats.
5. Compares runs (McNemar), with **run_B → run_C** as the main judge test.

**Primary metric:** Hit@k — at least one relevant chunk in the top-k.

**Primary comparison:** `run_B → run_C` — same reranker and hybrid weight; only the judge (+ corrective retrieval) differs.

---

## What we compare


| Question                      | Runs       | What changes                                           |
| ----------------------------- | ---------- | ------------------------------------------------------ |
| Does the reranker help?       | A → B      | B adds BGE reranker (hybrid 0.5)                       |
| **Does the LLM judge help?**  | **B → C**  | C enables judge + corrective loop                      |
| LLM judge vs thresholds only? | C_ctrl → C | C_ctrl sets `skip_llm_judge: true`                     |
| Hybrid weight effect?         | D, E vs B  | dense-only (0.0) vs sparse-only (1.0) — retrieval only |
| Top-k effect?                 | F vs B     | k=10 vs k=5 — retrieval only                           |


Ground truth is built from the **union** of chunks seen in runs A, B, and C, with labels keyed by `chunk_id`.

---

## Runs


| Run        | Reranker | Judge                              | hybrid | k   | Deploy with                      |
| ---------- | -------- | ---------------------------------- | ------ | --- | -------------------------------- |
| run_A      | no       | no                                 | 0.5    | 5   | `mmore retrieve` → port **8001** |
| run_B      | yes      | no                                 | 0.5    | 5   | `mmore retrieve`                 |
| run_C      | yes      | LLM judge                          | 0.5    | 5   | `mmore rag` → port **8000**      |
| run_C_ctrl | yes      | thresholds only (`skip_llm_judge`) | 0.5    | 5   | `mmore rag`                      |
| run_D      | yes      | no                                 | 0.0    | 5   | `mmore retrieve`                 |
| run_E      | yes      | no                                 | 1.0    | 5   | `mmore retrieve`                 |
| run_F      | yes      | no                                 | 0.5    | 10  | `mmore retrieve`                 |


Configs:

- `config/retrieve/run_*.yaml` — `python -m mmore retrieve --config-file …`
- `config/rag/run_C_api.yaml`, `run_C_ctrl_api.yaml` — `python -m mmore rag --config-file …`

**One YAML = one deployment.** You cannot reuse the same API URL for all runs (hybrid, reranker, and judge settings differ).

---

## Dependencies

### Install (this repo)

```bash
clone https://github.com/chadiaitekioui/mmore_retrieval_benchmark.git
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

That is enough for **steps 0, 1, 3, and 4**, and for **step 2 with OpenAI** (default).

Optional — only if step 2 loads a Hugging Face model **locally** (`ANNOTATOR_MODEL=org/model`):

```bash
pip install -r requirements-hf-annotate.txt
```

(`jobs/02_annotate.sh` runs this for you when the model id contains `/`.)

**MMORE is not in `requirements.txt`.** Deploy it separately (GPU, Milvus); this repo only calls it over HTTP during step 1.

---

## Step-by-step

Run from the **repo root** with the venv activated (`source .venv/bin/activate`).

### 0 — Prepare dataset

**Output:** `data/medxpertqa_200_mmore.jsonl` (+ `data/medxpertqa_200.jsonl` for answers)

```bash
bash jobs/00_prepare.sh
# pilot (30 questions): bash jobs/00_prepare.sh --pilot
```

No MMORE, no API key, no GPU.

---

### 1 — Collect results (×7 MMORE deployments)

**Needs (outside this repo):**

- Milvus index (e.g. `proc_demo.db`, collection `my_docs`)
- MMORE from branch `llm-as-a-judge`, deployed with the YAML for each run
- Network from the machine running collect to MMORE (see **Run:ai — two terminals** below)

**Output:** `results/<run>/chunks.json` (+ optional `raw_api.json`)

**All steps in one command:** with `MMORE_URL_run_`* per deployment — see `[jobs/README.md](jobs/README.md)`.

```bash
bash jobs/run_benchmark.sh
```

For **each** run: deploy MMORE with the matching config, wait for the API, then collect.


| Run        | Deploy config                    | Collect                                                    |
| ---------- | -------------------------------- | ---------------------------------------------------------- |
| run_A      | `config/retrieve/run_A.yaml`     | `bash jobs/01_collect.sh run_A http://localhost:8001`      |
| run_B      | `config/retrieve/run_B.yaml`     | `bash jobs/01_collect.sh run_B http://localhost:8001`      |
| run_C      | `config/rag/run_C_api.yaml`      | `bash jobs/01_collect.sh run_C http://localhost:8000`      |
| run_C_ctrl | `config/rag/run_C_ctrl_api.yaml` | `bash jobs/01_collect.sh run_C_ctrl http://localhost:8000` |
| run_D      | `config/retrieve/run_D.yaml`     | `bash jobs/01_collect.sh run_D http://localhost:8001`      |
| run_E      | `config/retrieve/run_E.yaml`     | `bash jobs/01_collect.sh run_E http://localhost:8001`      |
| run_F      | `config/retrieve/run_F.yaml`     | `bash jobs/01_collect.sh run_F http://localhost:8001`      |


#### Run:ai, two terminals (one way to do)

MMORE lives in its **own repo**; this benchmark is a **second clone** on the same PVC. You do not merge them. MMORE serves HTTP, this repo calls it.

1. Start **one GPU job** on Run:ai (both clones on the PVC, e.g. `/scratch/$USER/mmore` and `…/mmore-bench`).
2. Open **two Terminal windows on your Computer** and connect **both to that same job** (Run:ai UI *Connect* / *Shell*, or your cluster’s `runai bash <job>` — exact command depends on your setup).

**Terminal 1** (MMORE, leave running):

```bash
cd /scratch/$USER/mmore && source .venv/bin/activate
python -m mmore retrieve \
  --config-file /scratch/$USER/mmore-bench/config/retrieve/run_B.yaml
```

**Terminal 2** (benchmark, same job → `localhost`):

```bash
cd /scratch/$USER/mmore-bench && source .venv/bin/activate
bash jobs/01_collect.sh run_B http://localhost:8001
```

---

### 2 — Ground truth

**Output:** `data/ground_truth.json`

Labels each chunk: does it support the correct MCQ answer? Union of chunks from runs A, B, C.

Pick **one** annotator backend via `ANNOTATOR_MODEL` (default `gpt-4o-mini`):


| `ANNOTATOR_MODEL`                     | Backend                            | Install                          | Credentials            |
| ------------------------------------- | ---------------------------------- | -------------------------------- | ---------------------- |
| `gpt-4o-mini`, `gpt-4o`, … *(no `/`)* | OpenAI API                         | `requirements.txt` only          | `OPENAI_API_KEY`       |
| any name + `OPENAI_BASE_URL`          | OpenAI-compatible (Azure, vLLM, …) | `requirements.txt` only          | `OPENAI_API_KEY` + URL |
| `org/model` or `local`                | Hugging Face (local GPU)           | + `requirements-hf-annotate.txt` | `HF_TOKEN` if gated    |


**OpenAI (default):**

```bash
export OPENAI_API_KEY=sk-...
export ANNOTATOR_MODEL=gpt-4o-mini   # optional; also gpt-4o, etc.
bash jobs/02_annotate.sh
```

**OpenAI-compatible server:**

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-host/v1
export ANNOTATOR_MODEL=<model name on that server>
bash jobs/02_annotate.sh
```

**Hugging Face local (e.g. Llama-3.1-8B):**

```bash
export HF_TOKEN=...   # meta-llama/* is gated — accept license on HF first
export ANNOTATOR_MODEL=meta-llama/Llama-3.1-8B-Instruct
bash jobs/02_annotate.sh
```

---

### 3 — Metrics

**Output:** `results/<run>/metrics.json` (Hit@k, MRR, NDCG, judge stats)

```bash
bash jobs/03_metrics.sh
```

No MMORE, no annotator API.

---

### 4 — Comparison

**Output:** `results/summary.json` + printed table (McNemar; primary pair B→C)

```bash
bash jobs/04_compare.sh
```

---

## Limitations

- Ground truth is **answer-aware** (chunk must support the correct MCQ option); LLM annotator without human inter-rater agreement.
- OpenAI annotation is **paid** unless you use a compatible self-hosted endpoint or a local HF model.
- Judge↔hit correlation is a **secondary** indicator only.
- This repo does not start MMORE, Milvus, or Run:ai jobs for you — only documents how to call them.

