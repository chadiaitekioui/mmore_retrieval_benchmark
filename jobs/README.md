# Job scripts

Independent shell wrappers — see [README.md](../README.md) for order and prerequisites.

| Script | Purpose |
|--------|---------|
| `00_build_corpus.sh` | PLOS download + MMORE process/postprocess/index → `proc_demo.db` (Run:ai / GPU) |
| `00_prepare.sh` | Dataset → `data/*.jsonl` |
| `01_collect.sh RUN URL` | One run → `results/RUN/chunks.json` |
| `02_annotate.sh` | Ground truth (union A+B+C) — OpenAI, compatible API, or HF via `ANNOTATOR_MODEL` |
| `03_metrics.sh` | Metrics for all 7 runs |
| `04_compare.sh` | Table + McNemar → `results/summary.json` |
| `test_judge.sh` | Smoke test on 30 questions (RAG API must be up) |
| `run_benchmark.sh` | Full pipeline (00→04) in one command — see below |

### One-command full benchmark

Each run still needs the matching MMORE config deployed; set **one URL per deployment** (or use sequential mode on a single port):

```bash
export MMORE_URL_run_A=http://127.0.0.1:8011
export MMORE_URL_run_B=http://127.0.0.1:8012
export MMORE_URL_run_C=http://127.0.0.1:8000
export MMORE_URL_run_C_ctrl=http://127.0.0.1:8000
# … run_D, run_E, run_F
bash jobs/run_benchmark.sh
```

RAG runs can use `MMORE_RAG_URL_C` and `MMORE_RAG_URL_C_CTRL` instead of `MMORE_URL_run_C` / `MMORE_URL_run_C_ctrl`.

Single port + redeploy between runs:

```bash
export SEQUENTIAL_COLLECT=1
export MMORE_RETRIEVER_URL=http://127.0.0.1:8001
export MMORE_RAG_URL_C=http://127.0.0.1:8000
export MMORE_RAG_URL_C_CTRL=http://127.0.0.1:8000
bash jobs/run_benchmark.sh
```

Optional: `PARALLEL_COLLECT=1` (distinct URLs), `SKIP_PREPARE=1`, `SKIP_COLLECT=1`, `RUNAI=1` for cluster jobs.
