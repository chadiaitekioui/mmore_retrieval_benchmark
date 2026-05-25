# PLOS corpus → `proc_demo.db`

Reproducible scripts to build the Milvus index used by the retrieval benchmark (PLOS-1k / PLOS-5k style corpora).

Nothing under `data/`, `mmore_input/`, or `work/` is committed — only these scripts.

## MMORE pipeline (from your fork)

| Step | CLI | Input | Output |
|------|-----|-------|--------|
| 1 | `python -m mmore process --config-file …` | Directory of `.txt` files | `…/merged/merged_results.jsonl` |
| 2 | `python -m mmore postprocess --config-file … --input-data merged.jsonl` | JSONL (`MultimodalSample`) | Chunked JSONL |
| 3 | `python -m mmore index --config-file …` | Post-processed JSONL | `proc_demo.db` |

Supported **process** inputs: folders of `.pdf`, `.txt`, `.md`, `.html`, etc. (see MMORE docs). This corpus uses **`.txt`** built from PLOS API text.

Install MMORE from your fork (not PyPI):

```bash
pip install -e /path/to/mmore   # branch llm-as-a-judge
```

## Scripts

```bash
# 1) Download (API, free)
python corpus/download_plos.py --size 1000
python corpus/download_plos.py --size 5000

# 2) Convert → .txt tree for MMORE process
python corpus/convert_plos_to_mmore.py --input corpus/data/plos_1000.json

# 3) Full pipeline
export MMORE_ROOT=/path/to/mmore
bash corpus/build_index.sh 1000
# or
bash jobs/00_build_corpus.sh 1000   # Run:ai-oriented wrapper
```

## Benchmark env vars

After indexing, point the benchmark at the DB:

```bash
export DB_URI=/path/to/proc_demo.db
export DB_NAME=my_db
export COLLECTION_NAME=my_db
```

Then `bash jobs/00_prepare.sh` with the same `COLLECTION_NAME`, and run collect / annotate / metrics as in the root [README.md](../README.md).

## Index config

Templates live in `corpus/config/`. Defaults match the benchmark:

- File: `proc_demo.db` (`DB_URI`)
- Milvus database name: `my_db` (`DB_NAME`)
- Collection: `my_db` (`COLLECTION_NAME`)

Dense model: `BAAI/bge-small-en` + SPLADE (hybrid retrieval in benchmark configs).
