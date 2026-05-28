# Legacy configs (do not use for API deployment)

Files `run_*.yaml` in this folder are old **local RAG mode** examples (`mode: local`, `rag:` wrapper).

For the current benchmark, use only:

| Goal | File |
|------|------|
| Runs A, B, D, E, F (retrieval) | `../retrieve/run_*.yaml` → `mmore retrieve` |
| Runs C, C_ctrl (judge) | `../rag/run_C_api.yaml`, `run_C_ctrl_api.yaml` → `mmore rag` |

Legacy YAML is kept as reference for judge hyperparameters (full prompts in `run_C.yaml`).

Judge study (axes 2 & 3): see [`../study/README.md`](../study/README.md).
