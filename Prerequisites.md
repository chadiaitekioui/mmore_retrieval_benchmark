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