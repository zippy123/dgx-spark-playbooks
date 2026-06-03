# vLLM + Hermes Agent

> Run a local vLLM inference server on DGX Spark and connect Hermes Agent to it

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic Idea

This playbook sets up [vLLM](https://github.com/vllm-project/vllm) as a local OpenAI-compatible
inference server on your NVIDIA DGX Spark, then configures [Hermes Agent](https://hermes-agent.nousresearch.com)
to use it as its model backend. vLLM runs inside Docker using NVIDIA's CUDA 13.0 nightly image,
which includes full Blackwell (SM121) GPU support, FlashAttention, and FP8/NVFP4 quantization.

## What you'll accomplish

You will have vLLM serving a model on port 8000 with an OpenAI-compatible API, and Hermes Agent
configured to route all inference through that local endpoint — enabling fully offline, private AI
agent workflows on your DGX Spark.

## What to know before starting

- Basic familiarity with Docker and GPU workloads
- Hermes Agent installed and configured (run `hermes status` to verify)
- A HuggingFace account and token for model downloads

## Prerequisites

- DGX Spark device with Docker and NVIDIA Container Toolkit
- Hermes Agent installed (`~/.hermes/config.yaml` exists)
- HuggingFace account with a read token
- `~/.cache/huggingface/` directory for model caching

## Time & risk

**Duration**: 5-10 minutes for setup; 5-15 minutes for model download (depends on model size and
internet speed); 2-5 minutes for CUDA graph compilation on first start

**Risk level**: Low — only modifies `~/.hermes/config.yaml` (a backup exists at
`~/.hermes/config.yaml.bak.*`), easily reversible

**Rollback**: Stop the Docker container and restore the Hermes config backup

---

## Instructions

## Step 1. Log in to HuggingFace

**Description**: Authenticate with HuggingFace to enable faster, rate-limit-free model downloads.
The token is stored in `~/.cache/huggingface/token`.

```bash
hf auth login
```

Then export the token into your shell session:

```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

## Step 2. Start the vLLM server

**Description**: Launch vLLM in Docker with a model that meets Hermes Agent's 64K minimum context
requirement. `Qwen/Qwen3-8B` (BF16, ~16 GB) is a reliable default. The `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
flag is required because Qwen3-8B's config reports 40K max context but supports 65K via RoPE scaling.

```bash
docker run -d --gpus all \
  --name vllm-server \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:cu130-nightly \
  Qwen/Qwen3-8B \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.7 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

> **Note**: `--enable-auto-tool-choice` and `--tool-call-parser hermes` are required — Hermes Agent
> sends tool calls to the model and will get HTTP 400 errors without these flags.

## Step 3. Wait for the server to be ready

**Description**: The first start downloads model weights and compiles CUDA graphs. Monitor the logs
until you see `Application startup complete`.

```bash
docker logs vllm-server --follow
```

Typical timeline:
- Model download: 2-10 min (cached on subsequent starts)
- Weight loading: ~2 min
- CUDA graph compilation: ~1 min
- **Total first start**: ~5-15 min
- **Subsequent starts**: ~3 min

## Step 4. Verify the server is responding

**Description**: Confirm vLLM is serving the model correctly before configuring Hermes.

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Expected output includes `"id": "Qwen/Qwen3-8B"`.

## Step 5. Configure Hermes to use vLLM

**Description**: Update `~/.hermes/config.yaml` to point to the local vLLM endpoint. Edit the
`model` section at the top of the file:

```yaml
model:
  default: Qwen/Qwen3-8B
  provider: custom
  base_url: http://localhost:8000/v1
  context_length: 65536   # override: model config reports 40960 but vLLM serves 65536
```

Also update `custom_providers` near the bottom of the file:

```yaml
custom_providers:
- name: Local vLLM Qwen3-8B
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen3-8B
```

> **Important**: The `context_length: 65536` override is necessary because Hermes reads the context
> window from the model's HuggingFace config (which reports 40960), not from vLLM's
> `--max-model-len` flag. Without this override, Hermes will refuse to start with a
> "context window below minimum 64,000" error.

## Step 6. Restart Hermes

**Description**: Apply the config changes by restarting the Hermes gateway service.

```bash
hermes gateway restart
```

## Step 7. Verify the integration

**Description**: Run an end-to-end test to confirm Hermes is routing through vLLM correctly.

```bash
hermes chat -q "What model are you? /no_think"
```

Expected: Hermes responds identifying itself as `Qwen/Qwen3-8B`.

---

## Switching models

To swap the model, stop the container, update the Docker command and `config.yaml`, then restart:

```bash
docker rm -f vllm-server
# Update model name in the docker run command and config.yaml
hermes gateway restart
```

### Model compatibility notes

| Model | Context | Notes |
|-------|---------|-------|
| `Qwen/Qwen3-8B` | 65K (override) | Default recommendation; BF16, no quantization required |
| `Qwen/Qwen3-14B` | 65K (override) | Larger, better quality; ~28 GB BF16 |
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | 65K | Hybrid MoE NVFP4; use `--quantization compressed-tensors --moe-backend marlin` |

For NVFP4 models on DGX Spark (SM121), add these env vars to the docker run command:

```bash
-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
-e CUTE_DSL_ARCH=sm_121a \
-e VLLM_USE_FLASHINFER_MOE_FP4=0 \
-e FLASHINFER_DISABLE_VERSION_CHECK=1 \
-e VLLM_TEST_FORCE_FP8_MARLIN=1 \
```

---

## Step 8. (Optional) Persist the container across reboots

**Description**: Add a `--restart unless-stopped` flag so vLLM starts automatically after a reboot.

```bash
docker update --restart unless-stopped vllm-server
```

Or recreate with the restart policy included in the original `docker run` command by adding
`--restart unless-stopped`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| HTTP 400: `"auto" tool choice requires --enable-auto-tool-choice` | vLLM started without tool call flags | Add `--enable-auto-tool-choice --tool-call-parser hermes` to docker run |
| `context window below minimum 64,000` in Hermes | Model config reports context < 64K | Add `context_length: 65536` under `model:` in `~/.hermes/config.yaml` |
| `User-specified max_model_len is greater than derived max_model_len` | vLLM rejects 65K for models reporting 40K config | Add `-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` to docker run |
| `moe_backend='marlin' is not supported for unquantized MoE` | Wrong MoE backend for model's quantization type | Use `--moe-backend triton` for W4A16 models, `--moe-backend marlin` for W4A4 |
| Container exits immediately | vLLM entrypoint syntax error | Check `docker logs vllm-server` for the exact error |
| GPU memory OOM during weight load | Model too large for available memory | Reduce `--gpu-memory-utilization` or use a smaller/quantized model |
| `Application startup complete` never appears | CUDA graph compilation hanging | Wait 15-30 min on first start; check `docker logs` for progress |
