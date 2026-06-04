# vLLM + Hermes Agent

> Run a local vLLM inference server on DGX Spark and connect Hermes Agent to it

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Switching Models](#switching-models)
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

**Duration**: 5-10 minutes for setup; 15-25 minutes on first start (model download + CUDA graph
compilation); ~4-5 minutes on subsequent starts (cached)

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

**Description**: Launch vLLM in Docker serving `RedHatAI/Qwen3.6-35B-A3B-NVFP4` — a Qwen3.6
35B MoE model in NVFP4 (4-bit) quantization, natively accelerated on the DGX Spark's Blackwell
GB10 GPU. This is the validated production configuration.

```bash
docker run -d --gpus all \
  --name vllm-server \
  --restart unless-stopped \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm/vllm-openai:cu130-nightly \
  RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --quantization compressed-tensors \
  --moe-backend marlin \
  --port 8000 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3
```

> **Key flags explained:**
> - `--restart unless-stopped` — container auto-restarts after reboot
> - `CUTE_DSL_ARCH=sm_121a` — targets DGX Spark's Blackwell SM121 GPU
> - `VLLM_TEST_FORCE_FP8_MARLIN=1` — forces Marlin GEMM (FlashInfer CUTLASS broken on SM121)
> - `VLLM_USE_FLASHINFER_MOE_FP4=0` — disables broken FP4 MoE path on SM121
> - `--quantization compressed-tensors` — required for RedHatAI NVFP4 checkpoint format
> - `--moe-backend marlin` — correct backend for W4A4 NVFP4 MoE on Blackwell
> - `--reasoning-parser qwen3` — separates `<think>` reasoning from the actual response
> - `--enable-auto-tool-choice --tool-call-parser hermes` — required for Hermes Agent tool calls
> - `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` — allows 65K context (model config reports 40K max)

## Step 3. Wait for the server to be ready

**Description**: Monitor logs until startup is complete.

```bash
docker logs vllm-server --follow
```

Typical timeline:
- Model download (~24 GB): 10-20 min (skipped if cached)
- Weight loading: ~10 min
- CUDA graph compilation: ~3 min
- **Total first start**: ~25 min
- **Subsequent starts**: ~4-5 min

Look for: `Application startup complete`

## Step 4. Verify the server

```bash
curl -s http://localhost:8000/health && echo "OK"
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

## Step 5. Configure Hermes

**Description**: Update `~/.hermes/config.yaml`. Edit the `model` block at the top:

```yaml
model:
  default: RedHatAI/Qwen3.6-35B-A3B-NVFP4
  provider: custom
  base_url: http://localhost:8000/v1
  context_length: 65536
```

And the `custom_providers` block near the bottom:

```yaml
custom_providers:
- name: Local vLLM Qwen3.6-35B-NVFP4
  base_url: http://localhost:8000/v1
  model: RedHatAI/Qwen3.6-35B-A3B-NVFP4
```

> **Important**: `context_length: 65536` is required — Hermes reads context from the model's
> HuggingFace config (which reports 40,960), not from vLLM's `--max-model-len`. Without this
> override Hermes refuses to start with a "context window below minimum 64,000" error.

## Step 6. Restart Hermes

```bash
hermes gateway restart
```

## Step 7. Verify end-to-end

```bash
hermes chat -q "What model are you? /no_think"
```

Expected: Hermes responds identifying itself as `RedHatAI/Qwen3.6-35B-A3B-NVFP4`.

---

## Switching Models

Stop the container, update the model in the docker command and `config.yaml`, then restart:

```bash
docker rm -f vllm-server
# Edit model name in docker run and ~/.hermes/config.yaml
hermes gateway restart
```

### Model compatibility on DGX Spark (SM121)

| Model | Size | Quantization | vLLM flags |
|-------|------|-------------|------------|
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` *(recommended)* | ~24 GB | compressed-tensors NVFP4 | `--quantization compressed-tensors --moe-backend marlin` |
| `Qwen/Qwen3-8B` *(lightweight fallback)* | ~16 GB | BF16 | `--enable-auto-tool-choice --tool-call-parser hermes` |

For any NVFP4 model on SM121, always include:

```bash
-e CUTE_DSL_ARCH=sm_121a \
-e VLLM_USE_FLASHINFER_MOE_FP4=0 \
-e FLASHINFER_DISABLE_VERSION_CHECK=1 \
-e VLLM_TEST_FORCE_FP8_MARLIN=1 \
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| HTTP 400: `"auto" tool choice requires --enable-auto-tool-choice` | Missing tool call flags | Add `--enable-auto-tool-choice --tool-call-parser hermes` |
| `context window below minimum 64,000` | Hermes reads model config, not vLLM's max | Add `context_length: 65536` under `model:` in `~/.hermes/config.yaml` |
| `max_model_len greater than derived max_model_len` | Model config reports 40K max | Add `-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| `moe_backend='marlin' is not supported for unquantized MoE` | Wrong model — use W4A16 format, not W4A4 | Switch to `RedHatAI/Qwen3.6-35B-A3B-NVFP4` (compressed-tensors W4A4) |
| `moe_backend='triton' is not supported for NvFP4 MoE` | triton doesn't support NVFP4 MoE | Use `--moe-backend marlin` for NVFP4 models |
| `KeyError: 'layers.0.mlp.experts.w2_input_scale'` | nvidia/ checkpoint uses W4A16 (weight-only) | Use `RedHatAI/` checkpoint instead (W4A4) |
| Response contains raw `<think>` tags | Reasoning not parsed | Add `--reasoning-parser qwen3` |
| Container not running after reboot | Missing restart policy | Add `--restart unless-stopped` or run `docker update --restart unless-stopped vllm-server` |
| GPU memory shows `[N/A]` in nvidia-smi | Normal on GB10 unified memory | Not an error — GB10 shares CPU/GPU memory |
