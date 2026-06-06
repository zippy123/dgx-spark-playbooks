# vLLM + Hermes Agent

> Run a local vLLM inference server on DGX Spark and connect Hermes Agent to it

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Switching Models](#switching-models)
- [Remote Access from Another Machine](#remote-access-from-another-machine)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic Idea

This playbook sets up [vLLM](https://github.com/vllm-project/vllm) as a local OpenAI-compatible
inference server on your NVIDIA DGX Spark, then configures [Hermes Agent](https://hermes-agent.nousresearch.com)
to use it as its model backend. vLLM runs inside Docker using NVIDIA's CUDA 13.0 nightly image,
which includes full Blackwell (SM121) GPU support, FlashAttention, and FP8 quantization.

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

**Description**: Launch vLLM in Docker serving `Qwen/Qwen3.6-35B-A3B-FP8` — the official Qwen3.6
35B MoE model in FP8 quantization. FP8 is the recommended quantization on DGX Spark because
SM121 (GB10) has native FP8 hardware support but lacks the `cvt.e2m1x2` instruction needed for
native FP4 execution, making NVFP4 ~32% slower in practice.

```bash
docker run -d --gpus all \
  --name vllm-server \
  --restart unless-stopped \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  vllm/vllm-openai:cu130-nightly \
  Qwen/Qwen3.6-35B-A3B-FP8 \
  --port 8000 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

> **Key flags explained:**
> - `--restart unless-stopped` — container auto-restarts after reboot
> - `CUTE_DSL_ARCH=sm_121a` — targets DGX Spark's Blackwell SM121 GPU
> - `VLLM_MARLIN_USE_ATOMIC_ADD=1` — fixes non-deterministic Marlin kernel reductions on SM121
> - `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` — allows 65K context (model config reports 40K max)
> - `--max-num-batched-tokens 32768` — prefill budget; large enough for long prompts without excessive VRAM
> - `--enable-auto-tool-choice --tool-call-parser qwen3_xml` — required for Hermes Agent tool calls with Qwen3.6 FP8

### Why FP8 instead of NVFP4

DGX Spark's GB10 is SM121, which does **not** have native FP4 tensor core instructions
(`cvt.rn.satfinite.e2m1x2.f32`). NVFP4 falls back to Marlin dequant kernels (40.8 tok/s)
while FP8 runs natively (53.8 tok/s) — a 32% performance advantage. Additionally, Qwen3.6's
hybrid GDN (Gated DeltaNet) attention layers caused silent weight-loading bugs with community
NVFP4 checkpoints on both vLLM and SGLang, producing garbage output. FP8 avoids both issues.

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
  default: Qwen/Qwen3.6-35B-A3B-FP8
  provider: custom
  base_url: http://localhost:8000/v1
  context_length: 65536
```

And the `custom_providers` block near the bottom:

```yaml
custom_providers:
- name: Local vLLM Qwen3.6-35B-FP8
  base_url: http://localhost:8000/v1
  context_length: 65536
  model: Qwen/Qwen3.6-35B-A3B-FP8
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

Expected: Hermes responds identifying itself as `Qwen/Qwen3.6-35B-A3B-FP8`.

---

## Switching Models

Stop the container, update the model in the docker command and `config.yaml`, then restart:

```bash
docker rm -f vllm-server
# Edit model name in docker run and ~/.hermes/config.yaml
hermes gateway restart
```

### Model compatibility on DGX Spark (SM121)

| Model | Size | Quantization | Notes |
|-------|------|-------------|-------|
| `Qwen/Qwen3.6-35B-A3B-FP8` *(recommended)* | ~24 GB | FP8 | Native SM121 FP8 support, 53.8 tok/s, no extra vLLM quantization flags needed |
| `Qwen/Qwen3-8B` *(lightweight fallback)* | ~16 GB | BF16 | No quantization flags needed; good for testing |
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` *(not recommended)* | ~22 GB | compressed-tensors NVFP4 | 32% slower than FP8 on SM121; requires `--quantization compressed-tensors --moe-backend marlin` plus NVFP4 env vars; GDN weight-loading bugs in older vLLM/SGLang versions |

For any model on SM121, always include:

```bash
-e CUTE_DSL_ARCH=sm_121a \
-e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
```

---

## Remote Access from Another Machine

vLLM binds to `0.0.0.0:8000`, so any machine on your local network can use it as an
inference backend. This lets you run Hermes on a laptop (e.g. a MacBook) while the model
runs on the Spark — no cloud API needed.

### Step 1. Find the Spark's IP

On the Spark:

```bash
hostname -I | awk '{print $1}'
```

### Step 2. Test connectivity from the remote machine

```bash
curl -s http://<SPARK_IP>:8000/health && echo " OK"
```

If this fails, open the port on the Spark:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8000
```

### Step 3. Configure Hermes on the remote machine

```bash
hermes config set model.provider custom
hermes config set model.base_url http://<SPARK_IP>:8000/v1
hermes config set model.default "Qwen/Qwen3.6-35B-A3B-FP8"
hermes config set model.context_length 65536
```

### Step 4. Restart Hermes and verify

```bash
hermes gateway restart
hermes chat -q "What model are you? /no_think"
```

### Concurrency

vLLM handles multiple clients simultaneously via continuous batching. The current config
(`--max-num-seqs 4`) supports up to 4 concurrent requests. Multiple Hermes instances
(e.g. Spark + MacBook) share the same model without conflicts — vLLM queues overflow
requests automatically.

### Security note

The vLLM endpoint has **no authentication**. Anyone on your local network who can reach
port 8000 can use the model. To restrict access to a specific machine:

```bash
sudo ufw allow from <CLIENT_IP> to any port 8000
sudo ufw deny 8000
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| HTTP 400: `"auto" tool choice requires --enable-auto-tool-choice` | Missing tool call flags | Add `--enable-auto-tool-choice --tool-call-parser qwen3_xml` |
| `context window below minimum 64,000` | Hermes reads model config, not vLLM's max | Add `context_length: 65536` under `model:` in `~/.hermes/config.yaml` |
| `max_model_len greater than derived max_model_len` | Model config reports 40K max | Add `-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| NVFP4 model ~32% slower than expected | SM121 lacks native FP4 instructions | Switch to FP8: `Qwen/Qwen3.6-35B-A3B-FP8` |
| NVFP4 model produces `!!!!!!!!` garbage | GDN linear_attn weights silently dropped | Known bug in older vLLM/SGLang with community NVFP4 checkpoints; switch to FP8 |
| Response contains raw `<think>` tags | Reasoning not parsed | Add `--reasoning-parser qwen3` to vLLM command if needed |
| Container not running after reboot | Missing restart policy | Add `--restart unless-stopped` or run `docker update --restart unless-stopped vllm-server` |
| GPU memory shows `[N/A]` in nvidia-smi | Normal on GB10 unified memory | Not an error — GB10 shares CPU/GPU memory |
