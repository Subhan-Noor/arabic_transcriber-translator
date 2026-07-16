# WSL2 + vLLM (Aya Expanse 8B) Setup

The notebook runs on Windows, but **vLLM does not run natively on Windows**, so
the Aya translation server runs inside **WSL2 (Ubuntu)** and the notebook calls
it over HTTP at `http://localhost:8000/v1`.

## 0. WSL2 status (resolved on this machine)

WSL2 is working: `Ubuntu 24.04.3 LTS` starts on version 2 and the RTX 3070 is
visible via `nvidia-smi` inside Ubuntu.

If you ever see this error again ("WSL2 is not supported with your current
machine configuration... enable the 'Virtual Machine Platform' optional
component and ensure virtualization is enabled in the BIOS"), fix it with:

1. Elevated PowerShell: `wsl.exe --install --no-distribution`
2. Reboot Windows.
3. If it still fails, enable CPU virtualization (Intel VT-x / AMD SVM) in the
  BIOS/UEFI, then reboot again.
4. Verify: `wsl -d Ubuntu -e bash -lc "nvidia-smi | head -12"`

> **Already provisioned on this machine:** the venv exists at
> `~/local-translation-vllm/.venv` with `uv 0.11.29`, `vllm 0.25.1`,
> `bitsandbytes 0.49.2`, and `torch 2.11.0+cu130` (CUDA available). The only
> remaining step is the WSL-side `hf auth login` (section 3). Sections 1–2 are
> kept for reference / rebuilds.



## 1. Create the dedicated vLLM environment (inside WSL/Ubuntu)

A fresh environment is preferred because vLLM pins performance-sensitive
PyTorch/CUDA components.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

mkdir -p ~/local-translation-vllm
cd ~/local-translation-vllm

uv venv --python 3.12 .venv
source .venv/bin/activate
```



## 2. Install vLLM and bitsandbytes

The pinned declarations live in `requirements-vllm.txt` (in the repo root):

```bash
uv pip install vllm
uv pip install "bitsandbytes>=0.49.2"
```



## 3. Authenticate with Hugging Face (inside WSL)

Windows and WSL have separate home directories and require **separate** logins.

```bash
hf auth login
```

You must also have accepted the Aya Expanse 8B license on its Hugging Face model
page with the same account.

## 4. Confirm the server package

```bash
python -c "import vllm; print(vllm.__version__)"
```



## 5. Start the server (conservative 4-bit settings)

> **8 GB VRAM note.** `--gpu-memory-utilization` is a fraction of **total** GPU
> memory (8 GiB), but vLLM checks the resulting target against **free** memory
> at startup, and free memory fluctuates with whatever else is running on the
> Windows side (observed 6.9–7.5 GiB free across sessions). Keep utilization at
> or below `free_GiB / 8` (check first: `wsl -d Ubuntu -e bash -lc
> "nvidia-smi --query-gpu=memory.free --format=csv"`). Even once the utilization
> fraction *fits*, the fixed cost of loading Aya 8B's weights (even 4-bit) can
> consume nearly the whole reserved budget, leaving little/no room for the KV
> cache — see the "No available memory for the cache blocks" entry below. If
> that happens, raise utilization further (if free memory allows) and add
> `--enforce-eager` to skip the CUDA graph memory pool. First free the ceiling
> generally by closing GPU-hungry Windows apps (hardware-accelerated browsers,
> Discord, other CUDA/Python processes).

> **WSL2 UVA note.** vLLM's V1 engine uses pinned CPU memory (UVA) for some GPU
> buffers, but pinned memory is disabled by default on WSL2, which makes vLLM
> fail with `RuntimeError: UVA is not available`. Set
> `VLLM_WSL2_ENABLE_PIN_MEMORY=1` before launching to fix it
> ([vllm-project/vllm#43381](https://github.com/vllm-project/vllm/issues/43381)).

> **FlashInfer sampler note.** vLLM's default sampling kernel (FlashInfer) JIT
> compiles on first use and needs the `nvcc` compiler. This WSL venv only has
> the CUDA *runtime* (pulled in by `torch`), not the full toolkit, so JIT
> compilation fails with `RuntimeError: Could not find nvcc and default
> cuda_home='/usr/local/cuda' doesn't exist`. Set `VLLM_USE_FLASHINFER_SAMPLER=0`
> to fall back to vLLM's native PyTorch sampler (sampler-only change; no effect
> on the deterministic `temp=0` generation policy used here).

```bash
source ~/local-translation-vllm/.venv/bin/activate
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve CohereLabs/aya-expanse-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --quantization bitsandbytes \
  --max-model-len 3072 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enforce-eager
```

> Adjust `0.90` down toward `0.85` if `nvidia-smi` shows less than ~7.3 GiB free
> right before launching (keep it at or below `free_GiB / 8`).



### If it runs out of memory

Lower `--gpu-memory-utilization` further (it must stay below `free_GiB / 8`),
shrink the context, and offload weights to CPU RAM.

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve CohereLabs/aya-expanse-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --quantization bitsandbytes \
  --max-model-len 2560 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.83 \
  --cpu-offload-gb 4 \
  --enable-prefix-caching \
  --enforce-eager
```



### Startup error: "Free memory ... is less than desired GPU memory utilization"

```
ValueError: Free memory on device cuda:0 (6.93/8.0 GiB) on startup is less than
desired GPU memory utilization (0.9, 7.2 GiB).
```

Not a bug — the reservation target (`utilization × 8 GiB`) exceeded free memory.
Fix by (1) freeing Windows GPU memory to raise the free ceiling, then
(2) setting `--gpu-memory-utilization` at or below `free_GiB / 8` (e.g. `0.85`
when ~6.9 GiB is free).

### Startup error: "No available memory for the cache blocks"

```
(EngineCore pid=...) ValueError: No available memory for the cache blocks. Try
increasing `gpu_memory_utilization` when initializing the engine...
```

Happens *after* weights load successfully, during KV-cache sizing — the
`--gpu-memory-utilization` reservation was large enough to pass the earlier
free-memory check, but Aya 8B's weights (even 4-bit) plus activation/warmup
buffers consumed the entire reservation, leaving nothing for the cache. Fix, in
order: (1) recheck free VRAM with `nvidia-smi` and raise
`--gpu-memory-utilization` as high as `free_GiB / 8` allows; (2) add
`--enforce-eager` to skip the CUDA graph memory pool; (3) if still failing,
drop to the OOM-fallback command above (`--max-model-len 2560`,
`--cpu-offload-gb 4`), which shrinks the KV-cache requirement and offloads
weight memory to CPU RAM.

### Startup error: "RuntimeError: UVA is not available"

```
(EngineCore pid=...) RuntimeError: UVA is not available
```

WSL2 disables pinned CPU memory by default, and vLLM's V1 engine needs it for
UVA buffers. Fix: `export VLLM_WSL2_ENABLE_PIN_MEMORY=1` before running
`vllm serve` (see [vllm-project/vllm#43381](https://github.com/vllm-project/vllm/issues/43381)).
Persist it for future shells with
`echo 'export VLLM_WSL2_ENABLE_PIN_MEMORY=1' >> ~/.bashrc`.

### Startup error: "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist"

```
(EngineCore pid=...) RuntimeError: Could not find nvcc and default
cuda_home='/usr/local/cuda' doesn't exist
```

Raised from inside FlashInfer during vLLM's warmup pass (`flashinfer/jit/cpp_ext.py`),
not from your health check. FlashInfer is vLLM's default sampling kernel and
JIT-compiles on first use, which requires the `nvcc` compiler. This venv only
has the CUDA *runtime* (via `torch`), not the full toolkit. Fix: disable the
JIT sampler with `export VLLM_USE_FLASHINFER_SAMPLER=0` before `vllm serve`
(sampler-only fallback to vLLM's native PyTorch sampler; no effect on the
`temp=0` deterministic policy). Persist with
`echo 'export VLLM_USE_FLASHINFER_SAMPLER=0' >> ~/.bashrc`.

## 6. Health check from Windows

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
print(client.models.list())
```

See `docs/gpu_operating_rules.md` for the one-workload-at-a-time sequence
(stop this server before final NVENC rendering).