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

```bash
source ~/local-translation-vllm/.venv/bin/activate

vllm serve CohereLabs/aya-expanse-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --quantization bitsandbytes \
  --max-model-len 3072 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

### If it runs out of memory

```bash
vllm serve CohereLabs/aya-expanse-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --quantization bitsandbytes \
  --max-model-len 2560 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.88 \
  --cpu-offload-gb 4 \
  --enable-prefix-caching \
  --enforce-eager
```

## 6. Health check from Windows

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
print(client.models.list())
```

See `docs/gpu_operating_rules.md` for the one-workload-at-a-time sequence
(stop this server before final NVENC rendering).
