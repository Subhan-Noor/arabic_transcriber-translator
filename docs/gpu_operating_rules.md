# Safe GPU Operating Rules (RTX 3070, 8 GB VRAM)

This machine has a single NVIDIA RTX 3070 with **8 GB of VRAM**. That is enough
for **one** carefully quantized GPU workload at a time — not several at once.

Never let Faster-Whisper, the Aya vLLM server, and NVENC final rendering hold
GPU memory simultaneously.

## One-workload-at-a-time sequence

Run the end-to-end pipeline in this exact order:

```text
1. Run Faster-Whisper transcription.          (Windows / transcribe env)
2. Release Faster-Whisper (free its VRAM).    (Windows)
3. Confirm free VRAM with nvidia-smi.         (Windows and/or WSL)
4. Start the vLLM Aya server.                 (WSL2 / Ubuntu)
5. Translate.                                 (Windows notebook -> HTTP -> WSL)
6. Stop the vLLM server (Ctrl+C).             (WSL2)
7. Render the final video.                    (Windows / NVENC)
```

Do **not** leave Aya's vLLM server running while trying to load Faster-Whisper,
and do not start rendering (NVENC) while vLLM still holds memory.

## Releasing Faster-Whisper before translation

At the end of the transcription cell, explicitly release the model so its VRAM
is returned before the translation stage:

```python
import gc

del model
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Released the Faster-Whisper model before translation.")
```

`torch.cuda.empty_cache()` does not directly manage every CTranslate2
allocation, but deleting the `WhisperModel` object and collecting garbage
releases its model allocation.

## Checking VRAM

### Windows (host)

Verified working on this machine:

```powershell
nvidia-smi
```

Use this before starting vLLM (step 3) and after stopping it (before rendering)
to confirm memory has actually been freed.

### WSL2 (Ubuntu)

```bash
wsl -d Ubuntu -e bash -lc "nvidia-smi | head -12"
```

> **Status on this machine:** verified working — the RTX 3070 (8 GB) is visible
> inside Ubuntu. Use this before serving vLLM to confirm free VRAM.

## Stopping the translation server

Stop the vLLM server with `Ctrl+C` in the WSL terminal running `vllm serve`.
Confirm the released memory with `nvidia-smi` before running the final
rendering cell.

## If you hit CUDA out-of-memory

Work through these in order:

1. Confirm Faster-Whisper is unloaded.
2. Confirm no other model server is running (`nvidia-smi`).
3. Reduce `--max-model-len` (e.g. 3072 → 2560).
4. Reduce `--gpu-memory-utilization` slightly if startup allocation fails.
5. Keep `--max-num-seqs 1`.
6. Add `--cpu-offload-gb 4`.
7. Add `--enforce-eager`.
8. Restart WSL and retry.
