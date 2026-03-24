"""Central configuration for the YouTube Arabic Transcriber."""

from __future__ import annotations

from pathlib import Path

# Project root (this folder)
ROOT_DIR = Path(__file__).resolve().parent

# All transcription runs are saved here as individual sub-folders
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ---------------------------------------------------------------------------
# Faster-Whisper settings
# ---------------------------------------------------------------------------
# Model size: tiny | base | small | medium | large-v2
# Larger models are slower but produce more accurate Arabic transcripts.
FW_MODEL_SIZE = "large-v2"

# Inference device: "cuda" (NVIDIA GPU) or "cpu"
FW_DEVICE = "cuda"
