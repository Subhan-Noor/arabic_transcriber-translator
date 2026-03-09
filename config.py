"""
Central configuration for the YouTube translator CLI app.

All paths are relative to the project root (this folder).
"""

from __future__ import annotations

from pathlib import Path

# ---------- File paths ----------

ROOT_DIR = Path(__file__).resolve().parent

# Audio file downloaded from YouTube
VIDEO_PATH = ROOT_DIR / "video.mp3"

# Transcript files
TRANSCRIPT_RAW_AR_PATH = ROOT_DIR / "transcript_raw_ar.txt"
TRANSCRIPT_CLEAN_AR_PATH = ROOT_DIR / "transcript_clean_ar.txt"
TRANSCRIPT_RAW_AR_TIMED_PATH = ROOT_DIR / "transcript_raw_ar_timestamps.txt"

# Translation outputs
TRANSLATION_EN_PATH = ROOT_DIR / "translation_en.txt"
TRANSLATION_EN_POLISHED_PATH = ROOT_DIR / "translation_en_polished.txt"


# ---------- Faster-Whisper settings ----------

FW_MODEL_SIZE = "large-v2"  # e.g. tiny, base, small, medium, large-v2
FW_DEVICE = "cuda"        # "cuda" or "cpu"


# ---------- Ollama / LLM settings ----------

OLLAMA_BASE_URL = "http://localhost:11434"

# Default model for cleanup/editing steps (Arabic + English polish)
OLLAMA_MODEL = "command-r7b-arabic:7b"
OLLAMA_POLISH_MODEL = "qwen3:8b"

# Dedicated model for Arabic → English translation
OLLAMA_TRANSLATION_MODEL = "command-r7b-arabic:7b"
OLLAMA_TRANSLATION_MAX_TOKENS = 800
OLLAMA_TRANSLATION_TEMPERATURE = 0.1

OLLAMA_MAX_TOKENS = 900
OLLAMA_TEMPERATURE = 0.1

# Arabic cleanup (increase to avoid truncation on long chunks)
OLLAMA_CLEAN_AR_MAX_TOKENS = 900
OLLAMA_CLEAN_AR_TEMPERATURE = 0.1

