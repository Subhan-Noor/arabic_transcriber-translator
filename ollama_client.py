from __future__ import annotations

"""
Minimal Ollama HTTP client used for cleanup steps.
"""

from dataclasses import dataclass
from typing import Optional

import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_MAX_TOKENS, OLLAMA_TEMPERATURE


@dataclass
class OllamaConfig:
    base_url: str = OLLAMA_BASE_URL
    model: str = OLLAMA_MODEL
    max_tokens: int = OLLAMA_MAX_TOKENS
    temperature: float = OLLAMA_TEMPERATURE


def generate(
    prompt: str,
    model: Optional[str] = None,
    cfg: Optional[OllamaConfig] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Call Ollama's /api/generate endpoint and return the full response text.
    """
    cfg = cfg or OllamaConfig()
    model_name = model or cfg.model

    url = f"{cfg.base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "options": {
            "temperature": cfg.temperature if temperature is None else temperature,
            "num_predict": cfg.max_tokens if max_tokens is None else max_tokens,
        },
        "stream": True,
    }

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=300)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to contact Ollama at {url}: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

    chunks: list[str] = []
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = line.decode("utf-8")
        except UnicodeDecodeError:
            continue
        # Each line is a small JSON object; we only care about the 'response' field
        try:
            import json

            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        part = obj.get("response")
        if part:
            chunks.append(part)
    return "".join(chunks)

