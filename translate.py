from __future__ import annotations

"""
Translate a cleaned Arabic transcript file to English using a local LLM via Ollama.

Default input:  transcript_clean_ar.txt
Default output: translation_en.txt

Usage:
    python translate.py
    python translate.py input.txt output.txt
"""

import sys
from pathlib import Path
from typing import List

from config import (
    OLLAMA_TRANSLATION_MAX_TOKENS,
    OLLAMA_TRANSLATION_MODEL,
    OLLAMA_TRANSLATION_TEMPERATURE,
    TRANSCRIPT_CLEAN_AR_PATH,
    TRANSCRIPT_RAW_AR_TIMED_PATH,
    TRANSLATION_EN_PATH,
)
from ollama_client import generate as ollama_generate


def chunk_text(text: str, max_chars: int = 350) -> List[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                start = 0
                while start < len(para):
                    end = start + max_chars
                    chunks.append(para[start:end])
                    start = end
                current = ""

    if current:
        chunks.append(current)

    return chunks


def build_prompt(chunk: str, timestamp_context: str | None = None) -> str:
    timing_section = ""
    if timestamp_context:
        timing_section = (
            "Additional transcript timing context (for structure and resolving unclear phrases; "
            "do not reproduce timestamps in the output):\n"
            f"{timestamp_context}\n\n"
        )

    return (
        "You are an expert translator working from Arabic to English on Islamic content.\n"
        "Context:\n"
        "- The source text comes from Arabic audio of scholars, lessons, reminders, and fatawa.\n"
        "- The audio was first transcribed by ASR, then cleaned in Arabic; there may still be minor ASR artifacts.\n"
        "- This translation is one step in a multi-stage pipeline: raw ASR -> Arabic cleanup -> THIS TRANSLATION -> English polishing.\n"
        "- Your job is to produce a faithful English translation that a reader can understand clearly.\n\n"
        "Methodology and constraints:\n"
        "1. Translate accurately while preserving the meaning according to the understanding of the Salaf al-Salih.\n"
        "2. Do not introduce interpretations, theological changes, or wording that supports innovations (bid'ah).\n"
        "3. Maintain terminology consistent with the Qur'an, authentic Hadith, and the creed of Ahl al-Sunnah as understood by the early scholars.\n"
        "4. Preserve all content; do not summarize, omit details, or add new ideas.\n"
        "5. If the Arabic seems slightly garbled due to transcription, infer only the most likely intended meaning from the immediate context; do not invent missing content.\n"
        "6. If a phrase is unclear, translate it conservatively rather than creatively.\n"
        "7. Use clear, modern, readable English, but do not water down key technical or religious terms.\n"
        "8. Because this originates from speech, structure the English as clear written paragraphs and sentences while preserving the full meaning.\n"
        "9. Output must be English only, with no commentary, notes, or Arabic.\n"
        "10. Do not add explanatory sentences, sermons, moral reflections, or closing remarks that are not explicitly in the Arabic.\n\n"
        f"{timing_section}"
        "Arabic text to translate:\n"
        f"{chunk}\n\n"
        "Return only the English translation:\n"
    )


def translate_text_file(
    input_path: str | Path = TRANSCRIPT_CLEAN_AR_PATH,
    output_path: str | Path = TRANSLATION_EN_PATH,
    model_name: str = OLLAMA_TRANSLATION_MODEL,
) -> str:
    """
    Translate a cleaned Arabic transcript file to English using an Ollama-hosted LLM.

    - Uses simple paragraph-aware chunking to stay under context limits.
    - Streams progress: "Translating chunk i/N..."
    - Writes the final result to output_path and returns it.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input transcript not found: {input_file}")

    text = input_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Input transcript is empty.")

    timestamp_context = None
    timed_file = Path(TRANSCRIPT_RAW_AR_TIMED_PATH)
    if timed_file.exists():
        timestamp_context = timed_file.read_text(encoding="utf-8").strip()

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text chunks to translate (after preprocessing).")

    outputs: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Translating chunk {i}/{len(chunks)} (len={len(chunk)} chars)...", flush=True)
        prompt = build_prompt(chunk, timestamp_context=timestamp_context)
        translated = ollama_generate(
            prompt,
            model=model_name,
            max_tokens=OLLAMA_TRANSLATION_MAX_TOKENS,
            temperature=OLLAMA_TRANSLATION_TEMPERATURE,
        )
        outputs.append(translated.strip())

    translated_full = "\n\n".join(outputs)
    Path(output_path).write_text(translated_full, encoding="utf-8")
    return translated_full


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])

    if len(argv) >= 1:
        input_path = argv[0]
    else:
        input_path = str(TRANSCRIPT_CLEAN_AR_PATH)

    if len(argv) >= 2:
        output_path = argv[1]
    else:
        output_path = str(TRANSLATION_EN_PATH)

    try:
        translated = translate_text_file(input_path=input_path, output_path=output_path)
    except Exception as exc:
        print(f"Translation failed: {exc}", file=sys.stderr)
        return 1

    print("\n=== TRANSLATION COMPLETE ===\n")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}\n")
    print(translated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

