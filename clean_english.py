from __future__ import annotations

"""
Polish an English translation without changing meaning or summarizing.

Input : translation_en.txt
Output: translation_en_polished.txt
"""

from pathlib import Path
from typing import List

from config import (
    OLLAMA_POLISH_MODEL,
    TRANSLATION_EN_PATH,
    TRANSLATION_EN_POLISHED_PATH,
)
from ollama_client import generate as ollama_generate


def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
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


def build_prompt(chunk: str) -> str:
    return (
        "You are an editor for English texts that are already machine-translated from Arabic Islamic lectures.\n"
        "Instructions (according to the methodology of the Salaf al-Salih and the creed of Ahl al-Sunnah):\n"
        "1. Improve grammar, punctuation, and clarity of the existing English translation.\n"
        "2. Preserve ALL content and meaning; do NOT summarize, omit details, or add new ideas.\n"
        "3. Keep the style simple and readable, but faithful to the original Arabic meaning.\n"
        "4. Structure the text as clear, well-formed written paragraphs (not as raw speech), while keeping the full meaning.\n"
        "5. Do NOT translate into another language and do NOT add commentary.\n"
        "6. Ensure that the wording remains in line with the understanding of the early generations (Salaf al-Salih).\n"
        "7. Do not introduce interpretations, theological changes, or wording that supports innovations (bid'ah).\n"
        "8. Maintain terminology consistent with the Qur'an, authentic Hadith, and the creed of Ahl al-Sunnah as understood by the early scholars.\n"
        "9. Edit only what is already present. Do not add devotional phrases, explanations, or conclusions not found in the source translation.\n"
        "10. If a sentence is unclear, improve its English conservatively instead of rewriting its meaning freely.\n\n"
        "Current English translation to polish (it is already a translation of Arabic; just improve the English):\n"
        f"{chunk}\n\n"
        "Return only the polished English text:\n"
    )


def polish_english_translation(
    input_path: str | Path = TRANSLATION_EN_PATH,
    output_path: str | Path = TRANSLATION_EN_POLISHED_PATH,
) -> str:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input translation not found: {input_file}")

    text = input_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Input English translation is empty.")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text chunks to polish (after preprocessing).")

    outputs: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Polishing English chunk {i}/{len(chunks)} (len={len(chunk)} chars)...", flush=True)
        prompt = build_prompt(chunk)
        cleaned = ollama_generate(prompt, model=OLLAMA_POLISH_MODEL)
        outputs.append(cleaned.strip())

    polished_full = "\n\n".join(outputs)
    Path(output_path).write_text(polished_full, encoding="utf-8")
    return polished_full


def main() -> int:
    try:
        polished = polish_english_translation()
    except Exception as exc:
        print(f"English cleanup failed: {exc}")
        return 1

    print("\n=== ENGLISH CLEANUP COMPLETE ===\n")
    print(f"Input : {TRANSLATION_EN_PATH}")
    print(f"Output: {TRANSLATION_EN_POLISHED_PATH}\n")
    print(polished)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

