from __future__ import annotations

"""
Clean up a raw Arabic ASR transcript without translating or summarizing.

Input : transcript_raw_ar_timestamps.txt (with [MM:SS-MM:SS] timestamps)
Output: transcript_clean_ar.txt (Arabic only, no timestamps)
"""

import re
from pathlib import Path
from typing import List

from config import (
    OLLAMA_MODEL,
    OLLAMA_CLEAN_AR_MAX_TOKENS,
    OLLAMA_CLEAN_AR_TEMPERATURE,
    TRANSCRIPT_RAW_AR_PATH,
    TRANSCRIPT_RAW_AR_TIMED_PATH,
    TRANSCRIPT_CLEAN_AR_PATH,
)
from ollama_client import generate as ollama_generate


TIMESTAMP_RE = re.compile(r"^\[(\d{2}:\d{2}-\d{2}:\d{2})\]\s*(.*)$")


def chunk_lines(lines: List[str], max_lines: int = 5) -> List[List[str]]:
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        return []
    return [lines[i:i + max_lines] for i in range(0, len(lines), max_lines)]


def build_prompt(lines: List[str]) -> str:
    chunk = "\n".join(lines)
    return (
        "أنت مساعد لغوي يقوم بتنظيف نصوص من تفريغ صوتي عربي (ASR) لمحاضرات أو خطب شرعية.\n"
        "التعليمات (وفق منهج السلف الصالح وعقيدة أهل السنة والجماعة):\n"
        "1. أصلح فقط الأخطاء الواضحة جداً في التعرف الصوتي للكلمات العربية.\n"
        "2. إن لم تكن متأكداً من التصحيح، فابقِ العبارة قريبة جداً من الأصل ولا تخمّن.\n"
        "3. لا تُلخّص، ولا تحذف، ولا تعيد الصياغة بحرية، ولا تُترجم، ولا تُغيّر المعنى.\n"
        "4. لا تُدخل أي جملة جديدة غير موجودة في الأصل، ولا تُكمل الكلام من عندك.\n"
        "5. أبقِ ترتيب السطور كما هو تماماً، مع نفس عدد السطور.\n"
        "6. أبقِ كل طابع زمني [MM:SS-MM:SS] كما هو دون تغيير.\n"
        "7. بعد كل طابع زمني، أخرج النص العربي المنقّح فقط.\n"
        "8. احرص على أن يبقى المعنى منضبطاً مع فهم السلف الصالح، بلا تأويلات أو ألفاظ تدعم البدع.\n"
        "9. هذا نص تفريغ صوتي؛ المطلوب تنقيح محافظ جداً، لا كتابة نص جديد.\n\n"
        "النص:\n"
        f"{chunk}\n\n"
        "أخرج نفس السطور نفسها بعد التنقيح المحافظ، سطراً مقابل سطر:\n"
    )


def strip_timestamps(text: str) -> str:
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = TIMESTAMP_RE.match(line)
        if match:
            content = match.group(2).strip()
            if content:
                cleaned_lines.append(content)
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def clean_arabic_transcript(
    input_path: str | Path = TRANSCRIPT_RAW_AR_TIMED_PATH,
    output_path: str | Path = TRANSCRIPT_CLEAN_AR_PATH,
) -> str:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input transcript not found: {input_file}")

    text = input_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Input Arabic transcript is empty.")

    lines = [line for line in text.splitlines() if line.strip()]
    chunks = chunk_lines(lines)
    if not chunks:
        raise ValueError("No text chunks to clean (after preprocessing).")

    outputs: List[str] = []
    for i, chunk_lines_block in enumerate(chunks, start=1):
        print(
            f"Cleaning Arabic chunk {i}/{len(chunks)} (lines={len(chunk_lines_block)})...",
            flush=True,
        )
        prompt = build_prompt(chunk_lines_block)
        cleaned = ollama_generate(
            prompt,
            model=OLLAMA_MODEL,
            max_tokens=OLLAMA_CLEAN_AR_MAX_TOKENS,
            temperature=OLLAMA_CLEAN_AR_TEMPERATURE,
        )
        outputs.append(cleaned.strip())

    cleaned_with_timestamps = "\n".join(outputs)
    cleaned_full = strip_timestamps(cleaned_with_timestamps)
    Path(output_path).write_text(cleaned_full, encoding="utf-8")
    return cleaned_full


def main() -> int:
    try:
        cleaned = clean_arabic_transcript()
    except Exception as exc:
        print(f"Arabic cleanup failed: {exc}")
        return 1

    print("\n=== ARABIC CLEANUP COMPLETE ===\n")
    print(f"Input : {TRANSCRIPT_RAW_AR_TIMED_PATH}")
    print(f"Output: {TRANSCRIPT_CLEAN_AR_PATH}\n")
    print(cleaned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

