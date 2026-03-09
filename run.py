"""
Download a video from a URL and transcribe it to a raw Arabic transcript with timestamps.

Usage:
    python run.py <url>
    python run.py        (prompts for URL)

Outputs:
    - video.mp3
    - transcript_raw_ar.txt              (raw Arabic ASR)
    - transcript_raw_ar_timestamps.txt   (raw Arabic ASR with timestamps)
"""
import sys
from pathlib import Path

from download import download_mp3
from transcribe import transcribe, safe_print
from config import (
    VIDEO_PATH,
    TRANSCRIPT_RAW_AR_PATH,
    # TRANSCRIPT_CLEAN_AR_PATH,
    # TRANSLATION_EN_PATH,
    # TRANSLATION_EN_POLISHED_PATH,
)


def main() -> None:
    if len(sys.argv) > 1:
        url = sys.argv[1].strip()
    else:
        url = input("Enter video URL: ").strip()
    if not url:
        print("No URL provided.", file=sys.stderr)
        sys.exit(1)

    print("Downloading...", flush=True)
    video_path = download_mp3(url, output_path=str(VIDEO_PATH))
    print(f"Saved to {video_path}", flush=True)

    print("Transcribing (progress below; each line is one segment)...", flush=True)
    text = transcribe(str(VIDEO_PATH), verbose=True)
    Path(TRANSCRIPT_RAW_AR_PATH).write_text(text, encoding="utf-8")
    print(f"Transcript saved to {TRANSCRIPT_RAW_AR_PATH}", flush=True)
    safe_print(text)
    # The following steps (Arabic cleanup, translation, English polishing) are
    # intentionally disabled for now to focus on producing only the raw transcript.
    # print("\nCleaning Arabic transcript with Ollama...", flush=True)
    # cleaned_ar = clean_arabic_transcript()
    # print(f"Clean Arabic transcript saved to {TRANSCRIPT_CLEAN_AR_PATH}", flush=True)
    #
    # print("\nTranslating cleaned Arabic to English with Ollama (LLM)...", flush=True)
    # translated = translate_text_file(
    #     input_path=str(TRANSCRIPT_CLEAN_AR_PATH),
    #     output_path=str(TRANSLATION_EN_PATH),
    # )
    # print(f"Translation saved to {TRANSLATION_EN_PATH}", flush=True)
    #
    # print("\nPolishing English translation with Ollama...", flush=True)
    # polished = polish_english_translation(
    #     input_path=str(TRANSLATION_EN_PATH),
    #     output_path=str(TRANSLATION_EN_POLISHED_PATH),
    # )
    # print(f"Polished translation saved to {TRANSLATION_EN_POLISHED_PATH}", flush=True)
    #
    # # Preview final English
    # print("\n=== FINAL ENGLISH PREVIEW (first 1000 chars) ===\n")
    # print(polished[:1000])


if __name__ == "__main__":
    main()
