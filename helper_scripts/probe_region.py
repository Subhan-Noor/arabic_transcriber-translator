"""Re-transcribe a selected audio window with word-level timings.

Usage: python helper_scripts/probe_region.py input/example.mp3 5:11 7:11
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--language", default="ar")
    args = parser.parse_args()
    if not args.media.is_file():
        parser.error(f"Media file not found: {args.media}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from transcriber_utils import validate_time_range, format_timestamp
    try:
        start, end = validate_time_range(args.start, args.end)
    except ValueError as error:
        parser.error(str(error))
    import torch
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(download_model(args.model), device=device,
                         compute_type="float16" if device == "cuda" else "int8")
    segments, _ = model.transcribe(str(args.media), language=args.language,
        beam_size=5, temperature=0.0, condition_on_previous_text=False,
        vad_filter=False, clip_timestamps=f"{max(0, start-10)},{end+10}", word_timestamps=True)
    for seg in segments:
        print(f"[{format_timestamp(seg.start)}-{format_timestamp(seg.end)}] {seg.text.strip()}")
        for word in seg.words or []:
            mark = "*" if word.start < end and word.end > start else " "
            print(f"  {mark} {word.start:.2f}-{word.end:.2f} p={word.probability:.2f} {word.word}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
