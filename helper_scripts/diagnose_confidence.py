"""Dump segment confidence metrics for calibrating quality thresholds.

Usage: python helper_scripts/diagnose_confidence.py input/example.mp3
Writes output/diagnostics/<name>.tsv (ignored by Git).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--language", default="ar")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.media.is_file():
        parser.error(f"Media file not found: {args.media}")
    import torch
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = download_model(args.model)
    model = WhisperModel(model_path, device=device, compute_type="float16" if device == "cuda" else "int8")
    segments, info = model.transcribe(str(args.media), language=args.language, beam_size=5,
        temperature=0.0, condition_on_previous_text=False, vad_filter=True,
        word_timestamps=False, compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0, no_speech_threshold=0.6)
    dest = args.output or Path("output/diagnostics") / f"{args.media.stem}.tsv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = ["start\tend\tduration\tchars_per_second\tavg_logprob\tno_speech_prob\tcompression_ratio\ttext"]
    for seg in segments:
        duration = max(0, seg.end - seg.start)
        text = seg.text.strip().replace("\t", " ").replace("\n", " ")
        cps = len(text) / duration if duration else 0
        rows.append(f"{seg.start:.2f}\t{seg.end:.2f}\t{duration:.2f}\t{cps:.2f}\t{seg.avg_logprob:.3f}\t{seg.no_speech_prob:.3f}\t{seg.compression_ratio:.3f}\t{text}")
    dest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {dest} ({len(rows)-1} segments, {info.duration:.1f}s audio)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
