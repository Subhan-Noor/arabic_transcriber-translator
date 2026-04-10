"""Transcribe an audio/video file using Faster-Whisper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from tqdm import tqdm

from config import FW_DEVICE, FW_MODEL_SIZE
from timecode import format_timestamp

# Output file names written inside the per-run output directory.
_TRANSCRIPT_PLAIN = "transcript_raw_ar.txt"
_TRANSCRIPT_TIMED = "transcript_raw_ar_timestamps.txt"


def _build_model(model_size: Optional[str], device: Optional[str]) -> WhisperModel:
    size = model_size or FW_MODEL_SIZE
    dev = device or FW_DEVICE
    compute_type = "float16" if dev == "cuda" else "int8"
    print(f"  Loading Faster-Whisper '{size}' on {dev} ({compute_type}) …", flush=True)
    return WhisperModel(size, device=dev, compute_type=compute_type)


def transcribe(
    file_path: str | Path,
    output_dir: Path,
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    language: Optional[str] = "ar",
    verbose: bool = True,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> str:
    """
    Transcribe *file_path* and write two files into *output_dir*:

    - ``transcript_raw_ar.txt``            — plain text, space-joined
    - ``transcript_raw_ar_timestamps.txt`` — one ``[MM:SS-MM:SS]`` (or ``H:MM:SS``) line per segment

    *start_sec* / *end_sec* select a slice of the file (in seconds, aligned with the
    original media). If both are omitted, the whole file is transcribed.

    Returns the plain-text transcript.
    """
    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(model_size, device)

    use_clip = start_sec is not None or end_sec is not None
    clip_start = 0.0 if start_sec is None else float(start_sec)

    if not use_clip:
        clip_timestamps: str | list[float] = "0"
        vad_filter = True
    else:
        vad_filter = False
        if end_sec is None:
            clip_timestamps = str(clip_start)
        else:
            clip_timestamps = f"{clip_start},{float(end_sec)}"

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=vad_filter,
        clip_timestamps=clip_timestamps,
    )

    duration: float | None = getattr(info, "duration", None) if info is not None else None
    if duration:
        if use_clip:
            eff_end = float(end_sec) if end_sec is not None else duration
            eff_end = min(eff_end, duration)
            print(
                f"  Transcribing {format_timestamp(clip_start)} → {format_timestamp(eff_end)} "
                f"(file length {format_timestamp(duration)})",
                flush=True,
            )
        else:
            print(f"  Audio length: {format_timestamp(duration)}", flush=True)

    if use_clip:
        eff_end = float(end_sec) if end_sec is not None else (duration or 0.0)
        if duration:
            eff_end = min(eff_end, duration)
        bar_total = int(max(0.0, eff_end - clip_start)) or None
    else:
        bar_total = int(duration) if duration else None

    texts: list[str] = []
    timed_lines: list[str] = []

    with tqdm(
        total=bar_total,
        unit="s",
        unit_scale=True,
        desc="  Transcribing",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]"
        if bar_total
        else "{l_bar}{bar}| {n:.0f}s processed",
    ) as pbar:
        for seg in segments:
            text = seg.text.strip()
            seg_start = float(seg.start or 0)
            seg_end = float(seg.end or 0)

            if text:
                texts.append(text)
                line = (
                    f"[{format_timestamp(seg_start)}-{format_timestamp(seg_end)}] {text}"
                )
                timed_lines.append(line)
                if verbose:
                    tqdm.write(line)

            pbar.update(int(seg_end - seg_start))

    full_text = " ".join(texts)
    timed_text = "\n".join(timed_lines)

    (output_dir / _TRANSCRIPT_PLAIN).write_text(full_text, encoding="utf-8")
    (output_dir / _TRANSCRIPT_TIMED).write_text(timed_text, encoding="utf-8")

    return full_text


def safe_print(text: str) -> None:
    """Print Unicode text safely on Windows (avoids cp1252 encoding errors)."""
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe a single audio/video file.")
    parser.add_argument("file", help="Path to the audio or video file.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where transcripts are saved (default: current directory).",
    )
    parser.add_argument("--model", default=None, help="Faster-Whisper model size.")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--language", default="ar", help="Language code (default: ar).")
    parser.add_argument(
        "--start",
        metavar="TIME",
        default=None,
        help="Start time (e.g. 8:00, 0:35, 1:25:00). Default: beginning.",
    )
    parser.add_argument(
        "--end",
        metavar="TIME",
        default=None,
        help="End time (same formats as --start). Default: end of file.",
    )
    args = parser.parse_args()

    from timecode import parse_timestamp

    lang = None if args.language.lower() == "none" else args.language
    t0 = parse_timestamp(args.start) if args.start else None
    t1 = parse_timestamp(args.end) if args.end else None

    result = transcribe(
        args.file,
        output_dir=Path(args.output_dir),
        model_size=args.model,
        device=args.device,
        language=lang,
        verbose=True,
        start_sec=t0,
        end_sec=t1,
    )
    safe_print(result)
