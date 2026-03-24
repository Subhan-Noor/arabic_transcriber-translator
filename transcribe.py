"""Transcribe an audio/video file using Faster-Whisper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from config import FW_DEVICE, FW_MODEL_SIZE

# Output file names written inside the per-run output directory.
_TRANSCRIPT_PLAIN = "transcript_raw_ar.txt"
_TRANSCRIPT_TIMED = "transcript_raw_ar_timestamps.txt"


def _build_model(model_size: Optional[str], device: Optional[str]) -> WhisperModel:
    size = model_size or FW_MODEL_SIZE
    dev = device or FW_DEVICE
    compute_type = "float16" if dev == "cuda" else "int8"
    print(f"Loading Faster-Whisper '{size}' on {dev} ({compute_type}) …", flush=True)
    return WhisperModel(size, device=dev, compute_type=compute_type)


def transcribe(
    file_path: str | Path,
    output_dir: Path,
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    language: Optional[str] = "ar",
    verbose: bool = True,
) -> str:
    """
    Transcribe *file_path* and write two files into *output_dir*:

    - ``transcript_raw_ar.txt``            — plain text, one space-joined string
    - ``transcript_raw_ar_timestamps.txt`` — one ``[MM:SS-MM:SS] text`` line per segment

    Returns the plain-text transcript.
    """
    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(model_size, device)

    print("Transcribing …", flush=True)
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
    )

    if info is not None and getattr(info, "duration", None) is not None:
        dur: float = info.duration
        print(f"Audio length: {int(dur // 60)}m {int(dur % 60)}s", flush=True)

    def _fmt(t: float) -> str:
        m, s = divmod(int(t), 60)
        return f"{m:02d}:{s:02d}"

    texts: list[str] = []
    timed_lines: list[str] = []

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        texts.append(text)
        line = f"[{_fmt(float(seg.start or 0))}-{_fmt(float(seg.end or 0))}] {text}"
        timed_lines.append(line)
        if verbose:
            print(line, flush=True)

    full_text = " ".join(texts)
    timed_text = "\n".join(timed_lines)

    (output_dir / _TRANSCRIPT_PLAIN).write_text(full_text, encoding="utf-8")
    (output_dir / _TRANSCRIPT_TIMED).write_text(timed_text, encoding="utf-8")
    print(f"Transcripts saved to {output_dir}", flush=True)

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
    args = parser.parse_args()

    lang = None if args.language.lower() == "none" else args.language
    result = transcribe(
        args.file,
        output_dir=Path(args.output_dir),
        model_size=args.model,
        device=args.device,
        language=lang,
        verbose=True,
    )
    safe_print(result)
