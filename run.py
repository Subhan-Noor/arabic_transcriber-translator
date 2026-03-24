"""
YouTube Arabic Transcriber — main entry point.

Given a YouTube URL or a local video/audio file, this script:
  1. Downloads or extracts the audio.
  2. Transcribes it with Faster-Whisper.
  3. Saves all outputs in a dedicated sub-folder inside ``outputs/``.

Usage:
    python run.py "<youtube-url>"
    python run.py /path/to/lecture.mp4
    python run.py lecture.mp3 --language ar
    python run.py lecture.mp4 --model medium --device cpu
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from audio import extract_audio
from config import OUTPUTS_DIR
from download import download_mp3
from transcribe import safe_print, transcribe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_url(text: str) -> bool:
    return bool(re.match(r"https?://", text.strip()))


def _slugify(text: str, max_len: int = 60) -> str:
    """Return a filesystem-safe folder name derived from *text*."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)          # strip non-word chars
    text = re.sub(r"[\s_-]+", "_", text)           # collapse whitespace/dashes
    text = text.strip("_")
    return text[:max_len] or "transcription"


def _unique_dir(base: Path) -> Path:
    """Return *base* if unused, otherwise ``base_2``, ``base_3``, …"""
    if not base.exists():
        return base
    counter = 2
    while True:
        candidate = base.parent / f"{base.name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Transcribe Arabic audio from a YouTube URL or a local file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run.py "https://www.youtube.com/watch?v=xhGThib15IU"
  python run.py lecture.mp4
  python run.py lecture.mp4 --model medium --device cpu
  python run.py lecture.mp4 --language none   # auto-detect language
        """,
    )
    parser.add_argument(
        "input",
        help="YouTube URL  –or–  path to a local video/audio file "
             "(.mp4 .mkv .avi .mov .webm .mp3 .wav .flac .m4a …).",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="SIZE",
        help="Faster-Whisper model size: tiny | base | small | medium | large-v2. "
             "Defaults to the value set in config.py.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Inference device. Defaults to the value set in config.py.",
    )
    parser.add_argument(
        "--language",
        default="ar",
        metavar="LANG",
        help="Whisper language code (e.g. 'ar', 'en'). "
             "Pass 'none' to let Whisper auto-detect. Default: ar.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    lang = None if args.language.lower() == "none" else args.language
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — acquire audio and determine the output folder name
    # ------------------------------------------------------------------
    if _is_url(args.input):
        print("Detected YouTube URL — downloading audio …", flush=True)
        title, tmp_audio = download_mp3(args.input)
        folder_name = _slugify(title)
    else:
        src = Path(args.input)
        if not src.exists():
            print(f"Error: file not found: {src}", file=sys.stderr)
            sys.exit(1)
        folder_name = _slugify(src.stem)
        tmp_audio = None  # resolved below once output_dir is known

    output_dir = _unique_dir(OUTPUTS_DIR / folder_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_dir}", flush=True)

    # ------------------------------------------------------------------
    # Step 2 — place audio in the output folder
    # ------------------------------------------------------------------
    audio_dest = output_dir / "audio.mp3"

    if _is_url(args.input):
        # Move the temp download into the run folder
        tmp_audio.rename(audio_dest)
        audio_path = audio_dest
    else:
        src = Path(args.input)
        print(f"Extracting audio from '{src.name}' …", flush=True)
        audio_path = extract_audio(src, audio_dest)

    # ------------------------------------------------------------------
    # Step 3 — transcribe
    # ------------------------------------------------------------------
    print("Transcribing (each segment prints as it completes) …", flush=True)
    text = transcribe(
        audio_path,
        output_dir=output_dir,
        model_size=args.model,
        device=args.device,
        language=lang,
        verbose=True,
    )

    print(f"\nDone. All outputs saved to: {output_dir}", flush=True)
    safe_print(text)


if __name__ == "__main__":
    main()
