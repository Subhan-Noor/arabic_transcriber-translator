"""
YouTube Arabic Transcriber — main entry point.

Given a YouTube URL or a local video/audio file, this script:
  1. Downloads or extracts the audio.
  2. Transcribes it with Faster-Whisper.
  3. Saves all outputs in a dedicated sub-folder inside ``outputs/``.

Usage:
    python run.py "<youtube-url>"
    python run.py /path/to/lecture.mp4
    python run.py lecture.mp4 --language ar
    python run.py lecture.mp4 --start 8:00 --end 25:00
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from audio import extract_audio, probe_media_duration
from config import OUTPUTS_DIR
from download import download_mp3
from timecode import format_segment_folder_label, format_timestamp, parse_timestamp
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


def _resolve_segment(
    media_path: Path,
    start_raw: str | None,
    end_raw: str | None,
) -> tuple[float | None, float | None]:
    """
    Parse CLI time strings and clamp to *media_path* duration when known.

    Returns ``(start_sec, end_sec)`` where each may be ``None`` for
    "from beginning" / "to end of file" when no segment was requested.
    """
    if start_raw is None and end_raw is None:
        return None, None

    start_s = parse_timestamp(start_raw) if start_raw else 0.0
    end_s = parse_timestamp(end_raw) if end_raw else None

    dur = probe_media_duration(media_path)

    if end_s is None:
        if dur is not None:
            end_s = dur
    elif dur is not None and end_s > dur:
        print(
            f"  Warning: --end {format_timestamp(end_s)} is past file end "
            f"({format_timestamp(dur)}); clamped.",
            flush=True,
        )
        end_s = dur

    if dur is not None and start_s >= dur:
        print(
            f"Error: --start {format_timestamp(start_s)} is at or past the end "
            f"of the media ({format_timestamp(dur)}).",
            file=sys.stderr,
        )
        sys.exit(1)

    if end_s is not None and start_s >= end_s:
        print(
            "Error: start time must be before end time.",
            file=sys.stderr,
        )
        sys.exit(1)

    return start_s, end_s


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
  python run.py talk.mp4 --start 8:00 --end 25:00
  python run.py intro.mp4 --end 1:30        # from 0:00 through 1:30
  python run.py clip.mp4 --start 0:45         # from 0:45 to end of file
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
    parser.add_argument(
        "--start",
        metavar="TIME",
        default=None,
        help="Transcribe from this time onward. "
             "Formats: SS, M:SS, MM:SS, or H:MM:SS (e.g. 8:00, 0:35, 1:25:00).",
    )
    parser.add_argument(
        "--end",
        metavar="TIME",
        default=None,
        help="Transcribe up to this time (same formats as --start). "
             "Omit for end of file.",
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
        print("\n[1/2] Downloading audio …", flush=True)
        title, tmp_audio = download_mp3(args.input)
        base_folder = _slugify(title)
        probe_path = tmp_audio
    else:
        src = Path(args.input)
        if not src.exists():
            print(f"Error: file not found: {src}", file=sys.stderr)
            sys.exit(1)
        base_folder = _slugify(src.stem)
        tmp_audio = None
        probe_path = src

    start_s: float | None
    end_s: float | None
    start_s, end_s = _resolve_segment(probe_path, args.start, args.end)

    if args.start is not None or args.end is not None:
        base_folder = f"{base_folder}_{format_segment_folder_label(start_s, end_s)}"

    output_dir = _unique_dir(OUTPUTS_DIR / base_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output folder: {output_dir}", flush=True)

    # ------------------------------------------------------------------
    # Step 2 — place audio in the output folder
    # ------------------------------------------------------------------
    audio_dest = output_dir / "audio.mp3"

    if _is_url(args.input):
        tmp_audio.rename(audio_dest)
        audio_path = audio_dest
        print("  Audio saved.", flush=True)
    else:
        src = Path(args.input)
        print(f"\n[1/2] Extracting audio from '{src.name}' …", flush=True)
        audio_path = extract_audio(src, audio_dest)

    # ------------------------------------------------------------------
    # Step 3 — transcribe
    # ------------------------------------------------------------------
    print("\n[2/2] Transcribing …", flush=True)
    text = transcribe(
        audio_path,
        output_dir=output_dir,
        model_size=args.model,
        device=args.device,
        language=lang,
        verbose=True,
        start_sec=start_s,
        end_sec=end_s,
    )

    print(f"\nDone. Outputs saved to: {output_dir}", flush=True)
    safe_print(text)


if __name__ == "__main__":
    main()
