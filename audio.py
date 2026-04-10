"""Utilities for extracting audio from local video and audio files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

# All file extensions that can be fed to ffmpeg for audio extraction.
SUPPORTED_EXTENSIONS = {
    # Video
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv",
    # Audio (re-encode to normalised MP3)
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac",
}

_OUT_TIME_RE = re.compile(r"out_time_ms=(\d+)")


def probe_media_duration(input_path: Path) -> float | None:
    """Return the duration of *input_path* in seconds, or None if unknown."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def extract_audio(input_path: Path, output_path: Path) -> Path:
    """
    Convert *input_path* to an MP3 file at *output_path* using ffmpeg.

    A tqdm progress bar shows how much of the file has been processed.
    Returns *output_path*.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = probe_media_duration(input_path)

    cmd = [
        "ffmpeg",
        "-y",                    # overwrite output without prompting
        "-i", str(input_path),
        "-vn",                   # drop video stream
        "-acodec", "libmp3lame",
        "-q:a", "2",             # high-quality VBR (~190 kbps)
        "-progress", "pipe:1",   # stream progress key=value pairs to stdout
        "-nostats",              # suppress the default stderr stats line
        str(output_path),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        sys.exit(
            "Error: ffmpeg not found. Install it and ensure it is on your PATH.\n"
            "  Windows : winget install ffmpeg  (or https://ffmpeg.org/download.html)\n"
            "  macOS   : brew install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg"
        )

    total_s = int(duration) if duration else None
    last_s = 0.0

    with tqdm(
        total=total_s,
        unit="s",
        unit_scale=True,
        desc="  Extracting audio",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]"
        if total_s
        else "{l_bar}{bar}| {n:.0f}s elapsed",
    ) as pbar:
        for line in proc.stdout:  # type: ignore[union-attr]
            m = _OUT_TIME_RE.match(line.strip())
            if m:
                current_s = int(m.group(1)) / 1_000_000  # microseconds → seconds
                delta = current_s - last_s
                if delta > 0:
                    pbar.update(delta)
                    last_s = current_s

        proc.wait()

    if proc.returncode != 0:
        err = proc.stderr.read() if proc.stderr else ""  # type: ignore[union-attr]
        sys.exit(f"Error: ffmpeg exited with code {proc.returncode}:\n{err}")

    return output_path
