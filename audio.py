"""Utilities for extracting audio from local video and audio files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# All file extensions that can be fed to ffmpeg for audio extraction.
SUPPORTED_EXTENSIONS = {
    # Video
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv",
    # Audio (re-encode to normalised MP3)
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac",
}


def extract_audio(input_path: Path, output_path: Path) -> Path:
    """
    Convert *input_path* to an MP3 file at *output_path* using ffmpeg.

    Returns *output_path*.
    Exits with a clear error message if ffmpeg is not installed or fails.
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

    cmd = [
        "ffmpeg",
        "-y",                  # overwrite output without prompting
        "-i", str(input_path),
        "-vn",                 # drop video stream
        "-acodec", "libmp3lame",
        "-q:a", "2",           # high-quality VBR (~190 kbps)
        str(output_path),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        sys.exit(
            "Error: ffmpeg not found. Install it and ensure it is on your PATH.\n"
            "  Windows : winget install ffmpeg  (or https://ffmpeg.org/download.html)\n"
            "  macOS   : brew install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg"
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(
            f"Error: ffmpeg exited with code {exc.returncode}:\n"
            + exc.stderr.decode(errors="replace")
        )

    return output_path
