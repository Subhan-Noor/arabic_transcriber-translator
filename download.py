"""Download audio from a YouTube URL as MP3."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yt_dlp


def download_mp3(url: str) -> tuple[str, Path]:
    """
    Download audio from *url* as a 192 kbps MP3 into a temporary directory.

    Returns ``(video_title, mp3_path)``.
    The caller is responsible for moving the file to its final destination.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_transcribe_"))

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(tmp_dir / "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title: str = info.get("title") or "video"

    # yt-dlp names the MP3 after the video title; find it.
    candidates = list(tmp_dir.glob("*.mp3"))
    if not candidates:
        raise FileNotFoundError(
            f"yt-dlp did not produce an MP3 in {tmp_dir}. "
            "Ensure ffmpeg is installed and on your PATH."
        )

    return title, candidates[0]


if __name__ == "__main__":
    _url = input("Enter YouTube URL: ").strip()
    _title, _path = download_mp3(_url)
    print(f"Downloaded: {_title!r} → {_path}")
