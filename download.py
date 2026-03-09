import yt_dlp
from pathlib import Path

from config import VIDEO_PATH


def download_mp3(url: str, output_path: str | None = None) -> str:
    """
    Download audio from a URL as MP3.
    output_path: path to the output file (e.g. "video.mp3").
    Returns the path to the downloaded file.
    """
    target = Path(output_path) if output_path is not None else VIDEO_PATH
    p = target.resolve()
    outtmpl = str(p.parent / (p.stem + ".%(ext)s"))
    p.parent.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return str(p.resolve())


if __name__ == "__main__":
    youtube_url = input("Enter YouTube URL: ")
    download_mp3(youtube_url)