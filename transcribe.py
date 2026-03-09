import sys
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from config import (
    FW_DEVICE,
    FW_MODEL_SIZE,
    TRANSCRIPT_RAW_AR_PATH,
    TRANSCRIPT_RAW_AR_TIMED_PATH,
)


def _build_model(
    model_size: Optional[str] = None,
    device: Optional[str] = None,
) -> WhisperModel:
    size = model_size or FW_MODEL_SIZE
    dev = device or FW_DEVICE
    compute_type = "float16" if dev == "cuda" else "int8"
    print(f"Loading Faster-Whisper model='{size}' on device='{dev}' ({compute_type})", flush=True)
    return WhisperModel(size, device=dev, compute_type=compute_type)


def transcribe(
    file_path: str,
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    language: Optional[str] = "ar",
    verbose: bool = True,
) -> str:
    """
    Transcribe audio using Faster-Whisper.

    - Defaults to Arabic (`language='ar'`) for best accuracy on Arabic lectures.
    - Uses VAD to be more robust to noise and silence.
    - Prints each segment with timestamps when verbose=True.
    """
    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _build_model(model_size=model_size, device=device)

    print("Transcribing audio...", flush=True)
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
    )

    if info is not None and getattr(info, "duration", None) is not None:
        dur = info.duration
        mins = int(dur // 60)
        secs = int(dur % 60)
        print(f"Audio length: {mins}m {secs}s", flush=True)

    texts: list[str] = []
    lines: list[str] = []

    def _fmt(t: float) -> str:
        mins = int(t // 60)
        secs = int(t % 60)
        return f"{mins:02d}:{secs:02d}"

    for seg in segments:
        seg_text = seg.text.strip()
        if not seg_text:
            continue
        start = float(seg.start or 0.0)
        end = float(seg.end or 0.0)
        texts.append(seg_text)
        line = f"[{_fmt(start)}-{_fmt(end)}] {seg_text}"
        lines.append(line)
        if verbose:
            print(line, flush=True)

    full_text = " ".join(texts)
    timed_text = "\n".join(lines)

    # Always write a timestamped transcript for later use by LLMs/prompts.
    Path(TRANSCRIPT_RAW_AR_TIMED_PATH).write_text(timed_text, encoding="utf-8")
    print(f"Timestamped transcript saved to {TRANSCRIPT_RAW_AR_TIMED_PATH}", flush=True)

    return full_text


def safe_print(text: str) -> None:
    """Print Unicode text safely on Windows (avoids cp1252 encoding errors)."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Windows console often can't display Arabic etc.; write as UTF-8
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    audio_path = Path("video.mp3")
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    print("Transcribing (this may take a while)...", flush=True)
    text = transcribe(str(audio_path), verbose=True)
    Path(TRANSCRIPT_RAW_AR_PATH).write_text(text, encoding="utf-8")
    print(f"Transcript saved to {TRANSCRIPT_RAW_AR_PATH}", flush=True)
    safe_print(text)