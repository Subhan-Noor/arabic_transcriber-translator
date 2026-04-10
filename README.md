# YouTube Arabic Transcriber

A command-line tool that transcribes Arabic (and mixed Arabic/English) audio from
YouTube videos or local media files using **Faster-Whisper**.

Each run produces a self-contained output folder so nothing is ever overwritten.

## Output structure

```
outputs/
  some_video_title/
    audio.mp3                        # downloaded / extracted audio
    transcript_raw_ar.txt            # plain Arabic transcript
    transcript_raw_ar_timestamps.txt # [MM:SS-MM:SS] or [H:MM:SS-H:MM:SS] markers
  another_lecture/
    audio.mp3
    transcript_raw_ar.txt
    transcript_raw_ar_timestamps.txt
```

> **Work in progress:** LLM-based Arabic cleanup, Arabic → English translation,
> and English polishing are planned features. The transcription pipeline is
> fully functional today.

---

## 1. Prerequisites

### Python 3.11+

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### PyTorch with CUDA (NVIDIA GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Adjust the index URL to match your CUDA version (`cu128`, `cu126`, `cu121`).
Run `python check_cuda.py` to verify your setup.

### ffmpeg (required)

ffmpeg is used by both `yt-dlp` (audio extraction from YouTube) and this tool
(audio extraction from local video files).

```
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Debian / Ubuntu
sudo apt install ffmpeg
```

---

## 2. Usage

### Transcribe a YouTube video

```bash
python run.py "https://www.youtube.com/watch?v=xhGThib15IU"
```

### Transcribe a local video or audio file

Any format ffmpeg understands is accepted: `.mp4`, `.mkv`, `.mov`, `.avi`,
`.webm`, `.mp3`, `.wav`, `.flac`, `.m4a`, …

```bash
python run.py lecture.mp4
python run.py recording.mp3
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model SIZE` | Faster-Whisper model: `tiny` `base` `small` `medium` `large-v2` | `large-v2` (set in `config.py`) |
| `--device cuda\|cpu` | Inference device | `cuda` (set in `config.py`) |
| `--language LANG` | Whisper language code. Pass `none` to auto-detect | `ar` |
| `--start TIME` | Begin transcribing at this timestamp | start of file |
| `--end TIME` | Stop at this timestamp | end of file |

```bash
# Use a smaller/faster model on CPU
python run.py lecture.mp4 --model medium --device cpu

# Auto-detect language (mixed Arabic/English video)
python run.py lecture.mp4 --language none
```

### Transcribe part of a video (time range)

Use `--start` and/or `--end` with the same time formats you would use in a player:

| Form | Meaning | Example |
|------|---------|---------|
| `SS` | seconds only | `45` → 45 seconds |
| `M:SS` / `MM:SS` | minutes and seconds | `8:00`, `0:35` |
| `H:MM:SS` | hours, minutes, seconds | `1:25:00` |

```bash
# From 8:00 through 25:00 (timestamps in the transcript match the original video)
python run.py lecture.mp4 --start 8:00 --end 25:00

# First 90 seconds
python run.py clip.mp4 --end 1:30

# From 0:45 through the end of the file (output folder name includes the time range)
python run.py clip.mp4 --start 0:45
```

The full audio file is still downloaded or extracted; only the chosen window is sent to Whisper. If `ffprobe` can read the duration, `--end` is clamped to the file length and the output folder name includes the range (e.g. `…_8m00s-25m00s`).

### Transcribe only (skip download)

If you already have an audio file and just want a transcript:

```bash
python transcribe.py lecture.mp3 --output-dir outputs/my_lecture
python transcribe.py lecture.mp3 --output-dir outputs/snippet --start 2:00 --end 5:30
```

---

## 3. Tuning quality and speed

All defaults live in `config.py`:

| Setting | Default | Notes |
|---------|---------|-------|
| `FW_MODEL_SIZE` | `"large-v2"` | Best quality for Arabic. Use `"medium"` for faster runs. |
| `FW_DEVICE` | `"cuda"` | Switch to `"cpu"` if no GPU is available. |

---

## 4. Checking your GPU

```bash
python check_cuda.py
```
