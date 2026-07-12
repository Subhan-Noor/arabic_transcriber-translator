# Arabic Audio/Video Transcriber & Subtitler

Transcribe Arabic (and mixed Arabic/English) speech from YouTube videos or local
audio/video files using [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper),
then turn the transcript into subtitles — with an optional LLM-assisted
English translation step.

The project is notebook-based so you can run it either **locally** (with your
own GPU/CPU) or **for free on Google Colab** — no coding required beyond
editing a few configuration cells.

## Features

- Transcribe a local audio/video file *or* a YouTube URL (via `yt-dlp`)
- Plain-text and timestamped (`[MM:SS-MM:SS]`) transcripts
- Optional time-range transcription (e.g. only `8:00` → `25:00` of a file)
- Generate an **SRT** subtitle file from a timestamped transcript
- **Burn subtitles** directly into the video with `ffmpeg`
- A ready-made LLM prompt (`prompt.txt`) for translating the Arabic
  transcript into English while preserving the original timestamp
  boundaries — paste it into ChatGPT/Claude/etc. together with your
  transcript

## Which notebook should I use?

| Notebook | Where it runs | Best for |
|---|---|---|
| [`transcribe_notebook.ipynb`](transcribe_notebook.ipynb) | Your own machine (Jupyter) | Repeated use, local GPU, caches the Whisper model on disk so it never re-downloads |
| [`colab.ipynb`](colab.ipynb) | [Google Colab](https://colab.research.google.com/) | No local setup, free GPU, one-off transcriptions |

Both notebooks share the same core pipeline (Faster-Whisper → timestamped
transcript → SRT → captioned video); the Colab version additionally handles
file uploads/downloads in the browser instead of the local filesystem.

## Output structure

Each run writes into an `outputs/<name>/` folder next to the notebook, so
nothing is ever overwritten:

```
outputs/
  some_video_title/
    <name>_transcript.txt              # plain Arabic transcript
    <name>_transcript_timestamps.txt    # [MM:SS-MM:SS] timestamped transcript
    <name>.srt                          # (optional) subtitle file
    <name>_captioned.mp4                # (optional) video with burned-in captions
```

---

## 1. Local setup (`transcribe_notebook.ipynb`)

### Prerequisites

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your `PATH`
  (required for audio extraction and for burning subtitles into video)
- An NVIDIA GPU with CUDA is optional but strongly recommended for larger
  models; CPU also works, just slower

### Install

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you have an NVIDIA GPU, install a CUDA build of PyTorch (adjust the index
URL to match your CUDA version, e.g. `cu128`, `cu126`, `cu121`):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Run

```bash
jupyter notebook transcribe_notebook.ipynb
```

Run the cells top to bottom:

1. **Verify dependencies** — checks Python packages, ffmpeg, and GPU availability
2. **Configuration** — set model size, language, and optional time range
3. **Model cache** — downloads the Whisper model once, reused on every later run
4. **Input** — point at a local file *or* paste a YouTube URL (run one of the two cells)
5. **Transcribe** — runs Faster-Whisper and prints timestamped segments live
6. **Save transcripts** — writes the plain and timestamped `.txt` files
7. *(optional)* **Generate SRT** — converts a timestamped transcript into `.srt`
8. *(optional)* **Burn captions** — uses ffmpeg to produce a captioned `.mp4`

---

## 2. Google Colab (`colab.ipynb`)

1. Open [`colab.ipynb`](colab.ipynb) in Google Colab (upload it or open it
   directly from GitHub via `File → Open notebook → GitHub`)
2. Go to `Runtime → Change runtime type` and select a **T4 GPU** for the best speed
3. Run the cells top to bottom:
   - Install dependencies
   - Check GPU
   - Configuration (model size, language, time range)
   - Input — **either** upload a local file **or** paste a YouTube URL
   - Transcribe
   - Save & download the transcript
   - *(optional)* Generate an SRT and burn it into the video

---

## 3. Translating the transcript with an LLM

Once you have a timestamped transcript, copy the contents of
[`prompt.txt`](prompt.txt) into your LLM chat of choice (ChatGPT, Claude,
etc.) along with the transcript file. The prompt instructs the model to:

- Translate Arabic → English while preserving meaning and wording
- Keep the exact same `[MM:SS-MM:SS]` timestamp boundaries (so the output can
  be fed straight into the SRT generator cell)
- Avoid adding commentary, notes, or explanations

For long videos, split the transcript into ~20-minute chunks and translate
each one as a continuation of the previous part (the prompt already includes
guidance for this).

---

## 4. Tuning quality and speed

Set in the **Configuration** cell of either notebook:

| Setting | Options | Notes |
|---|---|---|
| `MODEL_SIZE` | `tiny` `base` `small` `medium` `large-v2` `large-v3` `large-v3-turbo` | Larger = more accurate, slower. `large-v3-turbo` is close to `large-v3` quality at a fraction of the time |
| `LANGUAGE` | `"ar"`, `"en"`, `"fr"`, … or `None` | `None` auto-detects from the first 30 seconds |
| `START_TIME` / `END_TIME` | `"8:00"`, `"1:25:00"`, `None` | Transcribe only part of a file; timestamps in the output still match the original |

---

## Contributing

Issues and pull requests are welcome. If you add a feature to one notebook,
please consider porting it to the other so the local and Colab versions stay
in sync.

## License

Released under the [MIT License](LICENSE).
