# YouTube Arabic Transcriber (CLI)

This is currently a **transcription-only** CLI tool. It:

- Downloads a YouTube video.
- Transcribes Arabic (and mostly-Arabic) audio using **Faster-Whisper**.
- Saves both a plain Arabic transcript and a version with timestamps.

> **Note:** LLM-based Arabic cleanup and translation (Arabic → English) are a **work in progress** and are currently disabled in `run.py`. The code for those stages exists (`clean_arabic.py`, `translate.py`, `clean_english.py`), but they are not executed by default.

Per run, `run.py` currently produces:

- `video.mp3`
- `transcript_raw_ar.txt`
- `transcript_raw_ar_timestamps.txt`

## 1. Environment setup

From the `youtube-translator` folder:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Make sure you installed the **CUDA** build of PyTorch so Faster-Whisper can use your RTX 3070:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

You can adjust the CUDA index URL (`cu128`, `cu126`, `cu121`) based on what works best on your machine.

## 2. (Optional) Ollama setup for future translation

The project includes scaffolding for Arabic cleanup and translation using **Ollama**, but those steps are not run by default.
If you want to experiment with them later:

```bash
ollama pull command-r7b-arabic:7b   # Arabic cleanup + translation
ollama pull qwen3:8b               # English polishing
ollama serve
```

The scripts assume Ollama is listening on `http://localhost:11434`. You can change this and the model names in `config.py`.

## 3. Usage

From the `youtube-translator` directory, with your virtualenv activated:

```bash
python run.py "<youtube-url>"
```

Example:

```bash
python run.py "https://www.youtube.com/watch?v=xhGThib15IU"
```

This will:

1. Download the video audio to `video.mp3`.
2. Transcribe it with Faster-Whisper → `transcript_raw_ar.txt` and `transcript_raw_ar_timestamps.txt`.

You can also run individual steps:

- Transcription only (writes `transcript_raw_ar.txt` + `transcript_raw_ar_timestamps.txt`):

  ```bash
  python transcribe.py
  ```

> Arabic cleanup (`clean_arabic.py`), translation (`translate.py`), and English polishing (`clean_english.py`) are present but considered **experimental** and are not part of the default `run.py` flow.

## 4. Tuning transcription quality and speed

- **Faster-Whisper** settings are in `config.py` (`FW_MODEL_SIZE`, `FW_DEVICE`).
  - You currently use `"large-v2"` on GPU for best quality; you can switch to `"medium"` or `"small"` for faster runs if needed.

