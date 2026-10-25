# Arabic Audio/Video Transcriber & Subtitler

Transcribe local Arabic or mixed-language audio/video with Faster-Whisper, translate
externally, and generate subtitles or captioned videos. Translation prompts specialize
in Islamic lectures; the transcription language is configurable.

## Choose a notebook

| Notebook | Workflow |
|---|---|
| [transcribe2.ipynb](transcribe2.ipynb) | Main local workflow: batch transcription, quality reports, approved redactions, SRT export, original-video captioning, and intro + looping-background videos. |
| [transcribe_nature.ipynb](transcribe_nature.ipynb) | The same transcription/translation workflow with randomized nature excerpts, centered subtitles, blur/color controls, an optional watermark and a silent design preview. |
| [colab.ipynb](colab.ipynb) | Hosted single-file transcription, uploads/YouTube input, SRT export and captions on the original video. No intro/background or nature renderer. |

The local notebooks accept files, not YouTube URLs. They share Python modules in
this repository: clone/download the whole project rather than just a notebook.
The Colab notebook is self-contained. No LLM API key is required: translation is
performed separately using your chosen assistant.

## Local setup

Use Python 3.11 or newer; Python 3.11 is the validation baseline. Install a full
[FFmpeg build](https://ffmpeg.org/download.html) with the `subtitles`/libass filter
and `ffprobe` on PATH for rendering and audio repair. The nature renderer also
requires the `ass`, `gblur`, `eq`, `scale`, `crop` and `fps` filters.

On Windows, open a terminal in the repository:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter notebook
```

On macOS/Linux, create the environment with `python3 -m venv .venv` and activate
with `source .venv/bin/activate`, then use the same install/run commands.
Select that environment as the notebook kernel. Start Jupyter from the repository
folder, so relative paths and shared imports resolve correctly.

CPU transcription uses int8. For NVIDIA GPU use, select an appropriate PyTorch
build using the [official installer](https://pytorch.org/get-started/locally/),
and follow [Faster-Whisper's GPU prerequisites](https://github.com/SYSTRAN/faster-whisper#gpu).
Faster-Whisper uses CTranslate2: a working CUDA PyTorch installation alone does
not guarantee that its CUDA/cuBLAS/cuDNN runtime libraries are available.
Rendering tests NVENC before use and falls back to CPU encoding when necessary.
H.264 is the public default; HEVC is optional.

Faster-Whisper, hf_xet and Jupyter are pinned to the local baseline. PyTorch is
left selectable for your operating system and CPU/CUDA installation. This is not
a complete environment lock; optional denoising also needs a compatible torchaudio
build. The local environment used during cleanup had Python 3.11.15,
Faster-Whisper 1.2.1, PyTorch 2.5.1+cu121 and FFmpeg 7.1.1.

## Transcribe and translate

1. Create `input/` and add media. Each filename stem must be unique:
   `lesson.mp3` and `lesson.mp4` in the same folder are rejected as ambiguous.
2. Run Cells 1, 2, 2B and 3 in either local notebook. Choose the model and language
   in Cell 2. `START_TIME`/`END_TIME` optionally restrict every input to a time range.
   Set `DEVICE="cpu"` and `COMPUTE_TYPE="int8"` after configuration if needed.
3. Inspect `transcript/arabic/<name>.txt` and any `<name>.flagged.txt` reports.
   These checks are heuristics, not proof of audio/transcription quality.
   Empty transcriptions produce a warning. Skipped inputs are not re-analyzed.
4. Optionally use Cell 4 to prepare a redaction review, approve regions, preview
   changes and apply them. Redact Arabic before translating. Original transcripts
   are backed up. A changed/removed detected report invalidates its old approvals.
5. Use [translation_prompt.md](translation_prompt.md) with a chat assistant, or
   [codex/README.md](codex/README.md) for the optional folder-based workflow.
   Fill `transcript/english/<name>.txt`. Preserve every timestamp, the number of
   blocks, and each block's meaning; keep audio-cutout placeholders unchanged.
6. Run Cell 9 to export SRT without video, or choose one of the rendering workflows below.

The model downloads once and is reused from cache. `OVERWRITE=False` skips existing
Arabic transcripts; English translations are never replaced by transcription.
`OVERWRITE=True` regenerates Arabic, plain text and the quality report. Review
existing translations after regenerating their source. Old reports are removed
when fresh analysis finds no alerts or quality checks are disabled.

```text
input/lesson.mp3
transcript/arabic/lesson.txt          # timestamped Arabic
transcript/arabic/lesson.plain.txt    # Arabic without timestamps
transcript/arabic/lesson.flagged.txt  # only when checks find a concern
transcript/english/lesson.txt         # blank until translated
transcript/english/lesson.srt         # generated on conversion/render
transcript/redactions.txt             # review decisions
output/lesson_final.mp4               # intro/background workflow
output/nature/lesson_final.mp4        # nature workflow
output/nature/lesson_final.clips.json # successful-render record and clip plan
```

Timestamped transcripts use whole seconds (`[MM:SS-MM:SS]` or
`[H:MM:SS-H:MM:SS]`). Subsecond ASR segments are given a minimum one-second caption
span to avoid zero-duration SRT cues; neighboring captions can overlap at this
precision. SRT conversion rejects malformed lines, nonpositive ranges and
out-of-order starts rather than silently omitting text. Existing SRT files are
regenerated from the current transcript when requested.

## Video assets: supply your own

No channel artwork, footage, intro, audio, fonts or watermark branding is bundled.
The `assets/` directory stays local and is ignored by Git. Existing channel assets
can be used without adding them to the repository.

### Intro + looping background (`transcribe2.ipynb`, Cell 8)

Create these files, or change `INTRO_PATH`/`BACKGROUND_PATH` in the cell:

- `assets/Intro.mp4`: a playable video of positive duration. Its audio plays first;
  if it has no audio, the renderer inserts silence for its duration.
- `assets/Background.mp4`: a playable video of positive duration. Its audio is
  discarded. Its picture loops beneath the main lecture audio.

Both videos are fitted to the configured canvas (default 1920×1080 at 24 fps).
The lecture starts after the intro; subtitles are shifted by that duration.
The final background loop plays to completion, with silence after the lecture
if needed, then the picture fades to black. Files are matched by stem across
`input/` and `transcript/english/`. Blank translations are skipped.

Set `OVERWRITE_FINAL=True` to rebuild existing exports after editing translations,
redactions or styling. Without it, completed video files are skipped. Partial or
failed encodes do not replace a previously completed file.

### Nature footage (`transcribe_nature.ipynb`)

Add your own video clips under `assets/nature/`; subfolders are supported. Each
usable clip must be slightly longer than `MIN_CLIP_SECONDS` (one output frame of
headroom). Defaults select excerpts of 8–20 seconds; adjust both length controls
in Cell 8A. Footage audio is discarded, and shorter clips are reported/skipped.
Use clips and fonts you are entitled to use.

Run **8A → 8B → 8C** for a silent design preview, or **8A → 8B → 8D** for final
translated videos. There is no intro by default. `INCLUDE_INTRO=True` uses your
`assets/Intro.mp4`; a silent intro is supported. Set your own `WATERMARK_TEXT`,
or leave it empty. Choose installed subtitle/watermark fonts; they are not bundled.

Cell 8A controls blur/color, subtitle size/margins, and background color, opacity,
padding and enable/disable. `SUBTITLE_BACKGROUND_MODE="block"` draws one box
around the whole caption; `"line"` draws individual line boxes.

Keep `.clips.json` completion records when renaming final nature videos. With
`OVERWRITE_FINAL=False`, either the normal export filename or its completion record
causes a skip. `OVERWRITE_FINAL=True` forces a fresh render and clip selection.

### Caption an existing video (`transcribe2.ipynb`, Cell 7)

Set `VIDEO_FILE`, `SRT_FILE` and `OUTPUT_FILE`, then run that optional cell.
It requires neither intro nor background footage. Generate the SRT with Cell 9
first, or supply an existing SRT. `OVERWRITE_CAPTIONED` controls replacement.

## Google Colab and YouTube

Open `colab.ipynb` in Colab and choose a GPU runtime if available. Run installation,
configuration, and either file upload or YouTube input; then transcribe and download
the plain/timestamped text. Translate externally, upload the translated text for
SRT conversion, and optionally burn it into a video.

YouTube input defaults to anonymous access. Downloads may fail because of access
restrictions or bot checks. The cell optionally accepts a cookies file; treat it
as a credential and do not save authentication output into the public notebook.
Uploading a local media file is the alternative.

For optional local downloading, install yt-dlp separately:

```text
python -m pip install yt-dlp
yt-dlp -x --audio-format mp3 -o "input/%(id)s.%(ext)s" "YOUTUBE_URL"
```

## Optional audio tools

Run helpers from the repository folder; they are independent of video rendering.

| Tool | Purpose / example |
|---|---|
| `helper_scripts/redact_flagged.py` | `review --interactive`, then `apply --dry-run`, then `apply`. Supports `--only NAME` and `--target arabic`. |
| `helper_scripts/probe_region.py` | `input/lesson.mp3 5:11 7:11` re-transcribes a window with word timings; `--model` and `--language` are configurable. |
| `helper_scripts/diagnose_confidence.py` | `input/lesson.mp3` writes metrics into ignored `output/diagnostics/`; uses the notebook's segment-level settings. |
| `helper_scripts/denoise_input.py` | Preview first; `--apply` writes enhanced files with original backups, or use `--out-dir input_cleaned`. Install `requirements-denoise.txt` separately. |
| `helper_scripts/strip_edges.py` | `lesson --start 00:20` previews coordinated audio/transcript trimming; add `--apply` to write. No personal trim presets ship. |

Denoising and trimming can change audio/transcripts. Keep the `.bak` originals.
Trimming prepares every replacement before applying it and rolls back ordinary
replacement failures; interruption/power loss during multiple file replacements
can still require recovery from backups. Re-running a cut applies it to the
current files, so preview each time. Re-render with overwrite enabled afterwards.

## Development and clean notebooks

```text
python -B -m unittest discover -s tests -v
python tools/clean_notebooks.py
python tools/clean_notebooks.py --check
```

Tests cover validation, ambiguity, report lifecycle, redaction, trim rollback and
encoder fallback, with synthetic rendering checks when FFmpeg is available.
They do not download Whisper models or read your channel media. Full speech
recognition and a hosted Colab session require separate end-to-end validation.
GitHub checks reject saved notebook outputs/execution counts and run the tests.
Clear outputs again after local use and before committing.

See [REMOVALS.md](REMOVALS.md) for every removed/replaced item and its reason.
Code is released under the [MIT License](LICENSE); supplied media and third-party
model/runtime licenses are separate.
