# Publication cleanup decisions

The old Git history and original tracked/untracked source files were archived
locally outside this repository before cleanup. The public repository starts
with one clean initial commit; the recovery archive is not published.

## Removed or replaced

| Item | Previous purpose | Why it is no longer needed |
|---|---|---|
| `transcribe_notebook.ipynb` | Older single-file transcription, SRT conversion, caption burning and intro/background rendering. | The batch notebook supersedes its pipeline. Plain-text export and existing-video caption burning are now available in `transcribe2.ipynb`; shared rendering replaces its older renderer. |
| `helper_scripts/rename_videos.py` | Personal bulk renaming of numbered channel exports. | Unrelated to transcription, executed renames immediately, and shipped with an empty title. Rename final exports normally; the nature edition retains completion records for renamed exports. |
| `helper_scripts/strip_english_lines.py` (previously untracked) | Removed every transcript line containing an ASCII Latin letter. | Could delete useful mixed-language speech and audio-cutout placeholders, overwriting without a backup. Review unwanted text explicitly instead. |
| `new_prompt.md` | Manual Arabic-to-English translation prompt. | Renamed to `translation_prompt.md` and aligned with the same-block timestamp policy. Its useful instructions remain. |
| `codex/README.txt` | Short, incomplete instructions for the optional assistant translation workflow. | Replaced with `codex/README.md`, including actual input/output folders and optional reference handling. |
| Public `translator` branch | Earlier experimental local-LLM translation pipeline, extra notebooks and setup guides. | Not used by the supported external-translation workflow. Archived locally; removed from the remote so it cannot retain the old notebook outputs as a public branch. |
| Saved outputs and execution counts in all retained notebooks | Prior transcripts, machine paths, tracebacks and embedded previews. | Generated per-run data belongs locally. These results are cleared and checked by CI. |
| Embedded translation prompts in local notebooks | Copies of the manual translation instructions. | Replaced with links to the maintained prompt, avoiding contradictory rules. |
| Local YouTube settings, unused slug/directory helpers and missing-cell references | Leftovers from the single-file/download workflow. | The local notebooks accept files; Colab provides YouTube input. The unused settings and references described cells that no longer existed. |
| Repeated lexical review section in `codex/system_prompt.md` | Duplicate translation guidance. | One copy remains. |
| Personal trimming presets in `strip_edges.py` | Twenty channel-specific 20-second cuts. | New users must explicitly choose their own cuts. Preview remains the default. |
| Mandatory `yt-dlp` and `tqdm` entries in local requirements | Download support and progress dependency. | Local notebooks do not download YouTube media or directly use tqdm. Faster-Whisper manages its own transitive dependencies; yt-dlp is optional and installed separately by Colab. |

## Kept deliberately

- `transcribe2.ipynb`: primary local batch workflow.
- `transcribe_nature.ipynb`: useful distinct nature-footage renderer, including preview,
  styling controls and completion records; outputs cleared and channel watermark made configurable.
- `colab.ipynb`: distinct hosted, single-file/YouTube workflow.
- `redact_flagged.py` and `probe_region.py`: repair tools referenced by the notebooks.
- `diagnose_confidence.py`: optional developer tool for checking threshold behavior.
- `denoise_input.py`: optional DeepFilterNet cleanup; separate requirements.
- `strip_edges.py`: optional coordinated trimming of audio and both transcripts.
- Manual and assistant translation prompts: different entry points with consistent rules.

## Local files that stay out of publication

`assets/`, `input/`, `output/`, `transcript/`, caches, environments, credentials,
backups, manifests and generated subtitles are ignored. Existing local media,
transcripts and channel assets are preserved. These are working data, not obsolete
source files; deleting them is not necessary for publication.
