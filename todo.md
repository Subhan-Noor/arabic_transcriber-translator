# TODO — Local Arabic→English Translation Stage

Task tracker derived from `../local_translation_notebook_development_plan.md`.
Target: add a local **Aya Expanse 8B (vLLM in WSL2)** translation stage to `transcribe2.ipynb`, between transcription and final rendering.

Legend: `[ ]` pending · `[x]` done · `[~]` in progress

---

## Phase 0 — Preserve a Baseline

- [x] Copy the working notebook to `transcribe2_before_local_translation.ipynb`.
- [x] Select three Arabic transcripts (≈1 min, ≈5 min, and one hard excerpt with technical terminology).
- [x] Save human-approved English translations for 20–40 timestamp blocks.
- [x] Store these as the initial **golden set** for later evaluation.

## Phase 1 — Windows Notebook Environment

- [x] `conda activate transcribe`. (env exists and is usable)
- [x] Confirm CUDA PyTorch (`torch.cuda.is_available()` is `True`, RTX 3070 detected). — verified: torch 2.5.1+cu121, RTX 3070.
- [x] Declare notebook-side dependencies (`requirements.txt`).
- [x] Declare optional dev dependencies (`requirements-dev.txt`).
- [x] Install notebook-side dependencies into the `transcribe` env. — all present (transformers, huggingface_hub, safetensors, sentencepiece, openai, httpx, pydantic, tqdm, faster_whisper, hf_xet).
- [x] `hf auth login` on Windows. — logged in (SSL_CERT_FILE fix applied).
- [x] Accept Aya Expanse 8B access terms on Hugging Face. — **MANUAL** (confirm the license is accepted on the model page with your account before serving).

## Phase 2 — WSL2 vLLM Environment

- [x] Install/verify WSL2. — Ubuntu 24.04.3 LTS starts, WSL version 2.
- [x] Confirm `nvidia-smi` sees the RTX 3070 inside Ubuntu. — verified (RTX 3070, 8 GB, driver 610.53).
- [x] Create dedicated uv venv (`~/local-translation-vllm/.venv`, Python 3.12). — uv 0.11.29, CPython 3.12.3.
- [x] Declare WSL dependencies (`requirements-vllm.txt`).
- [x] Install vLLM + bitsandbytes inside WSL. — vllm 0.25.1, bitsandbytes 0.49.2, torch 2.11.0+cu130 (CUDA True).
- [x] `hf auth login` inside WSL (separate from Windows). — **MANUAL** (needs your token; not logged in in WSL yet).
- [x] Confirm `import vllm; vllm.__version__`. — 0.25.1.

## Phase 3 — Safe GPU Operating Rules

- [x] Document/enforce the one-workload-at-a-time sequence (Whisper → release → vLLM → translate → stop → render). — see `docs/gpu_operating_rules.md`.
- [x] Verify `nvidia-smi` checks work on both Windows and WSL. — Windows verified (RTX 3070, 8 GB); WSL verified (RTX 3070 visible in Ubuntu).

## Phase 4 — Refactor Transcript Storage

- [x] Add `ARABIC_TRANSCRIPTS_DIR`, `ENGLISH_TRANSCRIPTS_DIR`, `TRANSLATION_RUNS_DIR` and create them. — in the config cell; `TRANSCRIPTS_DIR` kept as an alias to `transcript/arabic/`.
- [x] Point batch transcription output to `transcript/arabic/`. — via the `TRANSCRIPTS_DIR` alias; also releases the Whisper model at the end of the cell (GPU rule).
- [x] Point the final video cell to `transcript/english/`. — `TRANSCRIPT_FOLDER = r"transcript/english"`.
- [x] Confirm filename-stem matching still works (`input → arabic → english → output`). — matching is stem-based and unchanged; only the folder paths moved.

## Phase 5 — Prompt System

- [x] Keep canonical prompt (`prompts/islamic_translation_canonical.md`).
- [x] Create compact runtime prompt (`prompts/islamic_translation_runtime.md`).
- [x] Verify runtime prompt token size is ≈800–1,500 tokens. — was **1,118 tokens exact**; **1,349 tokens** after two rounds of Phase 15 tightening (Output Contract, then the Timestamp Contract fix below; chunk budget now ≤364 source tokens at the 3,072 limit). Still within target but with less headroom — worth revisiting if a future transcript needs a bigger chunk budget.
- [x] Add per-lecture context support (`translation_context/<name>.md`). — `CONTEXT_DIR` + `load_optional_context(stem)` (ignores `_`-prefixed files) + `translation_context/_TEMPLATE.md`.
- [x] Decide Aya prompt packaging (`single_user` vs `system`) after testing `apply_chat_template`. — **Decided: `AYA_PROMPT_MODE = "system"`**, confirmed on all 3 real transcripts (Phase 15): `single_user` mode transliterated the Arabic instead of translating it, wrapped output in markdown, and needed up to 13 retry-ladder attempts (recursing to single-block splits) to pass; `system` mode passed every file in 1-4 attempts. The runtime prompt's Output Contract was also tightened (no markdown, no transliteration, line must start with the `[` timestamp) — re-check meaning on the golden set during Phase 16.
- [x] **Real-data finding (Phase 15):** very short, fragment-only ASR blocks — one clause split across several ~1-second timestamp blocks, e.g. `1min.txt`'s `[00:28-00:31]`/`[00:31-00:32]`/`[00:32-00:33]` — tempted Aya to merge them into one output line, dropping timestamps (block-count validation error). Root cause: the runtime prompt's "you may move words between adjacent blocks" clause (Phase 5.2) was copied from the canonical prompt without its guard rails (`Do not move content across non-adjacent blocks`, `Keep short source blocks reasonably short`). Fixed by tightening the Timestamp Contract: every block gets its own line even if a fragment, never combine blocks, only a word or two may move across one boundary, plus a silent pre-response self-check. This reduced but did not eliminate first-attempt merges — the **Phase 12 retry ladder reliably recovers** the rest (reminder, or split-then-translate). Confirmed on all 3 real transcripts' first 8 blocks: `1min.txt` 2 attempts, `4min.txt` 1 attempt, `hard.txt` 4 attempts (needed a split) — all passed in `system` mode.

## Phase 6 — Translation Configuration Cell

- [x] Add config cell (backend, model id, vLLM URL/key, dirs, prompt paths). — new "Cell 4: Translation configuration" inserted before the final-video section, with a "Local translation" overview markdown cell above it.
- [x] Add run flags (`OVERWRITE_TRANSLATIONS`, `RESUME_TRANSLATIONS`, `MAX_RETRIES`). — plus `CONTINUE_ON_ERROR` for the later batch cell.
- [x] Set conservative `AYA_CONTEXT_LIMIT` and deterministic generation (`TEMPERATURE=0.0`, `TOP_P=1.0`). — **`AYA_CONTEXT_LIMIT = 2560`** (was `3072`, then `single_user`; see the real-batch finding below for why); also `AYA_PROMPT_MODE = "system"`.
- [x] **Real-batch finding:** Cell 7G raised `BadRequestError: ... maximum context length is 2560 tokens ... total of at least 2561` — the server was actually launched with the OOM-fallback `--max-model-len 2560` (`docs/wsl_vllm_setup.md` §5), but `AYA_CONTEXT_LIMIT` was still `3072`, so Cell 6 planned chunks against a context 512 tokens larger than the server really had (the Phase 12 reminder's ~41 extra tokens was a minor secondary contributor). Fixed: `AYA_CONTEXT_LIMIT = 2560`; `SAFETY_MARGIN_TOKENS` 128 → 160 (Cell 6) to absorb the reminder's overhead with room to spare; **`check_vllm_server` (Cell 7, run in 7B) now reads the server's live `max_model_len` and raises immediately if `AYA_CONTEXT_LIMIT` is set too high** — confirmed live that it catches the old 3072 value and passes cleanly at 2560, so this class of mismatch is now caught at the health check, not mid-batch. Re-verified end-to-end: all 25 chunks across all 3 real transcripts (`1min.txt`, `4min.txt`, `hard.txt`) translated successfully through the real retry ladder at the corrected limit, 0 unrecoverable failures. **Whenever the vLLM launch command changes `--max-model-len`, `AYA_CONTEXT_LIMIT` must be updated to match** — Cell 7B will now say so instead of failing deep in a batch.

## Phase 7 — Timestamp Parser

- [x] Implement `TimestampBlock` dataclass. — Cell 5 (frozen dataclass; exact timestamp string + text + source line number).
- [x] Implement `parse_timestamped_transcript` (exact timestamp preservation, reject malformed lines). — Cell 5; malformed lines raise `ValueError` with the line number; empty transcripts rejected.
- [x] Implement `blocks_to_text`. — Cell 5; round-trip (parse → rebuild) is lossless.
- [x] Add parser tests (standard, hour, Arabic punctuation, empty text, blank lines, malformed, duplicate, leading zeros). — inline self-tests run on every Cell 5 execution; all pass.

## Phase 8 — Token-Aware Chunking

- [x] Load the Aya tokenizer for planning only. — Cell 6; loads on Windows (gated download worked → license acceptance + `hf auth login` confirmed); clear "Aya is gated" error if it can't.
- [x] Implement `count_tokens` and `apply_chat_template` prompt measurement. — `count_tokens` + `measure_prompt_tokens` (fully rendered chat prompt); runtime prompt = 1,118 tokens exact.
- [x] Implement `estimate_output_tokens`. — `min(1400, max(256, 1.8·src + 128))`.
- [x] Implement chunk-budget math with safety margin (fail if budget ≤ 0). — `plan_source_budget` (binary search, since the output reserve grows with source size); raises `PromptTooLargeError` with the full token breakdown. At the 3,072 limit: ≤447 source tokens/chunk.
- [x] Implement greedy block chunking (never split a block initially). — `build_chunks`; an oversized single block is refused loudly, never split.
- [x] Implement small continuity-context builder (≤200–400 tokens, marked "do not output"). — `build_continuity_context`; strips timestamps, shrinks (then returns "") rather than exceed 400 tokens.

## Phase 9 — Shared Backend Interface

- [x] Implement `TranslationResult` dataclass. — Cell 7 (adds a truncation warning when `finish_reason == "length"`).
- [x] Define `TranslationBackend` protocol. — Cell 7.
- [x] Keep the batch pipeline backend-agnostic (call only `translate_chunk`). — interface in place; the Phase 14 batch cell must only call `translate_chunk()`.

## Phase 10 — Aya Expanse 8B via vLLM

- [x] Start vLLM server in WSL with conservative 4-bit settings. — **MANUAL** (needs the WSL-side `hf auth login` first; launch command in `docs/wsl_vllm_setup.md` §5 and the Cell 7 markdown).
- [x] Add OOM fallback launch command (lower `max-model-len`, cpu-offload, enforce-eager). — documented in `docs/wsl_vllm_setup.md` §5.
- [x] Health check the API from Windows (`client.models.list()`). — implemented as `check_vllm_server` + Cell 7B (also verifies the model id is actually served); raises the clear "cannot reach" message when the server is down; **passed live** against the running server (serving `CohereLabs/aya-expanse-8b`).
- [x] Implement `AyaVLLMBackend.translate_chunk`. — Cell 7; uses the same `build_chat_messages` packaging the chunk planner measures, so the token budget holds for the request actually sent.
- [x] Confirm deterministic generation policy (temp 0, one request at a time). — `TEMPERATURE=0.0`, `TOP_P=1.0`, sequential calls, client `max_retries=0` (retries are a Phase 12 pipeline decision), server `--max-num-seqs 1`.

## Phase 11 — Strict Validation

- [x] Implement `extract_timestamps`. — Cell 7C (lenient scan, used to diagnose unparseable output).
- [x] Implement `ValidationResult`. — Cell 7C.
- [x] Implement `validate_translation` (exact timestamp equality, block count, empty blocks, Arabic-script warning). — Cell 7C; also errors on unparseable output and on text invented for empty source blocks; empty blocks are only an error when the source block is non-empty.
- [x] Add extra warning checks (leading "Translation:", markdown/code fences, refusals, length anomalies, etc.). — Cell 7C: preamble, headings, code fences, notes, refusal language, echoed (untranslated) blocks, repeated consecutive lines, very short/long output, possible truncated final line. Self-tests run on every cell execution.

## Phase 12 — Retry & Recovery

- [x] Retry 1: resend chunk with a stronger timestamp-contract reminder. — Cell 7D (`translate_with_retries`; reminder appended to the runtime prompt, same packaging).
- [x] Retry 2: split failed chunk at a timestamp boundary and translate independently. — Cell 7D; halves recurse down to single blocks, each with its own reminder retry; continuity carried from the first half to the second.
- [x] Final failure: save raw responses to `translation_runs/`, mark file incomplete, never write partial output. — `ChunkTranslationError` carries every attempt (raw output + validation); the batch cell saves `chunk_NNNN_failed.json` + `INCOMPLETE` marker. `MAX_RETRIES` gates the ladder (0 = single attempt, 1 = + reminder, 2 = + split). Self-tests use a scripted fake backend (no server needed).

## Phase 13 — Checkpointing & Resume

- [x] Write a JSON checkpoint per validated chunk (hashes, model/backend, generation, timestamps, text, validation). — Cell 7E: `translation_runs/<model>/<stem>/chunk_NNNN.json`, atomic writes; plus `run.json` per lecture (source/prompt/context hashes, generation policy, chunk plan).
- [x] Implement resume rule (reuse only when source/prompt/context hashes + model/backend/generation match). — `checkpoint_key` + `load_chunk_checkpoint`; also matches the continuity-context hash (stricter than the plan) so re-translated earlier chunks invalidate later checkpoints; corrupt/invalid checkpoints are never trusted. Self-tests run on every cell execution.

## Phase 14 — Batch Translation Cell

- [x] Implement `translate_all_transcripts` (skip existing unless overwrite, per-file chunking, continuity carry-over). — Cell 7G; health-checks the server first, requires a passing smoke test (Phase 15 gate), prints per-chunk progress + a batch summary (translated/skipped/incomplete, reused chunks, retries, warnings).
- [x] Run whole-file validation before writing. — failure saves `final_validation_failed.json`, marks `INCOMPLETE`, never writes the English file.
- [x] Use atomic writes via a `.txt.tmp` temporary file. — `.txt.tmp` → `replace()`; verified end-to-end with scripted backends (clean run, skip, resume, overwrite, failure, continue-on-error, interrupted-run resume).

## Phase 15 — Smoke-Test Cell

- [x] Add a cell that translates 5–10 blocks from one file, prints token count, source, output, and validation. — Cell 7F: picks the first `transcript/arabic/*.txt` (or `SMOKE_TEST_FILE`), takes a token-safe first chunk of ≤`SMOKE_TEST_BLOCKS` blocks, prints rendered prompt tokens, source, and — per attempt — raw output + validation + usage/tok/s; `SMOKE_TEST_COMPARE_MODES = True` rehearses the other prompt packaging too and compares attempt counts.
- [x] Enforce: do not run batch until the smoke test passes. — Cell 7F sets `SMOKE_TEST_PASSED = True` only when the configured mode succeeds (raises otherwise); Cell 7G refuses to run unless the flag is set in the current kernel session.
- [x] **Design change:** the smoke test runs the chunk through `translate_with_retries` (Phase 12), the same call the batch cell makes, instead of one bare attempt. Reason: the first version did a single unretried call and failed on a chunk (the fragment-merge case above) that the retry ladder actually recovers — a false negative that would have blocked the batch on a chunk it could handle fine. **Verified live** on all 3 real transcripts (first 8 blocks each) in `system` mode: `1min.txt` passed in 2 attempts, `4min.txt` in 1, `hard.txt` in 4 (needed the split retry) — all unlocked `SMOKE_TEST_PASSED`.

## Phase 16 — Quality Evaluation

- [ ] Evaluate Aya against the golden set with consistent source/context/generation.
- [ ] Score the 10 manual categories on a 0–2 scale.
- [ ] Collect automatic statistics (valid-chunk ratio, retries, timestamp mismatches, tokens/s, peak VRAM, etc.).
- [ ] Apply the decision rule (promote Aya only if it passes; otherwise evaluate an alternative model).

## Phase 17 — Runtime Prompt Testing

- [ ] Test Variant A (full canonical prompt).
- [ ] Test Variant B (compact runtime prompt).
- [ ] Test Variant C (compact prompt + lecture context).
- [ ] Promote the runtime prompt only after it matches canonical on the golden set.

## Phase 18 — Operational Workflow

- [ ] Document the 9-step run workflow in the notebook (nvidia-smi → transcribe → release Whisper → start vLLM → smoke test → batch → summary → stop vLLM → render).

## Phase 19 — Error Handling

- [ ] Handle vLLM server unavailable with a clear message.
- [ ] Handle Aya access denied.
- [ ] Handle CUDA OOM (with the suggested mitigation order).
- [ ] Handle prompt-too-large (report full token budget breakdown, never truncate silently).
- [x] Handle invalid model output (save artifacts, retry or mark incomplete). — Cells 7D + 7G (attempt history saved to `chunk_NNNN_failed.json`, `INCOMPLETE` marker, `CONTINUE_ON_ERROR`).
- [x] Handle interrupted translation (resume from checkpoints). — Cells 7E + 7G (`RESUME_TRANSLATIONS` + per-chunk checkpoints).
- [x] Handle `AYA_CONTEXT_LIMIT` / server `--max-model-len` mismatch (not in the original plan, added after a real `BadRequestError` in Cell 7G). — `check_vllm_server` (Cell 7B) now compares `AYA_CONTEXT_LIMIT` against the server's live `max_model_len` and raises a clear, actionable error naming both values if the config assumes more context than the server actually has.

## Phase 20 — Security & Reproducibility

- [x] Add `.gitignore` (ignores `.env`, `transcript/`, `output/`, `translation_runs/`, media, checkpoints).
- [ ] Pin `requirements-windows.txt` after a working prototype (`pip freeze`).
- [ ] Pin `requirements-vllm.txt` versions after a working prototype (`uv pip freeze`).
- [ ] Pin the Hugging Face model revision/commit hash.
- [ ] Record prompt SHA-256 hashes per run.
- [ ] Confirm no HF tokens are stored in notebook source.

---

## Milestones

### Milestone 1 — Folder safety
- [x] Arabic transcripts written to `transcript/arabic/`.
- [~] English transcripts written to `transcript/english/`. — writer implemented (Cell 7G, atomic `.txt.tmp` → `.txt`); `transcript/arabic/` is still empty, so transcribe first, then run the batch.
- [x] Original Arabic files never overwritten. — Arabic and English use separate folders.
- [x] Final rendering reads only English files. — `TRANSCRIPT_FOLDER = transcript/english`.

### Milestone 2 — Parsing and validation
- [x] Timestamp parser preserves exact strings. — no normalization; leading zeros and duplicates preserved (Cell 5 self-tests).
- [x] Invalid transcript lines rejected. — `ValueError` with line number.
- [x] Validator detects changed timestamps. — Cell 7C (exact sequence equality; reordering also caught).
- [x] Validator detects changed block count. — Cell 7C.
- [x] Unit tests pass. — parser self-tests (Cell 5) + validator (7C), retry (7D), and checkpoint (7E) self-tests all pass; batch flow verified end-to-end with scripted backends.

### Milestone 3 — Aya smoke test
- [x] WSL sees the RTX 3070. — verified in Phase 2.
- [x] Aya access approved. — confirmed by a successful gated tokenizer download on Windows (acceptance is account-level, so WSL works too once its `hf auth login` is done).
- [x] vLLM starts with 4-bit quantization. — server running in WSL2, confirmed serving `CohereLabs/aya-expanse-8b`.
- [x] Windows notebook reaches the API. — `check_vllm_server` passed live from Windows.
- [x] One chunk translates successfully. — live smoke test: 8-block Arabic sample translated in `system` mode (~27 tok/s, ~12 s); re-run Cell 7F on a real lecture transcript once `transcript/arabic/` is populated.
- [x] Timestamp validation passes. — all 8 timestamps exact, 0 errors, 0 warnings on the live smoke output.

### Milestone 4 — Aya batch translation
- [ ] Token-aware chunking works. — needs the real server run.
- [ ] Continuity context is not repeated. — needs the real server run (validator + smoke test will catch repeats).
- [~] Checkpoints written. — implemented (Cell 7E) and verified with scripted backends; real-server run pending.
- [~] Interrupted runs resume. — implemented + verified with scripted backends (mid-run failure resumes from chunk checkpoints); real-server run pending.
- [~] Final transcript passes whole-file validation. — wired into Cell 7G (validates before writing); real-lecture run pending.

### Milestone 5 — Production workflow
- [ ] Prompt versions pinned.
- [ ] Model revisions pinned.
- [ ] Rendering succeeds from translated transcripts.
- [ ] At least one full lecture completes end to end.
- [ ] Human review confirms meaning and terminology.

---

## Recommended Implementation Order

1. [x] Refactor transcript directories (Phase 4).
2. [x] Add and test the timestamp parser (Phase 7).
3. [x] Add and test strict validation (Phase 11).
4. [x] Create the compact Aya runtime prompt (Phase 5).
5. [x] Create the backend interface (Phase 9).
6. [x] Start Aya via vLLM in WSL (Phase 10). — server up and health-checked from Windows.
7. [x] Translate and validate one small chunk (Phase 15). — smoke test passed live on all 3 real transcripts (Cell 7F, retry-aware); prompt packaging decided (`system`); Output Contract and Timestamp Contract both tightened based on real failures (transliteration/markdown, then fragment-block merging).
8. [x] Add token-aware chunking (Phase 8).
9. [x] Add checkpointing and batch translation (Phases 13–14).
10. [ ] Run Aya on the golden set (Phase 16).
11. [ ] Run one full lecture (Phase 18).
12. [ ] Update the notebook's top documentation (Phase 3.1).
13. [ ] Pin dependencies and model revisions (Phase 20).
