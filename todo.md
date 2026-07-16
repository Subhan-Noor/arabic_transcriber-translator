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
- [x] Verify runtime prompt token size is ≈800–1,500 tokens. — **1,118 tokens exact** by the Aya tokenizer (Cell 6); heuristic estimate was ≈1,020–1,327.
- [x] Add per-lecture context support (`translation_context/<name>.md`). — `CONTEXT_DIR` + `load_optional_context(stem)` (ignores `_`-prefixed files) + `translation_context/_TEMPLATE.md`.
- [~] Decide Aya prompt packaging (`single_user` vs `system`) after testing `apply_chat_template`. — both modes render cleanly through `apply_chat_template` (overhead 1,165 vs 1,166 tokens; Cell 6 diagnostics); default stays `AYA_PROMPT_MODE = "single_user"`; **final decision pending the smoke test (Phase 15)**, which needs the running vLLM server.

## Phase 6 — Translation Configuration Cell

- [x] Add config cell (backend, model id, vLLM URL/key, dirs, prompt paths). — new "Cell 4: Translation configuration" inserted before the final-video section, with a "Local translation" overview markdown cell above it.
- [x] Add run flags (`OVERWRITE_TRANSLATIONS`, `RESUME_TRANSLATIONS`, `MAX_RETRIES`). — plus `CONTINUE_ON_ERROR` for the later batch cell.
- [x] Set conservative `AYA_CONTEXT_LIMIT` and deterministic generation (`TEMPERATURE=0.0`, `TOP_P=1.0`). — `AYA_CONTEXT_LIMIT = 3072`; also `AYA_PROMPT_MODE = "single_user"`.

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

- [ ] Start vLLM server in WSL with conservative 4-bit settings. — **MANUAL** (needs the WSL-side `hf auth login` first; launch command in `docs/wsl_vllm_setup.md` §5 and the Cell 7 markdown).
- [x] Add OOM fallback launch command (lower `max-model-len`, cpu-offload, enforce-eager). — documented in `docs/wsl_vllm_setup.md` §5.
- [~] Health check the API from Windows (`client.models.list()`). — implemented as `check_vllm_server` + Cell 7B (also verifies the model id is actually served); confirmed it raises the clear "cannot reach" message when the server is down; needs the running server to pass.
- [x] Implement `AyaVLLMBackend.translate_chunk`. — Cell 7; uses the same `build_chat_messages` packaging the chunk planner measures, so the token budget holds for the request actually sent.
- [x] Confirm deterministic generation policy (temp 0, one request at a time). — `TEMPERATURE=0.0`, `TOP_P=1.0`, sequential calls, client `max_retries=0` (retries are a Phase 12 pipeline decision), server `--max-num-seqs 1`.

## Phase 11 — Strict Validation

- [ ] Implement `extract_timestamps`.
- [ ] Implement `ValidationResult`.
- [ ] Implement `validate_translation` (exact timestamp equality, block count, empty blocks, Arabic-script warning).
- [ ] Add extra warning checks (leading "Translation:", markdown/code fences, refusals, length anomalies, etc.).

## Phase 12 — Retry & Recovery

- [ ] Retry 1: resend chunk with a stronger timestamp-contract reminder.
- [ ] Retry 2: split failed chunk at a timestamp boundary and translate independently.
- [ ] Final failure: save raw responses to `translation_runs/`, mark file incomplete, never write partial output.

## Phase 13 — Checkpointing & Resume

- [ ] Write a JSON checkpoint per validated chunk (hashes, model/backend, generation, timestamps, text, validation).
- [ ] Implement resume rule (reuse only when source/prompt/context hashes + model/backend/generation match).

## Phase 14 — Batch Translation Cell

- [ ] Implement `translate_all_transcripts` (skip existing unless overwrite, per-file chunking, continuity carry-over).
- [ ] Run whole-file validation before writing.
- [ ] Use atomic writes via a `.txt.tmp` temporary file.

## Phase 15 — Smoke-Test Cell

- [ ] Add a cell that translates 5–10 blocks from one file, prints token count, source, output, and validation.
- [ ] Enforce: do not run batch until the smoke test passes.

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
- [ ] Handle invalid model output (save artifacts, retry or mark incomplete).
- [ ] Handle interrupted translation (resume from checkpoints).

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
- [ ] English transcripts written to `transcript/english/`. — folder + render path are wired; the actual writer arrives with the batch translation cell (Phase 14).
- [x] Original Arabic files never overwritten. — Arabic and English use separate folders.
- [x] Final rendering reads only English files. — `TRANSCRIPT_FOLDER = transcript/english`.

### Milestone 2 — Parsing and validation
- [x] Timestamp parser preserves exact strings. — no normalization; leading zeros and duplicates preserved (Cell 5 self-tests).
- [x] Invalid transcript lines rejected. — `ValueError` with line number.
- [ ] Validator detects changed timestamps. — Phase 11.
- [ ] Validator detects changed block count. — Phase 11.
- [~] Unit tests pass. — parser self-tests pass (Cell 5); validator tests arrive with Phase 11.

### Milestone 3 — Aya smoke test
- [x] WSL sees the RTX 3070. — verified in Phase 2.
- [x] Aya access approved. — confirmed by a successful gated tokenizer download on Windows (acceptance is account-level, so WSL works too once its `hf auth login` is done).
- [ ] vLLM starts with 4-bit quantization.
- [ ] Windows notebook reaches the API. — run Cell 7B once the server is up.
- [ ] One chunk translates successfully.
- [ ] Timestamp validation passes.

### Milestone 4 — Aya batch translation
- [ ] Token-aware chunking works.
- [ ] Continuity context is not repeated.
- [ ] Checkpoints written.
- [ ] Interrupted runs resume.
- [ ] Final transcript passes whole-file validation.

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
3. [ ] Add and test strict validation (Phase 11).
4. [x] Create the compact Aya runtime prompt (Phase 5).
5. [x] Create the backend interface (Phase 9).
6. [ ] Start Aya via vLLM in WSL (Phase 10). — **MANUAL**: WSL `hf auth login`, then the launch command in `docs/wsl_vllm_setup.md` §5.
7. [ ] Translate and validate one small chunk (Phase 15).
8. [x] Add token-aware chunking (Phase 8).
9. [ ] Add checkpointing and batch translation (Phases 13–14).
10. [ ] Run Aya on the golden set (Phase 16).
11. [ ] Run one full lecture (Phase 18).
12. [ ] Update the notebook's top documentation (Phase 3.1).
13. [ ] Pin dependencies and model revisions (Phase 20).
