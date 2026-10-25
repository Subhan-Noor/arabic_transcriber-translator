# Optional folder-based translation

The notebooks do not call an LLM API. These files are instructions you can give
to a coding assistant that has access to your local project folder.

1. Transcribe first, inspect quality reports, and redact bad audio before translating.
2. Use `codex/prompt.txt` as the task and `codex/system_prompt.md` as its translation rules.
3. Read timestamped sources from `transcript/arabic/<name>.txt`; exclude
   `.plain.txt`, `.flagged.txt`, SRT files and backups.
4. Write to matching `transcript/english/<name>.txt` files. Empty stubs may be
   filled; existing nonblank translations require explicit overwrite instructions.
5. Optionally supply references in `approved_translations/` and a glossary.
   Neither is required or bundled. Newly generated translations are not approved references.
6. Review the English, then run the SRT or final-video cell in either local notebook.

Preserve every timestamp, the block count, and the meaning assigned to each block.
Do not shift content between blocks. Preserve audio-cutout placeholders verbatim.
This is the same timestamp policy as `translation_prompt.md`, the manual chat prompt.

Keep references private if appropriate; `approved_translations/` is ignored by Git.
