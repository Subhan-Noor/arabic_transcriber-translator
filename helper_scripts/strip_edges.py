"""
Strip silent intro/outro subtitle fluff (and matching audio) from a named set.

Each media stem shares three files:
  input/<name>.mp3
  transcript/english/<name>.txt
  transcript/arabic/<name>.txt

Edit TRIMS below (one entry per stem), set DRY_RUN = False, then run:
  python helper_scripts/strip_edges.py

Or one-off without editing the config:
  python helper_scripts/strip_edges.py "fiqh (1)" --start 01:32
  python helper_scripts/strip_edges.py "fiqh (2)" --end 1:12:21
  python helper_scripts/strip_edges.py "fiqh (1)" --drop-start-lines 3

Preview first/last English lines to choose cut points:
  python helper_scripts/strip_edges.py --preview
  python helper_scripts/strip_edges.py --preview "fiqh (1)"
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ── customize this ────────────────────────────────────────────────────────────
# Per-stem cuts. Times are absolute in the *current* files (MM:SS or H:MM:SS).
# Use any combination of:
#   start              keep from this timestamp onward (strip earlier audio/cues)
#   end                keep until this timestamp (strip later audio/cues)
#   drop_start_lines   drop N cues from the start; cut audio at the next cue's start
#   drop_end_lines     drop N cues from the end; cut audio at the last kept cue's end
#
# Line counts are taken from transcript/english/<name>.txt.
TRIMS: dict[str, dict] = {}  # Optional explicit per-file presets; none ship with the project.

DRY_RUN = True          # set False to write changes
BACKUP = True           # copy originals to *.bak next to each file before overwrite
AUDIO_REENCODE = False  # False = stream-copy (fast); True = re-encode MP3 (cleaner cuts)
# ──────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "../input"
ENGLISH_DIR = ROOT / "../transcript" / "english"
ARABIC_DIR = ROOT / "../transcript" / "arabic"

TS_LINE = re.compile(r"^\[(\d+(?::\d{2})+)\s*-\s*(\d+(?::\d{2})+)\]\s*(.*)$")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def parse_ts(s: str | None) -> float | None:
    """Parse 'SS', 'M:SS', 'MM:SS', or 'H:MM:SS' into seconds."""
    if s is None or str(s).strip() == "":
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from transcriber_utils import parse_timestamp
    return parse_timestamp(s)



def fmt_ts(t: float) -> str:
    """Format seconds as MM:SS or H:MM:SS (matches notebook style)."""
    s = max(0, int(round(t)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def read_cues(path: Path) -> list[Cue]:
    text = path.read_text(encoding="utf-8-sig")
    cues: list[Cue] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = TS_LINE.match(line)
        if not m:
            continue
        start_s, end_s, body = m.groups()
        cues.append(Cue(parse_ts(start_s), parse_ts(end_s), body.strip()))
    return cues


def write_cues(path: Path, cues: list[Cue]) -> None:
    lines = [f"[{fmt_ts(c.start)}-{fmt_ts(c.end)}] {c.text}" for c in cues]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def find_audio(stem: str) -> Path | None:
    for ext in AUDIO_EXTS:
        candidate = INPUT_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def audio_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def resolve_cuts(
    english_cues: list[Cue],
    *,
    start: str | None = None,
    end: str | None = None,
    drop_start_lines: int = 0,
    drop_end_lines: int = 0,
) -> tuple[float, float | None]:
    """Return (keep_from, keep_until) in seconds. keep_until=None means end of file."""
    keep_from = 0.0
    keep_until: float | None = None

    if drop_start_lines < 0 or drop_end_lines < 0:
        raise ValueError("Line counts must be nonnegative.")

    if drop_start_lines:
        if drop_start_lines >= len(english_cues):
            raise ValueError(
                f"drop_start_lines={drop_start_lines} but only {len(english_cues)} cues"
            )
        keep_from = max(keep_from, english_cues[drop_start_lines].start)

    if drop_end_lines:
        if drop_end_lines >= len(english_cues):
            raise ValueError(
                f"drop_end_lines={drop_end_lines} but only {len(english_cues)} cues"
            )
        keep_until = english_cues[-(drop_end_lines + 1)].end

    if start is not None:
        keep_from = max(keep_from, parse_ts(start))
    if end is not None:
        end_sec = parse_ts(end)
        keep_until = end_sec if keep_until is None else min(keep_until, end_sec)

    if keep_until is not None and keep_until <= keep_from:
        raise ValueError(
            f"Invalid window: keep_from={fmt_ts(keep_from)} >= keep_until={fmt_ts(keep_until)}"
        )
    return keep_from, keep_until


def trim_cues(cues: list[Cue], keep_from: float, keep_until: float | None) -> list[Cue]:
    """Drop cues outside the window; clip partial overlaps; shift by -keep_from."""
    out: list[Cue] = []
    for c in cues:
        start, end = c.start, c.end
        if end <= keep_from:
            continue
        if keep_until is not None and start >= keep_until:
            continue
        if start < keep_from:
            start = keep_from
        if keep_until is not None and end > keep_until:
            end = keep_until
        if end < start:
            continue
        out.append(Cue(start - keep_from, end - keep_from, c.text))
    return out


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def trim_audio(src: Path, keep_from: float, keep_until: float | None, dest: Path) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if keep_from > 0:
        cmd += ["-ss", f"{keep_from:.3f}"]
    if keep_until is not None:
        cmd += ["-to", f"{keep_until:.3f}"]
    if AUDIO_REENCODE or src.suffix.lower() != ".mp3":
        # Re-encode when requested, or when stream-copy may not apply cleanly.
        if src.suffix.lower() == ".mp3":
            cmd += ["-c:a", "libmp3lame", "-q:a", "2"]
        else:
            cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(dest))
    subprocess.run(cmd, check=True)


def preview_stem(stem: str, head: int = 8, tail: int = 5) -> None:
    eng = ENGLISH_DIR / f"{stem}.txt"
    if not eng.exists():
        print(f"  missing: {eng}")
        return
    cues = read_cues(eng)
    audio = find_audio(stem)
    dur = f"{fmt_ts(audio_duration(audio))}" if audio else "no audio"
    print(f"\n=== {stem}  ({len(cues)} cues, audio {dur}) ===")
    print("--- first ---")
    for i, c in enumerate(cues[:head], 1):
        print(f"  {i:3d}  [{fmt_ts(c.start)}-{fmt_ts(c.end)}] {c.text}")
    if len(cues) > head + tail:
        print("  ...")
    print("--- last ---")
    start_i = max(head, len(cues) - tail)
    for i, c in enumerate(cues[start_i:], start_i + 1):
        print(f"  {i:3d}  [{fmt_ts(c.start)}-{fmt_ts(c.end)}] {c.text}")


def process_stem(
    stem: str,
    *,
    start: str | None = None,
    end: str | None = None,
    drop_start_lines: int = 0,
    drop_end_lines: int = 0,
) -> None:
    eng_path = ENGLISH_DIR / f"{stem}.txt"
    ar_path = ARABIC_DIR / f"{stem}.txt"
    audio_path = find_audio(stem)

    missing = []
    if not eng_path.exists():
        missing.append(str(eng_path))
    if not ar_path.exists():
        missing.append(str(ar_path))
    if audio_path is None:
        missing.append(str(INPUT_DIR / f"{stem}.{{mp3,...}}"))
    if missing:
        raise FileNotFoundError("Missing:\n  " + "\n  ".join(missing))

    eng_cues = read_cues(eng_path)
    ar_cues = read_cues(ar_path)
    keep_from, keep_until = resolve_cuts(
        eng_cues,
        start=start,
        end=end,
        drop_start_lines=drop_start_lines,
        drop_end_lines=drop_end_lines,
    )

    if keep_from <= 0 and keep_until is None:
        print(f"skip {stem}: nothing to strip")
        return

    if keep_from >= audio_duration(audio_path):
        raise ValueError("Start cut is at or beyond the audio duration.")
    new_eng = trim_cues(eng_cues, keep_from, keep_until)
    new_ar = trim_cues(ar_cues, keep_from, keep_until)
    dur = audio_duration(audio_path)
    until_label = fmt_ts(keep_until) if keep_until is not None else fmt_ts(dur)

    print(f"\n{stem}")
    print(f"  window     : {fmt_ts(keep_from)} -> {until_label}")
    print(f"  english    : {len(eng_cues)} -> {len(new_eng)} cues")
    print(f"  arabic     : {len(ar_cues)} -> {len(new_ar)} cues")
    print(f"  audio      : {audio_path.name}  ({fmt_ts(dur)} -> ~{fmt_ts((keep_until or dur) - keep_from)})")

    if new_eng:
        print(f"  new first  : [{fmt_ts(new_eng[0].start)}-{fmt_ts(new_eng[0].end)}] {new_eng[0].text}")
        print(f"  new last   : [{fmt_ts(new_eng[-1].start)}-{fmt_ts(new_eng[-1].end)}] {new_eng[-1].text}")

    if DRY_RUN:
        print("  dry-run only (set DRY_RUN = False, or pass --apply)")
        return

    if BACKUP:
        for p in (eng_path, ar_path, audio_path):
            bak = backup(p)
            print(f"  backup     : {bak.name}")

    # Prepare audio and both transcripts before replacing anything. Keep rollback
    # copies even with --no-backup so an ordinary replacement failure is recoverable.
    with tempfile.TemporaryDirectory(prefix="_trim_", dir=audio_path.parent) as temp:
        work = Path(temp)
        staged_audio = work / ("trimmed" + audio_path.suffix)
        trim_audio(audio_path, keep_from, keep_until, staged_audio)
        staged_eng, staged_ar = work / "english.txt", work / "arabic.txt"
        write_cues(staged_eng, new_eng)
        write_cues(staged_ar, new_ar)
        replacements = [(audio_path, staged_audio), (eng_path, staged_eng), (ar_path, staged_ar)]
        originals = {}
        for index, (dest, _) in enumerate(replacements):
            original = work / f"original_{index}"
            shutil.copy2(dest, original)
            originals[dest] = original
        replaced = []
        try:
            for dest, staged in replacements:
                # Stage beside the destination so os.replace stays on one filesystem.
                sibling = dest.with_name(dest.name + ".trim.tmp")
                try:
                    shutil.copy2(staged, sibling)
                    os.replace(sibling, dest)
                    replaced.append(dest)
                finally:
                    sibling.unlink(missing_ok=True)
        except Exception:
            for dest in reversed(replaced):
                shutil.copy2(originals[dest], dest)
            raise

    print("  done")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Strip intro/outro from matching audio + English/Arabic transcripts."
    )
    p.add_argument(
        "stem",
        nargs="?",
        help='Media stem, e.g. "fiqh (1)". Omit to process all entries in TRIMS.',
    )
    p.add_argument("--start", help="Keep from this timestamp (MM:SS or H:MM:SS).")
    p.add_argument("--end", help="Keep until this timestamp.")
    p.add_argument(
        "--drop-start-lines",
        type=int,
        default=0,
        help="Drop N English cues from the start; cut audio at the next cue.",
    )
    p.add_argument(
        "--drop-end-lines",
        type=int,
        default=0,
        help="Drop N English cues from the end; cut audio at the last kept cue.",
    )
    p.add_argument(
        "--preview",
        nargs="?",
        const="*",
        metavar="STEM",
        help="Show first/last English cues (all stems, or one STEM).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write changes even if DRY_RUN is True in the script.",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write .bak copies.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    global DRY_RUN, BACKUP

    args = build_parser().parse_args(argv)

    if args.apply:
        DRY_RUN = False
    if args.no_backup:
        BACKUP = False

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg/ffprobe not found on PATH.", file=sys.stderr)
        return 1

    if args.preview is not None:
        if args.preview == "*":
            stems = sorted({p.stem for p in ENGLISH_DIR.glob("*.txt")})
        else:
            stems = [args.preview]
        for stem in stems:
            preview_stem(stem)
        return 0

    jobs: list[tuple[str, dict]] = []
    if args.stem:
        jobs.append(
            (
                args.stem,
                {
                    "start": args.start,
                    "end": args.end,
                    "drop_start_lines": args.drop_start_lines,
                    "drop_end_lines": args.drop_end_lines,
                },
            )
        )
        if not any(
            [
                args.start,
                args.end,
                args.drop_start_lines,
                args.drop_end_lines,
            ]
        ):
            print(
                "Provide --start/--end and/or --drop-start-lines/--drop-end-lines, "
                "or put the stem in TRIMS and run without CLI cut flags.",
                file=sys.stderr,
            )
            return 1
    else:
        if not TRIMS:
            print(
                "TRIMS is empty. Edit the config at the top of strip_edges.py, "
                "or pass a stem plus cut flags.\n"
                "Tip: python helper_scripts/strip_edges.py --preview",
                file=sys.stderr,
            )
            return 1
        for stem, cfg in TRIMS.items():
            jobs.append((stem, cfg))

    for stem, cfg in jobs:
        try:
            process_stem(
                stem,
                start=cfg.get("start"),
                end=cfg.get("end"),
                drop_start_lines=int(cfg.get("drop_start_lines") or 0),
                drop_end_lines=int(cfg.get("drop_end_lines") or 0),
            )
        except Exception as exc:
            print(f"ERROR {stem}: {exc}", file=sys.stderr)
            return 1

    if DRY_RUN:
        print("\nDry run only. Re-run with --apply (or set DRY_RUN = False) to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
