#!/usr/bin/env python
"""Replace the subtitles inside bad-audio regions with a placeholder message.

Reads the confidence reports written by the transcribe cell
(`transcript/arabic/<name>.flagged.txt`), collects every flagged region into one
review file, and — only for the regions you approve — deletes the transcript
lines in that region and puts a single line in their place saying the audio cut
out.

Two steps, so nothing is ever changed without your say-so:

    python helper_scripts/redact_flagged.py review --interactive
        Walk through each flagged region, see the lines that would be removed,
        and approve or reject it. Decisions are stored in transcript/redactions.txt
        and are kept when you re-run `review` later.

    python helper_scripts/redact_flagged.py apply --dry-run
        Show exactly what would change.

    python helper_scripts/redact_flagged.py apply
        Write the changes. The untouched original is kept as <name>.txt.bak.

Run `review` without --interactive to just build the file and edit it by hand;
flip `approve: no` to `yes`, and adjust `region:` times or `message:` freely.
You can also add a stanza by hand for a bad region the detector missed.

A flagged region is usually a single transcript line spanning the whole bad
stretch, so redacting it is all-or-nothing for that line — there's no way to
drop half a line. If real speech resumes partway through the region (audio
"comes back" before the region ends), narrowing `region:` will not save those
words: it either still removes the whole line, or leaves the whole line
(garbled lead-in included) untouched. To actually keep them, add one or more
`keep:` lines to the stanza:

    keep: 7:02-7:11 | إلى أن يموت طيب دية المرأة على النفس من دية الرجل

This carves that sub-range out of the placeholder and inserts it verbatim with
its own timestamp instead. `helper_scripts/probe_region.py <media> <start>
<end>` re-transcribes just that window with word-level timestamps so you can
see exactly where recoverable speech begins, to fill in `keep:` accurately
rather than guessing.

Both transcript/arabic/ and transcript/english/ are updated when present, so
redacting before translating keeps the garbled text out of the translation, and
the placeholder survives into the final subtitles. Re-applying is a no-op.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover - older interpreters / odd streams
    pass

ROOT = Path(__file__).resolve().parents[1]
ARABIC_DIR = ROOT / "transcript/arabic"
ENGLISH_DIR = ROOT / "transcript/english"
REVIEW_FILE = ROOT / "transcript/redactions.txt"

# The subtitle shown in place of the removed lines.
DEFAULT_MESSAGE = "[Audio cuts out — this section could not be transcribed]"

# A transcript line is removed when at least this much of it lies inside the
# approved region, so a line that merely clips the boundary is left alone.
OVERLAP_MIN = 0.5

_TS = r"\d+(?::\d{2})+"
_TS_LINE = re.compile(rf"\[({_TS})\s*-\s*({_TS})\]\s*(.*)")
_FLAGGED_LINE = re.compile(rf"\[({_TS})-({_TS})\]\s*(SEVERE|warning)\s*:\s*(.*)", re.I)
_REPORT_FOR = re.compile(r"^#\s*Confidence report for\s+(.+?)\s*$")
_STANZA = re.compile(r"^---\s*(.+?)\s*\|\s*(" + _TS + r")\s*-\s*(" + _TS + r")\s*\|\s*(\S+)")
_RANGE = re.compile(rf"({_TS})\s*-\s*({_TS})")
_KEEP_LINE = re.compile(rf"^({_TS})\s*-\s*({_TS})\s*\|\s*(.*)$")


# ── timestamps ───────────────────────────────────────────────────────────────
def parse_ts(s: str) -> float:
    nums = [float(p) for p in str(s).strip().split(":")]
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


def fmt_ts(t: float) -> str:
    s = int(round(t))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# ── flagged reports ──────────────────────────────────────────────────────────
def load_flagged(only: str | None = None) -> list[dict]:
    """Every region from every transcript/arabic/*.flagged.txt."""
    regions = []
    for report in sorted(ARABIC_DIR.glob("*.flagged.txt")):
        report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
        stem = report.name[: -len(".flagged.txt")]
        if only and stem != only:
            continue
        media = stem
        for raw in report.read_text(encoding="utf-8").splitlines():
            hit = _REPORT_FOR.match(raw)
            if hit:
                media = hit.group(1)
                continue
            m = _FLAGGED_LINE.match(raw.strip())
            if m:
                regions.append({
                    "stem": stem,
                    "media": media,
                    "start": parse_ts(m.group(1)),
                    "end": parse_ts(m.group(2)),
                    "severity": m.group(3).upper(),
                    "detail": m.group(4).strip(),
                    "approve": False,
                    "message": DEFAULT_MESSAGE,
                    "keeps": [],
                    "source": "detected",
                    "report_hash": report_hash,
                })
    return regions


# ── review file ──────────────────────────────────────────────────────────────
def load_review() -> list[dict]:
    if not REVIEW_FILE.is_file():
        return []
    stanzas, cur = [], None
    for raw in REVIEW_FILE.read_text(encoding="utf-8").splitlines():
        head = _STANZA.match(raw)
        if head:
            cur = {
                "stem": head.group(1),
                "start": parse_ts(head.group(2)),
                "end": parse_ts(head.group(3)),
                "severity": head.group(4).upper(),
                "detail": "",
                "approve": False,
                "message": DEFAULT_MESSAGE,
                "keeps": [],
                "source": "manual",
                "report_hash": "",
            }
            stanzas.append(cur)
            continue
        if cur is None or not raw.strip() or raw.lstrip().startswith("#"):
            continue
        key, _, val = raw.partition(":")
        key, val = key.strip().lower(), val.strip()
        if key == "approve":
            cur["approve"] = val.lower() in {"yes", "y", "true", "1"}
        elif key == "message":
            cur["message"] = val or DEFAULT_MESSAGE
        elif key == "source":
            cur["source"] = val
        elif key == "report_hash":
            cur["report_hash"] = val
        elif key == "region":
            m = _RANGE.match(val)
            if m:
                cur["start"], cur["end"] = parse_ts(m.group(1)), parse_ts(m.group(2))
        elif key == "keep":
            m = _KEEP_LINE.match(val)
            if m and m.group(3).strip():
                cur["keeps"].append({"start": parse_ts(m.group(1)), "end": parse_ts(m.group(2)),
                                     "text": m.group(3).strip()})
    return stanzas


def save_review(stanzas: list[dict]) -> None:
    out = [
        "# Redaction review — which bad-audio regions get their subtitles replaced.",
        "#",
        "# Flip `approve: no` to `approve: yes` for every region you want redacted.",
        "# You may edit `region:` times and `message:` text, and add stanzas by hand",
        "# for regions the detector missed.",
        "#",
        "#   python helper_scripts/redact_flagged.py apply --dry-run   # preview",
        "#   python helper_scripts/redact_flagged.py apply             # write",
        "#",
        "# Re-running `review` refreshes the notes below and keeps your decisions.",
        "#",
        "# If real speech resumes partway through a region, narrowing `region:` will",
        "# NOT save those words (a region is usually one whole transcript line — it's",
        "# either kept or removed, never half). Add a `keep:` line instead to carve out",
        "# the good sub-range with its own text, e.g.:",
        "#   keep: 7:02-7:11 | the recovered words go here",
        "# helper_scripts/probe_region.py <media> <start> <end> finds the exact split.",
        "",
    ]
    for s in stanzas:
        out.append(f"--- {s['stem']} | {fmt_ts(s['start'])}-{fmt_ts(s['end'])} | {s['severity']}")
        if s.get("detail"):
            out.append(f"#   {s['detail']}")
        for note in s.get("notes", []):
            out.append(f"#   {note}")
        out.append(f"region: {fmt_ts(s['start'])}-{fmt_ts(s['end'])}")
        out.append(f"approve: {'yes' if s['approve'] else 'no'}")
        out.append(f"source: {s.get('source', 'manual')}")
        if s.get("report_hash"):
            out.append(f"report_hash: {s['report_hash']}")
        out.append(f"message: {s['message']}")
        for k in sorted(s.get("keeps", []), key=lambda k: k["start"]):
            out.append(f"keep: {fmt_ts(k['start'])}-{fmt_ts(k['end'])} | {k['text']}")
        out.append("")
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text("\n".join(out), encoding="utf-8")


# ── transcripts ──────────────────────────────────────────────────────────────
def targets_for(stem: str, target: str = "both") -> list[Path]:
    paths = []
    if target in {"both", "arabic"}:
        paths.append(ARABIC_DIR / f"{stem}.txt")
    if target in {"both", "english"}:
        paths.append(ENGLISH_DIR / f"{stem}.txt")
    return paths


def _entries(path: Path) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _TS_LINE.match(raw.strip())
        if m:
            rows.append({"start": parse_ts(m.group(1)), "end": parse_ts(m.group(2)),
                         "text": m.group(3).strip(), "raw": raw})
        else:
            rows.append({"start": None, "end": None, "text": None, "raw": raw})
    return rows


def _expected_lines(region: dict) -> list[dict]:
    """The line(s) a region should become: placeholder(s) around any `keep`
    sub-ranges, plus the kept sub-ranges verbatim with their own text. With no
    `keep` entries this is just the single whole-region placeholder — the
    original all-or-nothing behavior."""
    keeps = sorted(
        (k for k in region.get("keeps", []) if k["end"] > region["start"] and k["start"] < region["end"]),
        key=lambda k: k["start"],
    )
    out, cursor = [], region["start"]
    if region["end"] <= region["start"]:
        raise ValueError("Redaction region must end after it starts.")
    for k in keeps:
        k_start, k_end = max(k["start"], region["start"]), min(k["end"], region["end"])
        if k_end <= k_start or k_start < cursor:
            raise ValueError("Kept sub-ranges must have positive duration and must not overlap.")
        if k_start - cursor > 0.05:
            out.append({"start": cursor, "end": k_start, "text": region["message"]})
        out.append({"start": k_start, "end": k_end, "text": k["text"]})
        cursor = max(cursor, k_end)
    if region["end"] - cursor > 0.05:
        out.append({"start": cursor, "end": region["end"], "text": region["message"]})
    for e in out:
        e["raw"] = f"[{fmt_ts(e['start'])}-{fmt_ts(e['end'])}] {e['text']}"
    return out


def redact(path: Path, regions: list[dict], dry_run: bool = False) -> dict:
    """Drop the lines inside each region and insert the placeholder message
    (or, for any `keep` sub-ranges, the preserved text instead)."""
    if not path.is_file():
        return {"path": path, "status": "missing"}
    if not path.read_text(encoding="utf-8").strip():
        return {"path": path, "status": "empty"}

    expected = [e for reg in regions for e in _expected_lines(reg)]
    expected_raw = {e["raw"].strip() for e in expected}

    rows = _entries(path)
    removed: list[dict] = []
    kept: list[dict] = []
    already = set()
    for r in rows:
        raw = r["raw"].strip()
        # A line an earlier run already wrote — a placeholder or a kept
        # sub-range — is left exactly as-is, which is what keeps re-applying
        # a no-op instead of repeatedly removing and re-adding it.
        if raw in expected_raw:
            kept.append(r)
            already.add(raw)
            continue
        hit = False
        if r["start"] is not None:
            span = max(1e-9, r["end"] - r["start"])
            for reg in regions:
                if _overlap(r["start"], r["end"], reg["start"], reg["end"]) / span >= OVERLAP_MIN:
                    hit = True
                    break
        if hit:
            removed.append(r)
        else:
            kept.append(r)

    # New pieces go in chronological order; a region with nothing to remove
    # (a silent gap, say) still gets its placeholder.
    added = []
    for e in expected:
        if e["raw"].strip() in already:
            continue
        pos = next((j for j, r in enumerate(kept)
                    if r["start"] is not None and r["start"] >= e["start"]), len(kept))
        kept.insert(pos, e)
        added.append(e)

    if not removed and not added:
        return {"path": path, "status": "unchanged", "removed": [], "added": []}

    if not dry_run:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text("\n".join(r["raw"] for r in kept), encoding="utf-8")

    return {"path": path, "status": "dry-run" if dry_run else "written",
            "removed": removed, "added": added}


# ── steps ────────────────────────────────────────────────────────────────────
def _merge(flagged: list[dict], existing: list[dict]) -> list[dict]:
    """Keep decisions already made; add stanzas for newly flagged regions."""
    merged, used = [], set()
    for reg in flagged:
        match = None
        for i, old in enumerate(existing):
            if i in used or old["stem"] != reg["stem"]:
                continue
            if _overlap(reg["start"], reg["end"], old["start"], old["end"]) > 0:
                match = (i, old)
                break
        if match:
            i, old = match
            used.add(i)
            reg = {**reg, "start": old["start"], "end": old["end"],
                   "approve": old["approve"] and old.get("report_hash") == reg.get("report_hash"), "message": old["message"],
                   "keeps": old.get("keeps", [])}
        merged.append(reg)
    # Hand-written stanzas that match no flagged region are preserved.
    for i, old in enumerate(existing):
        if i not in used:
            stale = old.get("source") == "detected"
            merged.append({**old, "approve": False if stale else old["approve"],
                           "detail": "Report changed/removed; approval cleared." if stale else "(added by hand)"})
    merged.sort(key=lambda r: (r["stem"], r["start"]))
    return merged


def _annotate(regions: list[dict], target: str = "both") -> None:
    """Attach a note per region describing what it would remove."""
    for reg in regions:
        notes = []
        for path in targets_for(reg["stem"], target):
            if not path.is_file():
                notes.append(f"{path.as_posix()}: missing")
                continue
            if not path.read_text(encoding="utf-8").strip():
                notes.append(f"{path.as_posix()}: empty (not translated yet)")
                continue
            n = sum(
                1 for r in _entries(path)
                if r["start"] is not None
                and _overlap(r["start"], r["end"], reg["start"], reg["end"])
                / max(1e-9, r["end"] - r["start"]) >= OVERLAP_MIN
            )
            notes.append(f"{path.as_posix()}: would replace {n} line(s)")
        if reg.get("keeps"):
            span = ", ".join(f"{fmt_ts(k['start'])}-{fmt_ts(k['end'])}" for k in reg["keeps"])
            notes.append(f"keeping {len(reg['keeps'])} sub-range(s): {span}")
        reg["notes"] = notes


def review(interactive: bool = False, only: str | None = None, target: str = "both") -> list[dict]:
    flagged = load_flagged()
    if not flagged and not REVIEW_FILE.is_file():
        print(f"No flagged regions found in {ARABIC_DIR.as_posix()}/*.flagged.txt")
        return []
    regions = _merge(flagged, load_review())
    _annotate(regions, target)

    if interactive:
        print(f"{len(regions)} flagged region(s). y = redact, n = leave alone, q = stop.\n")
        selected = [reg for reg in regions if not only or reg["stem"] == only]
        for i, reg in enumerate(selected, 1):
            print("─" * 72)
            print(f"[{i}/{len(regions)}] {reg['stem']}  {fmt_ts(reg['start'])}-{fmt_ts(reg['end'])}"
                  f"  {reg['severity']}")
            if reg.get("detail"):
                print(f"  {reg['detail']}")
            for note in reg.get("notes", []):
                print(f"  {note}")
            for k in sorted(reg.get("keeps", []), key=lambda k: k["start"]):
                print(f"  keep {fmt_ts(k['start'])}-{fmt_ts(k['end'])}: {k['text']}")
            for path in targets_for(reg["stem"], target):
                if not path.is_file() or not path.read_text(encoding="utf-8").strip():
                    continue
                lines = [r for r in _entries(path)
                         if r["start"] is not None
                         and _overlap(r["start"], r["end"], reg["start"], reg["end"])
                         / max(1e-9, r["end"] - r["start"]) >= OVERLAP_MIN]
                if lines:
                    print(f"  ── would be removed from {path.as_posix()} ──")
                    for r in lines[:12]:
                        print(f"     {r['raw']}")
                    if len(lines) > 12:
                        print(f"     … {len(lines) - 12} more")
            cur = "yes" if reg["approve"] else "no"
            try:
                ans = input(f"  Redact this region? [y/n/q] (current: {cur}): ").strip().lower()
            except EOFError:
                print("  (no input available — stopping)")
                break
            if ans.startswith("q"):
                print("  Stopped; decisions so far are saved.")
                break
            if ans.startswith("y"):
                reg["approve"] = True
            elif ans.startswith("n"):
                reg["approve"] = False

    save_review(regions)
    n_ok = sum(1 for r in regions if r["approve"])
    print(f"\n{len(regions)} region(s) in {REVIEW_FILE.as_posix()} — {n_ok} approved.")
    if not interactive:
        print("Edit that file (approve: yes) or re-run with --interactive.")
    if n_ok:
        print("Then: python helper_scripts/redact_flagged.py apply --dry-run")
    return regions


def apply(dry_run: bool = False, only: str | None = None, target: str = "both") -> None:
    stanzas = [s for s in load_review() if s["approve"]]
    if only:
        stanzas = [s for s in stanzas if s["stem"] == only]
    if not stanzas:
        print(f"Nothing approved in {REVIEW_FILE.as_posix()} — set `approve: yes` on a region first.")
        return

    # Validate the whole request before writing either language. Detected approvals
    # are tied to the exact report reviewed; re-transcription requires re-approval.
    for region in stanzas:
        if region.get("source") == "detected":
            report = ARABIC_DIR / f"{region['stem']}.flagged.txt"
            if not report.is_file() or hashlib.sha256(report.read_bytes()).hexdigest() != region.get("report_hash"):
                raise ValueError(f"Stale approval for {region['stem']}; run review and approve the current report.")
        _expected_lines(region)
        english = ENGLISH_DIR / f"{region['stem']}.txt"
        if target in {"both", "english"} and english.is_file() and english.read_text(encoding="utf-8").strip():
            if any(re.search(r"[\u0600-\u06ff]", k["text"]) for k in region.get("keeps", [])):
                raise ValueError("Arabic keep: text cannot be inserted into an English translation. Apply --target arabic before translating, or use English recovery text with --target english.")

    by_stem: dict[str, list[dict]] = {}
    for s in stanzas:
        by_stem.setdefault(s["stem"], []).append(s)

    print(("DRY RUN — nothing will be written\n" if dry_run else "") +
          f"{len(stanzas)} approved region(s) across {len(by_stem)} file(s)")
    total_removed = total_added = 0
    for stem, regions in sorted(by_stem.items()):
        regions.sort(key=lambda r: r["start"])
        print(f"\n{'─' * 72}\n{stem}")
        for path in targets_for(stem, target):
            res = redact(path, regions, dry_run=dry_run)
            if res["status"] in {"missing", "empty"}:
                note = "not found" if res["status"] == "missing" else "empty (not translated yet)"
                print(f"  ↷ {path.as_posix()} — {note}, skipped")
                continue
            if res["status"] == "unchanged":
                print(f"  = {path.as_posix()} — already redacted, no change")
                continue
            print(f"  ✓ {path.as_posix()} — removed {len(res['removed'])} line(s), "
                  f"inserted {len(res['added'])}")
            for r in res["added"]:
                print(f"      + {r['raw']}")
            total_removed += len(res["removed"])
            total_added += len(res["added"])

    print(f"\n{'═' * 72}")
    verb = "would remove" if dry_run else "removed"
    print(f"{verb} {total_removed} line(s), inserted {total_added} placeholder(s)")
    if dry_run:
        print("Re-run without --dry-run to write the changes.")
    elif total_added:
        print("Originals kept as *.txt.bak — re-run the final-video cell to re-render.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)

    r = sub.add_parser("review", help="collect flagged regions and record approvals")
    r.add_argument("--interactive", action="store_true", help="approve region by region")

    a = sub.add_parser("apply", help="redact the approved regions")
    a.add_argument("--dry-run", action="store_true", help="show changes without writing")

    for p in (r, a):
        p.add_argument("--only", metavar="NAME", help="limit to one transcript stem, e.g. 20")
        p.add_argument("--target", choices=["both", "arabic", "english"], default="both",
                       help="which transcript folder(s) to touch (default: both)")

    args = ap.parse_args()
    if args.step == "review":
        review(interactive=args.interactive, only=args.only, target=args.target)
    else:
        apply(dry_run=args.dry_run, only=args.only, target=args.target)


if __name__ == "__main__":
    main()
