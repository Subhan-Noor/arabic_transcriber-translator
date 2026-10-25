"""Shared transcript handling for the local notebooks (standard library only)."""
from __future__ import annotations

import math
import re
import statistics
from pathlib import Path

MEDIA_EXTS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.opus', '.wma',
              '.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v'}
_TIME = r'\d+(?::\d{2}){1,2}'
_LINE = re.compile(rf'\[({_TIME})\s*-\s*({_TIME})\]\s*(.+)')


def parse_timestamp(value):
    if value is None:
        return None
    parts = str(value).strip().split(':')
    if not 1 <= len(parts) <= 3 or any(not re.fullmatch(r'\d+(?:\.\d+)?', p) for p in parts):
        raise ValueError(f'Invalid timestamp: {value!r}')
    nums = [float(p) for p in parts]
    if any(n >= 60 for n in nums[1:]):
        raise ValueError(f'Minutes/seconds must be below 60: {value!r}')
    seconds = sum(n * 60 ** i for i, n in enumerate(reversed(nums)))
    if not math.isfinite(seconds):
        raise ValueError(f'Invalid timestamp: {value!r}')
    return seconds


def validate_time_range(start, end):
    start, end = parse_timestamp(start), parse_timestamp(end)
    if end is not None and end <= (start or 0):
        raise ValueError('END_TIME must be later than START_TIME.')
    return start, end


def format_timestamp(seconds):
    value = float(seconds)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Time must be nonnegative and finite.')
    h, rem = divmod(int(round(value)), 3600)
    m, s = divmod(rem, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


def format_cue(start, end, text):
    """Keep the existing whole-second format without producing zero-length cues."""
    start, end = float(start), float(end)
    if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
        raise ValueError('A subtitle must have a positive duration.')
    first = int(round(start))
    last = max(first + 1, int(round(end)))
    return f'[{format_timestamp(first)}-{format_timestamp(last)}] {text.strip()}'


def parse_transcript(text):
    """Reject malformed nonblank lines rather than silently dropping captions."""
    cues, previous = [], -1.0
    for number, line in enumerate(text.lstrip('\ufeff').splitlines(), 1):
        if not line.strip():
            continue
        match = _LINE.fullmatch(line.strip())
        if not match:
            raise ValueError(f'Line {number}: expected [MM:SS-MM:SS] text.')
        start, end = (parse_timestamp(x) for x in match.group(1, 2))
        if end <= start:
            raise ValueError(f'Line {number}: end must be later than start.')
        if start < previous:
            raise ValueError(f'Line {number}: timestamps are out of order.')
        previous = start
        cues.append((start, end, match.group(3).strip()))
    if not cues:
        raise ValueError('No timestamped captions found.')
    return cues


def seconds_to_srt(seconds):
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def transcript_to_srt(text):
    return '\n'.join(f'{i}\n{seconds_to_srt(start)} --> {seconds_to_srt(end)}\n{body}\n'
                     for i, (start, end, body) in enumerate(parse_transcript(text), 1))


def convert_transcript_file(path):
    path = Path(path)
    result = transcript_to_srt(path.read_text(encoding='utf-8-sig'))
    dest = path.with_suffix('.srt')
    dest.write_text(result, encoding='utf-8')
    return dest


def list_media(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Media folder not found: {folder.resolve()}')
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTS)
    by_stem = {}
    for path in files:
        by_stem.setdefault(path.stem.casefold(), []).append(path)
    duplicates = [', '.join(p.name for p in group) for group in by_stem.values() if len(group) > 1]
    if duplicates:
        raise ValueError('Keep one media file per stem; ambiguous filenames: ' + '; '.join(duplicates))
    return files


def list_transcripts(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Transcript folder not found: {folder.resolve()}')
    return sorted(p for p in folder.glob('*.txt') if p.is_file()
                  and not p.name.endswith(('.flagged.txt', '.plain.txt')))


def pair_transcripts(media_folder, transcript_folder, only=None):
    media = {p.stem.casefold(): p for p in list_media(media_folder)}
    pairs, blank, unmatched = [], [], []
    for txt in list_transcripts(transcript_folder):
        if only is not None and txt.stem != Path(str(only)).stem:
            continue
        if not txt.read_text(encoding='utf-8-sig').strip():
            blank.append(txt.name)
        elif txt.stem.casefold() in media:
            pairs.append((txt, media[txt.stem.casefold()]))
        else:
            unmatched.append(txt.name)
    return pairs, blank, unmatched


def write_confidence_report(path, media_name, alerts):
    """Replace a report on every successful analysis, including clean runs."""
    path = Path(path)
    if not alerts:
        path.unlink(missing_ok=True)
        return
    severe = sum(a['severity'] == 'severe' for a in alerts)
    lines = [f'# Confidence report for {media_name}',
             f'# {severe} severe, {len(alerts) - severe} warning(s)', '']
    for alert in alerts:
        tag = 'SEVERE' if alert['severity'] == 'severe' else 'warning'
        lines.append(f"[{format_timestamp(alert['start'])}-{format_timestamp(alert['end'])}] {tag}: {alert['why']}")
        if alert.get('text'):
            lines.append(f"    transcribed as: {alert['text']}")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def confidence_alerts(rows, cov_start, cov_end, *, STALL_MIN_SECONDS=8.0, STALL_CPS_RATIO=0.25, GAP_MIN_SECONDS=20.0, LOW_AVG_LOGPROB=-0.55, HIGH_COMPRESSION_RATIO=3.0, HIGH_NO_SPEECH_PROB=0.6):
    """Find stretches of audio the model probably could not transcribe.

    `rows` are per-segment metrics collected during decoding. Returns alerts
    sorted by time, each with severity "severe" (audio is likely unusable) or
    "warn" (worth a look). Needs the whole file before judging, because the
    text-density test is relative to this file's own median.
    """
    alerts = []
    if not rows:
        return [{"severity": "warn", "start": cov_start, "end": max(cov_start + 1, cov_end or 0),
                 "why": "No speech was transcribed; check for silence or unusable audio before translating.", "text": ""}]

    # Median density over ordinary segments — short ones and bare interjections
    # would drag it down and mask real problems.
    ref = [r["cps"] for r in rows if r["dur"] >= 1.0 and r["nchars"] >= 3]
    median_cps = statistics.median(ref) if ref else 0.0

    # 1. Stalled segments: long, but almost nothing transcribed. This is what
    #    loud static/distortion looks like in Whisper's output.
    if median_cps > 0:
        floor = median_cps * STALL_CPS_RATIO
        for r in rows:
            if r["dur"] >= STALL_MIN_SECONDS and r["cps"] < floor:
                alerts.append({
                    "severity": "severe",
                    "start": r["start"],
                    "end": r["end"],
                    "why": (f"{r['dur']:.0f}s of audio yielded only {r['nchars']} characters "
                            f"({r['cps']:.2f} chars/s vs {median_cps:.1f} typical) — audio likely "
                            f"unintelligible: static, distortion, heavy noise, or music"),
                    "text": r["text"],
                })

    # 2. Stretches with no transcript at all.
    prev_end = cov_start
    for r in rows:
        gap = r["start"] - prev_end
        if gap >= GAP_MIN_SECONDS:
            alerts.append({
                "severity": "warn",
                "start": prev_end,
                "end": r["start"],
                "why": f"{gap:.0f}s with no transcribed speech — silence, or audio the model discarded",
                "text": "",
            })
        prev_end = max(prev_end, r["end"])
    if cov_end and cov_end - prev_end >= GAP_MIN_SECONDS:
        alerts.append({
            "severity": "warn",
            "start": prev_end,
            "end": cov_end,
            "why": f"{cov_end - prev_end:.0f}s at the end with no transcribed speech",
            "text": "",
        })

    # 3. Decode-window signals. Consecutive segments sharing the same values
    #    belong to one window, so collapse each run into a single alert instead
    #    of repeating it on every line of the window.
    def _why(r):
        out = []
        if r["alp"] is not None and r["alp"] < LOW_AVG_LOGPROB:
            out.append(f"low decode confidence (avg_logprob={r['alp']:.2f})")
        if r["cr"] is not None and r["cr"] > HIGH_COMPRESSION_RATIO:
            out.append(f"repetitive/looping output (compression_ratio={r['cr']:.2f})")
        if r["ns"] is not None and r["ns"] > HIGH_NO_SPEECH_PROB:
            out.append(f"probably not speech (no_speech_prob={r['ns']:.2f})")
        return "; ".join(out)

    run_why, run_start, run_end = "", 0.0, 0.0
    for r in list(rows) + [None]:
        why = _why(r) if r is not None else ""
        if why != run_why:
            if run_why:
                alerts.append({"severity": "warn", "start": run_start, "end": run_end,
                               "why": run_why, "text": ""})
            run_why = why
            if r is not None:
                run_start = r["start"]
        if r is not None:
            run_end = r["end"]

    alerts.sort(key=lambda a: (a["start"], a["severity"] != "severe"))
    return alerts
