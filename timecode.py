"""Parse and format user-friendly time codes (e.g. ``8:00``, ``1:25:30``)."""

from __future__ import annotations


def parse_timestamp(s: str) -> float:
    """
    Parse a time string into seconds.

    Accepted forms:

    - ``SS`` or ``S`` — seconds only (e.g. ``45`` → 45 s)
    - ``M:SS`` or ``MM:SS`` — minutes and seconds (e.g. ``8:00``, ``0:35``)
    - ``H:MM:SS`` — hours, minutes, seconds (e.g. ``1:25:00``)

    Whitespace around the string is ignored.
    """
    s = s.strip()
    if not s:
        raise ValueError("empty time string")

    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"invalid time value: {s!r}") from exc

    if len(nums) == 1:
        sec = nums[0]
        if sec < 0:
            raise ValueError("time must be non-negative")
        return sec

    if len(nums) == 2:
        minutes, seconds = nums
        if minutes < 0 or seconds < 0 or seconds >= 60:
            raise ValueError(f"invalid MM:SS value: {s!r}")
        return minutes * 60.0 + seconds

    if len(nums) == 3:
        hours, minutes, seconds = nums
        if hours < 0 or minutes < 0 or seconds < 0 or minutes >= 60 or seconds >= 60:
            raise ValueError(f"invalid H:MM:SS value: {s!r}")
        return hours * 3600.0 + minutes * 60.0 + seconds

    raise ValueError(
        f"too many ':' segments in {s!r}; use SS, M:SS, MM:SS, or H:MM:SS"
    )


def format_timestamp(seconds: float) -> str:
    """Format *seconds* as ``H:MM:SS`` if ≥ 1 hour, else ``MM:SS`` (zero-padded)."""
    if seconds < 0:
        seconds = 0.0
    s = int(round(seconds))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _folder_part(t: float) -> str:
    s = int(round(t))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m}m{sec:02d}s"


def format_range_suffix(start_s: float, end_s: float) -> str:
    """Filesystem-safe range label, e.g. ``8m00s-25m00s`` or ``1h05m30s-2h00m00s``."""
    return f"{_folder_part(start_s)}-{_folder_part(end_s)}"


def format_segment_folder_label(
    start_s: float | None,
    end_s: float | None,
) -> str:
    """
    Folder suffix for a partial transcription.

    - both bounds → ``8m00s-25m00s``
    - only start (open end) → ``from_8m00s``
    - only end (from 0) → ``until_25m00s``
    """
    if start_s is not None and end_s is not None:
        return format_range_suffix(start_s, end_s)
    if start_s is not None:
        return f"from_{_folder_part(start_s)}"
    if end_s is not None:
        return f"until_{_folder_part(end_s)}"
    raise ValueError("at least one of start_s, end_s must be set")
