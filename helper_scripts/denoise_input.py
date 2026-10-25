"""
Denoise every audio file in input/ with DeepFilterNet (pip / GPU).

Uses the Python package (`deepfilternet`), not the standalone Rust
`deep-filter` binary (that one is CPU-only). The model is moved to CUDA
automatically when PyTorch can see a GPU.

One-time setup (same env as the notebook, CUDA PyTorch already installed):
    python -m pip install deepfilternet

Check GPU:
    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"

Preview what would be processed:
    python helper_scripts/denoise_input.py

Write cleaned files (originals copied to *.bak first):
    python helper_scripts/denoise_input.py --apply

One file only (stem, with or without extension):
    python helper_scripts/denoise_input.py --apply usool
    python helper_scripts/denoise_input.py --apply "manhaj (1).mp3"

Write to a different folder instead of replacing input/:
    python helper_scripts/denoise_input.py --apply --out-dir input_cleaned

Gentler denoise (less distortion). Always re-run from the original *.bak:
    python helper_scripts/denoise_input.py --apply --from-backup --atten-lim 12 --mix 0.7

  --atten-lim N   cap noise reduction at N dB (try 6, 12, or 20; lower = milder)
  --mix F         blend enhanced with original, 0–1 (1 = fully enhanced; try 0.6–0.8)
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_DIR = (ROOT / "../input").resolve()

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".wma"}

# Lame VBR quality when writing MP3 back (2 ≈ ~190 kbps, same as strip_edges.py)
MP3_Q = "2"


def check_setup() -> int:
    """Print env / GPU / DeepFilterNet status. Returns 0 if ready to run."""
    print(f"Python : {sys.executable}")

    try:
        import torch
    except ImportError:
        print("torch is not installed in this environment.", file=sys.stderr)
        return 1

    print(f"torch  : {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU    : none — DeepFilterNet will run on CPU (slow).")
        print("         Install a CUDA PyTorch build if you expected a GPU.")

    try:
        import df  # noqa: F401
        from df.enhance import init_df  # noqa: F401
    except ImportError:
        print("DeepFilterNet is not installed.", file=sys.stderr)
        print("  python -m pip install deepfilternet", file=sys.stderr)
        return 1

    cli = shutil.which("deepFilter")
    rust = shutil.which("deep-filter")
    print(f"df pkg : installed")
    print(f"CLI    : {cli or 'deepFilter not on PATH (Python API will be used)'}")
    if rust and (not cli or Path(rust).resolve() != Path(cli).resolve()):
        print(f"note   : found Rust binary at {rust} — this script will not use it")

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH (needed to decode/encode mp3/m4a/…).", file=sys.stderr)
        return 1

    return 0


def output_path(src: Path, out_dir: Path) -> Path:
    """Map usool.wav.bak → out_dir/usool.wav, otherwise keep the filename."""
    if src.suffix.lower() == ".bak":
        return out_dir / src.name[: -len(".bak")]
    return out_dir / src.name


def list_audio(folder: Path, stem: str | None = None) -> list[Path]:
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if stem is None:
        return files
    key = Path(stem).stem
    matched = [p for p in files if p.stem == key]
    if not matched:
        # allow passing a relative path like input/usool.mp3
        candidate = Path(stem)
        if not candidate.is_absolute():
            candidate = candidate if candidate.is_file() else folder / candidate.name
        if candidate.is_file():
            return [candidate]
    return matched


def ffmpeg_to_wav(src: Path, dest: Path, *, mono: bool, sr: int | None = None) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-c:a", "pcm_s16le",
    ]
    if sr:
        cmd += ["-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd.append(str(dest))
    subprocess.run(cmd, check=True)


def load_wav(path: Path):
    """Read a PCM WAV without torchaudio (this env has no audio file backend)."""
    import torch

    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sw}")
    audio = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    if nch > 1:
        audio = audio.view(-1, nch).t().contiguous()
    else:
        audio = audio.view(1, -1)
    return audio, sr


def save_wav(path: Path, audio, sr: int) -> None:
    import torch

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    pcm = (audio.detach().cpu().clamp(-1, 1) * 32767).to(torch.int16)
    nch, nframes = pcm.shape
    interleaved = pcm.t().contiguous().view(-1).numpy().tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.setnframes(nframes)
        wf.writeframes(interleaved)


def wav_to_original_format(wav: Path, dest: Path) -> None:
    ext = dest.suffix.lower()
    if ext == ".wav":
        shutil.copy2(wav, dest)
        return
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(wav),
    ]
    if ext == ".mp3":
        cmd += ["-c:a", "libmp3lame", "-q:a", MP3_Q]
    elif ext in {".m4a", ".aac"}:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    elif ext == ".flac":
        cmd += ["-c:a", "flac"]
    elif ext == ".ogg":
        cmd += ["-c:a", "libvorbis", "-q:a", "6"]
    elif ext == ".opus":
        cmd += ["-c:a", "libopus", "-b:a", "128k"]
    elif ext == ".wma":
        cmd += ["-c:a", "wmav2", "-b:a", "192k"]
    else:
        raise ValueError(f"Unsupported output format: {ext}")
    cmd.append(str(dest))
    subprocess.run(cmd, check=True)


def enhance_file(
    model,
    df_state,
    wav_in: Path,
    wav_out: Path,
    *,
    chunk_seconds: float,
    atten_lim: float | None,
    mix: float,
) -> None:
    import torch
    from df.enhance import enhance
    from df.model import ModelParams

    df_sr = ModelParams().sr
    audio, sr = load_wav(wav_in)
    audio = audio.contiguous()
    if sr != df_sr:
        raise ValueError(f"Expected {df_sr} Hz WAV (ffmpeg should resample), got {sr}")
    duration = audio.shape[-1] / df_sr
    print(f"  duration: {duration / 60:.1f} min")
    if atten_lim is not None:
        print(f"  atten   : {atten_lim:g} dB cap")
    if mix < 1.0:
        print(f"  mix     : {mix:g} enhanced / {1 - mix:g} original")

    original = audio if mix < 1.0 else None

    # Long files hit CUDNN_STATUS_NOT_SUPPORTED if enhanced in one shot
    # (DeepFilterNet #123). 0 means "whole file".
    if chunk_seconds > 0 and duration > chunk_seconds:
        print(f"  chunks  : {chunk_seconds:g}s")
        audio = _enhance_chunked(
            model, df_state, audio, df_sr, chunk_seconds, atten_lim=atten_lim
        )
    else:
        audio = enhance(model, df_state, audio, pad=True, atten_lim_db=atten_lim)

    if original is not None:
        n = min(original.shape[-1], audio.shape[-1])
        audio = mix * audio[:, :n] + (1.0 - mix) * original[:, :n]

    save_wav(wav_out, audio.to("cpu"), df_sr)
    del audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _enhance_chunked(
    model, df_state, audio, sr: int, chunk_seconds: float, *, atten_lim: float | None
):
    """Overlap-add enhancement so long files fit in GPU / cuDNN limits."""
    import torch
    from df.enhance import enhance

    chunk = int(chunk_seconds * sr)
    overlap = min(int(0.5 * sr), chunk // 4)
    hop = max(chunk - overlap, 1)
    channels, total = audio.shape
    out = audio.new_zeros((channels, total))
    weight = audio.new_zeros(total)
    n_chunks = max(1, (total + hop - 1) // hop)

    pos = 0
    i = 0
    while pos < total:
        i += 1
        end = min(pos + chunk, total)
        if i == 1 or i % 10 == 0 or end >= total:
            print(f"  chunk   : {i}/{n_chunks}  ({pos / sr:.0f}s)", flush=True)
        piece = audio[:, pos:end].contiguous()
        enhanced = enhance(
            model, df_state, piece, pad=True, atten_lim_db=atten_lim
        )
        n = min(enhanced.shape[-1], end - pos)
        enhanced = enhanced[:, :n].contiguous()
        w = audio.new_ones(n)
        fade = min(overlap, n // 2) if overlap else 0
        if fade:
            ramp = torch.linspace(0, 1, fade, device=w.device, dtype=w.dtype)
            if pos > 0:
                w[:fade] = ramp
            if end < total:
                w[-fade:] = ramp.flip(0)
        out[:, pos : pos + n] += enhanced * w
        weight[pos : pos + n] += w
        if end >= total:
            break
        pos += hop

    weight = weight.clamp_min(1e-8)
    return out / weight


def process_one(
    src: Path,
    dest: Path,
    model,
    df_state,
    *,
    backup: bool,
    mono: bool,
    chunk_seconds: float,
    model_sr: int,
    atten_lim: float | None,
    mix: float,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if backup and dest.resolve() == src.resolve():
        bak = src.with_suffix(src.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(src, bak)
            print(f"  backup : {bak.name}")
        else:
            print(f"  backup : {bak.name} (already exists, left as-is)")

    with tempfile.TemporaryDirectory(prefix="dfn_") as tmp:
        tmp_dir = Path(tmp)
        wav_in = tmp_dir / "noisy.wav"
        wav_out = tmp_dir / "clean.wav"
        ffmpeg_to_wav(src, wav_in, mono=mono, sr=model_sr)
        enhance_file(
            model,
            df_state,
            wav_in,
            wav_out,
            chunk_seconds=chunk_seconds,
            atten_lim=atten_lim,
            mix=mix,
        )

        # Keep the real extension so ffmpeg / the wav copy path can see it
        # (usool.wav.tmp looks like format ".tmp" and ffmpeg refuses it).
        tmp_dest = dest.with_name(dest.stem + ".tmp" + dest.suffix)
        try:
            wav_to_original_format(wav_out, tmp_dest)
            tmp_dest.replace(dest)
        except Exception:
            if tmp_dest.exists():
                tmp_dest.unlink()
            raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Denoise audio in input/ with DeepFilterNet (Python / GPU)."
    )
    p.add_argument(
        "stem",
        nargs="?",
        help='Optional file stem or name, e.g. usool or "manhaj (1).mp3".',
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write cleaned files. Without this flag, only list what would run.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Folder of noisy audio (default: ../input).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write cleaned files here instead of replacing files in --input-dir.",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write .bak copies when replacing files in place.",
    )
    p.add_argument(
        "--pf",
        action="store_true",
        help="Enable DeepFilterNet post-filter (stronger NR, more artifacts).",
    )
    p.add_argument(
        "--atten-lim",
        type=float,
        default=None,
        metavar="DB",
        help="Cap noise reduction at this many dB (e.g. 12). Lower = milder, "
        "fewer artifacts. Default is unlimited (most aggressive).",
    )
    p.add_argument(
        "--mix",
        type=float,
        default=1.0,
        metavar="F",
        help="How much enhanced audio to keep, 0–1 (rest is the original). "
        "1 = fully enhanced. Try 0.6–0.8 if speech sounds watery.",
    )
    p.add_argument(
        "--from-backup",
        action="store_true",
        help="Read from *.bak (the original) instead of the already-denoised file.",
    )
    p.add_argument(
        "--keep-channels",
        action="store_true",
        help="Keep the original channel count instead of downmixing to mono (default: mono).",
    )
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=60.0,
        help="Enhance in N-second chunks (default 60). Long files fail cuDNN "
        "if run in one pass. Use 0 to disable chunking.",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Only print GPU / DeepFilterNet status and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    status = check_setup()
    if args.check:
        return status
    if status != 0:
        return status

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    files = list_audio(input_dir, args.stem)
    if not files:
        target = args.stem or f"*{tuple(AUDIO_EXTS)}"
        print(f"No audio files matched in {input_dir}: {target}", file=sys.stderr)
        return 1

    out_dir = args.out_dir.resolve() if args.out_dir else input_dir
    in_place = out_dir == input_dir
    mono = not args.keep_channels

    mix = args.mix
    if not math.isfinite(mix) or mix < 0 or mix > 1:
        print("--mix must be between 0 and 1.", file=sys.stderr)
        return 1

    if not math.isfinite(args.chunk_seconds) or args.chunk_seconds < 0:
        print("--chunk-seconds must be nonnegative and finite.", file=sys.stderr)
        return 1
    if args.atten_lim is not None and (not math.isfinite(args.atten_lim) or args.atten_lim < 0):
        print("--atten-lim must be nonnegative and finite.", file=sys.stderr)
        return 1

    if args.from_backup:
        restored = []
        missing = []
        for src in files:
            bak = src.with_suffix(src.suffix + ".bak")
            if bak.exists():
                restored.append(bak)
            else:
                missing.append(src.name)
        if missing:
            print(
                "No *.bak original for: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 1
        files = restored
        print("Source : *.bak originals")

    print(f"Input  : {input_dir}")
    print(f"Output : {out_dir}{'  (in-place)' if in_place else ''}")
    print(f"Files  : {len(files)}")
    print(f"Mono   : {mono}")
    print(f"Atten  : {args.atten_lim if args.atten_lim is not None else 'unlimited'} dB")
    print(f"Mix    : {mix:g}")
    if args.chunk_seconds:
        print(f"Chunks : {args.chunk_seconds:g}s")
    print()

    for src in files:
        dest = output_path(src, out_dir)
        print(f"  {src.name}  ->  {dest.name}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to denoise and write files.")
        return 0

    from df.enhance import init_df
    from df.model import ModelParams

    print("\nLoading DeepFilterNet…")
    # pip DeepFilterNet returns (model, df_state, suffix); newer git builds also return epoch
    loaded = init_df(
        post_filter=args.pf,
        log_level="INFO",
        log_file=None,
    )
    model, df_state = loaded[0], loaded[1]
    model_sr = ModelParams().sr

    for i, src in enumerate(files, 1):
        dest = output_path(src, out_dir)
        print(f"\n[{i}/{len(files)}] {src.name}  ->  {dest.name}")
        try:
            process_one(
                src,
                dest,
                model,
                df_state,
                backup=in_place and not args.no_backup and not args.from_backup,
                mono=mono,
                chunk_seconds=args.chunk_seconds,
                model_sr=model_sr,
                atten_lim=args.atten_lim,
                mix=mix,
            )
        except Exception as exc:
            print(f"ERROR {src.name}: {exc}", file=sys.stderr)
            return 1
        print("  done")

    print(f"\nCleaned {len(files)} file(s).")
    if in_place and not args.no_backup and not args.from_backup:
        print("Originals saved next to each file as *.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
