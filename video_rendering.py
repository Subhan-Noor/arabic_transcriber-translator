"""FFmpeg utilities shared by the standard local notebook and regression tests."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path


def run_ffmpeg(exe, args, cwd=None):
    tail = deque(maxlen=150)
    proc = subprocess.Popen([str(exe), '-y', '-nostdin', '-hide_banner', '-loglevel', 'error',
                             *map(str, args)], cwd=cwd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    try:
        for line in proc.stderr:
            tail.append(line)
        if proc.wait():
            raise RuntimeError('FFmpeg failed:\n' + ''.join(tail)[-6000:])
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        proc.stderr.close()


def find_ffmpeg(explicit=None):
    candidates = [Path(explicit).expanduser()] if explicit else []
    if not explicit:
        candidates += [Path(r'C:\ffmpeg\bin\ffmpeg.exe'), Path(r'C:\Program Files\ffmpeg\bin\ffmpeg.exe')]
        winget = Path.home() / 'AppData/Local/Microsoft/WinGet/Packages'
        if winget.is_dir():
            candidates += sorted(winget.glob('Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe'), reverse=True)
        if shutil.which('ffmpeg'):
            candidates.append(Path(shutil.which('ffmpeg')))
    for exe in candidates:
        if not exe.is_file():
            continue
        result = subprocess.run([str(exe), '-hide_banner', '-filters'], capture_output=True, text=True)
        filters = {p[1] for line in result.stdout.splitlines() if len(p := line.split()) >= 2}
        probe = exe.with_name('ffprobe.exe' if exe.suffix.lower() == '.exe' else 'ffprobe')
        if not probe.is_file() and shutil.which('ffprobe'):
            probe = Path(shutil.which('ffprobe'))
        if 'subtitles' in filters and probe.is_file():
            return exe.resolve(), probe.resolve()
    raise RuntimeError('Install a full FFmpeg build with the subtitles/libass filter and ffprobe, or set FFMPEG_PATH.')


def probe_media(ffprobe, path):
    result = subprocess.run([str(ffprobe), '-v', 'error', '-show_entries',
                             'format=duration:stream=codec_type,duration', '-of', 'json', str(path)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'Cannot read {path}: {result.stderr[-1000:]}')
    info = json.loads(result.stdout)
    duration = float(info.get('format', {}).get('duration', 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'Invalid media duration: {path}')
    return duration, {s['codec_type'] for s in info.get('streams', [])}


def choose_encoder(ffmpeg, quality=23, preset='veryfast', prefer_gpu=True, prefer_hevc=False, nvenc_preset='p4'):
    result = subprocess.run([str(ffmpeg), '-hide_banner', '-encoders'], capture_output=True, text=True)
    available = {p[1] for line in result.stdout.splitlines()
                 if len(p := line.split()) >= 2 and len(p[0]) == 6}
    gpu = ['hevc_nvenc', 'h264_nvenc'] if prefer_hevc else ['h264_nvenc', 'hevc_nvenc']
    for name in (gpu if prefer_gpu else []) + ['libx264', 'h264_mf']:
        if name not in available:
            continue
        if name == 'libx264':
            args = ['-c:v', name, '-crf', str(quality), '-preset', preset]
        elif name == 'h264_mf':
            args = ['-c:v', name, '-rate_control', 'quality', '-quality', str(max(10, min(100, 110 - quality * 3)))]
        else:
            args = ['-c:v', name, '-preset', nvenc_preset, '-rc', 'vbr', '-cq', str(quality), '-b:v', '0']
            if name == 'hevc_nvenc':
                args += ['-tag:v', 'hvc1']
        try:
            run_ffmpeg(ffmpeg, ['-f', 'lavfi', '-i', 'color=s=1920x1080:r=24', '-frames:v', '1',
                                *args, '-pix_fmt', 'yuv420p', '-f', 'null', '-'])
            return name, args
        except RuntimeError:
            print(f'{name} is unavailable on this system; trying another encoder.')
    raise RuntimeError('No working video encoder found. Install FFmpeg with libx264 support.')


def shift_srt(text, offset):
    def replace(match):
        h, m, s, ms = map(int, match.groups())
        value = h * 3600000 + m * 60000 + s * 1000 + ms + round(offset * 1000)
        h, rem = divmod(value, 3600000)
        m, rem = divmod(rem, 60000)
        s, ms = divmod(rem, 1000)
        return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'
    return re.sub(r'(\d{2,}):(\d{2}):(\d{2}),(\d{3})', replace, text)


def _encode_output(ffmpeg, args, output, work, quality, preset, prefer_gpu, prefer_hevc):
    name, encoding = choose_encoder(ffmpeg, quality, preset, prefer_gpu, prefer_hevc)
    partial = work / 'encoded.mp4'
    try:
        run_ffmpeg(ffmpeg, [*args, *encoding, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', partial], cwd=work)
    except RuntimeError:
        if not name.endswith('_nvenc'):
            raise
        print('GPU encoding failed during rendering; retrying with the CPU.')
        _, encoding = choose_encoder(ffmpeg, quality, preset, prefer_gpu=False)
        run_ffmpeg(ffmpeg, [*args, *encoding, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', partial], cwd=work)
    os.replace(partial, output)
    return output


def burn_subtitles(video, srt, output, *, ffmpeg_path=None, quality=23, preset='veryfast', prefer_gpu=True,
                   style='FontName=Arial,FontSize=18,BorderStyle=3,MarginV=40,Alignment=2'):
    video, srt, output = (Path(p).resolve() for p in (video, srt, output))
    if output in {video, srt}:
        raise ValueError('Output must differ from the input files.')
    if not video.is_file() or not srt.is_file():
        raise FileNotFoundError('Set VIDEO_FILE and SRT_FILE to existing files.')
    ffmpeg, _ = find_ffmpeg(ffmpeg_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='_caption_', dir=output.parent) as temp:
        work = Path(temp)
        shutil.copy2(srt, work / 'captions.srt')
        return _encode_output(ffmpeg, ['-i', video, '-vf', f"subtitles=captions.srt:force_style='{style}'",
                                      '-c:a', 'aac', '-b:a', '192k'], output, work,
                              quality, preset, prefer_gpu, False)


def render_loop_video(audio, srt, output, intro, background, *, ffmpeg_path=None, quality=23,
                      preset='veryfast', prefer_gpu=True, prefer_hevc=False, width=1920, height=1080,
                      fps=24, fade=1.5, style='FontName=Arial,FontSize=16,BorderStyle=3,MarginV=40,Alignment=2'):
    audio, srt, output, intro, background = (Path(p).resolve() for p in (audio, srt, output, intro, background))
    for path in (audio, srt, intro, background):
        if not path.is_file():
            raise FileNotFoundError(f'Required file missing: {path}. See README.md for channel asset requirements.')
    if output in {audio, srt, intro, background}:
        raise ValueError('Output must differ from the input files.')
    if width <= 0 or height <= 0 or width % 2 or height % 2 or fps <= 0 or fade < 0:
        raise ValueError('Use positive even dimensions, positive FPS, and nonnegative fade.')
    ffmpeg, ffprobe = find_ffmpeg(ffmpeg_path)
    audio_dur, audio_streams = probe_media(ffprobe, audio)
    intro_dur, intro_streams = probe_media(ffprobe, intro)
    bg_dur, bg_streams = probe_media(ffprobe, background)
    if 'audio' not in audio_streams or 'video' not in intro_streams or 'video' not in bg_streams:
        raise ValueError('Main media needs audio; intro and background need video.')
    loops = max(1, math.ceil(audio_dur / bg_dur))
    total = intro_dur + loops * bg_dur
    fade = min(fade, total)
    scale = (f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
             f'pad={width}:{height}:-1:-1:color=black,setsar=1,fps={fps},format=yuv420p,setpts=PTS-STARTPTS')
    intro_audio = ('[0:a]aresample=48000,aformat=channel_layouts=stereo,'
                   f'apad,atrim=duration={intro_dur:.6f},asetpts=PTS-STARTPTS[ia];'
                   if 'audio' in intro_streams else
                   f'anullsrc=r=48000:cl=stereo,atrim=duration={intro_dur:.6f}[ia];')
    filters = (f'[0:v]{scale}[iv];[1:v]{scale}[bv];[iv][bv]concat=n=2:v=1:a=0[v0];'
               f"[v0]subtitles=captions.srt:force_style='{style}'[v1];"
               + (f'[v1]fade=t=out:st={total-fade:.6f}:d={fade:.6f}[vout];' if fade else '[v1]null[vout];')
               + intro_audio + '[2:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[ma];'
               + f'[ia][ma]concat=n=2:v=0:a=1[a0];[a0]apad=whole_dur={total:.6f}[aout]')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='_render_', dir=output.parent) as temp:
        work = Path(temp)
        (work / 'captions.srt').write_text(shift_srt(srt.read_text(encoding='utf-8-sig'), intro_dur), encoding='utf-8')
        return _encode_output(ffmpeg, ['-i', intro, '-stream_loop', str(loops-1), '-i', background,
                                      '-i', audio, '-filter_complex', filters, '-map', '[vout]', '-map', '[aout]',
                                      '-c:a', 'aac', '-b:a', '192k', '-t', f'{total:.6f}'],
                              output, work, quality, preset, prefer_gpu, prefer_hevc)
