import ast
import contextlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import transcriber_utils as utils
import video_rendering as video
from helper_scripts import redact_flagged as redact
from helper_scripts import strip_edges as trim
from tools.clean_notebooks import clean_notebook

ROOT = Path(__file__).resolve().parents[1]


class TranscriptTests(unittest.TestCase):
    def test_notebook_batch_runs_empty_and_clean_decode(self):
        data = json.loads((ROOT/'transcribe2.ipynb').read_text(encoding='utf-8'))
        nature = json.loads((ROOT/'transcribe_nature.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(data['cells'][7]['source'], nature['cells'][7]['source'])
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        segment = SimpleNamespace(start=0, end=5, text='Example', avg_logprob=-0.1,
                                  no_speech_prob=0, compression_ratio=1)
        rows = []
        model = SimpleNamespace(transcribe=lambda *a, **k: (iter(rows), SimpleNamespace(duration=5)))
        modules = {'torch': fake_torch, 'faster_whisper': SimpleNamespace(WhisperModel=lambda *a, **k: model),
                   'IPython.display': SimpleNamespace(display=lambda *a: None, HTML=lambda x: x)}
        with tempfile.TemporaryDirectory() as temp, contextlib.chdir(temp), patch.dict(sys.modules, modules):
            folder = Path(temp); (folder/'input').mkdir(); (folder/'input/lesson.mp3').touch()
            scope = {'MODEL_PATH': 'fake-model'}
            exec(''.join(data['cells'][3]['source']), scope)
            code = ''.join(data['cells'][7]['source']).replace('OVERWRITE    = False', 'OVERWRITE    = True')
            exec(code, scope)
            report = folder/'transcript/arabic/lesson.flagged.txt'
            self.assertTrue(report.exists())
            english = folder/'transcript/english/lesson.txt'
            english.write_text('Existing translation', encoding='utf-8')
            rows.append(segment)
            exec(code, scope)
            self.assertFalse(report.exists())
            self.assertEqual(english.read_text(encoding='utf-8'), 'Existing translation')
            self.assertEqual((folder/'transcript/arabic/lesson.plain.txt').read_text(encoding='utf-8'), 'Example')

    def test_valid_transcript_and_hours(self):
        result = utils.transcript_to_srt('\ufeff[00:00-00:05] First\n\n[1:00:00-1:00:02] Last\n')
        self.assertIn('00:00:00,000 --> 00:00:05,000', result)
        self.assertIn('01:00:00,000 --> 01:00:02,000', result)

    def test_malformed_and_invalid_captions_fail(self):
        for sample in ('heading\n[00:00-00:05] Caption', '[00:04-00:04] Zero',
                       '[00:07-00:03] Backwards', '[00:61-01:02] Invalid',
                       '[00:10-00:20] A\n[00:05-00:09] B', '[00:00-00:05]  '):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                utils.transcript_to_srt(sample)

    def test_invalid_time_ranges(self):
        for first, last in (('-1', None), ('00:00', '00:00'), ('nan', '20'), ('00:61', None)):
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                utils.validate_time_range(first, last)

    def test_short_segments_still_have_duration(self):
        self.assertEqual(utils.parse_transcript(utils.format_cue(1.01, 1.1, 'Short')),
                         [(1.0, 2.0, 'Short')])

    def test_duplicate_stems_rejected_before_transcribing(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in ('Lesson.mp3', 'lesson.mp4'):
                (folder / name).touch()
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                utils.list_media(folder)

    def test_pairing_and_diagnostic_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            media, texts = folder / 'media', folder / 'texts'
            media.mkdir(); texts.mkdir()
            (media / 'lesson.mp3').touch()
            for name, text in {'lesson.txt': '[00:00-00:01] Hello', 'blank.txt': '',
                               'missing.txt': '[00:00-00:01] Missing',
                               'lesson.flagged.txt': 'report', 'lesson.plain.txt': 'plain'}.items():
                (texts / name).write_text(text, encoding='utf-8')
            pairs, blank, missing = utils.pair_transcripts(media, texts)
            self.assertEqual([(a.name, b.name) for a, b in pairs], [('lesson.txt', 'lesson.mp3')])
            self.assertEqual(blank, ['blank.txt'])
            self.assertEqual(missing, ['missing.txt'])

    def test_empty_decode_and_report_cleanup(self):
        alerts = utils.confidence_alerts([], 0, 60)
        self.assertEqual(len(alerts), 1)
        self.assertIn('No speech', alerts[0]['why'])
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / 'lesson.flagged.txt'
            utils.write_confidence_report(report, 'lesson.mp3', alerts)
            self.assertTrue(report.is_file())
            utils.write_confidence_report(report, 'lesson.mp3', [])
            self.assertFalse(report.exists())

    def test_notebook_syntax_and_empty_results(self):
        for file in ROOT.glob('*.ipynb'):
            data = json.loads(file.read_text(encoding='utf-8'))
            before = json.dumps(data, sort_keys=True)
            self.assertEqual(before, json.dumps(clean_notebook(data), sort_keys=True), file.name)
            for i, cell in enumerate(data['cells']):
                if cell['cell_type'] == 'code':
                    source = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                                       if not line.startswith('!'))
                    ast.parse(source, filename=f'{file.name} cell {i}')

    def test_colab_uses_same_parser(self):
        data = json.loads((ROOT / 'colab.ipynb').read_text(encoding='utf-8'))
        scope = {}
        exec(compile(''.join(data['cells'][4]['source']), '<Colab configuration>', 'exec'), scope)
        sample = '[00:00-00:01] Example\n[00:01-00:04] Continuation'
        self.assertEqual(scope['transcript_to_srt'](sample), utils.transcript_to_srt(sample))
        with self.assertRaises(ValueError):
            scope['transcript_to_srt']('heading\n' + sample)


class RedactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.arabic, self.english = self.root / 'arabic', self.root / 'english'
        self.arabic.mkdir(); self.english.mkdir()
        self.patches = [patch.object(redact, 'ARABIC_DIR', self.arabic),
                        patch.object(redact, 'ENGLISH_DIR', self.english),
                        patch.object(redact, 'REVIEW_FILE', self.root / 'review.txt')]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_redaction_backup_and_idempotence(self):
        path = self.arabic / 'lesson.txt'
        original = '[00:00-00:10] Garbled\n[00:10-00:20] Clear'
        path.write_text(original, encoding='utf-8')
        region = dict(start=0, end=10, message='[Audio cuts out]', keeps=[])
        redact.redact(path, [region])
        once = path.read_text(encoding='utf-8')
        self.assertEqual(redact.redact(path, [region])['status'], 'unchanged')
        self.assertEqual(path.read_text(encoding='utf-8'), once)
        self.assertEqual(path.with_suffix('.txt.bak').read_text(encoding='utf-8'), original)

    def test_changed_report_requires_approval_again(self):
        report = self.arabic / 'lesson.flagged.txt'
        report.write_text('[00:00-00:10] SEVERE: old\n', encoding='utf-8')
        regions = redact.load_flagged()
        regions[0]['approve'] = True
        redact.save_review(regions)
        report.write_text('[00:00-00:10] SEVERE: changed\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Stale approval'):
            redact.apply()
        updated = redact.review()
        self.assertFalse(updated[0]['approve'])

    def test_removed_report_clears_detected_approval(self):
        report = self.arabic / 'lesson.flagged.txt'
        report.write_text('[00:00-00:10] SEVERE: old\n', encoding='utf-8')
        regions = redact.load_flagged(); regions[0]['approve'] = True
        redact.save_review(regions); report.unlink()
        self.assertFalse(redact.review()[0]['approve'])

    def test_arabic_recovery_not_inserted_into_english(self):
        (self.english / 'lesson.txt').write_text('[00:00-00:10] English', encoding='utf-8')
        redact.save_review([dict(stem='lesson', start=0, end=10, severity='SEVERE', approve=True,
            message='[Audio cuts out]', keeps=[dict(start=5, end=10, text='عربي')])])
        with self.assertRaisesRegex(ValueError, 'Arabic keep'):
            redact.apply(target='both')


class TrimTests(unittest.TestCase):
    def test_replacement_failure_rolls_back_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            arabic, english, audio = folder/'arabic', folder/'english', folder/'input'
            for path in (arabic, english, audio):
                path.mkdir()
            original = '[00:00-00:10] First\n[00:10-00:20] Second\n'
            for path in (arabic, english):
                (path/'lesson.txt').write_text(original, encoding='utf-8')
            media = audio/'lesson.mp3'; media.write_bytes(b'original audio')
            real_replace = trim.os.replace
            def replace(src, dest):
                if Path(dest) == english/'lesson.txt':
                    raise OSError('replacement failure')
                return real_replace(src, dest)
            def encode(src, first, last, dest):
                dest.write_bytes(b'trimmed audio')
            with patch.multiple(trim, ARABIC_DIR=arabic, ENGLISH_DIR=english, INPUT_DIR=audio,
                                DRY_RUN=False, BACKUP=False), patch.object(trim, 'audio_duration', return_value=20), \
                 patch.object(trim, 'trim_audio', side_effect=encode), patch.object(trim.os, 'replace', side_effect=replace):
                with self.assertRaisesRegex(OSError, 'replacement failure'):
                    trim.process_stem('lesson', start='00:10')
            self.assertEqual(media.read_bytes(), b'original audio')
            for path in (arabic, english):
                self.assertEqual((path/'lesson.txt').read_text(encoding='utf-8'), original)

    def test_audio_failure_leaves_transcripts_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            arabic, english, audio = folder/'arabic', folder/'english', folder/'input'
            for p in (arabic, english, audio):
                p.mkdir()
            original = '[00:00-00:10] First\n[00:10-00:20] Second\n'
            for p in (arabic, english):
                (p/'lesson.txt').write_text(original, encoding='utf-8')
            (audio/'lesson.mp3').write_bytes(b'original audio')
            with patch.multiple(trim, ARABIC_DIR=arabic, ENGLISH_DIR=english, INPUT_DIR=audio,
                                DRY_RUN=False, BACKUP=False), patch.object(trim, 'audio_duration', return_value=20), \
                 patch.object(trim, 'trim_audio', side_effect=RuntimeError('encoder failure')):
                with self.assertRaisesRegex(RuntimeError, 'encoder failure'):
                    trim.process_stem('lesson', start='00:10')
            for p in (arabic, english):
                self.assertEqual((p/'lesson.txt').read_text(encoding='utf-8'), original)
            self.assertEqual((audio/'lesson.mp3').read_bytes(), b'original audio')

    def test_backup_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'test.txt'; path.write_text('original')
            trim.backup(path); path.write_text('changed'); trim.backup(path)
            self.assertEqual(path.with_suffix('.txt.bak').read_text(), 'original')


class EncoderTests(unittest.TestCase):
    def test_listed_but_unusable_gpu_falls_back(self):
        listing = subprocess.CompletedProcess([], 0, stdout=' V..... h264_nvenc GPU\n V..... libx264 CPU\n')
        calls = []
        def run(_, args, **kwargs):
            calls.append(args)
            if 'h264_nvenc' in args:
                raise RuntimeError('driver unavailable')
        with patch.object(video.subprocess, 'run', return_value=listing), patch.object(video, 'run_ffmpeg', side_effect=run):
            name, _ = video.choose_encoder(Path('ffmpeg'))
        self.assertEqual(name, 'libx264')
        self.assertEqual(len(calls), 2)

    def test_runtime_gpu_failure_retries_cpu_and_preserves_old_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); output = folder/'final.mp4'; output.write_bytes(b'old')
            def run(_, args, **kwargs):
                if 'h264_nvenc' in args:
                    raise RuntimeError('driver reset')
                Path(args[-1]).write_bytes(b'new')
            with patch.object(video, 'choose_encoder', side_effect=[('h264_nvenc', ['h264_nvenc']), ('libx264', ['libx264'])]), \
                 patch.object(video, 'run_ffmpeg', side_effect=run):
                video._encode_output('ffmpeg', [], output, folder, 23, 'fast', True, False)
            self.assertEqual(output.read_bytes(), b'new')
            with patch.object(video, 'choose_encoder', return_value=('libx264', ['libx264'])), \
                 patch.object(video, 'run_ffmpeg', side_effect=RuntimeError('bad input')):
                with self.assertRaises(RuntimeError):
                    video._encode_output('ffmpeg', [], output, folder, 23, 'fast', False, False)
            self.assertEqual(output.read_bytes(), b'new')


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg is not installed')
class SyntheticRenderingTests(unittest.TestCase):
    def test_standard_and_nature_rendering_with_silent_intro(self):
        ffmpeg, ffprobe = video.find_ffmpeg()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            intro, background, audio = folder/'intro.mp4', folder/'background.mp4', folder/'audio.wav'
            for p, color, duration in ((intro, 'blue', '0.25'), (background, 'green', '1.0')):
                video.run_ffmpeg(ffmpeg, ['-f', 'lavfi', '-i', f'color=c={color}:s=160x90:r=24:d={duration}',
                                          '-c:v', 'libx264', '-pix_fmt', 'yuv420p', p])
            video.run_ffmpeg(ffmpeg, ['-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.6', audio])
            srt = folder/'test.srt'; srt.write_text('1\n00:00:00,000 --> 00:00:01,000\nTest caption\n', encoding='utf-8')
            output = folder/'final.mp4'
            video.render_loop_video(audio, srt, output, intro, background, prefer_gpu=False,
                                     width=160, height=90, fade=0.1)
            duration, streams = video.probe_media(ffprobe, output)
            self.assertAlmostEqual(duration, 1.25, delta=0.1)
            self.assertTrue({'audio', 'video'} <= streams)
            burned = folder/'captioned.mp4'
            video.burn_subtitles(output, srt, burned, prefer_gpu=False)
            self.assertTrue(burned.is_file())
            colab = json.loads((ROOT/'colab.ipynb').read_text(encoding='utf-8'))
            fake_colab = SimpleNamespace(files=SimpleNamespace(download=lambda *a: None))
            scope = {'VIDEO_PATH': output, 'SRT_PATH': srt}
            with contextlib.chdir(folder), patch.dict(sys.modules, {'google.colab': fake_colab}):
                code = ''.join(colab['cells'][12]['source']).replace('PREFER_GPU = True', 'PREFER_GPU = False')
                exec(compile(code, '<Colab captions>', 'exec'), scope)
            self.assertTrue(scope['OUT_VIDEO'].is_file())
            data = json.loads((ROOT/'transcribe_nature.ipynb').read_text(encoding='utf-8'))
            scope = {}
            exec(compile(''.join(data['cells'][11]['source']), '<nature settings>', 'exec'), scope)
            exec(compile(''.join(data['cells'][12]['source']), '<nature helpers>', 'exec'), scope)
            scope.update(NATURE_FOLDER=str(folder), INCLUDE_INTRO=True, INTRO_PATH=str(intro),
                         TARGET_W=160, TARGET_H=90, MIN_CLIP_SECONDS=0.3, MAX_CLIP_SECONDS=0.5,
                         PREFER_GPU=False, FFMPEG_PATH=str(ffmpeg), WATERMARK_TEXT='', BLUR_SIGMA=0,
                         FADE_SECONDS=0.1, SUBTITLE_FONT='Arial')
            clips = scope['_prepare_nature']()
            nature = folder/'nature.mp4'
            scope['_render_nature'](audio, srt, nature, clips=clips, seed=2026)
            duration, streams = video.probe_media(ffprobe, nature)
            self.assertGreaterEqual(duration, 0.85)
            self.assertTrue({'audio', 'video'} <= streams)
            self.assertTrue(nature.with_suffix('.clips.json').is_file())


if __name__ == '__main__':
    unittest.main()
