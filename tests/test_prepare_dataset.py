"""
tests/test_prepare_dataset.py

Regression tests for ball_tracking/training/prepare_dataset.py's
video-normalization step, added 2026-08-15 after confirming via ffprobe
that a raw native .MOV capture and the same footage after WhatsApp's
own compression are measurably different files (resolution, codec,
bitrate) — independent of which physical device captured either one.
Training images must go through the SAME compressor real coach uploads
do (orchestrator.compress_video_file), or the model learns from one
visual regime and gets scored on another.
"""

import os

import pytest

from ball_tracking.training import prepare_dataset as pd


class TestNormalizedVideoPath:
    def test_compresses_with_no_fps_cap_when_not_cached(self, tmp_path, monkeypatch):
        """Real bug this guards against: stored ball_tracking_labels rows
        use frame_index values captured against the ORIGINAL video's own
        frame numbering (label_tool.py never compresses before display).
        Resampling fps during normalization would shift which frame
        lands at which index, silently misaligning a label with the
        wrong image — confirmed a real risk, not theoretical (at least
        one currently-labeled clip is a genuine ~120fps recording).
        max_fps must always be passed as None here, never the default."""
        monkeypatch.setattr(pd, "COMPRESSED_CACHE_DIR", str(tmp_path))
        captured = {}

        def _fake_compress(src, dest, **kwargs):
            captured["src"] = src
            captured["dest"] = dest
            captured["kwargs"] = kwargs
            with open(dest, "wb") as f:
                f.write(b"fake compressed video")

        monkeypatch.setattr(pd, "compress_video_file", _fake_compress)

        result = pd._normalized_video_path("C:/source/clip one.mp4", "clip one.mp4")

        assert captured["kwargs"].get("max_fps") is None
        assert result == os.path.join(str(tmp_path), "clip_one.mp4")
        assert os.path.exists(result)

    def test_reuses_cached_copy_without_recompressing(self, tmp_path, monkeypatch):
        """A source video's own content never changes once shot —
        re-compressing it on every single prepare_dataset.py run would
        be pure wasted ffmpeg time across dozens of clips."""
        monkeypatch.setattr(pd, "COMPRESSED_CACHE_DIR", str(tmp_path))
        cached_path = os.path.join(str(tmp_path), "clip_one.mp4")
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(cached_path, "wb") as f:
            f.write(b"already compressed")

        def _explode(*a, **k):
            raise AssertionError("compress_video_file should not be called when already cached")

        monkeypatch.setattr(pd, "compress_video_file", _explode)

        result = pd._normalized_video_path("C:/source/clip one.mp4", "clip one.mp4")

        assert result == cached_path
