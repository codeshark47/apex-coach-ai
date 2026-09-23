"""
tests/test_ball_tracking_status.py

Regression tests for tools/ball_tracking_status.py's staleness check
(2026-09-24, real confirmed bug): a label's stored pixel position is
only trustworthy if the video file it was clicked against hasn't
changed since. Confirmed twice, 10 days apart, on the same real clip
(WhatsApp Video 2026-08-14 at 5.12.16 PM.mp4) -- its file's mtime sits
after its earliest label's created_at, and the stored ground-truth
point for frame 306 lands in tree branches in the CURRENT file, nowhere
near the bowler. This had no automated check before; these tests cover
the one now built.
"""

import datetime
import os

import pytest

import tools.ball_tracking_status as bts


class TestParseCreatedAt:
    def test_parses_a_real_supabase_timestamp(self):
        result = bts._parse_created_at("2026-08-14T12:52:29.075411+00:00")
        assert result is not None
        assert result.year == 2026 and result.month == 8 and result.day == 14

    def test_parses_a_z_suffixed_timestamp(self):
        result = bts._parse_created_at("2026-08-14T12:52:29Z")
        assert result is not None

    def test_none_for_empty_or_missing(self):
        assert bts._parse_created_at(None) is None
        assert bts._parse_created_at("") is None

    def test_none_for_unparseable_garbage(self):
        assert bts._parse_created_at("not a timestamp") is None


class TestFindLocalVideo:
    def test_finds_a_real_file_under_input(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs("input")
        open(os.path.join("input", "clip.mp4"), "w").close()
        assert bts._find_local_video("clip.mp4") == os.path.join("input", "clip.mp4")

    def test_none_when_no_local_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs("input")
        assert bts._find_local_video("nowhere.mp4") is None


class TestReportStaleness:
    def _touch(self, path, mtime: datetime.datetime):
        open(path, "w").close()
        ts = mtime.timestamp()
        os.utime(path, (ts, ts))

    def test_flags_a_file_modified_well_after_its_earliest_label(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        os.makedirs("input")
        labeled_at = datetime.datetime(2026, 8, 14, 12, 52, 29, tzinfo=datetime.timezone.utc)
        modified_at = labeled_at + datetime.timedelta(hours=1)  # well past the 5-minute buffer
        self._touch(os.path.join("input", "stale_clip.mp4"), modified_at)

        rows = [{"source_video_filename": "stale_clip.mp4",
                 "created_at": labeled_at.isoformat()}]
        bts._report_staleness(rows)
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "stale_clip.mp4" in out

    def test_does_not_flag_a_file_modified_before_its_label(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        os.makedirs("input")
        labeled_at = datetime.datetime(2026, 8, 14, 12, 52, 29, tzinfo=datetime.timezone.utc)
        modified_at = labeled_at - datetime.timedelta(hours=1)  # file existed before labeling -- fine
        self._touch(os.path.join("input", "clean_clip.mp4"), modified_at)

        rows = [{"source_video_filename": "clean_clip.mp4",
                 "created_at": labeled_at.isoformat()}]
        bts._report_staleness(rows)
        out = capsys.readouterr().out
        assert "WARNING" not in out
        assert "No staleness flags" in out

    def test_a_small_mtime_gap_within_the_buffer_is_not_flagged(self, tmp_path, monkeypatch, capsys):
        """A plain copy/move can legitimately touch mtime by a few
        seconds without changing a single frame of content -- must not
        cry wolf on that."""
        monkeypatch.chdir(tmp_path)
        os.makedirs("input")
        labeled_at = datetime.datetime(2026, 8, 14, 12, 52, 29, tzinfo=datetime.timezone.utc)
        modified_at = labeled_at + datetime.timedelta(seconds=30)
        self._touch(os.path.join("input", "copied_clip.mp4"), modified_at)

        rows = [{"source_video_filename": "copied_clip.mp4",
                 "created_at": labeled_at.isoformat()}]
        bts._report_staleness(rows)
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_uses_the_earliest_label_when_a_clip_has_several(self, tmp_path, monkeypatch, capsys):
        """A clip's earliest label is the one that matters -- if the
        file was modified after the FIRST click but before a later one,
        that's still evidence the first click's stored position may no
        longer be trustworthy."""
        monkeypatch.chdir(tmp_path)
        os.makedirs("input")
        first_label = datetime.datetime(2026, 8, 14, 10, 0, 0, tzinfo=datetime.timezone.utc)
        second_label = datetime.datetime(2026, 8, 14, 13, 0, 0, tzinfo=datetime.timezone.utc)
        modified_at = datetime.datetime(2026, 8, 14, 12, 0, 0, tzinfo=datetime.timezone.utc)  # between the two
        self._touch(os.path.join("input", "multi_label_clip.mp4"), modified_at)

        rows = [
            {"source_video_filename": "multi_label_clip.mp4", "created_at": first_label.isoformat()},
            {"source_video_filename": "multi_label_clip.mp4", "created_at": second_label.isoformat()},
        ]
        bts._report_staleness(rows)
        out = capsys.readouterr().out
        assert "WARNING" in out  # caught via the earliest label, even though the later one is fine

    def test_a_clip_with_no_local_file_is_silently_skipped_not_flagged(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        os.makedirs("input")
        rows = [{"source_video_filename": "nowhere.mp4",
                 "created_at": "2026-08-14T12:52:29+00:00"}]
        bts._report_staleness(rows)
        out = capsys.readouterr().out
        assert "WARNING" not in out
        assert "0 labeled clip(s) have a local file" in out
