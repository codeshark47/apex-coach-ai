"""
tests/test_main_model_download.py

Regression test for a real bug found in a robustness audit (2026-09-13):
main.extract_video_landmarks used to download the pose-landmarker model
straight to its final path with no validation. A truncated download (or
an HTML error page saved as if it were the model) left a corrupt file
there permanently -- every subsequent run's `os.path.exists` check passed,
so it never re-downloaded and crashed deeper in the function on every
single call until someone manually deleted the file by hand.
"""

import os
from unittest.mock import patch

import main


def _fake_urlretrieve_factory(content: bytes):
    def _fake_urlretrieve(url, path):
        with open(path, "wb") as f:
            f.write(content)
    return _fake_urlretrieve


class TestModelDownloadValidation:
    def test_truncated_download_is_rejected_and_cleaned_up(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_video = tmp_path / "clip.mp4"
        fake_video.write_bytes(b"irrelevant -- fails before this is read")

        with patch("urllib.request.urlretrieve", side_effect=_fake_urlretrieve_factory(b"<html>error page</html>")):
            result = main.extract_video_landmarks(str(fake_video), str(tmp_path / "out.csv"))

        assert result["status"] == "error"
        assert "download" in result["error_message"].lower()
        # The poisoned partial file must not be left behind for the next
        # run to silently trust.
        assert not os.path.exists(os.path.join("models", "pose_landmarker_full.task"))
        assert not os.path.exists(os.path.join("models", "pose_landmarker_full.task.part"))

    def test_a_genuinely_valid_model_download_is_accepted_and_kept(self, tmp_path, monkeypatch):
        """A real, valid model download must be moved into place and kept
        for next time -- not just "big enough," genuinely loadable."""
        monkeypatch.chdir(tmp_path)
        fake_video = tmp_path / "corrupt.mp4"
        fake_video.write_bytes(b"garbage -- just needs to make it past the model-download step")
        real_model_path = os.path.join(
            os.path.dirname(os.path.abspath(main.__file__)), "models", "pose_landmarker_full.task"
        )
        real_model_bytes = open(real_model_path, "rb").read()

        with patch("urllib.request.urlretrieve", side_effect=_fake_urlretrieve_factory(real_model_bytes)):
            result = main.extract_video_landmarks(str(fake_video), str(tmp_path / "out.csv"))

        # The "video" itself is garbage, so this still ends in an error --
        # but a DIFFERENT one (unreadable video), proving the model itself
        # downloaded, validated, and loaded successfully.
        model_path = os.path.join("models", "pose_landmarker_full.task")
        assert os.path.exists(model_path)
        assert not os.path.exists(model_path + ".part")
        assert "model" not in (result.get("error_message") or "").lower()
