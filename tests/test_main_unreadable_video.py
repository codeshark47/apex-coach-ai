"""
tests/test_main_unreadable_video.py

Regression test for a real bug found in a robustness audit (2026-09-13):
a corrupted/truncated/0-byte upload, or a genuinely unreadable codec
(most likely when ffmpeg isn't on PATH — see
orchestrator.compress_video_file's own comment on that fallback), used to
fall all the way through extract_video_landmarks reporting
"status": "success" with an empty landmarks dataframe. Downstream code
(camera_angle_detection.estimate_camera_angle's own df.iloc[0]) assumes
at least one real frame exists and crashed on this silently-empty result
several calls later instead.

Deliberately a REAL file, not a mocked cv2.VideoCapture — a genuinely
corrupt file is what a real coach's bad upload looks like, and this
exercises the actual cv2/MediaPipe code path this bug lived in, not an
assumption about how it's mocked. Costs a few real seconds (loading the
real pose model), same tradeoff test_main_identity_walk.py's real-model
tests already accept elsewhere in this suite.
"""

import os

import main


def test_corrupt_video_file_returns_a_clear_error_not_a_silent_empty_success(tmp_path):
    fake_video = tmp_path / "corrupt.mp4"
    fake_video.write_bytes(b"this is not a real video file, just garbage bytes")
    out_csv = str(tmp_path / "landmarks.csv")

    result = main.extract_video_landmarks(str(fake_video), out_csv)

    assert result["status"] == "error"
    assert "error_message" in result
    assert result["error_message"]  # non-empty, real message
