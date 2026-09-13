"""
tests/test_batting_orchestrator_events.py

Regression test for a real bug found in a robustness audit (2026-09-13):
detect_batting_events used to return fabricated frame indices for a
too-short clip (see tests/test_batting_events.py). Fixed to return None
events with an "error" key instead -- this test covers the CALLER side:
extract_and_detect_batting_events must catch that signal and fail
cleanly, since several lines further down assume STANCE/BACKLIFT are
real ints (e.g. max(events["STANCE"], events["BACKLIFT"]), which raises
TypeError on None).
"""

import pandas as pd

import batting_orchestrator as bo


def test_caller_reports_failure_instead_of_proceeding_with_null_events(tmp_path, monkeypatch):
    def _fake_extract_video_landmarks(video_path, output_csv_path, **kwargs):
        pd.DataFrame({"frame": range(3)}).to_csv(output_csv_path, index=False)
        return {"status": "success", "fps": 30}

    monkeypatch.setattr(bo, "extract_video_landmarks", _fake_extract_video_landmarks)
    result = bo.extract_and_detect_batting_events(
        "fake.mp4", output_dir=str(tmp_path), camera_angle_override="side_on",
    )
    assert result["status"] == "failed"
    assert result["stage"] == "event_detection"
    assert "message" in result and result["message"]
