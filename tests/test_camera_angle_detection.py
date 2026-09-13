"""
tests/test_camera_angle_detection.py

Regression test for a real crash found in a robustness audit (2026-09-13):
estimate_camera_angle assumed at least one real landmark row always
exists. An empty dataframe (e.g. from a video main.extract_video_landmarks
couldn't read a single frame from) reached df.iloc[0] unguarded and raised
a raw IndexError instead of this function's own honest "unavailable"
result.
"""

import pandas as pd

import camera_angle_detection as cad


class TestEmptyDataframeGuard:
    def test_empty_dataframe_returns_unavailable_not_a_crash(self):
        df = pd.DataFrame(columns=[
            "LEFT_SHOULDER_x", "RIGHT_SHOULDER_x", "NOSE_y",
            "LEFT_ANKLE_y", "RIGHT_ANKLE_y",
        ])
        result = cad.estimate_camera_angle(df, 0, frame_width=640, frame_height=480)
        assert result.angle == "unavailable"
        assert result.ratio is None

    def test_empty_dataframe_with_missing_columns_still_returns_unavailable(self):
        """Belt-and-suspenders: an empty df with none of the required
        columns at all must still degrade gracefully, whichever check
        fires first."""
        df = pd.DataFrame()
        result = cad.estimate_camera_angle(df, 5, frame_width=640, frame_height=480)
        assert result.angle == "unavailable"
