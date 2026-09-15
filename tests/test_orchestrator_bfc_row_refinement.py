"""
tests/test_orchestrator_bfc_row_refinement.py

Regression tests for orchestrator._refine_bfc_row_raw (2026-09-15) — the
LAST of the 7 bowling metrics (rear_knee_angle, rear_hip_flexion) to get
the raw-re-extraction + identity-consistent-candidate treatment already
proven for release_height, head_stability, the annotated skeleton, and
front_knee_bracing/trunk_lean/hip_shoulder_separation.

Also covers a real, separate bug found while auditing this gap: the
_nearest_complete_row completeness check for bfc_row only ever verified
the trail HIP/KNEE/ANKLE columns, but calculate_rear_hip_flexion also
needs both SHOULDER columns — a row could pass as "complete" and still
fail inside that function.
"""

from unittest.mock import patch

import pandas as pd

import orchestrator as o


def _base_df(frames):
    rows = []
    for f in frames:
        rows.append({
            "frame": f,
            "NOSE_x": 0.25, "NOSE_y": 0.35,
            "RIGHT_HIP_x": 0.26, "RIGHT_HIP_y": 0.60,
            "RIGHT_KNEE_x": 0.26, "RIGHT_KNEE_y": 0.75,
            "RIGHT_ANKLE_x": 0.26, "RIGHT_ANKLE_y": 0.90,
            "LEFT_SHOULDER_x": 0.24, "RIGHT_SHOULDER_x": 0.28,
            "LEFT_SHOULDER_y": 0.40, "RIGHT_SHOULDER_y": 0.40,
        })
    return pd.DataFrame(rows)


class TestRefineBfcRowRaw:
    def test_a_genuine_refinement_is_merged_into_the_fallback_row(self):
        df = _base_df([30])
        fallback = df.iloc[0]
        raw = {30: [{"NOSE": (0.25, 0.35, 1.0), "RIGHT_HIP": (0.261, 0.601, 1.0), "RIGHT_KNEE": (0.261, 0.751, 1.0)}]}
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_bfc_row_raw("fake.mp4", 30.0, df, 30, "right", fallback)
        assert result["RIGHT_HIP_x"] == 0.261
        assert result["RIGHT_KNEE_x"] == 0.261

    def test_a_wrong_person_candidate_is_rejected_and_fallback_kept(self):
        df = _base_df([30])
        fallback = df.iloc[0]
        raw = {30: [{"NOSE": (0.70, 0.35, 1.0), "RIGHT_HIP": (0.70, 0.60, 1.0), "RIGHT_KNEE": (0.70, 0.75, 1.0)}]}  # bystander
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_bfc_row_raw("fake.mp4", 30.0, df, 30, "right", fallback)
        assert result["RIGHT_HIP_x"] == 0.26  # unchanged -- rejected

    def test_recovers_even_when_fallback_row_is_none(self):
        """The exact real scenario this exists for: _nearest_complete_row
        found NOTHING complete within +/-10 frames in the smoothed df
        (fallback_row=None), but a fresh, identity-validated raw pass can
        still recover it. Reference for identity validation comes from
        OTHER frames in df (not frame 30 itself, since that's what's
        missing)."""
        df = _base_df([25, 35])  # frame 30 itself has no row at all
        raw = {30: [{"NOSE": (0.25, 0.35, 1.0), "RIGHT_HIP": (0.261, 0.601, 1.0), "RIGHT_KNEE": (0.261, 0.751, 1.0),
                     "RIGHT_ANKLE": (0.261, 0.901, 1.0)}]}
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_bfc_row_raw("fake.mp4", 30.0, df, 30, "right", None)
        assert result is not None
        assert result["RIGHT_HIP_x"] == 0.261
        assert result["frame"] == 30

    def test_no_candidate_and_no_fallback_returns_none(self):
        df = _base_df([25, 35])
        raw = {}
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_bfc_row_raw("fake.mp4", 30.0, df, 30, "right", None)
        assert result is None

    def test_extraction_failure_falls_back_to_original(self):
        df = _base_df([30])
        fallback = df.iloc[0]
        with patch("orchestrator.extract_raw_landmarks_window", side_effect=Exception("boom")):
            result = o._refine_bfc_row_raw("fake.mp4", 30.0, df, 30, "right", fallback)
        assert result["RIGHT_HIP_x"] == 0.26

    def test_left_arm_bowler_uses_left_trail_side_landmarks(self):
        df = _base_df([30])
        captured = {}

        def _capture_needed(video_path, fps, needed, start, end):
            captured["needed"] = needed
            return {}

        with patch("orchestrator.extract_raw_landmarks_window", side_effect=_capture_needed):
            o._refine_bfc_row_raw("fake.mp4", 30.0, df, 30, "left", None)

        assert "LEFT_KNEE" in captured["needed"]
        assert "LEFT_ANKLE" in captured["needed"]
        assert "RIGHT_KNEE" not in captured["needed"]
        assert "LEFT_SHOULDER" in captured["needed"]
        assert "RIGHT_SHOULDER" in captured["needed"]


class TestNearestCompleteRowIncludesShoulders:
    def test_a_row_missing_shoulders_is_not_treated_as_complete(self):
        """REGRESSION: the old required_cols list for bfc_row only checked
        trail HIP/KNEE/ANKLE -- a row with those complete but NaN
        shoulders used to pass as "complete" even though calculate_rear_
        hip_flexion (which also needs both shoulders) would still fail on
        it silently."""
        df = pd.DataFrame([{
            "frame": 30,
            "RIGHT_HIP_x": 0.26, "RIGHT_HIP_y": 0.60,
            "RIGHT_KNEE_x": 0.26, "RIGHT_KNEE_y": 0.75,
            "RIGHT_ANKLE_x": 0.26, "RIGHT_ANKLE_y": 0.90,
            "LEFT_SHOULDER_x": float("nan"), "LEFT_SHOULDER_y": float("nan"),
            "RIGHT_SHOULDER_x": float("nan"), "RIGHT_SHOULDER_y": float("nan"),
        }])
        required_cols = [
            "RIGHT_HIP_x", "RIGHT_HIP_y", "RIGHT_KNEE_x", "RIGHT_KNEE_y",
            "RIGHT_ANKLE_x", "RIGHT_ANKLE_y",
            "LEFT_SHOULDER_x", "LEFT_SHOULDER_y", "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y",
        ]
        result = o._nearest_complete_row(df, 30, required_cols, max_search=0)
        assert result is None  # correctly rejected -- shoulders are NaN
