"""
tests/test_orchestrator_release_landmarks_and_head_stability.py

Regression tests for the same real, coach-reported identity-hijack bug
covered in tests/test_orchestrator_skeleton_refinement.py (2026-09-15),
extended to the other two raw-re-extraction "sharpen the reading"
functions that share the exact same vulnerability:

- _refine_release_landmarks_raw feeds the numeric Release Height metric.
- _refine_head_stability_window_raw feeds the numeric Head Stability
  metric.

Both call main.extract_raw_landmarks_window, an UNSEEDED, single-pose
MediaPipe pass with no idea which person the coach's seed clicks
identified. Without a check, either function can silently blend a
bystander's landmarks into what should be a single, consistently-
identified bowler's numeric metrics.
"""

from unittest.mock import patch

import pandas as pd

import orchestrator as o


def _base_df(frames):
    rows = []
    for f in frames:
        rows.append({
            "frame": f,
            "NOSE_x": 0.25, "NOSE_y": 0.55,
            "LEFT_HIP_x": 0.24, "RIGHT_HIP_x": 0.26,
            "LEFT_HIP_y": 0.60, "RIGHT_HIP_y": 0.60,
        })
    return pd.DataFrame(rows)


class TestRefineReleaseLandmarksRawIdentityConsistency:
    def test_a_different_person_far_away_is_rejected(self):
        """The bystander scenario: raw re-extraction at the BR frame
        lands on a person far from the seeded identity's own NOSE at
        that frame. Must return None for that row, not a mixed-identity
        reading, so the caller falls back to the smoothed value."""
        df = _base_df([10, 20])
        raw = {
            20: {"NOSE": (0.70, 0.55, 1.0), "RIGHT_WRIST": (0.71, 0.30, 1.0),
                 "LEFT_ANKLE": (0.69, 0.95, 1.0), "RIGHT_ANKLE": (0.72, 0.95, 1.0),
                 "LEFT_KNEE": (0.69, 0.80, 1.0), "RIGHT_KNEE": (0.72, 0.80, 1.0),
                 "LEFT_HIP": (0.69, 0.60, 1.0), "RIGHT_HIP": (0.72, 0.60, 1.0)},
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            br_row, height_row = o._refine_release_landmarks_raw(
                "fake.mp4", 30.0, "right", 20, 10, df
            )
        assert br_row is None

    def test_a_genuine_same_person_refinement_is_returned(self):
        """A real refinement (small correction, same person) must still
        come back as usable data."""
        df = _base_df([10, 20])
        raw = {
            20: {"NOSE": (0.252, 0.548, 1.0), "RIGHT_WRIST": (0.30, 0.30, 1.0),
                 "LEFT_ANKLE": (0.24, 0.95, 1.0), "RIGHT_ANKLE": (0.26, 0.95, 1.0),
                 "LEFT_KNEE": (0.24, 0.80, 1.0), "RIGHT_KNEE": (0.26, 0.80, 1.0),
                 "LEFT_HIP": (0.242, 0.602, 1.0), "RIGHT_HIP": (0.262, 0.598, 1.0)},
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            br_row, height_row = o._refine_release_landmarks_raw(
                "fake.mp4", 30.0, "right", 20, 10, df
            )
        assert br_row is not None
        assert br_row["NOSE_x"] == 0.252

    def test_no_df_passed_skips_the_check_for_backward_compatibility(self):
        """df defaults to None so any caller not yet passing it keeps
        working exactly as before (no identity check applied)."""
        raw = {
            20: {"NOSE": (0.70, 0.55, 1.0)},
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            br_row, height_row = o._refine_release_landmarks_raw(
                "fake.mp4", 30.0, "right", 20, 10
            )
        assert br_row is not None
        assert br_row["NOSE_x"] == 0.70

    def test_extraction_failure_returns_none_none(self):
        df = _base_df([10, 20])
        with patch("orchestrator.extract_raw_landmarks_window", side_effect=Exception("boom")):
            br_row, height_row = o._refine_release_landmarks_raw(
                "fake.mp4", 30.0, "right", 20, 10, df
            )
        assert br_row is None
        assert height_row is None


class TestRefineHeadStabilityWindowRawIdentityConsistency:
    def test_a_different_person_far_away_is_not_patched_in(self):
        df = _base_df([10, 11, 12])
        raw = {
            11: {"NOSE": (0.70, 0.55, 1.0), "LEFT_SHOULDER": (0.68, 0.50, 1.0),
                 "RIGHT_SHOULDER": (0.72, 0.50, 1.0)},
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_head_stability_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.25  # unchanged -- the wrong-person patch was rejected

    def test_a_genuine_same_person_refinement_is_patched_in(self):
        df = _base_df([10, 11, 12])
        raw = {
            11: {"NOSE": (0.252, 0.548, 1.0), "LEFT_SHOULDER": (0.20, 0.45, 1.0),
                 "RIGHT_SHOULDER": (0.30, 0.45, 1.0)},
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_head_stability_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.252  # the genuine refinement WAS applied

    def test_a_frame_with_no_existing_nose_reference_still_gets_patched(self):
        """No hips are in this function's landmark set at all, so a
        missing NOSE reference means nothing to check against -- the raw
        data is the only option and should still be used."""
        df = pd.DataFrame([{"frame": 20, "NOSE_x": float("nan"), "NOSE_y": float("nan"),
                             "LEFT_HIP_x": float("nan"), "RIGHT_HIP_x": float("nan"),
                             "LEFT_HIP_y": float("nan"), "RIGHT_HIP_y": float("nan")}])
        raw = {20: {"NOSE": (0.5, 0.5, 1.0)}}
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_head_stability_window_raw("fake.mp4", 30.0, df, 20, 20)

        row = result[result["frame"] == 20].iloc[0]
        assert row["NOSE_x"] == 0.5

    def test_extraction_failure_returns_original_df_unchanged(self):
        df = _base_df([10, 11])
        with patch("orchestrator.extract_raw_landmarks_window", side_effect=Exception("boom")):
            result = o._refine_head_stability_window_raw("fake.mp4", 30.0, df, 10, 11)
        pd.testing.assert_frame_equal(result, df)
