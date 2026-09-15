"""
tests/test_orchestrator_release_landmarks_and_head_stability.py

Regression tests for the same real, coach-reported identity-hijack bug
covered in tests/test_orchestrator_skeleton_refinement.py (2026-09-15),
extended to the other two raw-re-extraction "sharpen the reading"
functions that share the exact same vulnerability:

- _refine_release_landmarks_raw feeds the numeric Release Height metric.
- _refine_head_stability_window_raw feeds the numeric Head Stability
  metric.

Both call main.extract_raw_landmarks_window, an UNSEEDED MediaPipe pass
with no idea which person the coach's seed clicks identified, which
returns every candidate it detects per frame (not just its own top-
ranked pick -- trusting that ranking was itself part of the bug, since a
static/unblurred bystander often outranks the actual moving subject).
Without _select_identity_consistent_candidate choosing among them by
comparing to the seeded walk's own identity, either function could
silently blend a bystander's landmarks into what should be a single,
consistently-identified bowler's numeric metrics.
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
        """The bystander scenario: raw re-extraction at the BR frame's
        only candidate is a person far from the seeded identity's own
        NOSE at that frame. Must return None for that row, not a mixed-
        identity reading, so the caller falls back to the smoothed
        value."""
        df = _base_df([10, 20])
        raw = {
            20: [{"NOSE": (0.70, 0.55, 1.0), "RIGHT_WRIST": (0.71, 0.30, 1.0),
                  "LEFT_ANKLE": (0.69, 0.95, 1.0), "RIGHT_ANKLE": (0.72, 0.95, 1.0),
                  "LEFT_KNEE": (0.69, 0.80, 1.0), "RIGHT_KNEE": (0.72, 0.80, 1.0),
                  "LEFT_HIP": (0.69, 0.60, 1.0), "RIGHT_HIP": (0.72, 0.60, 1.0)}],
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
            20: [{"NOSE": (0.252, 0.548, 1.0), "RIGHT_WRIST": (0.30, 0.30, 1.0),
                  "LEFT_ANKLE": (0.24, 0.95, 1.0), "RIGHT_ANKLE": (0.26, 0.95, 1.0),
                  "LEFT_KNEE": (0.24, 0.80, 1.0), "RIGHT_KNEE": (0.26, 0.80, 1.0),
                  "LEFT_HIP": (0.242, 0.602, 1.0), "RIGHT_HIP": (0.262, 0.598, 1.0)}],
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            br_row, height_row = o._refine_release_landmarks_raw(
                "fake.mp4", 30.0, "right", 20, 10, df
            )
        assert br_row is not None
        assert br_row["NOSE_x"] == 0.252

    def test_the_correct_candidate_is_picked_even_when_ranked_second(self):
        """THE EXACT SCENARIO an external (Gemini) review flagged: MediaPipe's
        own top-ranked candidate is the wrong (static, unblurred) person,
        but the real bowler was ALSO detected in the same frame, just
        ranked second. Must select the identity-consistent one, not just
        reject the frame because the top pick was wrong."""
        df = _base_df([10, 20])
        raw = {
            20: [
                {"NOSE": (0.70, 0.55, 1.0)},  # ranked first -- the bystander, no other landmarks
                {"NOSE": (0.252, 0.548, 1.0), "RIGHT_WRIST": (0.30, 0.30, 1.0),
                 "LEFT_ANKLE": (0.24, 0.95, 1.0), "RIGHT_ANKLE": (0.26, 0.95, 1.0),
                 "LEFT_KNEE": (0.24, 0.80, 1.0), "RIGHT_KNEE": (0.26, 0.80, 1.0),
                 "LEFT_HIP": (0.242, 0.602, 1.0), "RIGHT_HIP": (0.262, 0.598, 1.0)},  # ranked second -- the bowler
            ],
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            br_row, height_row = o._refine_release_landmarks_raw(
                "fake.mp4", 30.0, "right", 20, 10, df
            )
        assert br_row is not None
        assert br_row["NOSE_x"] == 0.252

    def test_no_df_passed_skips_the_check_for_backward_compatibility(self):
        """df defaults to None so any caller not yet passing it keeps
        working (falls back to MediaPipe's own top-ranked candidate, no
        identity check applied)."""
        raw = {
            20: [{"NOSE": (0.70, 0.55, 1.0)}],
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
            11: [{"NOSE": (0.70, 0.55, 1.0), "LEFT_SHOULDER": (0.68, 0.50, 1.0),
                  "RIGHT_SHOULDER": (0.72, 0.50, 1.0)}],
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_head_stability_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.25  # unchanged -- the wrong-person patch was rejected

    def test_a_genuine_same_person_refinement_is_patched_in(self):
        df = _base_df([10, 11, 12])
        raw = {
            11: [{"NOSE": (0.252, 0.548, 1.0), "LEFT_SHOULDER": (0.20, 0.45, 1.0),
                  "RIGHT_SHOULDER": (0.30, 0.45, 1.0)}],
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_head_stability_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.252  # the genuine refinement WAS applied

    def test_the_correct_candidate_is_picked_even_when_ranked_second(self):
        """Same real scenario as the release-landmarks test above, applied
        to the head-stability window: the bystander is MediaPipe's top
        pick, the bowler is detected too but ranked second."""
        df = _base_df([10, 11, 12])
        raw = {
            11: [
                {"NOSE": (0.70, 0.55, 1.0), "LEFT_SHOULDER": (0.68, 0.50, 1.0), "RIGHT_SHOULDER": (0.72, 0.50, 1.0)},
                {"NOSE": (0.252, 0.548, 1.0), "LEFT_SHOULDER": (0.20, 0.45, 1.0), "RIGHT_SHOULDER": (0.30, 0.45, 1.0)},
            ],
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_head_stability_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.252

    def test_a_frame_with_no_reference_anywhere_nearby_is_rejected(self):
        """REGRESSION (2026-09-15): must FAIL CLOSED, not patch in an
        unverifiable identity -- see _select_identity_consistent_candidate's
        docstring for the real bug this protects against (the seeded walk
        frequently has no data exactly at the release frame, which is
        also exactly when the raw pass is most likely to find a
        bystander instead)."""
        df = pd.DataFrame([{"frame": 20, "NOSE_x": float("nan"), "NOSE_y": float("nan"),
                             "LEFT_HIP_x": float("nan"), "RIGHT_HIP_x": float("nan"),
                             "LEFT_HIP_y": float("nan"), "RIGHT_HIP_y": float("nan")}])
        raw = {20: [{"NOSE": (0.5, 0.5, 1.0)}]}
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_head_stability_window_raw("fake.mp4", 30.0, df, 20, 20)

        row = result[result["frame"] == 20].iloc[0]
        assert pd.isna(row["NOSE_x"])

    def test_a_frame_with_no_reference_of_its_own_falls_back_to_a_nearby_frame(self):
        """The frame being refined has no seeded NOSE of its own, but a
        nearby frame in the same df does -- that should still be used to
        validate identity rather than failing closed unnecessarily."""
        df = pd.DataFrame([
            {"frame": 18, "NOSE_x": 0.25, "NOSE_y": 0.55,
             "LEFT_HIP_x": 0.24, "RIGHT_HIP_x": 0.26, "LEFT_HIP_y": 0.60, "RIGHT_HIP_y": 0.60},
            {"frame": 20, "NOSE_x": float("nan"), "NOSE_y": float("nan"),
             "LEFT_HIP_x": float("nan"), "RIGHT_HIP_x": float("nan"),
             "LEFT_HIP_y": float("nan"), "RIGHT_HIP_y": float("nan")},
        ])
        raw = {20: [{"NOSE": (0.252, 0.548, 1.0)}]}
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_head_stability_window_raw("fake.mp4", 30.0, df, 20, 20)
        row = result[result["frame"] == 20].iloc[0]
        assert row["NOSE_x"] == 0.252

    def test_extraction_failure_returns_original_df_unchanged(self):
        df = _base_df([10, 11])
        with patch("orchestrator.extract_raw_landmarks_window", side_effect=Exception("boom")):
            result = o._refine_head_stability_window_raw("fake.mp4", 30.0, df, 10, 11)
        pd.testing.assert_frame_equal(result, df)

    def test_hip_fallback_recovers_a_candidate_with_no_nose(self):
        """REGRESSION (2026-09-15, found by an independent adversarial
        review): this function's own requested landmarks used to be
        NOSE/LEFT_SHOULDER/RIGHT_SHOULDER only -- no hips -- so a
        genuinely correct candidate detected with NOSE below the
        visibility cutoff (exactly the motion-blur case this whole window
        exists to help) had NOTHING for _raw_reference_point to check
        against (no NOSE, no hip pair either) and was silently rejected.
        Hips are now requested too, purely to give the identity check a
        second usable reference -- calculate_head_stability itself still
        never reads them."""
        df = _base_df([10, 11, 12])
        raw = {
            11: [{"LEFT_SHOULDER": (0.201, 0.451, 1.0), "RIGHT_SHOULDER": (0.301, 0.451, 1.0),
                  "LEFT_HIP": (0.241, 0.601, 1.0), "RIGHT_HIP": (0.261, 0.601, 1.0)}],  # no NOSE at all
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_head_stability_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["LEFT_SHOULDER_x"] == 0.201  # recovered via the hip-pair reference fallback
