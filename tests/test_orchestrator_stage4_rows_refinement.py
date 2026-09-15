"""
tests/test_orchestrator_stage4_rows_refinement.py

Regression tests for orchestrator._refine_stage4_rows_raw (2026-09-15).

Real coach-reported gap: Lead Knee Bracing, Trunk Lean, and Hip-Shoulder
Separation read straight from the seeded walk's own smoothed df and never
got the same raw-re-extraction sharpening release_height/head_stability/
the annotated skeleton already have -- so a frame where the seeded walk's
own continuity briefly dropped a needed landmark reported "Tracking Drop"
even when a fresh, identity-validated look at that exact frame could have
recovered it.

An external-AI suggestion misdiagnosed this as the SAME bug as the
bystander-hijack issue and proposed loosening
_select_identity_consistent_candidate's threshold -- verified directly
against the code that this was wrong: that function was never even
called in the code path computing these three metrics before this fix.
The real fix is extending the SAME proven raw-re-extraction +
identity-consistent-candidate pattern to these three metrics too, not
weakening the safety check that protects a different, already-fixed bug.
"""

from unittest.mock import patch

import pandas as pd

import orchestrator as o


def _base_df(frames):
    rows = []
    for f in frames:
        rows.append({
            "frame": f,
            "LEFT_HIP_x": 0.24, "RIGHT_HIP_x": 0.26, "LEFT_HIP_y": 0.60, "RIGHT_HIP_y": 0.60,
            "LEFT_SHOULDER_x": 0.24, "RIGHT_SHOULDER_x": 0.26, "LEFT_SHOULDER_y": 0.40, "RIGHT_SHOULDER_y": 0.40,
            "LEFT_KNEE_x": 0.24, "LEFT_KNEE_y": 0.75, "LEFT_ANKLE_x": 0.24, "LEFT_ANKLE_y": 0.90,
        })
    return pd.DataFrame(rows)


class TestRefineStage4RowsRaw:
    def test_a_genuine_refinement_is_merged_into_both_rows(self):
        df = _base_df([10, 20])
        raw = {
            10: [{"LEFT_HIP": (0.241, 0.601, 1.0), "RIGHT_HIP": (0.261, 0.601, 1.0),
                  "LEFT_KNEE": (0.241, 0.751, 1.0), "LEFT_ANKLE": (0.239, 0.899, 1.0)}],
            20: [{"LEFT_HIP": (0.241, 0.601, 1.0), "RIGHT_HIP": (0.261, 0.601, 1.0),
                  "LEFT_SHOULDER": (0.242, 0.402, 1.0)}],
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            ffc_row, br_row = o._refine_stage4_rows_raw("fake.mp4", 30.0, df, 10, 20, "left")

        assert ffc_row["LEFT_KNEE_x"] == 0.241
        assert br_row["LEFT_SHOULDER_x"] == 0.242

    def test_a_wrong_person_candidate_is_rejected_and_original_kept(self):
        """Same identity-consistency protection as the other _refine_*_raw
        functions -- a candidate far from the seeded identity must not be
        merged in, even for this new metric path."""
        df = _base_df([10, 20])
        raw = {
            10: [{"LEFT_HIP": (0.70, 0.60, 1.0), "RIGHT_HIP": (0.72, 0.60, 1.0),
                  "LEFT_KNEE": (0.70, 0.75, 1.0)}],  # bystander
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            ffc_row, br_row = o._refine_stage4_rows_raw("fake.mp4", 30.0, df, 10, 20, "left")

        assert ffc_row["LEFT_HIP_x"] == 0.24  # unchanged -- rejected

    def test_the_correct_candidate_is_picked_even_when_ranked_second(self):
        df = _base_df([10, 20])
        raw = {
            10: [
                {"LEFT_HIP": (0.70, 0.60, 1.0), "RIGHT_HIP": (0.72, 0.60, 1.0)},   # bystander, ranked first
                {"LEFT_HIP": (0.241, 0.601, 1.0), "RIGHT_HIP": (0.261, 0.601, 1.0)},  # bowler, ranked second
            ],
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            ffc_row, br_row = o._refine_stage4_rows_raw("fake.mp4", 30.0, df, 10, 20, "left")

        assert ffc_row["LEFT_HIP_x"] == 0.241

    def test_missing_frame_in_df_returns_none_for_that_row(self):
        df = _base_df([10])  # no row at frame 20
        raw = {}
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            ffc_row, br_row = o._refine_stage4_rows_raw("fake.mp4", 30.0, df, 10, 20, "left")

        assert ffc_row is not None
        assert br_row is None

    def test_extraction_failure_falls_back_to_original_rows(self):
        df = _base_df([10, 20])
        with patch("orchestrator.extract_raw_landmarks_window", side_effect=Exception("boom")):
            ffc_row, br_row = o._refine_stage4_rows_raw("fake.mp4", 30.0, df, 10, 20, "left")

        assert ffc_row["LEFT_HIP_x"] == 0.24
        assert br_row["LEFT_HIP_x"] == 0.24

    def test_right_arm_bowler_uses_right_lead_side_landmarks(self):
        """lead_side='right' (a left-arm bowler's lead leg) must request
        RIGHT_KNEE/RIGHT_ANKLE, not LEFT."""
        df = _base_df([10, 20])
        captured = {}

        def _capture_needed(video_path, fps, needed, start, end):
            captured["needed"] = needed
            return {}

        with patch("orchestrator.extract_raw_landmarks_window", side_effect=_capture_needed):
            o._refine_stage4_rows_raw("fake.mp4", 30.0, df, 10, 20, "right")

        assert "RIGHT_KNEE" in captured["needed"]
        assert "RIGHT_ANKLE" in captured["needed"]
        assert "LEFT_KNEE" not in captured["needed"]
