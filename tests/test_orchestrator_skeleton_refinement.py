"""
tests/test_orchestrator_skeleton_refinement.py

Regression test for a real, coach-reported bug (2026-09-15): even with
genuinely correct seed clicks (verified real detected bowler positions,
no misclick), the diagnostic freeze-frame / annotated-video skeleton at
Ball Release showed a stationary bystander instead of the bowler.

Root cause: main.extract_raw_landmarks_window (used by
orchestrator._refine_skeleton_window_raw to sharpen the skeleton for the
release-window video/images) runs its OWN completely separate, UNSEEDED,
single-pose MediaPipe pass -- it has no idea which person the coach's
seed clicks identified, and in a multi-person scene can lock onto a
different, more consistently-detected person entirely independently of
the correctly-seeded main identity walk. The old code blindly patched
whatever that pass found straight over the already-correct seeded
landmarks.

Fix: before trusting a frame's raw re-extraction, check it's still
plausibly the SAME person the seeded walk already placed there (NOSE, or
mid-hip as a fallback) -- reject frames where the raw pass lands somewhere
that isn't a small refinement of the already-known position, so a
completely different person can never silently overwrite a correct one.
"""

from unittest.mock import patch

import pandas as pd

import orchestrator as o


def _base_df(frames):
    """A minimal seeded/smoothed df with NOSE and both hips at a
    consistent "bowler" position for every frame -- standing in for the
    output of the correctly-seeded identity walk."""
    rows = []
    for f in frames:
        rows.append({
            "frame": f,
            "NOSE_x": 0.25, "NOSE_y": 0.55,
            "LEFT_HIP_x": 0.24, "RIGHT_HIP_x": 0.26,
            "LEFT_HIP_y": 0.60, "RIGHT_HIP_y": 0.60,
        })
    return pd.DataFrame(rows)


class TestRefineSkeletonWindowRawIdentityConsistency:
    def test_a_different_person_far_away_is_not_patched_in(self):
        """The exact real bug: the raw re-extraction locks onto a
        bystander at a position wildly different from the seeded walk's
        own already-correct identity. Must keep the original (seeded)
        values, not silently swap in the wrong person."""
        df = _base_df([10, 11, 12])
        # Bystander at (0.70, 0.55) -- 0.45 away from the seeded NOSE
        # position (0.25, 0.55), far beyond IDENTITY_CONSISTENCY_MAX_DIST.
        raw = {
            11: {"NOSE": (0.70, 0.55, 1.0), "LEFT_HIP": (0.69, 0.60, 1.0), "RIGHT_HIP": (0.71, 0.60, 1.0)},
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.25  # unchanged -- the wrong-person patch was rejected
        assert row11["NOSE_y"] == 0.55

    def test_a_genuine_same_person_refinement_is_patched_in(self):
        """A real refinement (small correction, same person) must still
        be applied -- this check must not block the feature it's
        protecting from doing its actual job."""
        df = _base_df([10, 11, 12])
        # A small, real refinement -- 0.01 away from the seeded position,
        # comfortably within tolerance.
        raw = {
            11: {"NOSE": (0.252, 0.548, 1.0), "LEFT_HIP": (0.242, 0.602, 1.0), "RIGHT_HIP": (0.262, 0.598, 1.0)},
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.252  # the genuine refinement WAS applied
        assert row11["NOSE_y"] == 0.548

    def test_falls_back_to_mid_hip_when_nose_is_missing(self):
        """NOSE can genuinely be absent (occlusion, turned head) -- the
        check must still work using mid-hip instead of just skipping the
        safety check entirely."""
        df = _base_df([10, 11])
        raw = {
            11: {"LEFT_HIP": (0.68, 0.60, 1.0), "RIGHT_HIP": (0.72, 0.60, 1.0)},  # far bystander, no NOSE key at all
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 11)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["LEFT_HIP_x"] == 0.24  # unchanged -- rejected via the mid-hip fallback check

    def test_a_frame_with_no_existing_reference_still_gets_patched(self):
        """If the seeded walk had NOTHING at this frame either (NaN), there's
        nothing already-correct to protect -- the raw re-extraction is the
        only data available and should still be used, not discarded."""
        df = pd.DataFrame([{"frame": 20, "NOSE_x": float("nan"), "NOSE_y": float("nan"),
                             "LEFT_HIP_x": float("nan"), "RIGHT_HIP_x": float("nan"),
                             "LEFT_HIP_y": float("nan"), "RIGHT_HIP_y": float("nan")}])
        raw = {20: {"NOSE": (0.5, 0.5, 1.0)}}
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 20, 20)

        row = result[result["frame"] == 20].iloc[0]
        assert row["NOSE_x"] == 0.5
        assert row["NOSE_y"] == 0.5

    def test_extraction_failure_returns_original_df_unchanged(self):
        df = _base_df([10, 11])
        with patch("orchestrator.extract_raw_landmarks_window", side_effect=Exception("boom")):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 11)
        pd.testing.assert_frame_equal(result, df)
