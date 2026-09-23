"""
tests/test_orchestrator_skeleton_refinement.py

Regression test for a real, coach-reported bug (2026-09-15): even with
genuinely correct seed clicks (verified real detected bowler positions,
no misclick), the diagnostic freeze-frame / annotated-video skeleton at
Ball Release showed a stationary bystander instead of the bowler.

Root cause, two layers, both fixed here:

1. main.extract_raw_landmarks_window (used by orchestrator._refine_
   skeleton_window_raw to sharpen the skeleton for the release-window
   video/images) runs its OWN completely separate, UNSEEDED MediaPipe
   pass -- it has no idea which person the coach's seed clicks
   identified. It ORIGINALLY also trusted MediaPipe's own single top-
   ranked candidate per frame (num_poses=1), which in a multi-person
   scene frequently favors a static, unblurred bystander over the
   actual moving/blurred subject -- entirely independently of the
   correctly-seeded main identity walk. The old code blindly patched
   whatever that top candidate was straight over the already-correct
   seeded landmarks.

2. Fixed in two layers: extract_raw_landmarks_window now returns EVERY
   detected candidate per frame (num_poses=3), and
   orchestrator._select_identity_consistent_candidate picks whichever
   candidate is still plausibly the SAME person the seeded walk already
   placed there (NOSE, or mid-hip as a fallback, searching nearby frames
   if this exact frame has no seeded reference of its own) -- rather
   than either blindly trusting MediaPipe's top pick (the original bug)
   or giving up the moment the top pick looks wrong (which would have
   missed the case where the real subject WAS detected, just not ranked
   first).
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
    """Every df below includes frame 9, one frame BEFORE the window
    passed to _refine_skeleton_window_raw (start_frame=10) -- needed
    since 2026-09-23 (see that function's own docstring on blanking the
    whole window upfront): frames INSIDE the window are blanked before
    processing begins, so frame 9 -- outside it, untouched -- is what
    lets these tests validate a candidate at frame 11 against a real,
    trustworthy reference, exactly like real run-up footage sitting just
    before FFC in an actual clip. None of these tests supply a raw
    candidate for frames 10/12 themselves, so those frames are expected
    to end up correctly blanked too (no assertions are made about them)."""

    def test_a_different_person_far_away_is_not_patched_in(self):
        """The exact real bug: the raw re-extraction's only candidate is
        a bystander at a position wildly different from the seeded
        walk's own already-correct identity. Must keep the original
        (seeded) values, not silently swap in the wrong person."""
        df = _base_df([9, 10, 11, 12])
        # Bystander at (0.70, 0.55) -- 0.45 away from the seeded NOSE
        # position (0.25, 0.55), far beyond IDENTITY_CONSISTENCY_MAX_DIST.
        raw = {
            11: [{"NOSE": (0.70, 0.55, 1.0), "LEFT_HIP": (0.69, 0.60, 1.0), "RIGHT_HIP": (0.71, 0.60, 1.0)}],
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        # BEHAVIOR CHANGE (2026-09-23): a rejected frame is now blanked
        # (NaN), not left at its stale original value -- see this
        # window's own "blank upfront" fix for why (a not-yet-processed
        # or rejected frame's original value can itself already be
        # wrong, and must never survive as a false "trusted" reference
        # for another frame's check).
        assert pd.isna(row11["NOSE_x"])  # rejected -- the wrong-person patch was blanked, not applied

    def test_a_genuine_same_person_refinement_is_patched_in(self):
        """A real refinement (small correction, same person) must still
        be applied -- this check must not block the feature it's
        protecting from doing its actual job."""
        df = _base_df([9, 10, 11, 12])
        # A small, real refinement -- 0.01 away from the seeded position,
        # comfortably within tolerance.
        raw = {
            11: [{"NOSE": (0.252, 0.548, 1.0), "LEFT_HIP": (0.242, 0.602, 1.0), "RIGHT_HIP": (0.262, 0.598, 1.0)}],
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.252  # the genuine refinement WAS applied
        assert row11["NOSE_y"] == 0.548

    def test_the_correct_candidate_is_picked_even_when_ranked_second(self):
        """THE EXACT SCENARIO an external (Gemini) review flagged, and the
        strongest reason multi-candidate selection matters over a plain
        accept/reject gate: MediaPipe's own top-ranked candidate (index 0)
        is the wrong person (a static, unblurred bystander scores higher
        confidence), but the real bowler was ALSO detected in the same
        frame, just ranked second. A single-candidate gate could only
        reject the frame outright; the real fix must find and use the
        correct candidate among several."""
        df = _base_df([9, 10, 11, 12])
        raw = {
            11: [
                {"NOSE": (0.70, 0.55, 1.0)},   # ranked first by MediaPipe -- the bystander
                {"NOSE": (0.252, 0.548, 1.0)},  # ranked second -- the actual bowler, close to seeded position
            ],
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.252  # the correct (second-ranked) candidate was selected
        assert row11["NOSE_y"] == 0.548

    def test_falls_back_to_mid_hip_when_nose_is_missing(self):
        """NOSE can genuinely be absent (occlusion, turned head) -- the
        check must still work using mid-hip instead of just skipping the
        safety check entirely."""
        df = _base_df([9, 10, 11])
        raw = {
            11: [{"LEFT_HIP": (0.68, 0.60, 1.0), "RIGHT_HIP": (0.72, 0.60, 1.0)}],  # far bystander, no NOSE key at all
        }
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 11)

        row11 = result[result["frame"] == 11].iloc[0]
        assert pd.isna(row11["LEFT_HIP_x"])  # rejected via the mid-hip fallback check, blanked not applied

    def test_a_frame_with_no_reference_anywhere_nearby_is_rejected(self):
        """REGRESSION (2026-09-15): the first version of this gate was
        permissive here ("nothing already-correct to protect, so use the
        raw data") -- that was wrong for exactly the highest-stakes case:
        real footage showed the seeded walk has NO data precisely at the
        release frame (motion blur), which is also exactly when the
        unseeded raw pass is most likely to lock onto a bystander. If
        NOTHING in the whole df within range has a usable reference, the
        raw patch must be REJECTED (never fabricate an identity), not
        silently accepted."""
        df = pd.DataFrame([{"frame": 20, "NOSE_x": float("nan"), "NOSE_y": float("nan"),
                             "LEFT_HIP_x": float("nan"), "RIGHT_HIP_x": float("nan"),
                             "LEFT_HIP_y": float("nan"), "RIGHT_HIP_y": float("nan")}])
        raw = {20: [{"NOSE": (0.5, 0.5, 1.0)}]}
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 20, 20)

        row = result[result["frame"] == 20].iloc[0]
        assert pd.isna(row["NOSE_x"])  # rejected -- no reference anywhere to trust it against

    def test_a_frame_with_no_reference_of_its_own_falls_back_to_a_nearby_frame(self):
        """The exact real-world case this fallback exists for: the frame
        being refined (e.g. the release frame itself) has no seeded data
        of its own (motion blur), but a NEARBY frame in the same seeded
        walk does -- that nearby frame's identity should still be used to
        validate the raw patch, rather than either giving up (rejecting
        good data) or blindly accepting (the original bug)."""
        df = pd.DataFrame([
            {"frame": 18, "NOSE_x": 0.25, "NOSE_y": 0.55,
             "LEFT_HIP_x": 0.24, "RIGHT_HIP_x": 0.26, "LEFT_HIP_y": 0.60, "RIGHT_HIP_y": 0.60},
            {"frame": 20, "NOSE_x": float("nan"), "NOSE_y": float("nan"),
             "LEFT_HIP_x": float("nan"), "RIGHT_HIP_x": float("nan"),
             "LEFT_HIP_y": float("nan"), "RIGHT_HIP_y": float("nan")},
        ])
        # A genuine same-person refinement at frame 20, close to frame 18's
        # known-good position.
        raw = {20: [{"NOSE": (0.252, 0.548, 1.0)}]}
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 20, 20)
        row = result[result["frame"] == 20].iloc[0]
        assert row["NOSE_x"] == 0.252  # validated against frame 18 and accepted

        # A bystander at frame 20 instead -- far from frame 18's known position.
        raw_wrong = {20: [{"NOSE": (0.70, 0.55, 1.0)}]}
        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw_wrong):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 20, 20)
        row = result[result["frame"] == 20].iloc[0]
        assert pd.isna(row["NOSE_x"])  # rejected -- inconsistent with frame 18's identity

    def test_extraction_failure_returns_original_df_unchanged(self):
        df = _base_df([10, 11])
        with patch("orchestrator.extract_raw_landmarks_window", side_effect=Exception("boom")):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 11)
        pd.testing.assert_frame_equal(result, df)
