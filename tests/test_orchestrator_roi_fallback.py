"""
tests/test_orchestrator_roi_fallback.py

Regression tests for orchestrator._roi_fallback_candidates (2026-09-19).

Real, evaluated response to an external-AI suggestion: a full-frame raw
re-extraction pass (main.extract_raw_landmarks_window) can find ZERO
candidates at a frame during a fast, motion-blurred delivery even when
the subject is genuinely there -- confirmed directly on this app's real
reproduction clip via a live detector scan. Also confirmed directly
(not a guess): re-running detection on a TIGHT crop centered on the
seeded walk's own nearest known-good position recovers a real,
continuously-plausible detection at several of those exact frames,
including the coach's own actual confirmed BFC/FFC/BR frames on the real
clip that drove this investigation.

Explicitly NOT the same idea as interpolating joint positions across the
gap (a different part of the same external suggestion, rejected) -- this
only searches a smaller region of REAL pixels for a REAL detection; every
candidate it finds still goes through the exact same
_select_identity_consistent_candidate validation as a full-frame
candidate.
"""

from unittest.mock import patch

import pandas as pd

import orchestrator as o


def _base_df(frames):
    rows = []
    for f in frames:
        rows.append({
            "frame": f,
            "NOSE_x": 0.30, "NOSE_y": 0.52,
            "LEFT_HIP_x": 0.29, "RIGHT_HIP_x": 0.31,
            "LEFT_HIP_y": 0.60, "RIGHT_HIP_y": 0.60,
        })
    return pd.DataFrame(rows)


class TestRoiFallbackCandidates:
    def test_calls_roi_extraction_centered_on_the_nearest_reference_row(self):
        """The crop center must be the seeded walk's own nearest
        already-confirmed position -- never a value invented by this
        function itself."""
        df = _base_df([50])  # reference frame 50, target frame 55 (5 away)
        captured = {}

        def _fake_roi(video_path, fps, landmark_names, frame_idx, center_xy_norm, radius_x, radius_y, num_poses=2):
            captured["frame_idx"] = frame_idx
            captured["center"] = center_xy_norm
            captured["radius_x"] = radius_x
            return [{"NOSE": (0.31, 0.52, 0.99)}]

        with patch("orchestrator.extract_raw_landmarks_at_frame_roi", side_effect=_fake_roi):
            result = o._roi_fallback_candidates("fake.mp4", 30.0, df, 55, ["NOSE"])

        assert captured["frame_idx"] == 55
        assert captured["center"] == (0.30, 0.52)  # frame 50's own NOSE position
        assert result == [{"NOSE": (0.31, 0.52, 0.99)}]

    def test_radius_grows_with_distance_from_the_reference_frame(self):
        df = _base_df([50])
        captured_radii = []

        def _fake_roi(video_path, fps, landmark_names, frame_idx, center_xy_norm, radius_x, radius_y, num_poses=2):
            captured_radii.append(radius_x)
            return []

        with patch("orchestrator.extract_raw_landmarks_at_frame_roi", side_effect=_fake_roi):
            o._roi_fallback_candidates("fake.mp4", 30.0, df, 51, ["NOSE"])  # 1 frame away
            o._roi_fallback_candidates("fake.mp4", 30.0, df, 70, ["NOSE"])  # 20 frames away

        assert captured_radii[1] > captured_radii[0]
        assert captured_radii[1] <= o._ROI_FALLBACK_MAX_RADIUS_X  # capped, never unbounded

    def test_no_reference_row_anywhere_nearby_returns_empty(self):
        df = pd.DataFrame([{"frame": 10, "NOSE_x": float("nan"), "NOSE_y": float("nan"),
                             "LEFT_HIP_x": float("nan"), "RIGHT_HIP_x": float("nan"),
                             "LEFT_HIP_y": float("nan"), "RIGHT_HIP_y": float("nan")}])
        with patch("orchestrator.extract_raw_landmarks_at_frame_roi") as mock_roi:
            result = o._roi_fallback_candidates("fake.mp4", 30.0, df, 50, ["NOSE"])
        assert result == []
        mock_roi.assert_not_called()  # nothing to center a crop on -- must not guess

    def test_falls_back_to_mid_hip_when_nose_is_missing(self):
        df = pd.DataFrame([{"frame": 50, "NOSE_x": float("nan"), "NOSE_y": float("nan"),
                             "LEFT_HIP_x": 0.24, "RIGHT_HIP_x": 0.26,
                             "LEFT_HIP_y": 0.60, "RIGHT_HIP_y": 0.60}])
        captured = {}

        def _fake_roi(video_path, fps, landmark_names, frame_idx, center_xy_norm, radius_x, radius_y, num_poses=2):
            captured["center"] = center_xy_norm
            return []

        with patch("orchestrator.extract_raw_landmarks_at_frame_roi", side_effect=_fake_roi):
            o._roi_fallback_candidates("fake.mp4", 30.0, df, 55, ["NOSE"])

        assert captured["center"] == (0.25, 0.60)  # mid-hip

    def test_extraction_exception_returns_empty_not_raises(self):
        df = _base_df([50])
        with patch("orchestrator.extract_raw_landmarks_at_frame_roi", side_effect=Exception("boom")):
            result = o._roi_fallback_candidates("fake.mp4", 30.0, df, 55, ["NOSE"])
        assert result == []


class TestRoiFallbackWiredIntoSkeletonRefinement:
    """Confirms the fallback is actually TRIED when the full-frame pass
    finds nothing at a frame -- not just that the helper function itself
    works in isolation."""

    def test_a_frame_absent_from_the_full_frame_pass_still_gets_patched_via_roi(self):
        df = _base_df([10, 11, 12])
        # Full-frame pass finds NOTHING at frame 11 (absent from the dict
        # entirely -- not an empty list, genuinely missing, matching a
        # real zero-candidate frame).
        raw_full_frame = {}

        def _fake_roi(video_path, fps, landmark_names, frame_idx, center_xy_norm, radius_x, radius_y, num_poses=2):
            if frame_idx == 11:
                return [{"NOSE": (0.301, 0.521, 0.99)}]
            return []

        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw_full_frame), \
             patch("orchestrator.extract_raw_landmarks_at_frame_roi", side_effect=_fake_roi):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.301  # recovered via the ROI fallback

    def test_roi_fallback_tried_when_full_frame_candidate_validates_but_is_incomplete(self):
        """REGRESSION (2026-09-21, found via a real end-to-end pipeline
        run): the full-frame pass can find a candidate that genuinely
        VALIDATES (it's the right person) but is missing some requested
        landmarks (real footage: nose/hip/shoulder detected across the
        whole frame, but knee/ankle too small/blurred at that scale) --
        while the SAME instant's ROI crop (zoomed in, better scale for
        the detector) finds all of them. A validated-but-partial
        full-frame hit must not block a more complete ROI reading from
        ever being tried."""
        df = _base_df([10, 11, 12])
        needed = ["NOSE", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "LEFT_ANKLE"]
        # Full-frame pass finds the RIGHT person, but only nose+hips --
        # no knee/ankle. This candidate WOULD validate (close position).
        raw_full_frame = {11: [{"NOSE": (0.301, 0.521, 0.99),
                                 "LEFT_HIP": (0.29, 0.60, 0.99), "RIGHT_HIP": (0.31, 0.60, 0.99)}]}

        def _fake_roi(video_path, fps, landmark_names, frame_idx, center_xy_norm, radius_x, radius_y, num_poses=2):
            if frame_idx == 11:
                # The ROI crop finds the SAME person, fully -- including
                # the knee/ankle the full-frame pass missed.
                return [{"NOSE": (0.302, 0.522, 0.99),
                         "LEFT_HIP": (0.291, 0.601, 0.99), "RIGHT_HIP": (0.311, 0.601, 0.99),
                         "LEFT_KNEE": (0.30, 0.75, 0.99), "LEFT_ANKLE": (0.30, 0.90, 0.99)}]
            return []

        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw_full_frame), \
             patch("orchestrator.extract_raw_landmarks_at_frame_roi", side_effect=_fake_roi):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        # The MORE COMPLETE (ROI) reading was used, not the partial
        # full-frame one -- knee/ankle are now present, not NaN.
        assert row11["LEFT_KNEE_x"] == 0.30
        assert row11["LEFT_ANKLE_x"] == 0.30

    def test_roi_fallback_tried_when_full_frame_pass_found_a_wrong_person(self):
        """REGRESSION (2026-09-19, found via a real end-to-end pipeline
        run): the full-frame pass very often finds SOMETHING at a hard
        frame -- typically the confidently-detected static bystander,
        not nothing at all. That non-empty (just wrong) candidate list
        must not prevent the ROI-crop fallback from being tried as a
        second attempt once the wrong candidate is rejected."""
        df = _base_df([10, 11, 12])
        # Full-frame pass DOES find something at frame 11 -- the bystander.
        raw_full_frame = {11: [{"NOSE": (0.75, 0.55, 0.99)}]}

        def _fake_roi(video_path, fps, landmark_names, frame_idx, center_xy_norm, radius_x, radius_y, num_poses=2):
            if frame_idx == 11:
                return [{"NOSE": (0.301, 0.521, 0.99)}]  # the real bowler, found via crop
            return []

        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw_full_frame), \
             patch("orchestrator.extract_raw_landmarks_at_frame_roi", side_effect=_fake_roi):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.301  # recovered via ROI fallback despite a wrong full-frame hit

    def test_a_wrong_person_found_via_roi_is_still_rejected(self):
        """The ROI fallback finding SOMETHING doesn't bypass identity
        validation -- a bystander found via the crop must still be
        rejected exactly like a bystander found via the full-frame pass."""
        df = _base_df([10, 11, 12])
        raw_full_frame = {}

        def _fake_roi(video_path, fps, landmark_names, frame_idx, center_xy_norm, radius_x, radius_y, num_poses=2):
            if frame_idx == 11:
                return [{"NOSE": (0.75, 0.55, 0.99)}]  # far from the seeded identity
            return []

        with patch("orchestrator.extract_raw_landmarks_window", return_value=raw_full_frame), \
             patch("orchestrator.extract_raw_landmarks_at_frame_roi", side_effect=_fake_roi):
            result = o._refine_skeleton_window_raw("fake.mp4", 30.0, df, 10, 12)

        row11 = result[result["frame"] == 11].iloc[0]
        assert row11["NOSE_x"] == 0.30  # unchanged -- rejected same as any other wrong candidate
