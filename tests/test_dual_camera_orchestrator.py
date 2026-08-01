"""
tests/test_dual_camera_orchestrator.py

Regression test for a real bug found during a broader audit:
dual_camera_orchestrator.run_dual_camera_analysis put "bowling_arm_detected"
at the TOP level of its return dict only. orchestrator.py's Single Camera
equivalent nests it INSIDE "biomechanical_metrics" instead, and
streamlit_app.py reads it via metrics.get("bowling_arm_detected") where
metrics IS biomechanical_metrics. For Dual Camera this silently returned
None everywhere it was read, meaning:

  1. Speed estimation never received the already-confirmed bowling arm
     (bowling_arm_override=metrics.get("bowling_arm_detected")) and fell
     back to its own internal auto-detection instead — able to disagree
     with the arm the rest of the report (event timing, release height,
     video overlay) is built on.
  2. The "Bowling arm (auto-detected/manually selected)" caption never
     displayed at all in Dual Camera mode.

This test isolates run_dual_camera_analysis's own dict-assembly logic
(the actual site of the bug) from the biomechanics calculations and video
rendering it calls out to (already covered elsewhere / not what broke) by
mocking those collaborators and using the side_precomputed/rear_precomputed
bypass to skip real video extraction entirely — real video files and
MediaPipe are not needed to catch this class of bug.
"""

from unittest.mock import patch

import pandas as pd

import dual_camera_orchestrator as dco


def _minimal_df(n_frames=20):
    return pd.DataFrame({"frame": range(n_frames)})


def _patch_all_collaborators(test_func):
    """Stubs every biomechanics/video-rendering collaborator
    run_dual_camera_analysis calls out to, so a test can exercise just
    its own dict-assembly logic in isolation."""
    patches = [
        patch.object(dco, "transcode_to_h264", side_effect=lambda p: p),
        patch.object(dco, "generate_fail_safe_video", return_value=None),
        patch.object(dco, "calculate_head_stability",
                     return_value={"status": "success", "deviation_index": 1.0, "tier": "Good"}),
        patch.object(dco, "calculate_hip_shoulder_separation",
                     return_value={"status": "success", "degrees": 20.0, "tier": "Good"}),
        patch.object(dco, "calculate_release_height_ratio_safe",
                     return_value={"status": "success", "ratio": 0.9, "classification": "Good"}),
        patch.object(dco, "_find_grounded_reference_near", return_value=None),
        patch.object(dco, "_nearest_complete_row", return_value=None),
        patch.object(dco, "calculate_trunk_lean",
                     return_value={"status": "success", "degrees": 30.0, "tier": "Good"}),
        patch.object(dco, "calculate_knee_bracing",
                     return_value={"status": "success", "degrees": 140.0, "tier": "Good"}),
    ]
    for p in reversed(patches):
        test_func = p(test_func)
    return test_func


class TestBowlingArmDetectedIsInBiomechanicalMetrics:
    @_patch_all_collaborators
    def test_bowling_arm_detected_is_nested_where_streamlit_app_reads_it(self, *mocks):
        events = {"BFC": 0, "FFC": 5, "BR": 10}
        side_precomputed = {
            "status": "success", "df": _minimal_df(), "fps": 30,
            "events": dict(events), "bowling_arm": "right",
        }
        rear_precomputed = {
            "status": "success", "df": _minimal_df(), "fps": 30,
            "events": dict(events),
        }

        result = dco.run_dual_camera_analysis(
            "fake_side.mp4", "fake_rear.mp4",
            side_precomputed=side_precomputed, rear_precomputed=rear_precomputed,
        )

        assert result["status"] == "success"
        # THE fix — this is what streamlit_app.py actually reads.
        assert result["biomechanical_metrics"]["bowling_arm_detected"] == "right"
        # Kept at the top level too, for backward compatibility.
        assert result["bowling_arm_detected"] == "right"

    @_patch_all_collaborators
    def test_left_arm_override_is_also_nested_correctly(self, *mocks):
        events = {"BFC": 0, "FFC": 5, "BR": 10}
        side_precomputed = {
            "status": "success", "df": _minimal_df(), "fps": 30,
            "events": dict(events), "bowling_arm": "right",
        }
        rear_precomputed = {
            "status": "success", "df": _minimal_df(), "fps": 30,
            "events": dict(events),
        }

        result = dco.run_dual_camera_analysis(
            "fake_side.mp4", "fake_rear.mp4", bowling_arm_override="left",
            side_precomputed=side_precomputed, rear_precomputed=rear_precomputed,
        )

        assert result["biomechanical_metrics"]["bowling_arm_detected"] == "left"


class TestKneeAngleImplausibilityGuard:
    """Regression test for a real bug found during a broader audit:
    orchestrator.py's Single Camera path has long rejected a near-zero
    knee-bracing reading as a collapsed-landmark tracking artifact (a
    human knee cannot physically fold to ~0 degrees mid-delivery), but
    Dual Camera computed the identical value from the same underlying
    calculate_knee_bracing() and never applied the same guard — a
    tracking-failure artifact Single Camera would report as "no data"
    was instead shown here as a real, severe (red/Critical) finding."""

    @_patch_all_collaborators
    def test_near_zero_knee_angle_is_nulled_not_reported_as_a_real_finding(self, *mocks):
        events = {"BFC": 0, "FFC": 5, "BR": 10}
        side_precomputed = {
            "status": "success", "df": _minimal_df(), "fps": 30,
            "events": dict(events), "bowling_arm": "right",
        }
        rear_precomputed = {
            "status": "success", "df": _minimal_df(), "fps": 30,
            "events": dict(events),
        }

        with patch.object(dco, "calculate_knee_bracing",
                           return_value={"status": "success", "degrees": 2.3,
                                         "tier": "Collapsing Knee Joint"}):
            result = dco.run_dual_camera_analysis(
                "fake_side.mp4", "fake_rear.mp4",
                side_precomputed=side_precomputed, rear_precomputed=rear_precomputed,
            )

        knee = result["biomechanical_metrics"]["front_knee_bracing"]
        assert knee["degrees"] is None
        assert knee["status"] == "error"

    @_patch_all_collaborators
    def test_genuine_low_but_plausible_knee_angle_is_kept(self, *mocks):
        """A real, if poor, knee-bracing reading (e.g. a collapsing but
        still anatomically real knee) must not be swept up by the same
        guard — only genuinely impossible near-zero readings."""
        events = {"BFC": 0, "FFC": 5, "BR": 10}
        side_precomputed = {
            "status": "success", "df": _minimal_df(), "fps": 30,
            "events": dict(events), "bowling_arm": "right",
        }
        rear_precomputed = {
            "status": "success", "df": _minimal_df(), "fps": 30,
            "events": dict(events),
        }

        with patch.object(dco, "calculate_knee_bracing",
                           return_value={"status": "success", "degrees": 130.0, "tier": "Acceptable"}):
            result = dco.run_dual_camera_analysis(
                "fake_side.mp4", "fake_rear.mp4",
                side_precomputed=side_precomputed, rear_precomputed=rear_precomputed,
            )

        knee = result["biomechanical_metrics"]["front_knee_bracing"]
        assert knee["degrees"] == 130.0
        assert knee["status"] == "success"
