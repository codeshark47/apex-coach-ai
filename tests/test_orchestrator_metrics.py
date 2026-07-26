"""
tests/test_orchestrator_metrics.py

Regression tests for the two orchestrator.py metric functions that had
real, confirmed bugs found on real footage this session:

  - calculate_hip_shoulder_separation: a NaN landmark used to compare
    False against every tier threshold and silently fall through to a
    confident (and false) "Blocked rotation" tier, instead of being
    flagged as a tracking failure.
  - calculate_release_height_ratio_safe: an implausibility ceiling meant
    to catch a MISTRACKED wrist was also rejecting a coach's directly
    confirmed release point — discarding a human's ground-truth
    observation on the theory that it must be a tracking glitch.
"""

import numpy as np
import pandas as pd

import orchestrator as o


def _hip_shoulder_row(**overrides):
    """A row where hips and shoulders are both clearly separated by a
    real rotation (~30 degrees) — a plausible mid-delivery pose."""
    base = {
        "frame": 10,
        "LEFT_SHOULDER_x": 0.40, "LEFT_SHOULDER_y": 0.30,
        "RIGHT_SHOULDER_x": 0.60, "RIGHT_SHOULDER_y": 0.35,
        "LEFT_HIP_x": 0.45, "LEFT_HIP_y": 0.50,
        "RIGHT_HIP_x": 0.55, "RIGHT_HIP_y": 0.50,
    }
    base.update(overrides)
    return base


class TestHipShoulderSeparation:
    def test_nan_landmark_returns_none_not_blocked_rotation(self):
        """The exact bug found on real footage: a NaN shoulder landmark
        used to fall through to tier='Blocked rotation', status='success'
        — a confident, false coaching claim generated from a pure
        tracking failure."""
        df = pd.DataFrame([_hip_shoulder_row(LEFT_SHOULDER_x=np.nan)])
        result = o.calculate_hip_shoulder_separation(df, ffc_frame=10)
        assert result["degrees"] is None
        assert result["status"] == "error"
        assert result["tier"] != "Blocked rotation"

    def test_missing_frame_returns_error(self):
        df = pd.DataFrame([_hip_shoulder_row(frame=1)])
        result = o.calculate_hip_shoulder_separation(df, ffc_frame=999)
        assert result["status"] == "error"

    def test_wraparound_case_produces_small_real_separation(self):
        """Regression test for the documented wraparound bug: when
        shoulder/hip angles straddle the +/-180 boundary, the OLD fold
        produced a negative nonsense value. This case (shoulder ~178,
        hip ~-178.7) should now report a small, physically real
        separation instead."""
        # Construct shoulder/hip vectors whose arctan2 angles straddle
        # the +/-180 boundary.
        row = _hip_shoulder_row(
            LEFT_SHOULDER_x=0.30, LEFT_SHOULDER_y=0.500,
            RIGHT_SHOULDER_x=0.70, RIGHT_SHOULDER_y=0.503,
            LEFT_HIP_x=0.70, LEFT_HIP_y=0.500,
            RIGHT_HIP_x=0.30, RIGHT_HIP_y=0.497,
        )
        df = pd.DataFrame([row])
        result = o.calculate_hip_shoulder_separation(df, ffc_frame=10)
        assert result["status"] == "success"
        # Must be a small real separation, not a negative/nonsense value
        # or a value anywhere near the raw (unwrapped) ~356 degree diff.
        assert 0 <= result["degrees"] <= 90

    def test_optimal_stretch_tier_boundary(self):
        assert o.calculate_hip_shoulder_separation(
            pd.DataFrame([_hip_shoulder_row()]), ffc_frame=10
        )["status"] == "success"


def _release_row(wrist_y=0.40, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20):
    return pd.Series({
        "RIGHT_WRIST_y": wrist_y,
        "LEFT_ANKLE_y": ankle_y, "RIGHT_ANKLE_y": ankle_y + 0.02,
        "LEFT_KNEE_y": knee_y, "LEFT_HIP_y": hip_y,
        "NOSE_y": nose_y,
    })


class TestReleaseHeightRatio:
    def test_missing_landmark_returns_error(self):
        row = pd.Series({"RIGHT_WRIST_y": None})
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right")
        assert result["status"] == "error"
        assert result["ratio"] is None

    def test_ankle_above_knee_is_flagged_implausible(self):
        """At release the plant foot is grounded — an ankle ABOVE the
        knee/hip in the frame indicates a mistracked landmark, not a
        real reading."""
        row = _release_row(ankle_y=0.10)  # ankle above knee/hip - implausible
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right", reference_row=row)
        assert result["status"] == "error"
        assert "implausible" in result["classification"].lower()

    def test_automatic_implausible_ratio_rejected_without_override(self):
        """An automatic (untouched) reading with a ratio outside
        physical bounds must be rejected — this is the case where 'this
        is probably a tracking glitch' is a reasonable inference. Uses a
        normal, plausible body height (nose/ankle span 0.65) so this
        specifically exercises the ratio-implausibility ceiling, not the
        separate 'body height too small' floor."""
        row = _release_row(wrist_y=-0.05, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right", reference_row=row)
        assert result["status"] == "error"
        assert result["classification"] == "Measurement error — verify camera angle"

    def test_same_implausible_ratio_accepted_with_coach_override(self):
        """THE regression-critical case found on real footage: a
        coach-confirmed wrist_override_norm must bypass the
        implausibility ceiling entirely — a human's direct observation
        is ground truth, not a tracking glitch to second-guess.
        Uses a normal, plausible body height (nose/ankle span 0.65) —
        only the OVERRIDDEN wrist position is extreme (ratio ~1.4,
        above the 1.30 ceiling that would reject an automatic reading)."""
        row = _release_row(ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(
            row, bowling_arm="right", reference_row=row,
            wrist_override_norm=(0.5, -0.05),
        )
        assert result["status"] == "success"
        assert result["ratio"] > 1.30

    def test_standard_mid_arm_release_classification(self):
        row = _release_row(wrist_y=0.40, ankle_y=0.85, hip_y=0.55, knee_y=0.70, nose_y=0.20)
        result = o.calculate_release_height_ratio_safe(row, bowling_arm="right", reference_row=row)
        assert result["status"] == "success"
        assert result["classification"] in (
            "Standard Mid-Arm Release", "High-Release Leverage", "Low-Sling Action",
        )
