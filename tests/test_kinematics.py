"""
tests/test_kinematics.py

Regression tests for the "never fabricate a number" invariant this whole
project is built on. Every case here traces back to a REAL bug found on
real footage this session: a tracking failure returning a numeric
placeholder (0.0, "0.00") that landed inside that metric's own "optimal"
band, so a genuine failure silently reported as a perfect score. These
tests exist to make sure that specific regression can never come back
silently.
"""

import numpy as np
import pandas as pd
import pytest

import kinematics as k


def _row(**overrides):
    """A minimally valid landmark row for a right-arm bowler's lead
    (LEFT) knee/hip/ankle and both shoulders — override individual
    fields per test."""
    base = {
        "LEFT_HIP_x": 0.50, "LEFT_HIP_y": 0.50,
        "LEFT_KNEE_x": 0.50, "LEFT_KNEE_y": 0.65,
        "LEFT_ANKLE_x": 0.50, "LEFT_ANKLE_y": 0.80,
        "RIGHT_HIP_x": 0.60, "RIGHT_HIP_y": 0.50,
        "RIGHT_SHOULDER_x": 0.60, "RIGHT_SHOULDER_y": 0.20,
        "LEFT_SHOULDER_x": 0.50, "LEFT_SHOULDER_y": 0.20,
    }
    base.update(overrides)
    return pd.Series(base)


class TestKneeBracing:
    def test_degenerate_denominator_returns_none_not_zero(self):
        """Hip, knee, and ankle all coincide at the same point — a real
        tracking collapse. Must return None, not a fabricated 0.0 that
        would fold into the 'Critical'/Collapsing tier as if it were a
        real (terrible) measurement."""
        row = _row(LEFT_HIP_x=0.5, LEFT_HIP_y=0.5,
                    LEFT_KNEE_x=0.5, LEFT_KNEE_y=0.5,
                    LEFT_ANKLE_x=0.5, LEFT_ANKLE_y=0.5)
        result = k.calculate_knee_bracing(row, lead_side="left")
        assert result["degrees"] is None
        assert result["status"] == "error"

    def test_missing_column_returns_none_not_zero(self):
        row = pd.Series({"LEFT_HIP_x": 0.5})  # missing everything else
        result = k.calculate_knee_bracing(row, lead_side="left")
        assert result["degrees"] is None
        assert result["status"] == "error"

    def test_elite_tier_boundary(self):
        # A straight leg: hip directly above knee, ankle directly below —
        # ~180 degrees.
        row = _row(LEFT_HIP_x=0.5, LEFT_HIP_y=0.3,
                    LEFT_KNEE_x=0.5, LEFT_KNEE_y=0.6,
                    LEFT_ANKLE_x=0.5, LEFT_ANKLE_y=0.9)
        result = k.calculate_knee_bracing(row, lead_side="left")
        assert result["status"] == "success"
        assert result["degrees"] >= 165.0
        assert result["tier"] == "Elite Rigid Extension"

    def test_lead_side_right_for_left_arm_bowler(self):
        """A left-arm bowler's lead leg is the RIGHT leg — regression
        test for the bug where this was hardcoded to LEFT regardless of
        bowling_arm, silently measuring the wrong leg for left-arm
        bowlers."""
        row = _row(RIGHT_HIP_x=0.5, RIGHT_HIP_y=0.3,
                    RIGHT_KNEE_x=0.5, RIGHT_KNEE_y=0.6,
                    RIGHT_ANKLE_x=0.5, RIGHT_ANKLE_y=0.9,
                    # LEFT leg deliberately degenerate — if the function
                    # measured the wrong leg, this test would fail loudly
                    LEFT_HIP_x=0.1, LEFT_HIP_y=0.1,
                    LEFT_KNEE_x=0.1, LEFT_KNEE_y=0.1,
                    LEFT_ANKLE_x=0.1, LEFT_ANKLE_y=0.1)
        result = k.calculate_knee_bracing(row, lead_side="right")
        assert result["status"] == "success"
        assert result["degrees"] >= 165.0


class TestTrunkLean:
    def test_nan_returns_none_not_zero(self):
        """The exact bug found on real footage: NaN inputs used to
        return 0.0, which sits inside the 'Optimal Upright Posture' band
        (0-8 degrees) — a tracking failure reported as a perfect score."""
        row = _row(LEFT_HIP_x=np.nan)
        result = k.calculate_trunk_lean(row)
        assert result["degrees"] is None
        assert result["status"] == "error"

    def test_genuine_zero_degree_lean_passes_through_as_success(self):
        """A REAL perfectly-upright bowler (shoulders directly above
        hips) must still report 0.0 and 'success' — the fix for the NaN
        case above must not have broken this legitimate case."""
        row = _row(LEFT_HIP_x=0.45, RIGHT_HIP_x=0.55, LEFT_HIP_y=0.5, RIGHT_HIP_y=0.5,
                    LEFT_SHOULDER_x=0.45, RIGHT_SHOULDER_x=0.55,
                    LEFT_SHOULDER_y=0.2, RIGHT_SHOULDER_y=0.2)
        result = k.calculate_trunk_lean(row)
        assert result["status"] == "success"
        assert result["degrees"] == 0.0
        assert result["tier"] == "Optimal Upright Posture"

    def test_implausible_angle_above_90_flagged_as_unreliable(self):
        """>90 degrees is physically nonsensical for lean-from-vertical —
        must be flagged as unreliable, not reported as a real extreme
        lean."""
        # Shoulders BELOW hips (inverted) forces an angle > 90.
        row = _row(LEFT_HIP_x=0.45, RIGHT_HIP_x=0.55, LEFT_HIP_y=0.3, RIGHT_HIP_y=0.3,
                    LEFT_SHOULDER_x=0.45, RIGHT_SHOULDER_x=0.55,
                    LEFT_SHOULDER_y=0.35, RIGHT_SHOULDER_y=0.7)
        result = k.calculate_trunk_lean(row)
        assert result["status"] == "error"
        assert "implausible" in result["tier"].lower()


def _head_row(frame, nose_x, sh_left_x=0.6, sh_right_x=0.4):
    """shoulder width defaults to 0.2 — a plausible, moderate camera
    distance, used across the normalization tests below."""
    return {"frame": frame, "NOSE_x": nose_x, "LEFT_SHOULDER_x": sh_left_x, "RIGHT_SHOULDER_x": sh_right_x}


class TestHeadStability:
    def test_empty_window_returns_none_not_zero(self):
        """The exact bug found on real footage: an empty tracking
        window used to return '0.00', which sits inside the 'Elite Fixed
        Gaze Focus' band — a tracking failure reported as a perfect
        result."""
        df = pd.DataFrame({"frame": [1, 2, 3], "NOSE_x": [0.1, 0.2, 0.3]})
        result = k.calculate_head_stability(df, start_frame=100, end_frame=200)
        assert result["deviation_index"] is None
        assert result["status"] == "error"

    def test_fewer_than_two_valid_points_returns_none(self):
        df = pd.DataFrame([
            _head_row(1, np.nan), _head_row(2, np.nan), _head_row(3, 0.3),
        ])
        result = k.calculate_head_stability(df, start_frame=1, end_frame=3)
        assert result["deviation_index"] is None
        assert result["status"] == "error"

    def test_genuine_stable_head_reports_elite_tier(self):
        df = pd.DataFrame([
            _head_row(1, 0.500), _head_row(2, 0.501), _head_row(3, 0.499), _head_row(4, 0.500),
        ])
        result = k.calculate_head_stability(df, start_frame=1, end_frame=4)
        assert result["status"] == "success"
        assert result["tier"] == "Elite Fixed Gaze Focus"
        assert result["recalibration_pending"] is True

    def test_erratic_head_movement_reports_correct_tier(self):
        df = pd.DataFrame([
            _head_row(1, 0.3), _head_row(2, 0.7), _head_row(3, 0.2), _head_row(4, 0.8),
        ])
        result = k.calculate_head_stability(df, start_frame=1, end_frame=4)
        assert result["status"] == "success"
        assert result["tier"] == "Erratic Lateral Head Drift"

    def test_camera_distance_no_longer_changes_the_reading(self):
        """THE core bug this fix targets: the exact same real head
        movement (as a fraction of body size) must read identically
        whether the bowler is filmed close to the camera (wide shoulder
        span) or further away (narrow shoulder span) — the old raw-
        pixel version would have reported these as different amounts of
        'drift' purely from framing, not real technique."""
        close_up = pd.DataFrame([
            _head_row(1, 0.48, sh_left_x=0.7, sh_right_x=0.3),   # width 0.4
            _head_row(2, 0.52, sh_left_x=0.7, sh_right_x=0.3),
        ])
        far_away = pd.DataFrame([
            _head_row(1, 0.49, sh_left_x=0.6, sh_right_x=0.4),   # width 0.2 (half as wide -> twice as far)
            _head_row(2, 0.51, sh_left_x=0.6, sh_right_x=0.4),
        ])
        close_result = k.calculate_head_stability(close_up, start_frame=1, end_frame=2)
        far_result = k.calculate_head_stability(far_away, start_frame=1, end_frame=2)
        assert close_result["status"] == far_result["status"] == "success"
        assert close_result["deviation_index"] == far_result["deviation_index"]

    def test_degenerate_shoulder_width_does_not_blow_up_the_ratio(self):
        """A near-zero shoulder width (mistracked/collapsed landmarks)
        must not be divided into, which would produce a huge, meaningless
        ratio — same 'None, not a fabricated number' guard as the rest of
        this codebase."""
        df = pd.DataFrame([
            _head_row(1, 0.5, sh_left_x=0.501, sh_right_x=0.500),  # width 0.001, below MIN_SHOULDER_WIDTH
            _head_row(2, 0.5, sh_left_x=0.501, sh_right_x=0.500),
        ])
        result = k.calculate_head_stability(df, start_frame=1, end_frame=2)
        assert result["deviation_index"] is None
        assert result["status"] == "error"
        assert result["tier"] == "Tracking Drop"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
