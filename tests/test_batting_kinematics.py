"""
tests/test_batting_kinematics.py

Same conventions as test_orchestrator_metrics.py: synthetic landmark
rows/DataFrames constructed directly (no real video/MediaPipe needed),
covering both the "clean success" and "tracking failure never fabricates
a value" cases for every batting_kinematics.py function.
"""

import numpy as np
import pandas as pd

import batting_kinematics as bk


class TestHeadMovement:
    def test_stable_head_is_still_over_the_ball(self):
        rows = [{"frame": i, "NOSE_x": 0.50 + (0.001 if i % 2 == 0 else -0.001)} for i in range(10)]
        df = pd.DataFrame(rows)
        result = bk.calculate_head_movement(df, stance_frame=0, contact_frame=9)
        assert result["status"] == "success"
        assert result["tier"] == "Head Still Over The Ball"

    def test_erratic_head_is_excess_drift(self):
        rows = [{"frame": i, "NOSE_x": 0.30 + i * 0.03} for i in range(10)]
        df = pd.DataFrame(rows)
        result = bk.calculate_head_movement(df, stance_frame=0, contact_frame=9)
        assert result["status"] == "success"
        assert result["tier"] == "Excess Head Drift"

    def test_empty_window_is_error_not_fabricated(self):
        df = pd.DataFrame([{"frame": 0, "NOSE_x": 0.5}])
        result = bk.calculate_head_movement(df, stance_frame=100, contact_frame=200)
        assert result["status"] == "error"
        assert result["deviation_index"] is None


def _foot_row(**overrides):
    base = {
        "LEFT_HEEL_x": 0.40, "LEFT_HEEL_y": 0.80,
        "LEFT_FOOT_INDEX_x": 0.40, "LEFT_FOOT_INDEX_y": 0.70,  # straight down the pitch
        "RIGHT_HEEL_x": 0.60, "RIGHT_HEEL_y": 0.80,
        "RIGHT_FOOT_INDEX_x": 0.60, "RIGHT_FOOT_INDEX_y": 0.70,
    }
    base.update(overrides)
    return pd.Series(base)


class TestFrontFootAlignment:
    def test_foot_pointing_down_the_pitch_is_aligned(self):
        row = _foot_row()
        result = bk.calculate_front_foot_alignment(row, front_side="left")
        assert result["status"] == "success"
        assert result["tier"] == "Aligned To The Line"
        assert result["degrees"] < 20.0

    def test_significantly_open_foot(self):
        # Foot pointing mostly sideways (large x change, small y change).
        row = _foot_row(LEFT_FOOT_INDEX_x=0.55, LEFT_FOOT_INDEX_y=0.79)
        result = bk.calculate_front_foot_alignment(row, front_side="left")
        assert result["status"] == "success"
        assert result["tier"] == "Significantly Open/Closed Foot"

    def test_zero_length_vector_is_error(self):
        row = _foot_row(LEFT_FOOT_INDEX_x=0.40, LEFT_FOOT_INDEX_y=0.80)  # same as heel
        result = bk.calculate_front_foot_alignment(row, front_side="left")
        assert result["status"] == "error"
        assert result["degrees"] is None

    def test_missing_landmark_is_error(self):
        row = _foot_row(LEFT_HEEL_x=np.nan)
        result = bk.calculate_front_foot_alignment(row, front_side="left")
        assert result["status"] == "error"


class TestWeightTransfer:
    def test_full_transfer_onto_front_foot(self):
        # stance_width = |0.30-0.70| = 0.40; hips move from 0.50 to 0.30
        # (all the way to the front ankle) = 0.20 displacement = 50%,
        # clearly above the 40% "Committed" threshold, not borderline.
        stance = {"frame": 0, "LEFT_ANKLE_x": 0.30, "RIGHT_ANKLE_x": 0.70,
                   "LEFT_HIP_x": 0.50, "RIGHT_HIP_x": 0.50}
        contact = {"frame": 10, "LEFT_ANKLE_x": 0.30, "RIGHT_ANKLE_x": 0.70,
                    "LEFT_HIP_x": 0.30, "RIGHT_HIP_x": 0.30}
        df = pd.DataFrame([stance, contact])
        result = bk.calculate_weight_transfer(df, stance_frame=0, contact_frame=10, front_side="left")
        assert result["status"] == "success"
        assert result["tier"] == "Committed Weight Transfer"
        assert result["percent"] > 40.0

    def test_no_transfer_stuck_on_back_foot(self):
        stance = {"frame": 0, "LEFT_ANKLE_x": 0.40, "RIGHT_ANKLE_x": 0.60,
                   "LEFT_HIP_x": 0.50, "RIGHT_HIP_x": 0.50}
        contact = {"frame": 10, "LEFT_ANKLE_x": 0.40, "RIGHT_ANKLE_x": 0.60,
                    "LEFT_HIP_x": 0.50, "RIGHT_HIP_x": 0.50}  # hips never moved
        df = pd.DataFrame([stance, contact])
        result = bk.calculate_weight_transfer(df, stance_frame=0, contact_frame=10, front_side="left")
        assert result["status"] == "success"
        assert result["tier"] == "Stuck On The Back Foot"
        assert result["percent"] == 0.0

    def test_missing_frame_is_error(self):
        df = pd.DataFrame([{"frame": 0, "LEFT_ANKLE_x": 0.4, "RIGHT_ANKLE_x": 0.6,
                             "LEFT_HIP_x": 0.5, "RIGHT_HIP_x": 0.5}])
        result = bk.calculate_weight_transfer(df, stance_frame=0, contact_frame=999, front_side="left")
        assert result["status"] == "error"

    def test_zero_stance_width_is_error_not_divide_crash(self):
        stance = {"frame": 0, "LEFT_ANKLE_x": 0.50, "RIGHT_ANKLE_x": 0.50,
                   "LEFT_HIP_x": 0.50, "RIGHT_HIP_x": 0.50}
        contact = {"frame": 10, "LEFT_ANKLE_x": 0.50, "RIGHT_ANKLE_x": 0.50,
                    "LEFT_HIP_x": 0.45, "RIGHT_HIP_x": 0.45}
        df = pd.DataFrame([stance, contact])
        result = bk.calculate_weight_transfer(df, stance_frame=0, contact_frame=10, front_side="left")
        assert result["status"] == "error"
        assert result["percent"] is None


class TestDownswingPlane:
    def test_straight_bat_downswing(self):
        # Path mostly downward (y increases) with a moderate horizontal
        # component -> angle from vertical should land in 10-35.
        backlift = {"frame": 0, "LEFT_WRIST_x": 0.45, "LEFT_WRIST_y": 0.30,
                    "RIGHT_WRIST_x": 0.47, "RIGHT_WRIST_y": 0.32}
        contact = {"frame": 5, "LEFT_WRIST_x": 0.55, "LEFT_WRIST_y": 0.65,
                   "RIGHT_WRIST_x": 0.57, "RIGHT_WRIST_y": 0.67}
        df = pd.DataFrame([backlift, contact])
        result = bk.calculate_downswing_plane(df, backlift_frame=0, contact_frame=5)
        assert result["status"] == "success"
        assert result["tier"] == "Straight-Bat Downswing"

    def test_round_the_body_swing(self):
        # Path mostly horizontal -> large angle from vertical.
        backlift = {"frame": 0, "LEFT_WRIST_x": 0.30, "LEFT_WRIST_y": 0.50,
                    "RIGHT_WRIST_x": 0.32, "RIGHT_WRIST_y": 0.50}
        contact = {"frame": 5, "LEFT_WRIST_x": 0.70, "LEFT_WRIST_y": 0.52,
                   "RIGHT_WRIST_x": 0.72, "RIGHT_WRIST_y": 0.52}
        df = pd.DataFrame([backlift, contact])
        result = bk.calculate_downswing_plane(df, backlift_frame=0, contact_frame=5)
        assert result["status"] == "success"
        assert result["tier"] == "Round-The-Body Swing"

    def test_steep_chopping_downswing(self):
        # Path almost entirely vertical -> small angle from vertical.
        backlift = {"frame": 0, "LEFT_WRIST_x": 0.50, "LEFT_WRIST_y": 0.30,
                    "RIGHT_WRIST_x": 0.50, "RIGHT_WRIST_y": 0.30}
        contact = {"frame": 5, "LEFT_WRIST_x": 0.501, "LEFT_WRIST_y": 0.65,
                   "RIGHT_WRIST_x": 0.501, "RIGHT_WRIST_y": 0.65}
        df = pd.DataFrame([backlift, contact])
        result = bk.calculate_downswing_plane(df, backlift_frame=0, contact_frame=5)
        assert result["status"] == "success"
        assert result["tier"] == "Steep/Chopping Downswing"

    def test_missing_frame_is_error(self):
        df = pd.DataFrame([{"frame": 0, "LEFT_WRIST_x": 0.5, "LEFT_WRIST_y": 0.5,
                             "RIGHT_WRIST_x": 0.5, "RIGHT_WRIST_y": 0.5}])
        result = bk.calculate_downswing_plane(df, backlift_frame=0, contact_frame=999)
        assert result["status"] == "error"


class TestTopElbowAngle:
    def test_controlled_top_elbow(self):
        row = pd.Series({
            "LEFT_SHOULDER_x": 0.40, "LEFT_SHOULDER_y": 0.30,
            "LEFT_ELBOW_x": 0.42, "LEFT_ELBOW_y": 0.45,
            "LEFT_WRIST_x": 0.50, "LEFT_WRIST_y": 0.55,
        })
        result = bk.calculate_top_elbow_angle(row, top_hand_side="left")
        assert result["status"] == "success"
        assert result["tier"] == "Controlled Top-Elbow"
        assert 100.0 <= result["degrees"] <= 160.0

    def test_collapsed_chicken_wing_elbow(self):
        # A sharply bent elbow: wrist folded back close to the shoulder side.
        row = pd.Series({
            "LEFT_SHOULDER_x": 0.40, "LEFT_SHOULDER_y": 0.30,
            "LEFT_ELBOW_x": 0.45, "LEFT_ELBOW_y": 0.45,
            "LEFT_WRIST_x": 0.38, "LEFT_WRIST_y": 0.33,
        })
        result = bk.calculate_top_elbow_angle(row, top_hand_side="left")
        assert result["status"] == "success"
        assert result["tier"] == "Collapsed (Chicken Wing) Elbow"

    def test_degenerate_landmarks_is_error(self):
        row = pd.Series({
            "LEFT_SHOULDER_x": 0.40, "LEFT_SHOULDER_y": 0.30,
            "LEFT_ELBOW_x": 0.40, "LEFT_ELBOW_y": 0.30,  # same point as shoulder
            "LEFT_WRIST_x": 0.50, "LEFT_WRIST_y": 0.55,
        })
        result = bk.calculate_top_elbow_angle(row, top_hand_side="left")
        assert result["status"] == "error"
        assert result["degrees"] is None
