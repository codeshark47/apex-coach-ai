"""
tests/test_batting_kinematics.py

Same conventions as test_orchestrator_metrics.py: synthetic landmark
rows/DataFrames constructed directly (no real video/MediaPipe needed),
covering both the "clean success" and "tracking failure never fabricates
a value" cases for every batting_kinematics.py function.
"""

import numpy as np
import pandas as pd
import pytest

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

    def test_tier_threshold_matches_metric_ranges_green_band(self):
        """BUG FOUND (2026-08-03, real coach test): this function used to
        cut its own "tier" text off at 0.015, while metric_ranges.py's
        actual green/OPTIMAL band goes up to 0.02 -- a real result of
        0.0162 landed inside the authoritative green band but showed the
        contradictory "Excess Head Drift" descriptor. A value in the
        0.015-0.02 gap must now report the tier consistent with
        metric_ranges, not a second, independent boundary."""
        import metric_ranges as mr
        # Alternating +-a around a fixed mean gives std_dev == a exactly
        # (population std, ddof=0) -- 0.0162 matches the real value from
        # the coach test that exposed this bug.
        rows = [{"frame": i, "NOSE_x": 0.50 + (0.0162 if i % 2 == 0 else -0.0162)} for i in range(10)]
        df = pd.DataFrame(rows)
        result = bk.calculate_head_movement(df, stance_frame=0, contact_frame=9)
        assert result["status"] == "success"
        value = float(result["deviation_index"])
        assert 0.015 < value <= 0.02, "test value must fall in the gap this bug lived in"
        assert result["tier"] == "Head Still Over The Ball"
        assert mr.classify("batting_head_movement", value) == "green"

    def test_empty_window_is_error_not_fabricated(self):
        df = pd.DataFrame([{"frame": 0, "NOSE_x": 0.5}])
        result = bk.calculate_head_movement(df, stance_frame=100, contact_frame=200)
        assert result["status"] == "error"
        assert result["deviation_index"] is None


def _axis_frames(front_ankle_stance=(0.60, 0.80), back_ankle=(0.40, 0.80),
                  front_ankle_contact=(0.62, 0.95), heel_contact=None, toe_contact=None):
    """
    Builds a minimal 2-row (stance, contact) DataFrame for exercising
    _derive_batting_axes/calculate_front_foot_alignment/
    detect_falling_over_risk. With these defaults: crease_vec (back->front
    at stance) = (0.20, 0), so local_x = (1, 0) ("off" direction) and
    local_y resolves to (0, 1) (stride moves +y, matching the default
    front_ankle_contact) -- i.e. a simple, axis-aligned basis, chosen so
    expected angles are easy to hand-verify. See
    TestFrontFootAlignmentRotationInvariance below for proof this isn't
    just a coincidence of axis-alignment.
    """
    stance = {"frame": 0, "LEFT_ANKLE_x": front_ankle_stance[0], "LEFT_ANKLE_y": front_ankle_stance[1],
              "RIGHT_ANKLE_x": back_ankle[0], "RIGHT_ANKLE_y": back_ankle[1],
              "NOSE_x": 0.50, "NOSE_y": 0.40}
    contact = {"frame": 10, "LEFT_ANKLE_x": front_ankle_contact[0], "LEFT_ANKLE_y": front_ankle_contact[1],
               "RIGHT_ANKLE_x": back_ankle[0], "RIGHT_ANKLE_y": back_ankle[1],
               "NOSE_x": 0.50, "NOSE_y": 0.40}
    if heel_contact is not None:
        contact["LEFT_HEEL_x"], contact["LEFT_HEEL_y"] = heel_contact
    if toe_contact is not None:
        contact["LEFT_FOOT_INDEX_x"], contact["LEFT_FOOT_INDEX_y"] = toe_contact
    return pd.DataFrame([stance, contact])


class TestFrontFootAlignment:
    def test_foot_pointing_straight_down_the_pitch_with_no_shot_selected(self):
        df = _axis_frames(heel_contact=(0.62, 0.80), toe_contact=(0.62, 0.95))
        result = bk.calculate_front_foot_alignment(df, stance_frame=0, contact_frame=10, front_side="left")
        assert result["status"] == "success"
        assert result["side"] == "Straight"
        assert abs(result["signed_degrees"]) < 1.0
        assert result["deviation_degrees"] < 5.0
        assert result["tier"] == "Aligned To The Line"

    def test_foot_open_toward_off_side_matches_cover_drive_target(self):
        # foot_vec = (0.08, 0.15) relative to local_x=(1,0)/local_y=(0,1)
        # -> signed_degrees ~ +28 degrees, matching cover_drive's target
        # center exactly (see SHOT_TARGET_CENTERS_DEGREES) -- the coach's
        # own worked example: a cover drive's toe should point off-side,
        # not straight down the pitch.
        df = _axis_frames(heel_contact=(0.62, 0.80), toe_contact=(0.70, 0.95))
        result = bk.calculate_front_foot_alignment(df, stance_frame=0, contact_frame=10,
                                                     front_side="left", shot_played="cover_drive")
        assert result["status"] == "success"
        assert result["side"] == "Off Side"
        assert 25.0 < result["signed_degrees"] < 31.0
        assert result["deviation_degrees"] < 2.0  # right on the cover-drive target
        assert result["tier"] == "On Target For The Shot"
        assert result["target_shot"] == "cover_drive"

    def test_same_foot_direction_is_off_target_for_straight_drive(self):
        # Same +28-degree foot direction as above, but the coach says a
        # STRAIGHT drive was played (target 0) -- now correctly flagged
        # as off-target, since alignment is relative to the shot. ~28
        # degrees off a 0-degree target lands in the "Slightly Off
        # Target" band (15-30); see test_wildly_open_foot_is_significantly_off_target
        # below for a deviation large enough to cross into "Significantly".
        df = _axis_frames(heel_contact=(0.62, 0.80), toe_contact=(0.70, 0.95))
        result = bk.calculate_front_foot_alignment(df, stance_frame=0, contact_frame=10,
                                                     front_side="left", shot_played="straight_drive")
        assert result["status"] == "success"
        assert result["deviation_degrees"] > 15.0
        assert result["tier"] == "Slightly Off Target"

    def test_wildly_open_foot_is_significantly_off_target(self):
        # foot_vec = (0.15, 0.05) -> a much wider angle from straight
        # (~72 degrees), clearly beyond even the "slightly off" band for
        # a straight drive (target 0).
        df = _axis_frames(heel_contact=(0.62, 0.80), toe_contact=(0.77, 0.85))
        result = bk.calculate_front_foot_alignment(df, stance_frame=0, contact_frame=10,
                                                     front_side="left", shot_played="straight_drive")
        assert result["status"] == "success"
        assert result["deviation_degrees"] > 30.0
        assert result["tier"] == "Significantly Off Target"

    def test_foot_open_toward_leg_side(self):
        df = _axis_frames(heel_contact=(0.62, 0.80), toe_contact=(0.54, 0.95))
        result = bk.calculate_front_foot_alignment(df, stance_frame=0, contact_frame=10, front_side="left")
        assert result["status"] == "success"
        assert result["side"] == "Leg Side"
        assert result["signed_degrees"] < 0

    def test_zero_length_foot_vector_is_error(self):
        df = _axis_frames(heel_contact=(0.62, 0.80), toe_contact=(0.62, 0.80))  # same as heel
        result = bk.calculate_front_foot_alignment(df, stance_frame=0, contact_frame=10, front_side="left")
        assert result["status"] == "error"
        assert result["signed_degrees"] is None

    def test_missing_landmark_is_error(self):
        df = _axis_frames(heel_contact=(0.62, 0.80), toe_contact=(0.62, 0.95))
        df.loc[df["frame"] == 0, "LEFT_ANKLE_x"] = np.nan
        result = bk.calculate_front_foot_alignment(df, stance_frame=0, contact_frame=10, front_side="left")
        assert result["status"] == "error"

    def test_unrecognized_shot_falls_back_to_dead_straight_target(self):
        df = _axis_frames(heel_contact=(0.62, 0.80), toe_contact=(0.62, 0.95))
        result = bk.calculate_front_foot_alignment(df, stance_frame=0, contact_frame=10,
                                                     front_side="left", shot_played="reverse_scoop")
        assert result["status"] == "success"
        assert result["target_shot"] is None
        assert result["deviation_degrees"] < 5.0  # straight foot vs. dead-straight fallback target

    def test_forward_defense_target_is_straight(self):
        """Forward defense's own real coaching cue is "get the front foot
        (and head) to the line of the ball" -- same straight target as a
        straight drive, not exempted like the other defensive/back-foot
        shot (backward_defense)."""
        df = _axis_frames(heel_contact=(0.62, 0.80), toe_contact=(0.62, 0.95))
        result = bk.calculate_front_foot_alignment(df, stance_frame=0, contact_frame=10,
                                                     front_side="left", shot_played="forward_defense")
        assert result["status"] == "success"
        assert result["target_shot"] == "forward_defense"
        assert result["deviation_degrees"] < 5.0
        assert result["tier"] == "On Target For The Shot"


class TestFrontFootAlignmentNotApplicableShots:
    """Back-foot/horizontal-bat/unorthodox shots (full taxonomy the coach
    asked for in the dropdown) where "front foot points toward the shot"
    isn't a real coaching concept -- must report no deviation/target
    rather than silently scoring against an inapplicable straight-line
    default, but must still report the real measured foot direction."""

    @pytest.mark.parametrize("shot_key", sorted(bk.NOT_APPLICABLE_SHOTS))
    def test_not_applicable_shot_reports_direction_but_no_deviation(self, shot_key):
        df = _axis_frames(heel_contact=(0.62, 0.80), toe_contact=(0.70, 0.95))
        result = bk.calculate_front_foot_alignment(df, stance_frame=0, contact_frame=10,
                                                     front_side="left", shot_played=shot_key)
        assert result["status"] == "success"
        assert result["deviation_degrees"] is None
        assert result["target_shot"] == shot_key
        assert result["tier"] == "Not Applicable For This Shot"
        assert result["signed_degrees"] is not None  # real direction still reported
        assert result["side"] == "Off Side"


class TestFrontFootAlignmentRotationInvariance:
    def test_signed_angle_is_unchanged_under_an_arbitrary_camera_rotation(self):
        """
        The whole point of deriving the axis from the STANCE frame's own
        ankle line (rather than assuming a fixed image axis means "down
        the pitch") is that it should work regardless of where the camera
        is standing. Proves that by rotating an entire clean scenario by
        an arbitrary angle and confirming the exact same signed angle
        comes out -- this is what makes side-on and front-on/rear-on both
        usable without hand-coding two separate formulas.
        """
        theta = np.radians(37.0)
        c, s = np.cos(theta), np.sin(theta)

        def rot(x, y):
            return (c * x - s * y, s * x + c * y)

        df = pd.DataFrame([
            {"frame": 0, "LEFT_ANKLE_x": rot(0.60, 0.80)[0], "LEFT_ANKLE_y": rot(0.60, 0.80)[1],
             "RIGHT_ANKLE_x": rot(0.40, 0.80)[0], "RIGHT_ANKLE_y": rot(0.40, 0.80)[1]},
            {"frame": 10, "LEFT_ANKLE_x": rot(0.62, 0.95)[0], "LEFT_ANKLE_y": rot(0.62, 0.95)[1],
             "RIGHT_ANKLE_x": rot(0.40, 0.80)[0], "RIGHT_ANKLE_y": rot(0.40, 0.80)[1],
             "LEFT_HEEL_x": rot(0.62, 0.80)[0], "LEFT_HEEL_y": rot(0.62, 0.80)[1],
             "LEFT_FOOT_INDEX_x": rot(0.70, 0.95)[0], "LEFT_FOOT_INDEX_y": rot(0.70, 0.95)[1]},
        ])
        result = bk.calculate_front_foot_alignment(df, stance_frame=0, contact_frame=10, front_side="left")
        assert result["status"] == "success"
        assert 25.0 < result["signed_degrees"] < 31.0  # same ~28 degrees as the unrotated case


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

    def test_implausible_percent_is_error_not_a_fabricated_number(self):
        """BUG FOUND (2026-08-03, real coach test): a genuine session
        produced 497% -- physically impossible (hips can't move ~5x the
        stance width). Caused by a small-but-nonzero stance_width (not
        caught by the earlier "< 1e-6" literal-zero check) amplifying an
        ordinary hip displacement. Must now report this as an honest
        tracking failure, not a fabricated-looking extreme number."""
        stance = {"frame": 0, "LEFT_ANKLE_x": 0.490, "RIGHT_ANKLE_x": 0.510,
                   "LEFT_HIP_x": 0.500, "RIGHT_HIP_x": 0.500}
        contact = {"frame": 10, "LEFT_ANKLE_x": 0.490, "RIGHT_ANKLE_x": 0.510,
                    "LEFT_HIP_x": 0.400, "RIGHT_HIP_x": 0.400}
        df = pd.DataFrame([stance, contact])
        result = bk.calculate_weight_transfer(df, stance_frame=0, contact_frame=10, front_side="left")
        assert result["status"] == "error"
        assert result["percent"] is None
        assert result["tier"] == "Tracking Drop"

    def test_zero_stance_width_is_error_not_divide_crash(self):
        stance = {"frame": 0, "LEFT_ANKLE_x": 0.50, "RIGHT_ANKLE_x": 0.50,
                   "LEFT_HIP_x": 0.50, "RIGHT_HIP_x": 0.50}
        contact = {"frame": 10, "LEFT_ANKLE_x": 0.50, "RIGHT_ANKLE_x": 0.50,
                    "LEFT_HIP_x": 0.45, "RIGHT_HIP_x": 0.45}
        df = pd.DataFrame([stance, contact])
        result = bk.calculate_weight_transfer(df, stance_frame=0, contact_frame=10, front_side="left")
        assert result["status"] == "error"
        assert result["percent"] is None

    def test_direction_uses_contact_frame_front_ankle_not_stance(self):
        """
        BUG FIX (2026-08-03): this used to read the front ankle's STANCE
        position to decide which direction was "toward the front foot" --
        wrong, because the whole point of a front-foot shot is that the
        foot itself moves forward during the stride. Here the front ankle
        starts BEHIND the stance hip-center (x=0.45 < hip 0.50) but
        strides all the way to x=0.65 (past the hip) by contact -- the
        hips moving to x=0.60 is a genuine, full forward transfer, and
        must score as strongly positive using the CONTACT ankle position,
        not flip sign using the stale stance position.
        """
        stance = {"frame": 0, "LEFT_ANKLE_x": 0.45, "RIGHT_ANKLE_x": 0.35,
                   "LEFT_HIP_x": 0.50, "RIGHT_HIP_x": 0.50}
        contact = {"frame": 10, "LEFT_ANKLE_x": 0.65, "RIGHT_ANKLE_x": 0.35,
                    "LEFT_HIP_x": 0.60, "RIGHT_HIP_x": 0.60}
        df = pd.DataFrame([stance, contact])
        result = bk.calculate_weight_transfer(df, stance_frame=0, contact_frame=10, front_side="left")
        assert result["status"] == "success"
        assert result["percent"] > 0


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


class TestFrontKneeFlexion:
    """New metric (2026-08-03), same Law-of-Cosines formula as
    kinematics.calculate_knee_bracing and this module's own
    calculate_top_elbow_angle -- reusing the exact same numeric shapes
    from TestTopElbowAngle above (shoulder/elbow/wrist -> hip/knee/ankle)
    since it's the identical formula, just a different joint."""

    def test_athletic_front_knee_flex(self):
        row = pd.Series({
            "LEFT_HIP_x": 0.40, "LEFT_HIP_y": 0.30,
            "LEFT_KNEE_x": 0.42, "LEFT_KNEE_y": 0.45,
            "LEFT_ANKLE_x": 0.50, "LEFT_ANKLE_y": 0.55,
        })
        result = bk.calculate_front_knee_flexion(row, front_side="left")
        assert result["status"] == "success"
        assert result["tier"] == "Athletic Front-Knee Flex"
        assert 100.0 <= result["degrees"] <= 170.0

    def test_collapsed_front_knee(self):
        row = pd.Series({
            "LEFT_HIP_x": 0.40, "LEFT_HIP_y": 0.30,
            "LEFT_KNEE_x": 0.45, "LEFT_KNEE_y": 0.45,
            "LEFT_ANKLE_x": 0.38, "LEFT_ANKLE_y": 0.33,
        })
        result = bk.calculate_front_knee_flexion(row, front_side="left")
        assert result["status"] == "success"
        assert result["tier"] == "Collapsed Front Knee"

    def test_near_zero_angle_is_implausible_not_fabricated(self):
        # kh and ka point in nearly the same direction from the knee ->
        # angle ~0, anatomically impossible -- a landmark-collapse
        # artifact, must report as a tracking failure, not a real 0.
        row = pd.Series({
            "LEFT_HIP_x": 0.50, "LEFT_HIP_y": 0.50,
            "LEFT_KNEE_x": 0.50, "LEFT_KNEE_y": 0.60,
            "LEFT_ANKLE_x": 0.50, "LEFT_ANKLE_y": 0.55,
        })
        result = bk.calculate_front_knee_flexion(row, front_side="left")
        assert result["status"] == "error"
        assert result["degrees"] is None
        assert result["tier"] == "Tracking Drop"

    def test_degenerate_landmarks_is_error(self):
        row = pd.Series({
            "LEFT_HIP_x": 0.40, "LEFT_HIP_y": 0.30,
            "LEFT_KNEE_x": 0.40, "LEFT_KNEE_y": 0.30,  # same point as hip
            "LEFT_ANKLE_x": 0.50, "LEFT_ANKLE_y": 0.55,
        })
        result = bk.calculate_front_knee_flexion(row, front_side="left")
        assert result["status"] == "error"
        assert result["degrees"] is None


class TestFallingOverRisk:
    """
    Compound fault detector: the coach's own worked example -- head AND
    front foot both drifting toward the danger side of the ball's actual
    line. Uses the same self-calibrating off/leg axis as front-foot
    alignment (local_x=(1,0), local_y=(0,1) with these fixture values --
    see _axis_frames's docstring above).
    """

    def _frames(self, front_ankle_contact_x, nose_contact_x):
        stance = {"frame": 0, "LEFT_ANKLE_x": 0.60, "LEFT_ANKLE_y": 0.80,
                   "RIGHT_ANKLE_x": 0.40, "RIGHT_ANKLE_y": 0.80,
                   "NOSE_x": 0.50, "NOSE_y": 0.40}
        contact = {"frame": 10, "LEFT_ANKLE_x": front_ankle_contact_x, "LEFT_ANKLE_y": 0.95,
                    "RIGHT_ANKLE_x": 0.40, "RIGHT_ANKLE_y": 0.80,
                    "NOSE_x": nose_contact_x, "NOSE_y": 0.40}
        return pd.DataFrame([stance, contact])

    def test_flagged_when_head_and_foot_both_drift_toward_the_danger_side(self):
        # front ankle +0.05 (25% of the 0.20 stance width), nose +0.06
        # (30%) -- both well past the 15% threshold, same direction as
        # an off-stump ball's danger side.
        df = self._frames(front_ankle_contact_x=0.65, nose_contact_x=0.56)
        result = bk.detect_falling_over_risk(df, stance_frame=0, contact_frame=10,
                                              front_side="left", ball_line="off")
        assert result["status"] == "success"
        assert result["flagged"] is True
        assert result["reason"] is not None
        assert result["head_shift_pct"] > 15.0
        assert result["foot_cross_pct"] > 15.0

    def test_not_flagged_when_drift_is_away_from_the_danger_side(self):
        df = self._frames(front_ankle_contact_x=0.55, nose_contact_x=0.44)
        result = bk.detect_falling_over_risk(df, stance_frame=0, contact_frame=10,
                                              front_side="left", ball_line="off")
        assert result["status"] == "success"
        assert result["flagged"] is False

    def test_not_flagged_when_drift_is_below_threshold(self):
        df = self._frames(front_ankle_contact_x=0.605, nose_contact_x=0.505)
        result = bk.detect_falling_over_risk(df, stance_frame=0, contact_frame=10,
                                              front_side="left", ball_line="off")
        assert result["status"] == "success"
        assert result["flagged"] is False

    def test_middle_line_is_not_applicable(self):
        df = self._frames(front_ankle_contact_x=0.65, nose_contact_x=0.56)
        result = bk.detect_falling_over_risk(df, stance_frame=0, contact_frame=10,
                                              front_side="left", ball_line="middle")
        assert result["status"] == "not_applicable"
        assert result["flagged"] is False

    def test_no_ball_line_is_not_applicable(self):
        df = self._frames(front_ankle_contact_x=0.65, nose_contact_x=0.56)
        result = bk.detect_falling_over_risk(df, stance_frame=0, contact_frame=10,
                                              front_side="left", ball_line=None)
        assert result["status"] == "not_applicable"

    def test_implausible_drift_is_error_not_a_fabricated_number(self):
        """BUG FOUND (2026-08-03, real coach test): a genuine session
        produced head_shift_pct=146.3%, foot_cross_pct=73.6% -- the head
        allegedly drifted MORE than the entire stance width sideways,
        physically impossible for a real batting shot. Same root cause as
        the weight_transfer 497% bug: a small-but-nonzero stance_width
        denominator amplifying an ordinary movement into an absurd
        percentage. Must report this as an honest tracking failure, not a
        fabricated-looking extreme number."""
        df = self._frames(front_ankle_contact_x=0.85, nose_contact_x=0.56)  # foot drift = 125%
        result = bk.detect_falling_over_risk(df, stance_frame=0, contact_frame=10,
                                              front_side="left", ball_line="off")
        assert result["status"] == "error"
        assert result["flagged"] is False
        assert result["head_shift_pct"] is None
        assert result["foot_cross_pct"] is None

    def test_leg_side_ball_mirrors_the_check(self):
        # Danger side flips for a leg-stump ball -- drifting toward LEG
        # (negative local_x) is now the dangerous direction.
        df = self._frames(front_ankle_contact_x=0.55, nose_contact_x=0.44)
        result = bk.detect_falling_over_risk(df, stance_frame=0, contact_frame=10,
                                              front_side="left", ball_line="leg")
        assert result["status"] == "success"
        assert result["flagged"] is True
