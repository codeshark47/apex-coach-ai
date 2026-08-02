"""
tests/test_batting_events.py

Synthetic-data tests for batting_events.py's phase-detection heuristics
— same reasoning as test_batting_kinematics.py, no real video/MediaPipe
needed. These cover the DETECTION LOGIC, not whether the thresholds are
tuned against real footage (they're explicitly not yet — see this
module's own docstring).
"""

import pandas as pd

import batting_events as be


class TestDetectBattingHand:
    def test_left_wrist_higher_is_left_handed(self):
        df = pd.DataFrame([{"frame": 0, "LEFT_WRIST_y": 0.30, "RIGHT_WRIST_y": 0.55}])
        assert be.detect_batting_hand(df, stance_frame=0) == "left"

    def test_right_wrist_higher_is_right_handed(self):
        df = pd.DataFrame([{"frame": 0, "LEFT_WRIST_y": 0.55, "RIGHT_WRIST_y": 0.30}])
        assert be.detect_batting_hand(df, stance_frame=0) == "right"

    def test_missing_stance_frame_defaults_to_left(self):
        df = pd.DataFrame([{"frame": 5, "LEFT_WRIST_y": 0.55, "RIGHT_WRIST_y": 0.30}])
        assert be.detect_batting_hand(df, stance_frame=0) == "left"


def _synthetic_swing_df(n_frames=30, contact_idx=20):
    """
    A wrist-midpoint trajectory that's flat (stance) for the first few
    frames, rises to a high point (backlift) partway through, then makes
    one large, fast movement into contact_idx (the swing), then settles
    (follow-through) — enough structure for detect_batting_events to find
    a genuine stance start, backlift peak, and contact frame.
    """
    rows = []
    for i in range(n_frames):
        if i < 5:
            lx, ly, rx, ry = 0.50, 0.55, 0.52, 0.55  # stance, roughly still
        elif i < 12:
            # backlift: rising (smaller y = higher in frame)
            frac = (i - 5) / 6
            lx, ly, rx, ry = 0.48 - 0.05 * frac, 0.55 - 0.25 * frac, 0.50 - 0.05 * frac, 0.55 - 0.25 * frac
        elif i < contact_idx:
            # slow-ish approach toward contact
            frac = (i - 12) / max(1, (contact_idx - 12))
            lx = 0.43 + 0.05 * frac
            ly = 0.30 + 0.10 * frac
            rx = 0.45 + 0.05 * frac
            ry = 0.30 + 0.10 * frac
        elif i == contact_idx:
            # the fast swing: one big jump = the peak-speed frame
            lx, ly, rx, ry = 0.60, 0.65, 0.62, 0.65
        else:
            lx, ly, rx, ry = 0.61, 0.66, 0.63, 0.66  # follow-through, roughly still
        rows.append({"frame": i, "LEFT_WRIST_x": lx, "LEFT_WRIST_y": ly,
                     "RIGHT_WRIST_x": rx, "RIGHT_WRIST_y": ry})
    return pd.DataFrame(rows)


class TestDetectBattingEvents:
    def test_short_clip_uses_proportional_fallback(self):
        df = pd.DataFrame([{"frame": i, "LEFT_WRIST_x": 0.5, "LEFT_WRIST_y": 0.5,
                             "RIGHT_WRIST_x": 0.5, "RIGHT_WRIST_y": 0.5} for i in range(5)])
        events = be.detect_batting_events(df, fps=30)
        assert events["CONTACT_confidence"] == "low"
        assert 0 <= events["STANCE"] <= events["CONTACT"] <= events["FOLLOW_THROUGH"] < 5

    def test_contact_lands_on_the_peak_speed_frame(self):
        df = _synthetic_swing_df(n_frames=30, contact_idx=20)
        events = be.detect_batting_events(df, fps=30)
        assert events["CONTACT"] == 20
        assert events["CONTACT_confidence"] == "high"

    def test_backlift_is_before_contact_and_a_real_high_point(self):
        df = _synthetic_swing_df(n_frames=30, contact_idx=20)
        events = be.detect_batting_events(df, fps=30)
        assert events["STANCE"] <= events["BACKLIFT"] < events["CONTACT"]

    def test_follow_through_is_after_contact_and_within_bounds(self):
        df = _synthetic_swing_df(n_frames=30, contact_idx=20)
        events = be.detect_batting_events(df, fps=30)
        assert events["CONTACT"] < events["FOLLOW_THROUGH"] <= 29

    def test_stance_is_first_real_detection_not_a_gap_filled_frame(self):
        rows = _synthetic_swing_df(n_frames=30, contact_idx=20).to_dict("records")
        # First 3 frames have no real wrist detection at all (NaN).
        for r in rows[:3]:
            r["LEFT_WRIST_x"] = float("nan")
            r["RIGHT_WRIST_x"] = float("nan")
        df = pd.DataFrame(rows)
        events = be.detect_batting_events(df, fps=30)
        assert events["STANCE"] == 3
