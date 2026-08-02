"""
batting_events.py

Batting-equivalent of orchestrator.py's detect_bowling_arm/
detect_delivery_events — auto-detects which side is the batter's leading
side, and the key phase frames (stance, backlift, contact, follow-
through) from the pose landmarks alone.

HONESTY NOTE (matches this whole project's standard): detect_delivery_events
in orchestrator.py reached its current reliability only after extensive
tuning against many real bowling clips, over the course of this entire
project. This batting equivalent is a first-pass heuristic, not tuned
against real batting footage yet — same reasoning as batting_kinematics.py.
It WILL need real-footage validation and correction before the auto-
detected frames here should be trusted the way bowling's are; the
Streamlit UI's coach-confirmation step (same auto_detected/coach_confirmed
pattern already used for bowling) is what makes that safe to ship now
rather than waiting.

CONTACT is the single most important event here — the batting equivalent
of Ball Release for bowling. Without ball-tracking data (a separate,
already-scoped future feature), there is no way to see exactly where the
ball is, so contact is approximated as the moment of PEAK BAT-PROXY
SPEED (the midpoint of both wrists, since both hands grip the handle
together) — for an attacking shot, bat speed peaks at or very near
contact, the same kind of kinematic proxy already used for "ball
release" in the bowling pipeline (found via wrist peak position, not by
seeing the ball either).
"""

import numpy as np
import pandas as pd


def detect_batting_hand(df: pd.DataFrame, stance_frame: int) -> str:
    """
    Returns 'left' or 'right' — the batter's LEADING side, which by
    cricket convention is both the top hand on the bat handle AND the
    front/leading foot (a right-handed batter has the left hand on top
    and the left foot as the front foot; a left-handed batter is the
    mirror image). The top hand sits physically higher on the handle
    than the bottom hand at address, so at the stance frame, whichever
    wrist has the smaller (higher-in-frame) y position is the leading
    side. Defaults to 'left' (right-handed batter, the more common case)
    if the stance frame can't be read.
    """
    try:
        rows = df[df["frame"] == stance_frame]
        if rows.empty:
            return "left"
        row = rows.iloc[0]
        l_y = float(row["LEFT_WRIST_y"])
        r_y = float(row["RIGHT_WRIST_y"])
        if np.isnan(l_y) or np.isnan(r_y):
            return "left"
        return "left" if l_y <= r_y else "right"
    except Exception:
        return "left"


def detect_batting_events(df: pd.DataFrame, fps: float) -> dict:
    """
    Returns {"STANCE": int, "BACKLIFT": int, "CONTACT": int,
    "FOLLOW_THROUGH": int, "CONTACT_confidence": "high"|"low"}.
    """
    total_frames = len(df)

    if total_frames < 10:
        return {
            "STANCE": 0,
            "BACKLIFT": int(total_frames * 0.3),
            "CONTACT": int(total_frames * 0.6),
            "FOLLOW_THROUGH": max(0, total_frames - 1),
            "CONTACT_confidence": "low",
        }

    lwx = df["LEFT_WRIST_x"].interpolate(method="linear").bfill().ffill().values
    lwy = df["LEFT_WRIST_y"].interpolate(method="linear").bfill().ffill().values
    rwx = df["RIGHT_WRIST_x"].interpolate(method="linear").bfill().ffill().values
    rwy = df["RIGHT_WRIST_y"].interpolate(method="linear").bfill().ffill().values

    mid_x = (lwx + rwx) / 2
    mid_y = (lwy + rwy) / 2

    # STANCE: first frame with a REAL (non-gap-filled) detection of both
    # wrists — the earliest point there's anything trustworthy to measure
    # from at all.
    had_real = (~df["LEFT_WRIST_x"].isna()).values & (~df["RIGHT_WRIST_x"].isna()).values
    real_indices = np.flatnonzero(had_real)
    stance_idx = int(real_indices[0]) if len(real_indices) else 0

    # CONTACT: frame of peak bat-proxy speed (frame-to-frame displacement
    # of the wrist midpoint), searched only from stance onward so an
    # earlier, unrelated hand movement (adjusting the grip, tapping the
    # bat) can't win by being large but irrelevant. Prefers a REAL
    # detection window over a gap-filled one, same reasoning as the
    # bowling release-point search: motion blur right at the fastest part
    # of the swing is exactly where MediaPipe is most likely to briefly
    # lose the wrist, and a frozen forward-filled value can look like a
    # (fake) local peak.
    dx = np.diff(mid_x, prepend=mid_x[0])
    dy = np.diff(mid_y, prepend=mid_y[0])
    speed = np.hypot(dx, dy)
    speed[:stance_idx] = 0.0

    real_mask = had_real.copy()
    real_mask[:stance_idx] = False
    if real_mask.any():
        candidate_speed = np.where(real_mask, speed, -1.0)
        contact_idx = int(np.argmax(candidate_speed))
        contact_confidence = "high"
    else:
        contact_idx = int(np.argmax(speed))
        contact_confidence = "low"

    # BACKLIFT: the highest point (smallest y — image y increases
    # downward) the bat-proxy reaches BEFORE contact — the top of the
    # swing's backswing, the natural start point for measuring the
    # downswing's path/plane.
    if contact_idx > stance_idx:
        pre_contact_y = mid_y[stance_idx:contact_idx]
        backlift_idx = stance_idx + int(np.argmin(pre_contact_y))
    else:
        backlift_idx = stance_idx

    # FOLLOW-THROUGH: a fixed short window after contact (~0.3s) —
    # reference only (video trimming context), not used by any metric
    # calculation, so it doesn't need the same precision as CONTACT.
    follow_through_idx = min(total_frames - 1, contact_idx + int(round(fps * 0.3)))

    return {
        "STANCE": stance_idx,
        "BACKLIFT": backlift_idx,
        "CONTACT": contact_idx,
        "FOLLOW_THROUGH": follow_through_idx,
        "CONTACT_confidence": contact_confidence,
    }
