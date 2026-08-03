"""
batting_kinematics.py

Pure per-frame/windowed metric calculations for batting technique analysis
— the batting equivalent of kinematics.py, following the exact same
pattern: every function takes already-extracted MediaPipe landmarks (a
pandas Series for a single frame, or a DataFrame + frame range for a
window) and returns {"...": value, "tier": str, "status": "success"|"error"},
never fabricating a number when tracking failed (see the "None, not 0.0"
convention used throughout kinematics.py, kept identically here).

FIRST VERSION, same honesty standard as everything else in this project:
these thresholds are grounded in widely-taught batting coaching principles
(head over the ball, front foot toward the line, weight transfer onto the
front foot, a straight-bat downswing, a controlled top-elbow through
contact) but have NOT been tuned against real batting footage the way
kinematics.py's bowling metrics were, over many real clips, across this
whole project. Treat these ranges as a reasoned starting point to
validate against real footage, not as already-proven constants.

KNOWN LIMITATION (documented, not hidden): MediaPipe tracks body
landmarks only — it does not track the bat itself. "Downswing plane"
below uses the midpoint of both wrists as a proxy for the bat handle's
path, which is a reasonable approximation (both hands grip the handle
together) but is NOT the same as tracking the bat's blade/face angle
directly. True bat-face-angle metrics would need a bat object detector —
a separate bootstrap, the same shape as the ball-tracking one.
"""

import numpy as np
import pandas as pd


def calculate_head_movement(df: pd.DataFrame, stance_frame: int, contact_frame: int) -> dict:
    """
    "Get your head over the ball" is one of the most universal batting
    coaching cues — excess head movement away from the ball's line
    between stance and contact is a classic, well-documented technical
    fault. Reuses kinematics.calculate_head_stability's exact approach
    (stddev of NOSE_x across the window) rather than reimplementing it —
    this metric is already generic, not bowling-specific; it just needs a
    batting-relevant window (stance -> contact) instead of a bowling one
    (BFC -> BR).
    """
    from kinematics import calculate_head_stability
    result = calculate_head_stability(df, stance_frame, contact_frame)
    # Relabel to batting-relevant tier language without changing the
    # underlying (proven, reused) computation at all.
    #
    # BUG FOUND (2026-08-03, real coach test): this used 0.015 as its own
    # cutoff for the tier TEXT, while metric_ranges.py's actual green band
    # for batting_head_movement is (0.0, 0.02) — a value like 0.0162 (a
    # real result from that test) landed inside the green/OPTIMAL band by
    # the authoritative classify() call, but showed "Excess Head Drift" as
    # its descriptor, a self-contradictory pair that reached the Gemini
    # prompt as "ZONE: OPTIMAL (descriptor: Excess Head Drift)". Same class
    # of bug already fixed for bowling metrics (ZONE from metric_ranges is
    # the only authoritative source; the tier/descriptor text must never
    # imply a different verdict) — matching the boundary here instead of
    # inventing a second, independent one.
    if result.get("status") == "success":
        std_dev = float(result["deviation_index"])
        tier = "Head Still Over The Ball" if std_dev <= 0.02 else "Excess Head Drift"
        return {"deviation_index": result["deviation_index"], "tier": tier, "status": "success"}
    return {"deviation_index": None, "tier": result.get("tier", "Data Deficit"), "status": "error"}


def calculate_front_foot_alignment(row: pd.Series, front_side: str) -> dict:
    """
    Angle of the front foot (HEEL -> FOOT_INDEX vector) relative to the
    image's vertical axis. Assumes the camera is positioned behind the
    stumps looking down the pitch (the setup this whole batting feature
    was scoped around) — in that view, "down the pitch" runs roughly
    vertically in frame, so a front foot pointing straight down the pitch
    (classic "step to the line of the ball") reads as a SMALL angle from
    vertical, while an open or closed foot position (across the line)
    reads as a LARGER angle. front_side: 'left' or 'right' — whichever
    foot is the front foot for this batter's stance.
    """
    try:
        side = "LEFT" if front_side == "left" else "RIGHT"
        heel = np.array([float(row[f"{side}_HEEL_x"]), float(row[f"{side}_HEEL_y"])])
        toe = np.array([float(row[f"{side}_FOOT_INDEX_x"]), float(row[f"{side}_FOOT_INDEX_y"])])
        vec = toe - heel
        if np.linalg.norm(vec) == 0 or np.isnan(vec).any():
            return {"degrees": None, "tier": "Tracking Drop", "status": "error"}

        angle = round(float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1])))), 1)
        if np.isnan(angle):
            return {"degrees": None, "tier": "Tracking Drop", "status": "error"}

        if angle <= 20.0:
            tier = "Aligned To The Line"
        elif angle <= 35.0:
            tier = "Slightly Open/Closed"
        else:
            tier = "Significantly Open/Closed Foot"
        return {"degrees": angle, "tier": tier, "status": "success"}
    except Exception:
        return {"degrees": None, "tier": "Data Deficit", "status": "error"}


def calculate_weight_transfer(df: pd.DataFrame, stance_frame: int, contact_frame: int,
                               front_side: str) -> dict:
    """
    "Transfer your weight into the shot" — measures how far the mid-hip
    point has moved TOWARD the front foot between stance and contact,
    normalized by stance width (ankle-to-ankle distance at stance), so
    the result is a scale-independent percentage rather than raw pixels.
    A batter whose weight never shifts forward (a common fault, "falling
    over" or staying stuck on the back foot) shows near-0%; a full
    transfer onto the front foot shows a larger positive percentage.
    front_side: 'left' or 'right' — the front foot's side, used to know
    which direction "toward the front foot" actually points for this
    batter's stance.
    """
    try:
        stance_rows = df[df["frame"] == stance_frame]
        contact_rows = df[df["frame"] == contact_frame]
        if stance_rows.empty or contact_rows.empty:
            return {"percent": None, "tier": "Window Missing", "status": "error"}
        stance_row = stance_rows.iloc[0]
        contact_row = contact_rows.iloc[0]

        front = "LEFT" if front_side == "left" else "RIGHT"
        back = "RIGHT" if front_side == "left" else "LEFT"

        stance_width = abs(float(stance_row[f"{front}_ANKLE_x"]) - float(stance_row[f"{back}_ANKLE_x"]))
        if stance_width < 1e-6 or np.isnan(stance_width):
            return {"percent": None, "tier": "Tracking Drop", "status": "error"}

        stance_hip_x = (float(stance_row["LEFT_HIP_x"]) + float(stance_row["RIGHT_HIP_x"])) / 2
        contact_hip_x = (float(contact_row["LEFT_HIP_x"]) + float(contact_row["RIGHT_HIP_x"])) / 2
        front_ankle_x = float(stance_row[f"{front}_ANKLE_x"])

        # Positive = hips moved toward the front ankle; negative = moved
        # further away from it (weight going the wrong way).
        direction = 1.0 if front_ankle_x >= stance_hip_x else -1.0
        displacement = (contact_hip_x - stance_hip_x) * direction
        percent = round(float(displacement / stance_width) * 100, 1)
        if np.isnan(percent):
            return {"percent": None, "tier": "Tracking Drop", "status": "error"}

        # IMPLAUSIBILITY GUARD (2026-08-03, found on a real coach test: a
        # genuine session produced 497%, which would mean the hips moved
        # nearly 5x the entire stance width — not physically possible for
        # a real batting shot). The 1e-6 check above only catches a
        # LITERALLY zero-width stance; it doesn't catch a small-but-nonzero
        # stance_width (e.g. from an imprecisely chosen stance frame or
        # noisy ankle tracking), which still blows up an ordinary hip
        # displacement into an absurd percentage when divided. Same
        # reasoning as the knee-bracing near-zero-angle guard elsewhere in
        # this codebase: an engineering ceiling based on what's physically
        # plausible, not a cited biomechanics constant. Beyond this, the
        # result is far more likely a degenerate denominator than a real
        # measurement — honest "couldn't measure this" beats a fabricated-
        # looking extreme number.
        WEIGHT_TRANSFER_IMPLAUSIBLE_CEILING = 150.0
        if abs(percent) > WEIGHT_TRANSFER_IMPLAUSIBLE_CEILING:
            return {"percent": None, "tier": "Tracking Drop", "status": "error"}

        if percent >= 40.0:
            tier = "Committed Weight Transfer"
        elif percent >= 20.0:
            tier = "Partial Weight Transfer"
        else:
            tier = "Stuck On The Back Foot"
        return {"percent": percent, "tier": tier, "status": "success"}
    except Exception:
        return {"percent": None, "tier": "Data Deficit", "status": "error"}


def calculate_downswing_plane(df: pd.DataFrame, backlift_frame: int, contact_frame: int) -> dict:
    """
    Proxy for "straight bat" downswing plane — see this module's docstring
    for why this uses the midpoint of both wrists (both hands grip the
    handle together) as a stand-in for the bat handle's path, not the
    bat's actual blade/face angle. Measures the angle of the path from
    the backlift position to the contact position, relative to vertical.
    Too flat (swinging around the body) or too steep/chopping (an
    exaggerated near-vertical downswing) are both real, commonly-coached
    faults — hence a "band" classification (a middle range is best),
    same shape as hip_shoulder_separation in metric_ranges.py.
    """
    try:
        back_rows = df[df["frame"] == backlift_frame]
        contact_rows = df[df["frame"] == contact_frame]
        if back_rows.empty or contact_rows.empty:
            return {"degrees": None, "tier": "Window Missing", "status": "error"}
        back_row = back_rows.iloc[0]
        contact_row = contact_rows.iloc[0]

        def _wrist_mid(row):
            return np.array([
                (float(row["LEFT_WRIST_x"]) + float(row["RIGHT_WRIST_x"])) / 2,
                (float(row["LEFT_WRIST_y"]) + float(row["RIGHT_WRIST_y"])) / 2,
            ])

        back_pt = _wrist_mid(back_row)
        contact_pt = _wrist_mid(contact_row)
        vec = contact_pt - back_pt
        if np.linalg.norm(vec) == 0 or np.isnan(vec).any():
            return {"degrees": None, "tier": "Tracking Drop", "status": "error"}

        angle = round(float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1])))), 1)
        if np.isnan(angle):
            return {"degrees": None, "tier": "Tracking Drop", "status": "error"}

        if 10.0 <= angle <= 35.0:
            tier = "Straight-Bat Downswing"
        elif angle < 10.0:
            tier = "Steep/Chopping Downswing"
        else:
            tier = "Round-The-Body Swing"
        return {"degrees": angle, "tier": tier, "status": "success"}
    except Exception:
        return {"degrees": None, "tier": "Data Deficit", "status": "error"}


def calculate_top_elbow_angle(row: pd.Series, top_hand_side: str) -> dict:
    """
    Elbow angle (Law of Cosines, same formula as
    kinematics.calculate_knee_bracing) of the TOP hand at the point of
    contact. A collapsed ("chicken wing") top elbow is a well-known loss-
    of-control fault; a fully rigid/locked elbow is also atypical for a
    fluid shot — hence a "band" shape rather than pure higher-is-better.
    top_hand_side: 'left' or 'right' — whichever hand is the top hand on
    the handle for this batter (see detect_batting_hand in
    batting_events.py).
    """
    try:
        side = "LEFT" if top_hand_side == "left" else "RIGHT"
        s = np.array([float(row[f"{side}_SHOULDER_x"]), float(row[f"{side}_SHOULDER_y"])])
        e = np.array([float(row[f"{side}_ELBOW_x"]), float(row[f"{side}_ELBOW_y"])])
        w = np.array([float(row[f"{side}_WRIST_x"]), float(row[f"{side}_WRIST_y"])])

        es, ew = s - e, w - e
        denom = np.linalg.norm(es) * np.linalg.norm(ew)
        if denom == 0 or np.isnan(denom):
            return {"degrees": None, "tier": "Tracking Drop", "status": "error"}

        cos_theta = np.dot(es, ew) / denom
        angle = round(float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))), 1)

        if 100.0 <= angle <= 160.0:
            tier = "Controlled Top-Elbow"
        elif angle < 100.0:
            tier = "Collapsed (Chicken Wing) Elbow"
        else:
            tier = "Locked/Rigid Elbow"
        return {"degrees": angle, "tier": tier, "status": "success"}
    except Exception:
        return {"degrees": None, "tier": "Data Deficit", "status": "error"}
