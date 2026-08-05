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

DUAL CAMERA ANGLE (2026-08-03, coach requirement: nets sessions often
can't be filmed side-on because the net physically obstructs that shot,
so front-on/rear-on footage must work too — see camera_angle_detection.py
for how the app tells the two apart). front_foot_alignment below is
redesigned to be self-calibrating rather than assuming a fixed image axis
means "down the pitch": it derives its own reference axis per clip from
the STANCE frame's own back-ankle -> front-ankle line (which approximates
the crease direction for almost any normal stance, regardless of where
the camera is standing), then measures the front foot's direction at
CONTACT relative to that derived axis. This works the same way whether
the camera happened to be side-on, front-on, or somewhere in between,
which is exactly why this approach was chosen over hand-coding two
separate "vertical means X" assumptions per camera angle. See
_derive_batting_axes for the full derivation and its documented
assumptions/limitations.

weight_transfer and downswing_plane were NOT redesigned this way — both
fundamentally need the batter's forward stride (down the pitch) to
project onto a real image axis, which side-on filming gives cleanly but
front-on/rear-on foreshortens into the depth axis (invisible to 2D pose
landmarks). Rather than inventing an unvalidated front-on formula for
these two, the honest choice is: keep computing them as before (no
metric silently disappears), but the calling orchestrator marks them as
reduced-confidence when the camera angle is front_or_rear, so the coach
and the AI coaching report both treat them with appropriate caution —
same "disclose, don't gate" pattern this project already uses for
release-detection confidence in the bowling pipeline.
"""

import numpy as np
import pandas as pd


def _derive_batting_axes(df: pd.DataFrame, stance_frame: int, contact_frame: int,
                          front_side: str):
    """
    Derives a per-clip, self-calibrating 2D coordinate frame for "off side"
    vs. "down the pitch", from body landmarks alone — no stumps/crease
    calibration click exists yet, so this is the best available signal.

    ASSUMPTION (documented, not hidden): at STANCE, a batter's back-ankle
    and front-ankle sit close to the crease line, side by side (normal
    shoulder-width stance) — the vector from the back ankle to the front
    ankle therefore approximates the crease direction in image space,
    regardless of where the camera is standing. This can be wrong for an
    unusually open/closed/exaggerated stance; there is no way to detect
    that from pose landmarks alone, so it is accepted as a reasoned
    approximation, same honesty standard as every other first-pass metric
    in this module.

    local_x: unit vector along the derived crease line, pointing from the
    LEG side toward the OFF side. Sign is fixed by cricket convention: for
    a right-handed batter (front_side == "left"), the front (left) foot
    sits marginally toward the off side of the back (right) foot in a
    normal stance — so back_ankle -> front_ankle already points leg -> off.
    This holds symmetrically for a left-handed batter by the same
    convention (front_side is already defined as the LEADING side in
    batting_events.detect_batting_hand, independent of literal left/right).

    local_y: unit vector approximating "down the pitch, toward the
    bowler" — the perpendicular to local_x, with its sign resolved using
    the REAL, measured stride direction (front ankle's own displacement
    from stance to contact) rather than guessed, since a batter playing
    any shot at all moves the front foot toward the ball at least
    slightly. If the front foot barely moves (e.g. a pure back-foot
    shot), this sign resolution has no real signal to use and keeps
    whichever rotation was picked arbitrarily — a known, disclosed edge
    case, not a hidden failure.

    Returns {"local_x": np.array([x, y]), "local_y": np.array([x, y]),
    "status": "success"} or {"status": "error"} on any tracking failure.
    """
    try:
        stance_rows = df[df["frame"] == stance_frame]
        contact_rows = df[df["frame"] == contact_frame]
        if stance_rows.empty or contact_rows.empty:
            return {"status": "error"}
        stance_row = stance_rows.iloc[0]
        contact_row = contact_rows.iloc[0]

        front = "LEFT" if front_side == "left" else "RIGHT"
        back = "RIGHT" if front_side == "left" else "LEFT"

        front_ankle_stance = np.array([float(stance_row[f"{front}_ANKLE_x"]),
                                        float(stance_row[f"{front}_ANKLE_y"])])
        back_ankle_stance = np.array([float(stance_row[f"{back}_ANKLE_x"]),
                                       float(stance_row[f"{back}_ANKLE_y"])])
        crease_vec = front_ankle_stance - back_ankle_stance
        crease_norm = np.linalg.norm(crease_vec)
        if crease_norm == 0 or np.isnan(crease_norm):
            return {"status": "error"}
        local_x = crease_vec / crease_norm  # leg -> off, by convention above

        candidate_y = np.array([-local_x[1], local_x[0]])  # one of two perpendiculars

        front_ankle_contact = np.array([float(contact_row[f"{front}_ANKLE_x"]),
                                         float(contact_row[f"{front}_ANKLE_y"])])
        stride_vec = front_ankle_contact - front_ankle_stance
        if np.dot(candidate_y, stride_vec) < 0:
            candidate_y = -candidate_y
        y_norm = np.linalg.norm(candidate_y)
        if y_norm == 0 or np.isnan(y_norm):
            return {"status": "error"}
        local_y = candidate_y / y_norm

        if np.isnan(local_x).any() or np.isnan(local_y).any():
            return {"status": "error"}

        return {"local_x": local_x, "local_y": local_y, "status": "success"}
    except Exception:
        return {"status": "error"}


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
        # BUG FIX (2026-08-05): threshold moved from 0.02 to 0.08 to match
        # metric_ranges.py's batting_head_movement green band — that band
        # itself moved because the underlying calculate_head_stability
        # value changed scale entirely (now normalized by shoulder width,
        # not raw camera-distance-dependent pixels). Same "keep this in
        # sync with metric_ranges, don't invent a second boundary" fix
        # already applied here once before.
        tier = "Head Still Over The Ball" if std_dev <= 0.08 else "Excess Head Drift"
        return {
            "deviation_index": result["deviation_index"], "tier": tier, "status": "success",
            "recalibration_pending": result.get("recalibration_pending", False),
        }
    return {"deviation_index": None, "tier": result.get("tier", "Data Deficit"), "status": "error"}


# Coach-selectable named shots -> target front-foot direction, in degrees
# from "straight down the pitch" (positive = toward off side, negative =
# toward leg side) — see _derive_batting_axes for how that reference axis
# itself is computed. EXPERT-APPROXIMATED coaching heuristic (the user's
# own worked example: a cover drive's front foot should point "between
# mid-off and extra cover", not toward point) — NOT empirically validated
# against real footage or a cited biomechanics source. A reasoned
# starting point to refine once real footage/coach feedback is available,
# same honesty standard as every other threshold in this module.
#
# Deliberately covers only shots where "front foot points toward the
# shot" is a real, established coaching cue: the front-foot drives, plus
# forward defense (its own well-known cue is "get the front foot AND head
# to the line of the ball" -- i.e. straight, same as a straight drive's
# target). See NOT_APPLICABLE_SHOTS below for the rest of a full cricket
# shot vocabulary the coach asked for in the dropdown, where this specific
# metric's premise doesn't hold.
SHOT_TARGET_CENTERS_DEGREES = {
    "straight_drive": 0.0,
    "off_drive": 12.0,
    "cover_drive": 28.0,
    "on_drive": -12.0,
    "flick_leg_glance": -28.0,
    "forward_defense": 0.0,
}
FRONT_FOOT_DEVIATION_TOLERANCE = 15.0  # degrees either side of the shot's target center

# Shots selectable in the UI (for the falling-over check, the report's own
# record, and future metrics) where THIS metric's premise -- "the front
# foot should point toward the fielding position you're hitting" -- simply
# doesn't apply: back-foot shots (weight goes back, not into a forward
# stride), horizontal-bat shots (the bat swings across the line; footwork
# is about clearing room, not aiming the front foot), and unorthodox shots
# with no consistent footwork pattern at all. Scoring these against a
# fake target would be a worse, more misleading result than reporting
# "not applicable" honestly -- same principle as returning None instead
# of a fabricated number on a tracking failure elsewhere in this module.
NOT_APPLICABLE_SHOTS = frozenset({
    "backward_defense",
    "square_cut", "late_cut",
    "pull_shot", "hook_shot",
    "standard_sweep", "reverse_sweep", "slog_sweep",
    "scoop_ramp", "switch_hit",
})


def calculate_front_foot_alignment(df: pd.DataFrame, stance_frame: int, contact_frame: int,
                                    front_side: str, shot_played: str = None) -> dict:
    """
    Signed angle of the front foot (HEEL -> FOOT_INDEX vector, at CONTACT)
    relative to a per-clip "down the pitch" axis derived from the STANCE
    frame (see _derive_batting_axes) — positive = pointing toward the off
    side, negative = toward the leg side, 0 = dead straight.

    shot_played: one of SHOT_TARGET_CENTERS_DEGREES's keys, or None/
    unrecognized. When given, this is the coach-confirmed shot the batter
    was playing (there's no ball-tracking data yet to infer intended shot
    direction automatically) — the coaching insight the user specifically
    asked for: "if he's hitting a cover drive, the toe should be... in
    between mid-off and extra cover", not just "pointing straight down
    the pitch" regardless of the shot. When shot_played is None/
    unrecognized, deviation falls back to measuring against dead-straight
    (0 degrees) — the original, shot-agnostic behavior. When shot_played
    is in NOT_APPLICABLE_SHOTS (a back-foot/horizontal-bat/unorthodox
    shot where "front foot toward the shot" isn't a real coaching
    concept), deviation_degrees is None and the tier says so explicitly,
    rather than silently scoring against an inapplicable straight-line
    default.
    """
    try:
        axes = _derive_batting_axes(df, stance_frame, contact_frame, front_side)
        if axes["status"] != "success":
            return {"signed_degrees": None, "side": None, "deviation_degrees": None,
                     "tier": "Tracking Drop", "status": "error"}
        local_x, local_y = axes["local_x"], axes["local_y"]

        contact_rows = df[df["frame"] == contact_frame]
        if contact_rows.empty:
            return {"signed_degrees": None, "side": None, "deviation_degrees": None,
                     "tier": "Window Missing", "status": "error"}
        contact_row = contact_rows.iloc[0]

        side = "LEFT" if front_side == "left" else "RIGHT"
        heel = np.array([float(contact_row[f"{side}_HEEL_x"]), float(contact_row[f"{side}_HEEL_y"])])
        toe = np.array([float(contact_row[f"{side}_FOOT_INDEX_x"]), float(contact_row[f"{side}_FOOT_INDEX_y"])])
        foot_vec = toe - heel
        if np.linalg.norm(foot_vec) == 0 or np.isnan(foot_vec).any():
            return {"signed_degrees": None, "side": None, "deviation_degrees": None,
                     "tier": "Tracking Drop", "status": "error"}

        signed_degrees = round(float(np.degrees(np.arctan2(
            np.dot(foot_vec, local_x), np.dot(foot_vec, local_y)
        ))), 1)
        if np.isnan(signed_degrees):
            return {"signed_degrees": None, "side": None, "deviation_degrees": None,
                     "tier": "Tracking Drop", "status": "error"}

        side_label = "Straight" if abs(signed_degrees) < 3.0 else (
            "Off Side" if signed_degrees > 0 else "Leg Side"
        )

        if shot_played in NOT_APPLICABLE_SHOTS:
            # Report the real, measured foot direction (still genuinely
            # useful information) but deliberately no target/deviation —
            # see NOT_APPLICABLE_SHOTS's own comment for why scoring one
            # of these against a fake target would be worse than not
            # scoring it at all.
            return {
                "signed_degrees": signed_degrees,
                "side": side_label,
                "deviation_degrees": None,
                "target_shot": shot_played,
                "tier": "Not Applicable For This Shot",
                "status": "success",
            }

        target_center = SHOT_TARGET_CENTERS_DEGREES.get(shot_played, 0.0)
        deviation_degrees = round(abs(signed_degrees - target_center), 1)

        if deviation_degrees <= FRONT_FOOT_DEVIATION_TOLERANCE:
            tier = "On Target For The Shot" if shot_played in SHOT_TARGET_CENTERS_DEGREES \
                else "Aligned To The Line"
        elif deviation_degrees <= 2 * FRONT_FOOT_DEVIATION_TOLERANCE:
            tier = "Slightly Off Target"
        else:
            tier = "Significantly Off Target"

        return {
            "signed_degrees": signed_degrees,
            "side": side_label,
            "deviation_degrees": deviation_degrees,
            "target_shot": shot_played if shot_played in SHOT_TARGET_CENTERS_DEGREES else None,
            "tier": tier,
            "status": "success",
        }
    except Exception:
        return {"signed_degrees": None, "side": None, "deviation_degrees": None,
                 "tier": "Data Deficit", "status": "error"}


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
        # BUG FIX (2026-08-03, found while redesigning this module for dual
        # camera-angle support): this used to read the front ankle's
        # STANCE position as the "target" direction — but the whole point
        # of a front-foot shot is that the front foot MOVES toward the
        # bowler during the stride, so the direction "toward the front
        # foot" should be measured using where that foot actually ends up
        # (CONTACT), not where it started. Using the stance position could
        # silently pick the wrong sign whenever the stride carried the
        # front ankle's x-position past the hip's stance x-position (a
        # real, plausible outcome for a full stride), flipping a genuine
        # forward weight transfer into a "negative" (wrong-direction)
        # reading.
        front_ankle_x = float(contact_row[f"{front}_ANKLE_x"])

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


def calculate_front_knee_flexion(row: pd.Series, front_side: str) -> dict:
    """
    Front (lead) knee angle at contact — Law of Cosines HIP-KNEE-ANKLE,
    identical formula to kinematics.calculate_knee_bracing (bowling's lead
    knee at front-foot-contact/release). New metric added while extending
    this module for dual camera-angle support: reliably measurable from a
    SIDE-ON view specifically (the knee flexes in the plane a side-on
    camera sees edge-on-free), giving side-on footage a genuinely useful
    signal bowling's own front_knee_bracing metric already proves out.
    A collapsed front knee at contact is a common, real coaching fault
    (loss of a stable base to drive through the shot); a fully locked/
    rigid knee is also atypical for a fluid, balanced shot.
    """
    try:
        side = "LEFT" if front_side == "left" else "RIGHT"
        h = np.array([float(row[f"{side}_HIP_x"]), float(row[f"{side}_HIP_y"])])
        k = np.array([float(row[f"{side}_KNEE_x"]), float(row[f"{side}_KNEE_y"])])
        a = np.array([float(row[f"{side}_ANKLE_x"]), float(row[f"{side}_ANKLE_y"])])

        kh, ka = h - k, a - k
        denom = np.linalg.norm(kh) * np.linalg.norm(ka)
        if denom == 0 or np.isnan(denom):
            return {"degrees": None, "tier": "Tracking Drop", "status": "error"}

        cos_theta = np.dot(kh, ka) / denom
        angle = round(float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))), 1)

        # ANATOMICAL PLAUSIBILITY GUARD — same reasoning as this codebase's
        # other near-zero-angle guards (bowling's knee-bracing, batting's
        # weight-transfer ceiling): a real human knee cannot fold to
        # anywhere near 0 degrees, so a value this low is a landmark
        # collapse artifact, not a real reading.
        KNEE_ANGLE_IMPLAUSIBLE_THRESHOLD = 5.0
        if angle < KNEE_ANGLE_IMPLAUSIBLE_THRESHOLD:
            return {"degrees": None, "tier": "Tracking Drop", "status": "error"}

        if angle < 100.0:
            tier = "Collapsed Front Knee"
        elif angle <= 170.0:
            tier = "Athletic Front-Knee Flex"
        else:
            tier = "Locked/Rigid Front Knee"
        return {"degrees": angle, "tier": tier, "status": "success"}
    except Exception:
        return {"degrees": None, "tier": "Data Deficit", "status": "error"}


def detect_falling_over_risk(df: pd.DataFrame, stance_frame: int, contact_frame: int,
                              front_side: str, ball_line: str = None) -> dict:
    """
    Compound technical-fault detector for the specific pattern the coach
    described: the batter's HEAD falling laterally toward a delivery's
    line, combined with the FRONT FOOT crossing over the same way (a
    "scissor"/blocked shape) — classic "falling over" / "playing around
    the front pad", a well-known coaching red flag rather than a single
    metric on its own. Neither head drift nor foot crossover alone proves
    this fault; it's specifically the co-occurrence, toward the danger
    side of the ball actually bowled, that the coach flagged.

    ball_line: coach-reported line of the delivery — "off", "middle", or
    "leg" (there's no mature ball-tracking signal yet to detect this
    automatically). Only "off" and "leg" are evaluated (mirror images of
    the same fault); "middle"/None/unrecognized skips the check rather
    than guessing which direction would be "wrong" for a straight ball.

    Reuses _derive_batting_axes for the same self-calibrating off/leg
    axis as calculate_front_foot_alignment — this is why the compound
    check works under EITHER camera angle without needing to know which
    one was used: the axis is derived from body geometry, not the camera.

    Returns {"flagged": bool, "reason": str|None, "head_shift_pct": float|None,
    "foot_cross_pct": float|None, "status": "success"|"error"|"not_applicable"}.
    """
    if ball_line not in ("off", "leg"):
        return {"flagged": False, "reason": None, "head_shift_pct": None,
                 "foot_cross_pct": None, "status": "not_applicable"}
    try:
        axes = _derive_batting_axes(df, stance_frame, contact_frame, front_side)
        if axes["status"] != "success":
            return {"flagged": False, "reason": None, "head_shift_pct": None,
                     "foot_cross_pct": None, "status": "error"}
        local_x = axes["local_x"]  # leg -> off

        stance_rows = df[df["frame"] == stance_frame]
        contact_rows = df[df["frame"] == contact_frame]
        if stance_rows.empty or contact_rows.empty:
            return {"flagged": False, "reason": None, "head_shift_pct": None,
                     "foot_cross_pct": None, "status": "error"}
        stance_row, contact_row = stance_rows.iloc[0], contact_rows.iloc[0]

        front = "LEFT" if front_side == "left" else "RIGHT"
        back = "RIGHT" if front_side == "left" else "LEFT"
        stance_width = abs(float(stance_row[f"{front}_ANKLE_x"]) - float(stance_row[f"{back}_ANKLE_x"]))
        if stance_width < 1e-6 or np.isnan(stance_width):
            return {"flagged": False, "reason": None, "head_shift_pct": None,
                     "foot_cross_pct": None, "status": "error"}

        nose_stance = np.array([float(stance_row["NOSE_x"]), float(stance_row["NOSE_y"])])
        nose_contact = np.array([float(contact_row["NOSE_x"]), float(contact_row["NOSE_y"])])
        front_ankle_stance = np.array([float(stance_row[f"{front}_ANKLE_x"]), float(stance_row[f"{front}_ANKLE_y"])])
        front_ankle_contact = np.array([float(contact_row[f"{front}_ANKLE_x"]), float(contact_row[f"{front}_ANKLE_y"])])
        if np.isnan(nose_stance).any() or np.isnan(nose_contact).any() \
                or np.isnan(front_ankle_stance).any() or np.isnan(front_ankle_contact).any():
            return {"flagged": False, "reason": None, "head_shift_pct": None,
                     "foot_cross_pct": None, "status": "error"}

        # Signed shift along the off/leg axis, normalized by stance width
        # into a comparable percentage, same normalization convention as
        # calculate_weight_transfer.
        head_shift_pct = round(float(np.dot(nose_contact - nose_stance, local_x) / stance_width) * 100, 1)
        foot_cross_pct = round(float(np.dot(front_ankle_contact - front_ankle_stance, local_x) / stance_width) * 100, 1)

        # IMPLAUSIBILITY GUARD (2026-08-03, found on a real coach test: a
        # genuine session produced head_shift_pct=146.3%, foot_cross_pct=
        # 73.6% — meaning the head allegedly drifted MORE than the entire
        # stance width sideways, well beyond any real batting motion. Same
        # root cause as the weight_transfer 497% bug this codebase already
        # fixed once: the "< 1e-6" check above only catches a literally-
        # zero stance width, not a small-but-nonzero one that still blows
        # up an ordinary lateral movement into an absurd percentage when
        # divided. Lateral (off/leg) drift is inherently a SMALLER motion
        # than the forward stride weight_transfer measures, so this uses a
        # tighter ceiling than that metric's 150.0 — an engineering choice,
        # not a cited biomechanics constant, same honesty standard as
        # every other guard in this module.
        FALLING_OVER_IMPLAUSIBLE_CEILING = 100.0
        if abs(head_shift_pct) > FALLING_OVER_IMPLAUSIBLE_CEILING or abs(foot_cross_pct) > FALLING_OVER_IMPLAUSIBLE_CEILING:
            return {"flagged": False, "reason": None, "head_shift_pct": None,
                     "foot_cross_pct": None, "status": "error"}

        # EXPERT-APPROXIMATED thresholds (not empirically validated): both
        # the head AND the front foot must drift a meaningful amount
        # (>15% of stance width) in the SAME direction, toward the side
        # the ball line makes dangerous, before this is flagged as the
        # named compound fault rather than ordinary shot-to-shot movement.
        DRIFT_THRESHOLD_PCT = 15.0
        danger_sign = 1.0 if ball_line == "off" else -1.0  # off ball -> danger is toward off side
        head_toward_danger = head_shift_pct * danger_sign
        foot_toward_danger = foot_cross_pct * danger_sign

        flagged = head_toward_danger > DRIFT_THRESHOLD_PCT and foot_toward_danger > DRIFT_THRESHOLD_PCT
        reason = None
        if flagged:
            side_word = "off" if ball_line == "off" else "leg"
            reason = (
                f"Head and front foot both drifted toward the {side_word} side on a "
                f"{side_word}-stump line delivery — classic 'falling over'/playing around "
                f"the front pad. This exposes the stumps and is a genuine red flag, not a "
                f"minor style quirk."
            )
        return {
            "flagged": flagged, "reason": reason,
            "head_shift_pct": head_shift_pct, "foot_cross_pct": foot_cross_pct,
            "status": "success",
        }
    except Exception:
        return {"flagged": False, "reason": None, "head_shift_pct": None,
                 "foot_cross_pct": None, "status": "error"}
