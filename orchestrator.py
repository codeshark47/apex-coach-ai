import os
import subprocess
import pandas as pd
import numpy as np
import cv2

from main import extract_video_landmarks
from kinematics import (
    calculate_knee_bracing,
    calculate_trunk_lean,
    calculate_head_stability
)


def detect_bowling_arm(df: pd.DataFrame) -> str:
    """
    Auto-detects which arm is the bowling arm by comparing vertical
    range of motion of each wrist across the whole clip. The bowling arm
    swings through a dramatically larger vertical arc during delivery
    than the non-bowling arm, which stays comparatively still. Returns
    'right' or 'left'. Defaults to 'right' (the original hardcoded
    assumption) if landmarks are unavailable.
    """
    try:
        r_wrist_y = df["RIGHT_WRIST_y"].dropna()
        l_wrist_y = df["LEFT_WRIST_y"].dropna()
        r_range = float(r_wrist_y.max() - r_wrist_y.min()) if len(r_wrist_y) > 1 else 0.0
        l_range = float(l_wrist_y.max() - l_wrist_y.min()) if len(l_wrist_y) > 1 else 0.0
        return "left" if l_range > r_range else "right"
    except Exception:
        return "right"


def detect_delivery_events(df: pd.DataFrame, fps: int, bowling_arm: str = "right") -> dict:
    """
    Robust physical milestone detection using velocity windows.
    bowling_arm: 'right' or 'left' — determines which wrist is used for
    release detection and which ankle is front/back foot. The lead (front)
    foot is always the ankle OPPOSITE the bowling arm.
    """
    total_frames = len(df)

    if total_frames < 10:
        return {
            "BFC": 0,
            "FFC": int(total_frames * 0.4),
            "BR": int(total_frames * 0.8)
        }

    bowl_side = "RIGHT" if bowling_arm == "right" else "LEFT"
    lead_side = "LEFT" if bowling_arm == "right" else "RIGHT"

    wrist_y = df[f"{bowl_side}_WRIST_y"].interpolate(method="linear").bfill().ffill().values
    lead_ankle_y = df[f"{lead_side}_ANKLE_y"].interpolate(method="linear").bfill().ffill().values
    back_ankle_y = df[f"{bowl_side}_ANKLE_y"].interpolate(method="linear").bfill().ffill().values

    # FRONT-FOOT PLANT (FFC), found independently of the wrist: the start
    # of the LAST sustained stretch where the lead ankle stops moving and
    # stays grounded. A normal running stride touches down and lifts again
    # within a couple of frames; the actual delivery-stride plant stays
    # down through release, so it's the only point in the clip where the
    # ankle holds still for a sustained duration. This matters because
    # searching for release (the wrist's highest point) across the WHOLE
    # clip can lock onto an earlier arm-raise during the bowler's
    # gather/jump instead of the real release — constraining the release
    # search to start only after this plant fixes that.
    # Tried a percentile-based threshold (relative to this clip's own
    # distribution) instead of a fixed fraction of total range, on the
    # theory it would self-normalize better across videos. Verified
    # against real footage: it backfired — it became so strict that only
    # an artificial dead-flat stretch survived (see the real-detection
    # guard below), collapsing FFC to frame ~2. Reverted to the looser
    # range-based floor, which verified correctly on real footage once
    # combined with that guard.
    plateau_window = max(2, int(round(fps * 0.12)))
    ankle_range = float(np.nanmax(lead_ankle_y) - np.nanmin(lead_ankle_y))
    stability_floor = max(ankle_range * 0.04, 1e-6)
    rolling_std = pd.Series(lead_ankle_y).rolling(
        window=plateau_window, center=False, min_periods=plateau_window
    ).std().values
    is_stable = rolling_std < stability_floor

    # A frame can only count as "stable" if it — and the rest of its
    # plateau_window — had a REAL detection before any gap-filling. Before
    # the bowler has entered the frame, every point is missing and gets
    # backfilled to one repeated constant (zero variance by construction),
    # which otherwise looks MORE "stable" than genuine stillness and can
    # get mistaken for the plant. Verified: this is what actually excludes
    # the empty-frame stretch, not the threshold formula.
    had_real_detection = (~df[f"{lead_side}_ANKLE_y"].isna()).values
    window_all_real = pd.Series(had_real_detection).rolling(
        window=plateau_window, center=False, min_periods=plateau_window
    ).min().astype(bool).values
    is_stable = is_stable & window_all_real
    plant_end_candidates = np.where(is_stable)[0]

    if len(plant_end_candidates) > 0:
        # Walk back from the end of the LAST stable stretch to where it
        # started — that start is the moment the foot actually touched
        # down and stopped moving.
        ffc_idx = int(plant_end_candidates[-1])
        while ffc_idx > 0 and is_stable[ffc_idx - 1]:
            ffc_idx -= 1
        ffc_idx = max(0, ffc_idx - plateau_window + 1)
    else:
        # No clear plateau found (short/noisy clip) — fall back to the
        # single-frame peak, restricted to the back half of the clip to
        # avoid an early running stride.
        half = int(total_frames * 0.5)
        ffc_idx = half + int(np.argmax(lead_ankle_y[half:]))

    # BALL RELEASE (BR): the bowling wrist's highest point, searched ONLY
    # in a realistic window after front-foot plant — never before it,
    # since release always follows the plant.
    #
    # WIDENED from 0.4s: verified on real rear-view footage where the
    # front-ankle "sustained stillness" plant-detector locked onto an
    # intermediate running stride instead of the true final plant (a
    # multi-stride run-up can show several strides that briefly look
    # "stable" the same way a genuine plant does) — FFC landed ~1 full
    # second early. The true release was still a single, clean, entirely
    # unambiguous global minimum in the wrist trajectory (no competing
    # arm-raise anywhere else in the clip), it just fell outside the old
    # narrow window. A wider window tolerates FFC being somewhat off
    # without reintroducing the original bug this window exists to
    # prevent — that bug was an arm-raise during the gather/jump, which
    # happens BEFORE BFC/FFC chronologically, so it's structurally
    # unreachable by widening a window that only searches FORWARD from FFC.
    br_search_window = max(2, int(round(fps * 1.2)))
    br_search_end = min(total_frames, ffc_idx + br_search_window)
    br_slice = wrist_y[ffc_idx:br_search_end]

    # Prefer a REAL (non-gap-filled) detection for the peak. Verified on
    # real footage: the fastest part of the arm swing (right at release)
    # is exactly where MediaPipe is most likely to briefly lose the wrist
    # to motion blur, which then gets forward-filled to a frozen, stale
    # value — that stale flat value can look like an unbeatable "peak" to
    # a simple argmin and win the search even though it's not a real,
    # current position. Only fall back to the filled data if the whole
    # window has no real detection at all.
    wrist_had_real = (~df[f"{bowl_side}_WRIST_y"].isna()).values

    # ANATOMICAL PLAUSIBILITY GATE: a bowling arm stays substantially
    # extended through the delivery swing (a sharply bent elbow mid-swing
    # is literally the legal-delivery threshold in cricket law, not a real
    # technique variant). Verified directly on real footage: the exact
    # frames where the raw wrist position jitters wildly frame-to-frame
    # (motion blur right at release) also show anatomically-impossible
    # elbow flexion collapsing to ~45-60 degrees — a second, independent
    # signal confirming the wrist landmark itself was lost/confused there,
    # catching cases the gap-filled/interpolated check above misses
    # (MediaPipe reported these frames with normal-looking confidence, so
    # they were never flagged as missing — just wrong). This gate cannot
    # by itself explain a cross-environment frame discrepancy on identical
    # code, but it removes a real, demonstrated source of an unreliable
    # peak candidate from the search regardless of which frame the
    # decoder-level noise happens to land on.
    sx = df[f"{bowl_side}_SHOULDER_x"].values
    sy = df[f"{bowl_side}_SHOULDER_y"].values
    ex = df[f"{bowl_side}_ELBOW_x"].values
    ey = df[f"{bowl_side}_ELBOW_y"].values
    wx = df[f"{bowl_side}_WRIST_x"].values
    wy = df[f"{bowl_side}_WRIST_y"].values
    se_x, se_y = sx - ex, sy - ey
    we_x, we_y = wx - ex, wy - ey
    elbow_norm = np.hypot(se_x, se_y) * np.hypot(we_x, we_y)
    with np.errstate(invalid="ignore", divide="ignore"):
        elbow_cos = np.clip((se_x * we_x + se_y * we_y) / elbow_norm, -1, 1)
    elbow_angle_deg = np.degrees(np.arccos(elbow_cos))
    ELBOW_MIN_PLAUSIBLE_DEG = 90.0
    elbow_plausible = np.nan_to_num(elbow_angle_deg, nan=0.0) >= ELBOW_MIN_PLAUSIBLE_DEG
    wrist_had_real = wrist_had_real & elbow_plausible

    real_mask_slice = wrist_had_real[ffc_idx:br_search_end]

    # RELEASE-DETECTION CONFIDENCE: an honest, decode-independent signal
    # for how much of the search window had usable wrist data (real
    # detection AND anatomically plausible), regardless of which exact
    # frame the peak search lands on. Verified directly against real
    # footage with genuine motion blur at release: different video
    # decoders (different OpenCV builds, or the same clip re-encoded)
    # can each land on a DIFFERENT release frame for the same delivery,
    # because the underlying wrist signal is genuinely ambiguous at the
    # pixel level in that window — not because of a bug in this search.
    # No amount of tuning the search logic itself closes that gap, since
    # it's a property of the source footage (motion blur), not the
    # algorithm. What the app CAN do honestly is detect and disclose it,
    # rather than presenting a specific frame number with false
    # confidence. Below BR_CONFIDENCE_FLOOR of the window being usable,
    # flag low confidence so the UI can warn that release-frame-dependent
    # numbers (release height, speed) may be off by a few frames here.
    br_plausible_fraction = float(real_mask_slice.mean()) if len(real_mask_slice) > 0 else 0.0
    BR_CONFIDENCE_FLOOR = 0.6
    br_confidence = "high" if br_plausible_fraction >= BR_CONFIDENCE_FLOOR else "low"

    # PROMINENCE, not just the single lowest point: a real release swing
    # rises substantially from its own recent baseline (arm coming up
    # from a low, resting position). Verified on real footage this
    # matters once the window above was widened to tolerate FFC timing
    # error — the wider window could otherwise catch a shallow,
    # insignificant dip later on where tracking had briefly drifted onto
    # a bystander doing something unrelated (no real arm-raise dynamics,
    # just a flat, low-amplitude trace with a technically-lower point).
    # For each frame, baseline = the highest real wrist_y (lowest arm
    # position) seen so far since the window started; prominence = how
    # far the current point has risen above that running baseline. The
    # frame with the GREATEST prominence is the real swing-up, not
    # necessarily the frame with the single lowest absolute value.
    if len(br_slice) > 0 and real_mask_slice.any():
        real_slice = np.where(real_mask_slice, br_slice, np.nan)
        running_baseline = np.full(len(real_slice), np.nan)
        current_max = -np.inf
        for i, v in enumerate(real_slice):
            if not np.isnan(v):
                current_max = max(current_max, v)
            if current_max > -np.inf:
                running_baseline[i] = current_max
        prominence = running_baseline - real_slice
        if np.any(~np.isnan(prominence)):
            peak_relative_idx = int(np.nanargmax(prominence))
            # ONSET, not the deepest point of the swing: verified on real
            # footage (a frame showing the actual ball still at the
            # fingertips) that true release happens when the arm FIRST
            # reaches near-full extension — the wrist continues rising
            # slightly further afterward under its own follow-through
            # momentum even though the ball has already left the hand, so
            # the geometric deepest point can land several frames after
            # the real release. Take the EARLIEST frame (proportional
            # threshold, so it scales with each delivery's own swing
            # size) that already reached most of the peak's own
            # prominence — scanning forward from the window start rather
            # than walking backward from the peak, since the swing can
            # briefly dip back down mid-rise (a real "double-dip" in the
            # wrist trajectory, not tracking noise) before its final
            # deepest point, which would stop a backward walk short of
            # the true, earlier onset.
            peak_prominence = prominence[peak_relative_idx]
            ONSET_FRACTION = 0.85
            threshold = ONSET_FRACTION * peak_prominence
            onset_idx = peak_relative_idx
            for i in range(0, peak_relative_idx + 1):
                if not np.isnan(prominence[i]) and prominence[i] >= threshold:
                    onset_idx = i
                    break
            br_idx = ffc_idx + onset_idx
        else:
            br_idx = ffc_idx + int(np.argmin(br_slice))
    elif len(br_slice) > 0:
        br_idx = ffc_idx + int(np.argmin(br_slice))
    else:
        br_idx = min(ffc_idx + 1, total_frames - 1)

    bfc_lookback = max(0, ffc_idx - int(fps * 0.5))
    bfc_window = back_ankle_y[bfc_lookback:ffc_idx]

    if len(bfc_window) > 0:
        bfc_idx = bfc_lookback + int(np.argmax(bfc_window))
    else:
        bfc_idx = max(0, ffc_idx - int(fps * 0.2))

    if bfc_idx == ffc_idx:
        ffc_idx += 1
    if ffc_idx == br_idx:
        br_idx += 1

    return {
        "BFC": int(max(1, bfc_idx)),
        "FFC": int(max(2, ffc_idx)),
        "BR": int(min(br_idx, total_frames - 1)),
        "BR_confidence": br_confidence,
        "BR_plausible_fraction": round(br_plausible_fraction, 2),
    }


# ALIAS: dual_camera_orchestrator.py imports this function under the name
# "embedded_detect_events" (likely written against an earlier/different
# naming convention that never matched this file). Rather than renaming
# detect_delivery_events itself — which risks breaking anything else that
# may depend on the current name — both names now point to the same
# function. Zero behavior change, fixes the ImportError from dual camera mode.
embedded_detect_events = detect_delivery_events


def calculate_hip_shoulder_separation(df: pd.DataFrame, ffc_frame: int) -> dict:
    """
    Measures rotational separation between hip and shoulder planes at FFC.
    Uses arctan2 method — correct for rear-view and side-view footage.

    BUG FIX (was): the old code took abs(shoulder_angle - hip_angle) directly.
    Since each angle individually is in (-180, 180], their raw difference can
    range up to 360 degrees. The old "if >90: separation = 180-separation"
    fold assumed its input was already safely in [0, 180] — whenever the two
    angles straddled the +/-180 boundary (e.g. shoulder_angle=178,
    hip_angle=-178.74, raw abs diff=356.74), that fold produced a NEGATIVE
    nonsense value (180-356.74 = -176.74) instead of the small real
    separation the wraparound actually represented (3.26 degrees here).
    Fix: wrap the angle difference into (-180, 180] BEFORE folding.
    Verified against known non-wraparound cases to produce identical
    results to the old formula, and against the exact failing case to now
    produce a physically plausible value.
    """
    try:
        row = df[df["frame"] == ffc_frame].iloc[0]

        shoulder_angle = np.degrees(np.arctan2(
            row["LEFT_SHOULDER_y"] - row["RIGHT_SHOULDER_y"],
            row["LEFT_SHOULDER_x"] - row["RIGHT_SHOULDER_x"]
        ))
        hip_angle = np.degrees(np.arctan2(
            row["LEFT_HIP_y"] - row["RIGHT_HIP_y"],
            row["LEFT_HIP_x"] - row["RIGHT_HIP_x"]
        ))

        raw_diff = shoulder_angle - hip_angle
        wrapped_diff = (raw_diff + 180) % 360 - 180  # safely in (-180, 180]
        separation = abs(wrapped_diff)                # safely in [0, 180]

        # Hip/shoulder lines are undirected axes, not directed vectors, so a
        # separation beyond 90 degrees represents the same physical twist as
        # its complement — fold into [0, 90].
        if separation > 90:
            separation = 180 - separation

        separation = round(separation, 2)

        if separation >= 25.0:
            tier = "Optimal stretch"
        elif separation >= 15.0:
            tier = "Moderate separation"
        else:
            tier = "Blocked rotation"

        return {"degrees": separation, "tier": tier, "status": "success"}

    except Exception as e:
        return {
            "degrees": None,
            "tier": "Calculation error",
            "status": "error",
            "error_message": str(e)
        }


def calculate_release_height_ratio_safe(br_row: pd.Series, bowling_arm: str = "right") -> dict:
    """
    Calculates release height leverage ratio with expanded real-world tolerances.
    Prevents N/A dropouts on high-arm actions or varied camera distances.
    bowling_arm: 'right' or 'left' — determines which wrist measures release.
    """
    try:
        bowl_side = "RIGHT" if bowling_arm == "right" else "LEFT"
        y_wrist = br_row.get(f"{bowl_side}_WRIST_y")
        y_head = br_row.get("NOSE_y")

        # Use the FRONT/LEAD ankle (opposite the bowling arm) — that's the
        # foot planted on the ground at release, same "lead_side" convention
        # used elsewhere (detect_delivery_events, calculate_knee_bracing).
        # Previously this just preferred whichever ankle was labeled "LEFT"
        # regardless of which leg that actually was — for a left-arm
        # bowler the left ankle is the TRAILING leg, which normally lifts
        # during follow-through, so checking it against the "ankle should
        # be grounded" rule below was flagging normal motion as an error.
        lead_side = "LEFT" if bowling_arm == "right" else "RIGHT"
        trail_side = "RIGHT" if lead_side == "LEFT" else "LEFT"
        y_ankle_lead = br_row.get(f"{lead_side}_ANKLE_y")
        y_ankle_trail = br_row.get(f"{trail_side}_ANKLE_y")
        if y_ankle_lead is not None and not pd.isna(y_ankle_lead):
            y_ankle, ankle_side = y_ankle_lead, lead_side
        elif y_ankle_trail is not None and not pd.isna(y_ankle_trail):
            y_ankle, ankle_side = y_ankle_trail, trail_side
        else:
            y_ankle, ankle_side = None, None

        if any(v is None or pd.isna(v) for v in [y_wrist, y_head, y_ankle]):
            return {
                "ratio": None,
                "classification": "Landmark missing",
                "status": "error",
                "error_message": "One or more landmarks missing on BR frame."
            }

        body_height = abs(float(y_ankle) - float(y_head))

        debug_raw = {
            "y_wrist": round(float(y_wrist), 4),
            "y_head": round(float(y_head), 4),
            "y_ankle": round(float(y_ankle), 4),
            "body_height": round(float(body_height), 4),
            "bowl_side_used": bowl_side,
            "ankle_side_used": ankle_side,
        }

        # Numerical-stability floor only — a near-zero denominator blows the
        # ratio up regardless of whether tracking is good. This is NOT a
        # mistracking signal by itself: a video's head-to-ankle span in the
        # frame depends heavily on camera distance, so a flat cutoff here
        # (previously 0.35) rejects legitimately well-tracked videos just
        # because they were filmed wider or further away.
        if body_height < 0.05:
            return {
                "ratio": None,
                "classification": "Body height too small",
                "status": "error",
                "error_message": (
                    f"Body height span ({round(body_height, 3)}) too small to "
                    f"divide by reliably."
                ),
                "debug_raw": debug_raw
            }

        # Real mistracking check: at ball release the front/plant foot is on
        # the ground, so the ankle landmark should sit below both the knee
        # and hip in the frame (larger y = lower in image coordinates). This
        # holds regardless of camera distance/framing, unlike a raw span
        # cutoff, so it catches an actual mistracked ankle without punishing
        # videos filmed wider or further away.
        y_knee = br_row.get(f"{ankle_side}_KNEE_y")
        y_hip = br_row.get(f"{ankle_side}_HIP_y")
        if y_knee is not None and not pd.isna(y_knee) and y_hip is not None and not pd.isna(y_hip):
            if float(y_ankle) < float(y_knee) or float(y_ankle) < float(y_hip):
                return {
                    "ratio": None,
                    "classification": "Ankle landmark implausible",
                    "status": "error",
                    "error_message": (
                        f"{ankle_side} ankle is not below the {ankle_side.lower()} "
                        f"knee/hip on the BR frame — landmark is likely "
                        f"mistracked rather than the reading being real."
                    ),
                    "debug_raw": {**debug_raw,
                                  "y_knee": round(float(y_knee), 4),
                                  "y_hip": round(float(y_hip), 4)}
                }

        ratio = round(abs(float(y_ankle) - float(y_wrist)) / body_height, 4)

        if ratio > 1.30 or ratio < 0.30:
            return {
                "ratio": None,
                "classification": "Measurement error — verify camera angle",
                "status": "error",
                "error_message": f"Ratio {ratio} outside physical bounds.",
                "debug_raw": debug_raw
            }

        if ratio >= 0.85:
            classification = "High-Release Leverage"
        elif ratio >= 0.75:
            classification = "Standard Mid-Arm Release"
        else:
            classification = "Low-Sling Action"

        return {"ratio": ratio, "classification": classification, "status": "success", "debug_raw": debug_raw}

    except Exception as e:
        return {
            "ratio": None,
            "classification": "Calculation error",
            "status": "error",
            "error_message": str(e)
        }


def _draw_rounded_rect(img, pt1, pt2, color, radius):
    """Filled rounded rectangle — cv2 has no native support for this."""
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(img, (cx, cy), radius, color, -1)


def _draw_panel(frame, pt1, pt2, fill_color=(15, 15, 15), radius=14,
                 fill_alpha=0.55, shadow_offset=6, shadow_alpha=0.35):
    """
    Rounded panel with a soft drop shadow, alpha-blended onto frame in
    place. Replaces the old flat sharp-cornered rectangle, which read as
    a basic "programmer UI" overlay rather than a broadcast graphic.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = frame.shape[:2]

    shadow = frame.copy()
    sx1, sy1 = x1 + shadow_offset, y1 + shadow_offset
    sx2, sy2 = min(x2 + shadow_offset, w - 1), min(y2 + shadow_offset, h - 1)
    _draw_rounded_rect(shadow, (sx1, sy1), (sx2, sy2), (0, 0, 0), radius)
    frame[:] = cv2.addWeighted(shadow, shadow_alpha, frame, 1 - shadow_alpha, 0)

    fill = frame.copy()
    _draw_rounded_rect(fill, (x1, y1), (x2, y2), fill_color, radius)
    frame[:] = cv2.addWeighted(fill, fill_alpha, frame, 1 - fill_alpha, 0)


def generate_fail_safe_video(video_path: str, output_path: str,
                              df: pd.DataFrame, events: dict,
                              slow_motion_factor: float = 4.0,
                              bowling_arm: str = "right"):
    """
    Generates annotated skeleton overlay video using mp4v codec.

    Skeleton style: cyan connective lines and magenta joint markers with a
    soft glow (drawn as a darker, thicker under-stroke plus a brighter,
    thinner over-stroke), closer to the reference look requested, instead
    of flat single-pixel green/red.

    slow_motion_factor: the output video is written with its FPS divided by
    this factor. All frames are kept (nothing skipped or duplicated) — the
    same frame count now plays back over a longer real-world duration,
    which is the correct way to get true slow motion without needing frame
    interpolation. Default 4.0 = 4x slower than real time.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    output_fps = max(1, int(round(source_fps / slow_motion_factor)))

    # LOGO WATERMARK: subtle brand mark in a fixed corner, low opacity so
    # it doesn't distract from the analysis itself. Loaded once, resized
    # once — composited per-frame is just an alpha blend, cheap.
    logo_rgba = None
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apex_logo.png.png")
    if os.path.exists(logo_path):
        raw_logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if raw_logo is not None and raw_logo.shape[2] == 4:
            logo_w = max(40, int(width * 0.09))
            logo_h = int(raw_logo.shape[0] * (logo_w / raw_logo.shape[1]))
            logo_rgba = cv2.resize(raw_logo, (logo_w, logo_h), interpolation=cv2.INTER_AREA)
    LOGO_MARGIN = 14
    LOGO_OPACITY = 0.6

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        output_fps, (width, height)
    )

    df = df.copy()
    landmark_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    # Same fix as main.py's gap-fill: time-based limit (not a fixed frame
    # count) plus limit_area="inside" so a genuinely long tracking loss
    # (e.g. an arm occluded during running) stays real NaN — which the
    # drawing loop below already skips gracefully — instead of getting
    # padded into a fabricated, frozen position that then gets drawn as a
    # stray disconnected limb.
    draw_gap_fill_limit = max(1, int(round(source_fps * 0.1)))
    df[landmark_cols] = df[landmark_cols].interpolate(
        method="linear", limit=draw_gap_fill_limit, limit_direction="both", limit_area="inside"
    )

    # UNIFIED BROADCAST PALETTE: one clean look across the whole skeleton
    # (was: green legs, white upper body, orange joints — a color-by-limb
    # scheme with no real informational purpose that read as a debug
    # overlay rather than a broadcast graphic). Matches a commercial
    # reference sample: bold white bones with a soft dark contact-shadow
    # for legibility on any background, and cyan joints with a white
    # outline ring so they read as clean dots rather than tiny smudges.
    BONE_SHADOW = (25, 25, 25)
    BONE_CORE = (245, 245, 245)
    JOINT_OUTLINE = (255, 255, 255)
    JOINT_CORE = (235, 195, 50)

    # RESOLUTION SCALE: every fixed pixel size below (joint radius, bone
    # width) was tuned against a 1080x1920 test clip. Verified directly on
    # a real, smaller 848x478 clip the founder submitted: those same fixed
    # sizes made the joints so large relative to a more distant, smaller
    # on-screen bowler that they physically overlapped into one solid
    # blob, hiding his whole body. Scale every size by frame dimensions
    # relative to that original tuning resolution instead of using it
    # as-is, clamped so it neither vanishes on tiny clips nor balloons on
    # huge ones.
    render_scale = max(0.35, min(1.5, min(width, height) / 1080.0))

    def _rs(px):
        return max(1, int(round(px * render_scale)))
    # Kept for the release-badge accent color elsewhere (unchanged meaning,
    # just no longer used to color-code limbs in the skeleton itself).
    LEG_LINE_GLOW = BONE_SHADOW
    LEG_LINE_CORE = BONE_CORE
    UPPER_LINE_GLOW = BONE_SHADOW
    UPPER_LINE_CORE = BONE_CORE

    LEG_CONNECTIONS = {
        ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
        ("LEFT_ANKLE", "LEFT_HEEL"), ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
        ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
        ("RIGHT_ANKLE", "RIGHT_HEEL"), ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
        ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
        ("LEFT_HIP", "RIGHT_HIP"),
    }

    connections = [
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),
        ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_SHOULDER", "LEFT_ELBOW"),
        ("LEFT_ELBOW", "LEFT_WRIST"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("RIGHT_ELBOW", "RIGHT_WRIST"),
        ("LEFT_HIP", "LEFT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_HIP", "RIGHT_KNEE"),
        ("RIGHT_KNEE", "RIGHT_ANKLE"),
        ("LEFT_ANKLE", "LEFT_HEEL"),
        ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
        ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
        ("RIGHT_ANKLE", "RIGHT_HEEL"),
        ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
        ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
        ("NOSE", "LEFT_SHOULDER"),
        ("NOSE", "RIGHT_SHOULDER"),
    ]
    joint_nodes = ["LEFT_KNEE", "RIGHT_KNEE", "LEFT_HIP", "RIGHT_HIP",
                   "LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE",
                   "LEFT_SHOULDER", "RIGHT_SHOULDER", "NOSE"]

    total_frames = int(df["frame"].max()) + 1 if len(df) else 0

    def _row_knee_angle(r):
        try:
            h = np.array([float(r["LEFT_HIP_x"]), float(r["LEFT_HIP_y"])])
            k = np.array([float(r["LEFT_KNEE_x"]), float(r["LEFT_KNEE_y"])])
            a = np.array([float(r["LEFT_ANKLE_x"]), float(r["LEFT_ANKLE_y"])])
            kh, ka = h - k, a - k
            denom = np.linalg.norm(kh) * np.linalg.norm(ka)
            if denom == 0 or np.isnan(denom):
                return np.nan
            cos_theta = np.dot(kh, ka) / denom
            return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
        except Exception:
            return np.nan

    knee_by_frame = {int(r["frame"]): _row_knee_angle(r) for _, r in df.iterrows()}
    knee_series = pd.Series([knee_by_frame.get(i, np.nan) for i in range(total_frames)])
    knee_series = knee_series.interpolate(limit_direction="both")
    knee_arr = knee_series.to_numpy()

    # --- RELEASE HEIGHT % at BR ---
    # Reuses calculate_release_height_ratio_safe (same function that feeds the
    # report card) instead of a separate formula, so the number burned into
    # the video can never disagree with the report, respects bowling_arm
    # instead of always assuming right-arm, and shows nothing rather than a
    # fabricated number when tracking is unreliable.
    release_height_pct = None
    # Normalized (0-1) points for the release-height line drawn on the BR
    # frame: from the bowling-arm wrist straight down to ground level (the
    # same ankle landmark already used to calculate the ratio above), so
    # the line visually matches the number rather than being a separate
    # guess. Only set when the ratio itself is trustworthy — same
    # "don't show it if we don't trust it" rule as the number.
    release_line_pts = None
    br_frame = events.get("BR")
    if br_frame is not None:
        br_rows = df[df["frame"] == br_frame]
        if not br_rows.empty:
            br_row_for_release = br_rows.iloc[0]
            rh = calculate_release_height_ratio_safe(br_row_for_release, bowling_arm=bowling_arm)
            if rh.get("ratio") is not None:
                release_height_pct = rh["ratio"] * 100
                dbg = rh.get("debug_raw") or {}
                bowl_side = dbg.get("bowl_side_used")
                wrist_x = br_row_for_release.get(f"{bowl_side}_WRIST_x") if bowl_side else None
                if (wrist_x is not None and not pd.isna(wrist_x)
                        and "y_wrist" in dbg and "y_ankle" in dbg):
                    release_line_pts = {
                        "wrist_x": float(wrist_x),
                        "wrist_y": dbg["y_wrist"],
                        "ground_y": dbg["y_ankle"],
                    }

    # --- ESTIMATED ELBOW EXTENSION (2D approximation of ICC's 15-degree law) ---
    # NOTE: official ICC assessment requires 3D motion capture; this 2D
    # estimate can be distorted by camera angle/arm rotation toward or away
    # from the lens, and should be presented as an approximation, not an
    # official ruling.
    def _row_elbow_angle(r):
        try:
            s = np.array([float(r["RIGHT_SHOULDER_x"]), float(r["RIGHT_SHOULDER_y"])])
            e = np.array([float(r["RIGHT_ELBOW_x"]), float(r["RIGHT_ELBOW_y"])])
            w = np.array([float(r["RIGHT_WRIST_x"]), float(r["RIGHT_WRIST_y"])])
            es, ew = s - e, w - e
            denom = np.linalg.norm(es) * np.linalg.norm(ew)
            if denom == 0 or np.isnan(denom):
                return np.nan
            cos_theta = np.dot(es, ew) / denom
            return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
        except Exception:
            return np.nan

    elbow_extension_deg = None
    bfc_frame = events.get("BFC")
    if bfc_frame is not None and br_frame is not None and bfc_frame < br_frame:
        window_df = df[(df["frame"] >= bfc_frame) & (df["frame"] <= br_frame)]
        best_frame, best_diff = None, None
        for _, r in window_df.iterrows():
            try:
                sh_y = float(r["RIGHT_SHOULDER_y"])
                wr_y = float(r["RIGHT_WRIST_y"])
                diff = abs(wr_y - sh_y)
                if best_diff is None or diff < best_diff:
                    best_diff, best_frame = diff, r
            except Exception:
                continue
        br_rows2 = df[df["frame"] == br_frame]
        if best_frame is not None and not br_rows2.empty:
            angle_horizontal = _row_elbow_angle(best_frame)
            angle_release = _row_elbow_angle(br_rows2.iloc[0])
            if not np.isnan(angle_horizontal) and not np.isnan(angle_release):
                elbow_extension_deg = max(0.0, angle_release - angle_horizontal)

    ANGLE_MIN, ANGLE_MAX = 0.0, 180.0
    ELITE_MIN = 165.0
    # SHRUNK from 38% of frame height to a slim ~16% strip — the old chart
    # dominated a third of the video, which read as a dashboard bolted onto
    # footage rather than a broadcast graphic. The data itself (live knee
    # angle) stays, just far less visually loud. A separate, even slimmer
    # ZONE_BAR sits below it — a static phase legend inspired directly by
    # the commercial reference sample, which has no live chart at all but
    # does have this clean bottom orientation strip.
    # Floors LOWERED from 46/120 to 30/90: those were tuned as "don't go
    # unreadably small" minimums against a 1920-tall test clip, but on a
    # real, shorter 478px-tall clip they backfired — forcing the bottom
    # overlay to ~35% of frame height instead of the intended ~20%,
    # because the fixed floor overrode the percentage entirely. Still a
    # legibility floor, just one that doesn't dominate genuinely small
    # footage.
    ZONE_BAR_H = max(30, int(height * 0.045))
    PANEL_H = max(90, int(height * 0.16))
    ZONE_BAR_TOP = height - ZONE_BAR_H
    PANEL_TOP = ZONE_BAR_TOP - PANEL_H
    MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 55, 20, 24, 20

    def x_to_px(fidx):
        span = max(total_frames - 1, 1)
        return MARGIN_L + int((fidx / span) * (width - MARGIN_L - MARGIN_R))

    def y_to_px(angle):
        clipped = max(ANGLE_MIN, min(ANGLE_MAX, angle))
        frac = (clipped - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)
        usable = PANEL_H - MARGIN_T - MARGIN_B
        return MARGIN_T + int((1 - frac) * usable)

    chart_base = np.full((PANEL_H, width, 3), (16, 16, 18), dtype=np.uint8)
    elite_y = y_to_px(ELITE_MIN)
    cv2.rectangle(chart_base, (MARGIN_L, MARGIN_T), (width - MARGIN_R, elite_y),
                  (28, 55, 28), -1)
    cv2.putText(chart_base, f"ELITE {ELITE_MIN:.0f}+", (MARGIN_L + 8, MARGIN_T + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 205, 130), 1, cv2.LINE_AA)
    # Fewer gridlines (was every 45deg incl. labels at each) — a compact
    # panel doesn't have room for a dense axis without feeling cramped.
    for g in (0, 90, 180):
        gy = y_to_px(g)
        cv2.line(chart_base, (MARGIN_L, gy), (width - MARGIN_R, gy), (48, 48, 50), 1, cv2.LINE_AA)
        cv2.putText(chart_base, f"{g}", (6, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (150, 150, 150), 1, cv2.LINE_AA)
    prev_pt = None
    for fidx in range(total_frames):
        val = knee_arr[fidx] if fidx < len(knee_arr) else np.nan
        if np.isnan(val):
            prev_pt = None
            continue
        pt = (x_to_px(fidx), y_to_px(val))
        if prev_pt is not None:
            cv2.line(chart_base, prev_pt, pt, JOINT_CORE, 2, cv2.LINE_AA)
        prev_pt = pt
    cv2.putText(chart_base, "LEAD KNEE ANGLE", (MARGIN_L, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)

    PHASE_BADGE_WINDOW = max(3, int(round(source_fps * 0.5)))

    # Tracks the bowler's last-known horizontal screen position (0-1) so
    # the info badge can be placed on whichever side he ISN'T on — a
    # fixed top-left badge was covering him directly at release on
    # tightly-framed footage, exactly the moment a coach most wants an
    # unobstructed view. Persists across any frame with no tracking data
    # so the badge doesn't flicker sides during a brief gap.
    last_known_bowler_x = None

    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        f_match = df[df["frame"] == f_idx]
        if not f_match.empty:
            row = f_match.iloc[0]
            torso_x_cols = ["NOSE_x", "LEFT_HIP_x", "RIGHT_HIP_x"]
            torso_x_vals = [float(row[c]) for c in torso_x_cols if not pd.isna(row[c])]
            if torso_x_vals:
                last_known_bowler_x = sum(torso_x_vals) / len(torso_x_vals)
            for partA, partB in connections:
                try:
                    if pd.isna(row[f"{partA}_x"]) or pd.isna(row[f"{partB}_x"]):
                        continue
                    xA = int(float(row[f"{partA}_x"]) * width)
                    yA = int(float(row[f"{partA}_y"]) * height)
                    xB = int(float(row[f"{partB}_x"]) * width)
                    yB = int(float(row[f"{partB}_y"]) * height)
                    if (0 < xA < width and 0 < yA < height and
                            0 < xB < width and 0 < yB < height):
                        cv2.line(frame, (xA, yA), (xB, yB), BONE_SHADOW, _rs(6), cv2.LINE_AA)
                        cv2.line(frame, (xA, yA), (xB, yB), BONE_CORE, _rs(3), cv2.LINE_AA)
                except Exception:
                    continue

            # SPINE: MediaPipe has no single spine landmark, so this is a
            # virtual centerline from the shoulder midpoint to the hip
            # midpoint. Without it the torso was just a hollow box (shoulder
            # line + two side lines + hip line) with nothing down the
            # middle, which is what made the skeleton look unrigged/amateur
            # rather than like a proper body frame.
            try:
                spine_cols = ["LEFT_SHOULDER_x", "LEFT_SHOULDER_y", "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y",
                              "LEFT_HIP_x", "LEFT_HIP_y", "RIGHT_HIP_x", "RIGHT_HIP_y"]
                if not any(pd.isna(row[c]) for c in spine_cols):
                    neck_x = (float(row["LEFT_SHOULDER_x"]) + float(row["RIGHT_SHOULDER_x"])) / 2
                    neck_y = (float(row["LEFT_SHOULDER_y"]) + float(row["RIGHT_SHOULDER_y"])) / 2
                    midhip_x = (float(row["LEFT_HIP_x"]) + float(row["RIGHT_HIP_x"])) / 2
                    midhip_y = (float(row["LEFT_HIP_y"]) + float(row["RIGHT_HIP_y"])) / 2
                    sx1, sy1 = int(neck_x * width), int(neck_y * height)
                    sx2, sy2 = int(midhip_x * width), int(midhip_y * height)
                    if 0 < sx1 < width and 0 < sy1 < height and 0 < sx2 < width and 0 < sy2 < height:
                        cv2.line(frame, (sx1, sy1), (sx2, sy2), BONE_SHADOW, 6, cv2.LINE_AA)
                        cv2.line(frame, (sx1, sy1), (sx2, sy2), BONE_CORE, 3, cv2.LINE_AA)
                        cv2.circle(frame, (sx1, sy1), _rs(9), JOINT_OUTLINE, -1, cv2.LINE_AA)
                        cv2.circle(frame, (sx1, sy1), _rs(6), JOINT_CORE, -1, cv2.LINE_AA)
            except Exception:
                pass

            for node in joint_nodes:
                try:
                    if pd.isna(row[f"{node}_x"]):
                        continue
                    nx = int(float(row[f"{node}_x"]) * width)
                    ny = int(float(row[f"{node}_y"]) * height)
                    if 0 < nx < width and 0 < ny < height:
                        cv2.circle(frame, (nx, ny), _rs(9), JOINT_OUTLINE, -1, cv2.LINE_AA)
                        cv2.circle(frame, (nx, ny), _rs(6), JOINT_CORE, -1, cv2.LINE_AA)
                except Exception:
                    continue

        # Composited BEFORE the chart panel overwrites the bottom of the
        # frame, and anchored bottom-right of the VIDEO area specifically
        # (not the chart panel) — the badge only ever appears in the top
        # area, so this placement can never collide with it regardless of
        # which side the badge dynamically picks.
        if logo_rgba is not None:
            lh, lw = logo_rgba.shape[:2]
            lx1 = width - lw - LOGO_MARGIN
            ly1 = PANEL_TOP - lh - LOGO_MARGIN
            if lx1 > 0 and ly1 > 0:
                roi = frame[ly1:ly1 + lh, lx1:lx1 + lw]
                logo_bgr = logo_rgba[:, :, :3]
                logo_alpha = (logo_rgba[:, :, 3:4].astype(np.float32) / 255.0) * LOGO_OPACITY
                frame[ly1:ly1 + lh, lx1:lx1 + lw] = (
                    logo_bgr.astype(np.float32) * logo_alpha
                    + roi.astype(np.float32) * (1 - logo_alpha)
                ).astype(np.uint8)

        panel_region = frame[PANEL_TOP:PANEL_TOP + PANEL_H, 0:width]
        blended = cv2.addWeighted(panel_region, 0.35, chart_base, 0.65, 0)
        frame[PANEL_TOP:PANEL_TOP + PANEL_H, 0:width] = blended

        cur_val = knee_arr[f_idx] if f_idx < len(knee_arr) else np.nan
        if not np.isnan(cur_val):
            px, py_rel = x_to_px(f_idx), y_to_px(cur_val)
            py = PANEL_TOP + py_rel
            cv2.line(frame, (px, PANEL_TOP + MARGIN_T), (px, PANEL_TOP + PANEL_H - MARGIN_B),
                     JOINT_CORE, 1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 5, JOINT_OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 3, JOINT_CORE, -1, cv2.LINE_AA)
            cv2.putText(frame, f"{cur_val:.0f}deg", (width - 130, PANEL_TOP + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, JOINT_CORE, 2, cv2.LINE_AA)

        status_text = "RUN-UP"
        if f_idx >= events["BR"]:
            status_text = "FOLLOW-THROUGH"
        elif f_idx >= events["FFC"]:
            status_text = "BALL RELEASE"
        elif f_idx >= events["BFC"]:
            status_text = "DELIVERY STRIDE"

        # ZONE PROGRESS BAR: a slim, mostly-static phase legend inspired by
        # the commercial reference sample (which has no live chart at all,
        # just this bottom strip) — but goes one step further by actually
        # tracking a live position marker along it, since real BFC/FFC/BR
        # boundaries are already known here and the reference apparently
        # doesn't have that data to show. Tick x-positions are evenly
        # spaced (matching the reference's even spacing, not scaled to each
        # phase's actual duration) — this is a schematic orientation aid,
        # not a second data chart.
        zone_bg = frame[ZONE_BAR_TOP:ZONE_BAR_TOP + ZONE_BAR_H, 0:width].copy()
        cv2.rectangle(zone_bg, (0, 0), (width, ZONE_BAR_H), (14, 14, 16), -1)
        frame[ZONE_BAR_TOP:ZONE_BAR_TOP + ZONE_BAR_H, 0:width] = cv2.addWeighted(
            frame[ZONE_BAR_TOP:ZONE_BAR_TOP + ZONE_BAR_H, 0:width], 0.25, zone_bg, 0.75, 0
        )
        zone_labels = ["RUN-UP", "STRIDE", "RELEASE", "FOLLOW-THROUGH"]
        zone_bounds = [0, events.get("BFC", 0), events.get("FFC", 0), events.get("BR", 0),
                       max(total_frames - 1, 1)]
        bar_x0, bar_x1 = MARGIN_L + 10, width - MARGIN_R - 10
        # FIX: was ZONE_BAR_TOP + ZONE_BAR_H//2 + 6, then labels drawn
        # BELOW that at +th+10 — on a shorter clip (lower ZONE_BAR_H,
        # see the floor fix above) that pushed label text past the
        # bottom edge of the frame entirely, clipping it off. Anchor the
        # tick line near the TOP of the bar and the label baseline to a
        # fixed offset from the frame's own bottom edge instead, which is
        # correct by construction regardless of ZONE_BAR_H.
        bar_y = ZONE_BAR_TOP + max(10, int(ZONE_BAR_H * 0.32))
        label_baseline_y = height - 8
        # 5 boundary x-positions (matching the 5 zone_bounds frame indices)
        # for the marker to interpolate across 4 segments; labels are drawn
        # at each segment's MIDPOINT, since each label names a span (e.g.
        # "RELEASE" = the FFC-to-BR window), not a single instant.
        boundary_xs = [int(bar_x0 + (bar_x1 - bar_x0) * i / 4) for i in range(5)]
        label_xs = [(boundary_xs[i] + boundary_xs[i + 1]) // 2 for i in range(4)]
        cv2.line(frame, (boundary_xs[0], bar_y), (boundary_xs[-1], bar_y), (90, 90, 95), 2, cv2.LINE_AA)

        # Current phase index (0-3), used to highlight both the label and
        # find where along its segment the live marker sits.
        phase_idx = 0
        for i in range(3):
            if f_idx >= zone_bounds[i + 1]:
                phase_idx = i + 1
        phase_idx = min(phase_idx, 3)

        for i, (tx, label) in enumerate(zip(label_xs, zone_labels)):
            active = (i == phase_idx)
            tick_color = JOINT_CORE if active else (130, 130, 135)
            cv2.circle(frame, (tx, bar_y), _rs(5) if active else _rs(3), tick_color, -1, cv2.LINE_AA)
            font_scale = (0.42 if active else 0.38) * render_scale
            text_color = (245, 245, 245) if active else (150, 150, 150)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx_centered = max(2, min(width - tw - 2, tx - tw // 2))
            cv2.putText(frame, label, (tx_centered, label_baseline_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

        # Live marker: piecewise-linear position within the current phase's
        # own frame span, mapped onto that segment's boundary-to-boundary
        # x-range (not the label positions, which sit at segment midpoints).
        seg_lo, seg_hi = zone_bounds[phase_idx], zone_bounds[phase_idx + 1]
        seg_frac = 0.0 if seg_hi <= seg_lo else (f_idx - seg_lo) / (seg_hi - seg_lo)
        seg_frac = max(0.0, min(1.0, seg_frac))
        marker_x = int(boundary_xs[phase_idx] + seg_frac * (boundary_xs[phase_idx + 1] - boundary_xs[phase_idx]))
        cv2.circle(frame, (marker_x, bar_y), _rs(7), JOINT_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(frame, (marker_x, bar_y), _rs(5), JOINT_CORE, -1, cv2.LINE_AA)
        # Was plain text drawn straight onto the video with no background —
        # low contrast and hard to read against a bright sky, reading more
        # like a debug label than a broadcast lower-third. Same rounded
        # pill treatment as the other badges for a consistent look.
        (status_w, status_h), _ = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        _draw_panel(frame, (12, 12), (12 + status_w + 24, 12 + status_h + 20),
                    radius=status_h // 2 + 6, shadow_offset=3, fill_alpha=0.5)
        cv2.putText(frame, status_text, (24, 12 + status_h + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (235, 235, 235), 2, cv2.LINE_AA)

        event_labels = [("BFC", "CONTACT", (60, 225, 90)),
                         ("FFC", "CONTACT", (60, 225, 90)),
                         ("BR", "RELEASE", JOINT_CORE)]
        # Pick whichever event is actually CLOSEST to this frame, not just
        # the first one in the list that's within range. BFC/FFC/BR often
        # land within a few frames of each other for a real delivery, so
        # "first match wins" was showing the earlier event's badge
        # (CONTACT) even on the exact BR frame itself, hiding the RELEASE
        # badge and the release-height line/text that only draw when the
        # RELEASE badge is showing.
        active_badge = None
        best_distance = None
        for key, label, color in event_labels:
            ev_frame = events.get(key)
            if ev_frame is None:
                continue
            distance = abs(f_idx - ev_frame)
            if distance <= PHASE_BADGE_WINDOW and (best_distance is None or distance < best_distance):
                active_badge = (label, color, ev_frame)
                best_distance = distance

        if active_badge is not None:
            label, color, ev_frame = active_badge
            badge_val = knee_arr[ev_frame] if ev_frame < len(knee_arr) else np.nan
            is_release = (label == "RELEASE")
            # SHRUNK from 340x150/130 — smaller, less boxy footprint, more
            # in line with the reference sample's restraint, while keeping
            # every number the old, larger version showed.
            box_w = 290
            box_h = 128 if is_release else 108
            # Default to the left (matches the old fixed behavior) unless
            # the bowler is known to be on the left half of frame, in
            # which case the badge moves to the right so it never sits
            # directly on top of him.
            if last_known_bowler_x is not None and last_known_bowler_x < 0.5:
                box_x = width - box_w - 20
            else:
                box_x = 20
            box_y = 40
            _draw_panel(frame, (box_x, box_y), (box_x + box_w, box_y + box_h))
            cv2.putText(frame, label, (box_x + 14, box_y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "KNEE", (box_x + 14, box_y + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
            if not np.isnan(badge_val):
                cv2.putText(frame, f"{badge_val:.0f}deg", (box_x + 85, box_y + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2, cv2.LINE_AA)
            if is_release:
                cv2.putText(frame, "RELEASE HEIGHT", (box_x + 14, box_y + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
                if release_height_pct is not None:
                    cv2.putText(frame, f"{release_height_pct:.0f}%", (box_x + 165, box_y + 104),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, JOINT_CORE, 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "N/A", (box_x + 165, box_y + 104),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
                # ELBOW EXTENSION DISABLED — 2D-derived readings were producing
                # false positives (e.g. 81deg on a visibly legal action).
                # Needs a real accuracy fix before showing this to users again.

                # RELEASE HEIGHT LINE: a vertical line from the bowling-arm
                # wrist straight down to ground level (same ankle landmark
                # used to calculate the ratio above), drawn only on the
                # actual BR frame — no ball tracking involved, this is a
                # visual read-out of the same number shown as text.
                if f_idx == ev_frame and release_line_pts is not None:
                    lx = int(release_line_pts["wrist_x"] * width)
                    ly_wrist = int(release_line_pts["wrist_y"] * height)
                    ly_ground = int(release_line_pts["ground_y"] * height)
                    if (0 < lx < width and 0 < ly_wrist < height
                            and 0 < ly_ground < height):
                        DROP_LINE_COLOR = (60, 60, 255)  # BGR — highly visible red
                        cv2.line(frame, (lx, ly_wrist), (lx, ly_ground),
                                 DROP_LINE_COLOR, 2, cv2.LINE_AA)
                        cv2.circle(frame, (lx, ly_wrist), 5, DROP_LINE_COLOR, -1, cv2.LINE_AA)
                        cv2.circle(frame, (lx, ly_ground), 5, DROP_LINE_COLOR, -1, cv2.LINE_AA)
                        label = "Ball Release Height"
                        (label_w, label_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        # FIX: was centered on the line's vertical midpoint,
                        # which sits right on the torso — verified on real
                        # footage this landed the label directly over the
                        # bowler's body. Place it beside the line at wrist
                        # height instead (right if there's room, else left),
                        # same "avoid the subject" logic already used for
                        # the phase badge above.
                        label_x_right = lx + 14
                        if label_x_right + label_w + 10 < width:
                            label_x = label_x_right
                        else:
                            label_x = max(6, lx - label_w - 14)
                        label_y = min(max(ly_wrist + label_h // 2, label_h + 6), height - 6)
                        _draw_panel(frame, (label_x - 6, label_y - label_h - 6),
                                    (label_x + label_w + 6, label_y + 6),
                                    radius=6, shadow_offset=3)
                        cv2.putText(frame, label, (label_x, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, DROP_LINE_COLOR, 2, cv2.LINE_AA)

        out.write(frame)

        # FREEZE ON RELEASE: hold the exact Ball Release frame (with its
        # skeleton/badge already drawn) for a beat before playback
        # continues, like a broadcast replay highlight. Duplicates the same
        # already-rendered frame at the output's own fps, so it doesn't
        # touch source detection/tracking at all.
        if br_frame is not None and f_idx == br_frame:
            RELEASE_FREEZE_SECONDS = 0.8
            freeze_frame_count = max(1, int(round(output_fps * RELEASE_FREEZE_SECONDS)))
            for _ in range(freeze_frame_count):
                out.write(frame)

        f_idx += 1
    cap.release()
    out.release()


def transcode_to_h264(input_path: str) -> str:
    """
    Transcodes mp4v video to H264 for browser playback using ffmpeg.

    FIX (was): the old code hardcoded a Windows-only path
    (r"C:\\ffmpeg\\bin\\ffmpeg.exe") for os.name == "nt", and just the bare
    "ffmpeg" command otherwise — with no check that either actually exists.
    If ffmpeg wasn't at that exact path (or not on PATH on Linux), the
    subprocess call would fail, the exception was silently swallowed, and
    the function returned the original untranscoded mp4v video with NO
    warning anywhere — which may not play back correctly in a browser.

    Fix: use shutil.which() to actually locate ffmpeg on PATH, on any OS.
    If it's genuinely not installed/found, that's surfaced with a clear
    log line instead of failing silently.
    """
    import shutil

    base, ext = os.path.splitext(input_path)
    web_safe_path = f"{base}_h264{ext}"

    if os.path.exists(web_safe_path):
        try:
            os.remove(web_safe_path)
        except OSError:
            pass

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print(
            "WARNING: ffmpeg not found on PATH. Video will be served in its "
            "original codec, which may not play back correctly in all browsers. "
            "Install ffmpeg and ensure it's on PATH to fix this."
        )
        return input_path

    cmd = [
        ffmpeg_bin, "-y", "-i", input_path,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "medium",
        "-profile:v", "baseline", "-level", "3.0",
        "-an", web_safe_path
    ]

    try:
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo
        )

        if result.returncode == 0 and os.path.exists(web_safe_path):
            return web_safe_path
        else:
            print(
                f"WARNING: ffmpeg transcode failed (exit code {result.returncode}): "
                f"{result.stderr.decode(errors='ignore')[:300]}"
            )
    except Exception as e:
        print(f"WARNING: ffmpeg transcode raised an exception: {e}")

    return input_path


def run_complete_bowling_analysis(video_path: str,
                                   output_dir: str = "output",
                                   bowling_arm_override: str = None,
                                   seed_point: tuple = None,
                                   seed_frame_index: int = 0,
                                   extra_seeds: list = None) -> dict:
    """
    Core orchestration loop.
    Extracts landmarks, detects events, calculates all 5 biomechanical
    metrics, generates annotated video, and returns unified payload.

    seed_point/seed_frame_index: optional coach click identifying the
    bowler in a reference frame, passed straight through to
    extract_video_landmarks — see that function's docstring.

    extra_seeds: optional list of (frame_index, point) pairs — lets a
    coach re-confirm the bowler's identity at additional points later
    in the clip if tracking is lost for a long stretch (e.g. a bystander
    standing between the bowler and camera for several seconds). Each
    seed only has to survive the gap to its nearest neighboring seed
    instead of one seed carrying the whole video — see
    main._walk_from_seed for how zones are split between seeds.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "landmarks.csv")

    # STAGE 1 — LANDMARK EXTRACTION
    extraction = extract_video_landmarks(video_path, csv_path,
                                          seed_point=seed_point,
                                          seed_frame_index=seed_frame_index,
                                          extra_seeds=extra_seeds)

    if extraction["status"] == "error":
        return {
            "status": "failed",
            "stage": "perception",
            "message": extraction["error_message"]
        }

    df = pd.read_csv(csv_path)
    fps = extraction["fps"]

    # STAGE 2 — BOWLING ARM DETECTION + EVENT DETECTION
    if bowling_arm_override in ("left", "right"):
        bowling_arm = bowling_arm_override
    else:
        bowling_arm = detect_bowling_arm(df)
    events = detect_delivery_events(df, fps, bowling_arm=bowling_arm)

    # STAGE 3 — FRAME VALIDATION
    ffc_rows = df[df["frame"] == events["FFC"]]
    if ffc_rows.empty:
        return {
            "status": "failed",
            "stage": "frame_extraction",
            "message": (
                f"FFC frame {events['FFC']} not found in landmark data. "
                f"Video may be too short or landmarks dropped."
            )
        }
    ffc_row = ffc_rows.iloc[0]

    br_rows = df[df["frame"] == events["BR"]]
    if br_rows.empty:
        return {
            "status": "failed",
            "stage": "frame_extraction",
            "message": f"BR frame {events['BR']} not found in landmark data."
        }
    br_row = br_rows.iloc[0]
    lead_side = "left" if bowling_arm == "right" else "right"

    # STAGE 4 — METRIC CALCULATIONS
    knee_analysis = calculate_knee_bracing(ffc_row, lead_side=lead_side)
    knee_at_release = calculate_knee_bracing(br_row, lead_side=lead_side)
    lean_analysis = calculate_trunk_lean(br_row)
    head_stability = calculate_head_stability(df, events["BFC"], events["BR"])
    hip_separation = calculate_hip_shoulder_separation(df, events["FFC"])
    release_height = calculate_release_height_ratio_safe(br_row, bowling_arm=bowling_arm)

    # FFC-to-Release knee angle delta ("yielding knee" check flagged in
    # external biomechanical audit): a static single-frame knee angle at
    # FFC can't show whether the knee then BENDS (yields) before release,
    # which is a real, separate coaching concern from the FFC angle alone.
    knee_delta = None
    knee_delta_status = "unavailable"
    if knee_analysis.get("status") == "success" and knee_at_release.get("status") == "success":
        knee_delta = round(knee_at_release["degrees"] - knee_analysis["degrees"], 1)
        knee_delta_status = "yielding" if knee_delta < -5.0 else ("braced" if knee_delta >= 0 else "minor_yield")

    # STAGE 5 — VIDEO GENERATION
    raw_output_video = os.path.join(output_dir, "annotated_raw.mp4")
    generate_fail_safe_video(video_path, raw_output_video, df, events, bowling_arm=bowling_arm)
    web_safe_video_file = transcode_to_h264(raw_output_video)

    # STAGE 6 — SAFE KEY EXTRACTION
    # FIX: the old `.get(a) or .get(b) or .get(c)` pattern has a falsy-zero
    # bug — if a metric's real value is exactly 0.0 (e.g. a perfectly
    # upright 0.0-degree trunk lean, a genuinely ideal result), Python
    # treats 0.0 as falsy and the `or` chain incorrectly keeps searching,
    # silently turning a great result into None/N/A. Explicit None-checks
    # fix this — verified this is the actual key kinematics.py returns
    # ("degrees" for both trunk_lean and knee_bracing).
    def _first_non_none(d: dict, *keys):
        for k in keys:
            v = d.get(k)
            if v is not None:
                return v
        return None

    trunk_lean_val = _first_non_none(lean_analysis, "trunk_lean_degrees", "degrees", "angle")
    knee_bracing_val = _first_non_none(knee_analysis, "front_knee_angle", "degrees", "angle")

    # ANATOMICAL PLAUSIBILITY GUARD (knee bracing only):
    # A human knee cannot physically be at ~0 degrees mid-delivery — that
    # would mean the joint folded completely in on itself. A near-zero
    # reading here almost always means the hip/knee/ankle landmarks
    # collapsed onto nearly the same point due to tracking failure (e.g.
    # on a very short/low-quality clip), and arccos(~1) returned ~0 as a
    # pure math artifact of that degeneracy, not a real measurement.
    # Confirmed against a real clip: this exact scenario produced "0.0°"
    # after the falsy-zero fix above started correctly passing through
    # real zero values — this guard distinguishes a genuine 0-degree
    # result (which never happens for THIS metric) from degenerate math.
    # NOTE: this threshold (5 degrees) is an engineering choice based on
    # basic human anatomy, not a cited biomechanics constant.
    # IMPORTANT: this guard is intentionally NOT applied to trunk_lean —
    # 0 degrees of trunk lean is a real, genuinely ideal result (a
    # perfectly upright bowler), so the same "near-zero is implausible"
    # logic would be wrong there.
    KNEE_ANGLE_IMPLAUSIBLE_THRESHOLD = 5.0  # degrees
    if knee_bracing_val is not None and knee_bracing_val < KNEE_ANGLE_IMPLAUSIBLE_THRESHOLD:
        knee_bracing_val = None

    # STAGE 7 — RETURN UNIFIED PAYLOAD
    return {
        "status": "success",
        "video_metadata": {
            "source_file": os.path.basename(video_path),
            "fps": fps,
            "total_frames": len(df)
        },
        "time_indices": {
            "back_foot_contact_frame": events["BFC"],
            "front_foot_contact_frame": events["FFC"],
            "ball_release_frame": events["BR"],
            "ball_release_confidence": events.get("BR_confidence", "high"),
            "ball_release_plausible_fraction": events.get("BR_plausible_fraction", 1.0),
        },
        "biomechanical_metrics": {
            "trunk_lean": {
                "degrees": trunk_lean_val,
                "tier": (lean_analysis.get("classification") or
                         lean_analysis.get("tier") or "Unknown"),
                "status": lean_analysis.get("status", "error"),
                "critique": lean_analysis.get("critique", "N/A")
            },
            "front_knee_bracing": {
                "degrees": knee_bracing_val,
                "tier": (knee_analysis.get("classification") or
                         knee_analysis.get("tier") or "Unknown"),
                "status": knee_analysis.get("status", "error"),
                "critique": knee_analysis.get("critique", "N/A"),
                "degrees_at_release": knee_at_release.get("degrees"),
                "yield_delta_degrees": knee_delta,
                "yield_status": knee_delta_status
            },
            "hip_shoulder_separation": hip_separation,
            "bowling_arm_detected": bowling_arm,
            "release_height": {
                "ratio": release_height.get("ratio"),
                "classification": (release_height.get("classification") or
                                    release_height.get("tier") or "Unknown"),
                "status": release_height.get("status", "error"),
                "debug_raw": release_height.get("debug_raw")
            },
            "head_stability": {
                "value": (head_stability.get("deviation_index") or
                          head_stability.get("value")),
                "classification": (head_stability.get("tier") or
                                    head_stability.get("classification") or
                                    "Unknown"),
                "status": head_stability.get("status", "error")
            }
        },
        "annotated_video_output": web_safe_video_file.replace("\\", "/")
    }