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
    br_search_window = max(2, int(round(fps * 0.4)))
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
    real_mask_slice = wrist_had_real[ffc_idx:br_search_end]
    if len(br_slice) > 0 and real_mask_slice.any():
        candidate_slice = np.where(real_mask_slice, br_slice, np.inf)
        br_idx = ffc_idx + int(np.argmin(candidate_slice))
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
        "BR": int(min(br_idx, total_frames - 1))
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

    LEG_LINE_GLOW   = (20, 100, 20)
    LEG_LINE_CORE   = (60, 225, 90)
    UPPER_LINE_GLOW = (70, 70, 70)
    UPPER_LINE_CORE = (235, 235, 235)
    JOINT_GLOW = (10, 140, 210)
    JOINT_CORE = (0, 210, 255)

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
    PANEL_H = max(190, int(height * 0.38))
    PANEL_TOP = height - PANEL_H
    MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 65, 20, 34, 26

    def x_to_px(fidx):
        span = max(total_frames - 1, 1)
        return MARGIN_L + int((fidx / span) * (width - MARGIN_L - MARGIN_R))

    def y_to_px(angle):
        clipped = max(ANGLE_MIN, min(ANGLE_MAX, angle))
        frac = (clipped - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)
        usable = PANEL_H - MARGIN_T - MARGIN_B
        return MARGIN_T + int((1 - frac) * usable)

    chart_base = np.full((PANEL_H, width, 3), (18, 18, 18), dtype=np.uint8)
    elite_y = y_to_px(ELITE_MIN)
    cv2.rectangle(chart_base, (MARGIN_L, MARGIN_T), (width - MARGIN_R, elite_y),
                  (30, 70, 30), -1)
    for g in range(0, 181, 45):
        gy = y_to_px(g)
        cv2.line(chart_base, (MARGIN_L, gy), (width - MARGIN_R, gy), (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(chart_base, f"{g}", (8, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (170, 170, 170), 1, cv2.LINE_AA)
    prev_pt = None
    for fidx in range(total_frames):
        val = knee_arr[fidx] if fidx < len(knee_arr) else np.nan
        if np.isnan(val):
            prev_pt = None
            continue
        pt = (x_to_px(fidx), y_to_px(val))
        if prev_pt is not None:
            cv2.line(chart_base, prev_pt, pt, (0, 210, 255), 2, cv2.LINE_AA)
        prev_pt = pt
    cv2.putText(chart_base, "LEAD KNEE ANGLE", (MARGIN_L, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA)

    PHASE_BADGE_WINDOW = max(3, int(round(source_fps * 0.5)))

    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        f_match = df[df["frame"] == f_idx]
        if not f_match.empty:
            row = f_match.iloc[0]
            for partA, partB in connections:
                try:
                    if pd.isna(row[f"{partA}_x"]) or pd.isna(row[f"{partB}_x"]):
                        continue
                    xA = int(float(row[f"{partA}_x"]) * width)
                    yA = int(float(row[f"{partA}_y"]) * height)
                    xB = int(float(row[f"{partB}_x"]) * width)
                    yB = int(float(row[f"{partB}_y"]) * height)
                    is_leg = (partA, partB) in LEG_CONNECTIONS or (partB, partA) in LEG_CONNECTIONS
                    glow = LEG_LINE_GLOW if is_leg else UPPER_LINE_GLOW
                    core = LEG_LINE_CORE if is_leg else UPPER_LINE_CORE
                    if (0 < xA < width and 0 < yA < height and
                            0 < xB < width and 0 < yB < height):
                        cv2.line(frame, (xA, yA), (xB, yB), glow, 2, cv2.LINE_AA)
                        cv2.line(frame, (xA, yA), (xB, yB), core, 1, cv2.LINE_AA)
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
                        cv2.line(frame, (sx1, sy1), (sx2, sy2), UPPER_LINE_GLOW, 2, cv2.LINE_AA)
                        cv2.line(frame, (sx1, sy1), (sx2, sy2), UPPER_LINE_CORE, 1, cv2.LINE_AA)
                        cv2.circle(frame, (sx1, sy1), 3, JOINT_GLOW, -1, cv2.LINE_AA)
                        cv2.circle(frame, (sx1, sy1), 1, JOINT_CORE, -1, cv2.LINE_AA)
            except Exception:
                pass

            for node in joint_nodes:
                try:
                    if pd.isna(row[f"{node}_x"]):
                        continue
                    nx = int(float(row[f"{node}_x"]) * width)
                    ny = int(float(row[f"{node}_y"]) * height)
                    if 0 < nx < width and 0 < ny < height:
                        cv2.circle(frame, (nx, ny), 3, JOINT_GLOW, -1, cv2.LINE_AA)
                        cv2.circle(frame, (nx, ny), 1, JOINT_CORE, -1, cv2.LINE_AA)
                except Exception:
                    continue

        panel_region = frame[PANEL_TOP:PANEL_TOP + PANEL_H, 0:width]
        blended = cv2.addWeighted(panel_region, 0.35, chart_base, 0.65, 0)
        frame[PANEL_TOP:PANEL_TOP + PANEL_H, 0:width] = blended

        cur_val = knee_arr[f_idx] if f_idx < len(knee_arr) else np.nan
        if not np.isnan(cur_val):
            px, py_rel = x_to_px(f_idx), y_to_px(cur_val)
            py = PANEL_TOP + py_rel
            cv2.line(frame, (px, PANEL_TOP + MARGIN_T), (px, PANEL_TOP + PANEL_H - MARGIN_B),
                     (0, 210, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 6, (0, 140, 210), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 3, (0, 230, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, f"{cur_val:.0f}deg", (width - 150, PANEL_TOP + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, cv2.LINE_AA)

        status_text = "RUN-UP"
        if f_idx >= events["BR"]:
            status_text = "FOLLOW-THROUGH"
        elif f_idx >= events["FFC"]:
            status_text = "BALL RELEASE"
        elif f_idx >= events["BFC"]:
            status_text = "DELIVERY STRIDE"
        cv2.putText(frame, status_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (200, 200, 200), 2, cv2.LINE_AA)

        event_labels = [("BFC", "CONTACT", (60, 225, 90)),
                         ("FFC", "CONTACT", (60, 225, 90)),
                         ("BR", "RELEASE", (0, 210, 255))]
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
            box_w = 340
            box_h = 150 if is_release else 130
            box_x, box_y = 20, 50
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h),
                          (15, 15, 15), -1)
            frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
            cv2.putText(frame, label, (box_x + 16, box_y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
            cv2.putText(frame, "KNEE", (box_x + 16, box_y + 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 2, cv2.LINE_AA)
            if not np.isnan(badge_val):
                cv2.putText(frame, f"{badge_val:.0f}deg", (box_x + 100, box_y + 88),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
            if is_release:
                cv2.putText(frame, "RELEASE HEIGHT", (box_x + 16, box_y + 118),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
                if release_height_pct is not None:
                    cv2.putText(frame, f"{release_height_pct:.0f}%", (box_x + 190, box_y + 122),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "N/A", (box_x + 190, box_y + 122),
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
                        label_x = min(max(lx + 10, 0), width - label_w - 10)
                        label_y = min(max(ly_wrist + (ly_ground - ly_wrist) // 2, label_h + 6), height - 6)
                        cv2.rectangle(frame, (label_x - 6, label_y - label_h - 6),
                                      (label_x + label_w + 6, label_y + 6), (15, 15, 15), -1)
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
                                   bowling_arm_override: str = None) -> dict:
    """
    Core orchestration loop.
    Extracts landmarks, detects events, calculates all 5 biomechanical
    metrics, generates annotated video, and returns unified payload.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "landmarks.csv")

    # STAGE 1 — LANDMARK EXTRACTION
    extraction = extract_video_landmarks(video_path, csv_path)

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
            "ball_release_frame": events["BR"]
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