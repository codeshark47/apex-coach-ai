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


def detect_delivery_events(df: pd.DataFrame, fps: int) -> dict:
    """Robust physical milestone detection using velocity windows."""
    total_frames = len(df)

    if total_frames < 10:
        return {
            "BFC": 0,
            "FFC": int(total_frames * 0.4),
            "BR": int(total_frames * 0.8)
        }

    r_wrist_y = df["RIGHT_WRIST_y"].interpolate(method="linear").bfill().ffill().values
    br_idx = int(np.argmin(r_wrist_y))

    if br_idx <= 5 or br_idx >= total_frames - 2:
        br_idx = int(total_frames * 0.75)

    l_ankle_y = df["LEFT_ANKLE_y"].interpolate(method="linear").bfill().ffill().values
    lookback_start = max(0, br_idx - int(fps * 0.6))
    ffc_window = l_ankle_y[lookback_start:br_idx]

    if len(ffc_window) > 0:
        ffc_idx = lookback_start + int(np.argmax(ffc_window))
    else:
        ffc_idx = max(0, br_idx - int(fps * 0.3))

    r_ankle_y = df["RIGHT_ANKLE_y"].interpolate(method="linear").bfill().ffill().values
    bfc_lookback = max(0, ffc_idx - int(fps * 0.5))
    bfc_window = r_ankle_y[bfc_lookback:ffc_idx]

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


def calculate_release_height_ratio_safe(br_row: pd.Series) -> dict:
    """
    Calculates release height leverage ratio with expanded real-world tolerances.
    Prevents N/A dropouts on high-arm actions or varied camera distances.
    """
    try:
        y_wrist = br_row.get("RIGHT_WRIST_y")
        y_head = br_row.get("NOSE_y")
        y_ankle = br_row.get("LEFT_ANKLE_y") or br_row.get("RIGHT_ANKLE_y")

        if any(v is None or pd.isna(v) for v in [y_wrist, y_head, y_ankle]):
            return {
                "ratio": None,
                "classification": "Landmark missing",
                "status": "error",
                "error_message": "One or more landmarks missing on BR frame."
            }

        body_height = abs(float(y_ankle) - float(y_head))
        if body_height < 0.05:
            return {
                "ratio": None,
                "classification": "Body height too small",
                "status": "error",
                "error_message": "Body height span too small — bowler may be out of frame."
            }

        ratio = round(abs(float(y_ankle) - float(y_wrist)) / body_height, 4)

        if ratio > 1.30 or ratio < 0.30:
            return {
                "ratio": None,
                "classification": "Measurement error — verify camera angle",
                "status": "error",
                "error_message": f"Ratio {ratio} outside physical bounds."
            }

        if ratio >= 0.85:
            classification = "High-Release Leverage"
        elif ratio >= 0.75:
            classification = "Standard Mid-Arm Release"
        else:
            classification = "Low-Sling Action"

        return {"ratio": ratio, "classification": classification, "status": "success"}

    except Exception as e:
        return {
            "ratio": None,
            "classification": "Calculation error",
            "status": "error",
            "error_message": str(e)
        }


def generate_fail_safe_video(video_path: str, output_path: str,
                              df: pd.DataFrame, events: dict,
                              slow_motion_factor: float = 4.0):
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
    df[landmark_cols] = df[landmark_cols].interpolate(
        method="linear", limit=3, limit_direction="both"
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

    ANGLE_MIN, ANGLE_MAX = 0.0, 180.0
    ELITE_MIN = 165.0
    PANEL_H = max(140, int(height * 0.30))
    PANEL_TOP = height - PANEL_H
    MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 65, 20, 34, 22

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
    for g in range(0, 181, 30):
        gy = y_to_px(g)
        cv2.line(chart_base, (MARGIN_L, gy), (width - MARGIN_R, gy), (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(chart_base, f"{g}", (8, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (150, 150, 150), 1, cv2.LINE_AA)
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
                        cv2.line(frame, (xA, yA), (xB, yB), glow, 5, cv2.LINE_AA)
                        cv2.line(frame, (xA, yA), (xB, yB), core, 2, cv2.LINE_AA)
                except Exception:
                    continue
            for node in joint_nodes:
                try:
                    if pd.isna(row[f"{node}_x"]):
                        continue
                    nx = int(float(row[f"{node}_x"]) * width)
                    ny = int(float(row[f"{node}_y"]) * height)
                    if 0 < nx < width and 0 < ny < height:
                        cv2.circle(frame, (nx, ny), 4, JOINT_GLOW, -1, cv2.LINE_AA)
                        cv2.circle(frame, (nx, ny), 2, JOINT_CORE, -1, cv2.LINE_AA)
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
        active_badge = None
        for key, label, color in event_labels:
            ev_frame = events.get(key)
            if ev_frame is not None and abs(f_idx - ev_frame) <= PHASE_BADGE_WINDOW:
                active_badge = (label, color, ev_frame)
                break

        if active_badge is not None:
            label, color, ev_frame = active_badge
            badge_val = knee_arr[ev_frame] if ev_frame < len(knee_arr) else np.nan
            box_w, box_h = 300, 130
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
                                   output_dir: str = "output") -> dict:
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

    # STAGE 2 — EVENT DETECTION
    events = detect_delivery_events(df, fps)

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

    # STAGE 4 — METRIC CALCULATIONS
    knee_analysis = calculate_knee_bracing(ffc_row)
    lean_analysis = calculate_trunk_lean(br_row)
    head_stability = calculate_head_stability(df, events["BFC"], events["BR"])
    hip_separation = calculate_hip_shoulder_separation(df, events["FFC"])
    release_height = calculate_release_height_ratio_safe(br_row)

    # STAGE 5 — VIDEO GENERATION
    raw_output_video = os.path.join(output_dir, "annotated_raw.mp4")
    generate_fail_safe_video(video_path, raw_output_video, df, events)
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
                "critique": knee_analysis.get("critique", "N/A")
            },
            "hip_shoulder_separation": hip_separation,
            "release_height": {
                "ratio": release_height.get("ratio"),
                "classification": (release_height.get("classification") or
                                    release_height.get("tier") or "Unknown"),
                "status": release_height.get("status", "error")
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