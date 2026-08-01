"""
speed_estimation.py

Two distinct outputs, kept separate on purpose:

1. Phase durations (BFC->FFC->BR), in seconds. These are 100% real —
   derived only from frame indices and fps, no assumptions required.
   Always computed, always available.

2. Estimated release-arm speed (km/h). This requires real-world scale
   (see calibration.py). If no calibration is supplied, this function
   returns status="not_calibrated" and NO numeric value — it will not
   invent a stride-length constant to fake a number.

IMPORTANT HONESTY NOTE (surfaced to the UI, not hidden):
   What's measured here is the speed of the bowling wrist/hand landmark
   in the frames around ball release. This is a strong correlate of ball
   speed but is NOT the same measurement a speed gun/radar takes — a
   radar tracks the ball itself, which leaves the hand faster than the
   hand moves due to wrist snap and finger action. Label this as "Release
   Arm Speed (estimate)", not "Ball Speed", anywhere it's displayed.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

import monitoring


def compute_phase_durations(events: dict, fps: float) -> dict:
    """
    events: {"BFC": int, "FFC": int, "BR": int} frame indices
    fps: frames per second of the source video

    Returns seconds for each phase. Always real, never estimated.
    """
    if not fps or fps <= 0:
        raise ValueError("fps must be a positive number to compute timing.")

    bfc, ffc, br = events["BFC"], events["FFC"], events["BR"]

    return {
        "bfc_to_ffc_seconds": round((ffc - bfc) / fps, 4),
        "ffc_to_br_seconds": round((br - ffc) / fps, 4),
        "bfc_to_br_seconds": round((br - bfc) / fps, 4),
        "fps": fps,
    }


def _wrist_pixel_series(df: pd.DataFrame, wrist: str, frame_width: int, frame_height: int):
    x = df[f"{wrist}_x"].interpolate(method="linear").bfill().ffill().values * frame_width
    y = df[f"{wrist}_y"].interpolate(method="linear").bfill().ffill().values * frame_height
    return x, y


def _select_bowling_arm(df: pd.DataFrame, br_idx: int, fps: float,
                         frame_width: int, frame_height: int) -> str:
    """
    Picks whichever wrist has greater speed at the BR frame — that's the
    bowling arm. Real measurement, not an assumption of handedness.
    """
    window = max(2, int(fps * 0.05))  # ~50ms either side
    lo = max(0, br_idx - window)
    hi = min(len(df) - 1, br_idx + window)

    speeds = {}
    for wrist in ("RIGHT_WRIST", "LEFT_WRIST"):
        if f"{wrist}_x" not in df.columns:
            continue
        x, y = _wrist_pixel_series(df, wrist, frame_width, frame_height)
        if hi <= lo:
            continue
        dx = x[hi] - x[lo]
        dy = y[hi] - y[lo]
        dt = (hi - lo) / fps
        speeds[wrist] = math.hypot(dx, dy) / dt if dt > 0 else 0.0

    if not speeds:
        raise ValueError("No wrist landmark columns found in landmark data.")

    return max(speeds, key=speeds.get)


def _extract_raw_wrist_window(video_path: str, wrist_name: str, br_idx: int,
                               fps: float, window_frames: int) -> dict:
    """
    Re-extracts RAW (unsmoothed) pixel positions for one wrist landmark,
    directly from the source video, for a window around br_idx.

    Why this exists: the saved landmarks CSV has already been through
    Hampel-filter outlier rejection AND a 5-frame rolling-mean smoothing
    pass — exactly right for a stable-looking skeleton and reliable event
    timing, but verified on real footage that it also DILUTES the true
    peak instantaneous velocity of a fast, brief motion like a bowling
    release swing (a smoothed 5-frame average of a sharp spike is,
    definitionally, much lower than the spike itself). Re-extracting raw
    positions just for this small window avoids that dilution.

    Processes every frame from 0 up to the window end (not just the
    window itself) to preserve VIDEO mode's real temporal continuity —
    only the window's positions are kept, but detection needs the earlier
    frames to have "warmed up" properly.

    Returns {frame_index: (x_px, y_px, visibility)}.
    """
    import os
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = os.path.join("models", "pose_landmarker_full.task")
    landmark_index = 16 if wrist_name == "RIGHT_WRIST" else 15

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.4,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ms_per_frame = 1000.0 / fps
    window_end = br_idx + window_frames

    positions = {}
    idx = 0
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok or idx > window_end:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts = int(round(idx * ms_per_frame))
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        res = landmarker.detect_for_video(img, ts)
        if idx >= br_idx - window_frames and res.pose_landmarks:
            lm = res.pose_landmarks[0][landmark_index]
            positions[idx] = (lm.x * frame_w, lm.y * frame_h, lm.visibility)
        idx += 1
    cap.release()
    landmarker.close()
    return positions


def _fit_slope_and_r_squared(t: np.ndarray, values: np.ndarray) -> tuple:
    """
    Least-squares slope of values vs t, and the fit's R^2 (1.0 = every
    point sits exactly on the line, 0.0 = no better than a flat mean).
    A constant (zero-variance) input is treated as a perfect fit with
    zero slope — genuinely stationary, not evidence of bad tracking.
    """
    if np.allclose(values, values[0]):
        return 0.0, 1.0
    slope, intercept = np.polyfit(t, values, 1)
    predicted = slope * t + intercept
    ss_res = np.sum((values - predicted) ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return float(slope), float(r_squared)


def _corroborated_peak_speed_px_s(positions: dict, fps: float,
                                   fit_radius: int = 2, min_r_squared: float = 0.9) -> Optional[float]:
    """
    Peak velocity (px/s) from raw per-frame positions, using a locally
    fitted straight line (least squares) instead of a single frame-to-
    frame hop — a real, structural blind spot was found in the previous
    "corroborated hop" approach this replaces, from real reported cases
    where a correctly-calibrated setup reported BOTH physically
    impossible speeds (~600km/h; fastest recorded deliveries are
    ~161km/h) on some clips AND suspiciously slow ones (~5-20km/h,
    implausible for genuine bowling effort) on others.

    THE BLIND SPOT: the old method required a hop's speed to be matched
    by a comparably-fast NEIGHBORING hop before trusting it — built to
    catch a jump-then-snap-back tracking glitch, which does produce two
    adjacent, similarly-large hops. But the genuine peak wrist speed AT
    ball release is, by definition, a brief, sharp spike: release is the
    one instant the arm moves fastest, meaningfully faster than the
    frames just before/after it — that's what makes it the release
    point. That exact signature (one frame far exceeding its immediate
    neighbors) is what the old check was built to reject as noise. So on
    a real, fast, clean swing, the TRUE peak often failed its own
    "comparable neighbor" test and got discarded — while an unrelated,
    genuinely noisy corner of the window that happened to have two
    similarly-sized adjacent hops could still pass. This is consistent
    with results swinging wildly between too-fast (a glitch that DID
    corroborate itself) and too-slow (the real peak rejected, a weaker
    reading accepted instead) — not two separate bugs, one shared cause.

    THE FIX: for each candidate center frame, fit a straight line (least
    squares) to x(t) and y(t) separately over a small window of
    `2*fit_radius+1` consecutive frames centered on it, and use the
    FITTED slope as that frame's velocity — a standard technique
    (equivalent to a simplified Savitzky-Golay derivative) for
    estimating the derivative of a noisy signal. Real physical motion is
    locally smooth even during rapid acceleration, so a short window's
    true trend is well approximated by a line; a single erratic frame
    corrupts that line's RESIDUAL rather than silently passing through
    unquestioned like a raw two-point difference would. R^2 (fit
    quality, the weaker of the x/y axes) gates which candidates are
    trusted: genuine coherent motion fits a line well (high R^2); a
    window containing a tracking glitch, or pure noise, does not (low
    R^2). The reported peak is the highest velocity among windows
    meeting min_r_squared — if none do, returns None (an honest "no
    reliable window" signal), never the least-trustworthy reading.

    Needs at least 2*fit_radius+1 frames of raw positions to evaluate
    even one candidate window; the caller should pass a raw-position
    window comfortably wider than that (e.g. the ~80ms-either-side
    window already used around release) so multiple candidate centers
    exist and a glitch in one doesn't have to poison the only option.
    """
    frames = sorted(positions.keys())
    span = 2 * fit_radius + 1
    if len(frames) < span:
        return None

    frame_to_xy = {f: (positions[f][0], positions[f][1]) for f in frames}
    best_speed = None

    for center in frames:
        window_frames = [center + d for d in range(-fit_radius, fit_radius + 1)]
        if any(f not in frame_to_xy for f in window_frames):
            continue

        t = np.array(window_frames, dtype=float) / fps
        xs = np.array([frame_to_xy[f][0] for f in window_frames])
        ys = np.array([frame_to_xy[f][1] for f in window_frames])

        vx, r2_x = _fit_slope_and_r_squared(t, xs)
        vy, r2_y = _fit_slope_and_r_squared(t, ys)
        # The weaker of the two axes' fits — a good fit on x alone
        # doesn't matter if y is scattered noise (or vice versa).
        r2 = min(r2_x, r2_y)
        if r2 < min_r_squared:
            continue

        speed = math.hypot(vx, vy)
        if best_speed is None or speed > best_speed:
            best_speed = speed

    return best_speed


def compute_release_arm_speed(df: pd.DataFrame, events: dict, fps: float,
                               frame_width: int, frame_height: int,
                               meters_per_pixel: Optional[float],
                               video_path: Optional[str] = None,
                               bowling_arm_override: Optional[str] = None) -> dict:
    """
    Returns:
      {"status": "not_calibrated"}   -- if meters_per_pixel is None
      {"status": "success", "kmh": ..., "mps": ..., "bowling_arm": ...}
      {"status": "error", "message": ...}

    IMPORTANT: uses PEAK frame-to-frame instantaneous velocity near release,
    not net displacement across the whole window. A bowling arm swings
    through a curved arc — over a wide time window, the straight-line
    distance between the start and end point (the "chord") is much
    shorter than the actual path length traveled, which would silently
    underestimate speed by several-fold. Peak instantaneous velocity
    between consecutive frames is the physically correct quantity for
    "how fast was the hand moving at release."

    video_path: when given, re-extracts RAW (unsmoothed) positions for
    the velocity window directly from the source video instead of using
    the already-smoothed df — see _extract_raw_wrist_window for why this
    matters. Falls back to the smoothed df if not given or if raw
    re-extraction fails for any reason, so this stays backward compatible.

    bowling_arm_override: 'right' or 'left' — the arm already resolved by
    orchestrator.py (manual selection or its own auto-detect). When given,
    this is used directly instead of re-detecting from wrist speed, so the
    speed number can never disagree with the arm the rest of the report
    (event timing, release height, video overlay) is built on.
    """
    if meters_per_pixel is None:
        return {
            "status": "not_calibrated",
            "message": (
                "Camera not calibrated for this setup — run calibration once "
                "to enable speed estimates."
            ),
        }

    try:
        br_idx = int(events["BR"])
        if br_idx <= 0 or br_idx >= len(df):
            return {"status": "error", "message": "BR frame out of range."}

        if bowling_arm_override in ("left", "right"):
            bowling_arm = "RIGHT_WRIST" if bowling_arm_override == "right" else "LEFT_WRIST"
            if f"{bowling_arm}_x" not in df.columns:
                return {"status": "error", "message": f"No {bowling_arm} landmark data available."}
        else:
            bowling_arm = _select_bowling_arm(df, br_idx, fps, frame_width, frame_height)
        window = max(3, int(fps * 0.08))  # ~80ms either side of release

        peak_px_per_s = None
        used_raw = False
        if video_path:
            try:
                raw_positions = _extract_raw_wrist_window(video_path, bowling_arm, br_idx, fps, window)
                peak_px_per_s = _corroborated_peak_speed_px_s(raw_positions, fps)
                used_raw = peak_px_per_s is not None
            except Exception:
                peak_px_per_s = None

        if peak_px_per_s is None:
            # BUG FIX: this fallback used to take a raw, unprotected max()
            # over every frame-to-frame speed in the window — exactly the
            # same "trust the single worst reading" mistake fixed in
            # _corroborated_peak_speed_px_s above, just on smoothed data
            # instead of raw. Smoothing alone doesn't reliably prevent a
            # spurious spike (a 5-frame rolling mean still passes through
            # a large chunk of a genuine one-frame tracking glitch), so
            # this path gets the identical corroboration check.
            x, y = _wrist_pixel_series(df, bowling_arm, frame_width, frame_height)
            lo = max(0, br_idx - window)
            hi = min(len(df) - 1, br_idx + window)
            if hi - lo < 2:
                return {"status": "error", "message": "Insufficient frames around BR for a velocity estimate."}
            positions = {i: (x[i], y[i], None) for i in range(lo, hi + 1)}
            peak_px_per_s = _corroborated_peak_speed_px_s(positions, fps)

        if peak_px_per_s is None:
            return {
                "status": "error",
                "message": (
                    "Tracking around release was too unstable for a reliable "
                    "speed estimate — no consistent frame-to-frame motion was "
                    "found (common with heavy motion blur right at release). "
                    "Double-check the confirmed release point, or re-shoot at "
                    "a higher shutter speed / frame rate if possible."
                ),
            }

        mps = peak_px_per_s * meters_per_pixel
        kmh = mps * 3.6

        if kmh <= 0 or kmh > 200:
            # Physically implausible for a human bowling delivery — surface
            # as an error rather than a silently wrong number. 200km/h is a
            # generous upper bound (fastest recorded deliveries are ~161km/h).
            return {
                "status": "error",
                "message": (
                    f"Computed speed ({kmh:.1f} km/h) is outside plausible "
                    f"bounds — check calibration accuracy."
                ),
            }

        return {
            "status": "success",
            "kmh": round(kmh, 1),
            "mps": round(mps, 2),
            "bowling_arm": bowling_arm,
            "used_raw_reextraction": used_raw,
        }
    except Exception as e:
        monitoring.capture(e)
        return {"status": "error", "message": str(e)}


def compute_release_height_absolute(release_height_debug: Optional[dict], frame_height: int,
                                     meters_per_pixel: Optional[float]) -> dict:
    """
    Absolute vertical release height (bowling wrist above ground/ankle
    level) in real-world units — a companion to the always-available
    body-proportion ratio (release height as a % of the bowler's own
    body height), computed from the exact same landmark data at the
    exact same release-onset frame, just converted through a real-world
    scale (e.g. stump height, 71.12cm / 28in, via calibration.py)
    instead of expressed relative to body size.

    release_height_debug: the "debug_raw" dict already returned by
    calculate_release_height_ratio_safe (y_wrist, y_ankle, both
    normalized 0-1) — reused rather than re-deriving the landmarks, so
    this can never disagree with the ratio on WHICH points it measured.

    Returns {"status": "not_calibrated"} if no calibration is set —
    never invents a scale. Never returns a value further than 2.5m off
    the ground, which is beyond any human's standing reach.
    """
    if meters_per_pixel is None:
        return {
            "status": "not_calibrated",
            "message": (
                "Camera not calibrated for this setup — calibrate using a "
                "known real-world distance (e.g. stump height, 71.12cm) to "
                "enable this."
            ),
        }
    if not release_height_debug or "y_wrist" not in release_height_debug or "y_ankle" not in release_height_debug:
        return {"status": "error", "message": "Release landmark data unavailable."}

    try:
        vertical_norm = abs(float(release_height_debug["y_ankle"]) - float(release_height_debug["y_wrist"]))
        vertical_m = vertical_norm * frame_height * meters_per_pixel
        vertical_cm = vertical_m * 100

        if vertical_cm <= 0 or vertical_cm > 250:
            return {
                "status": "error",
                "message": (
                    f"Computed height ({vertical_cm:.0f}cm) is outside plausible "
                    f"bounds — check calibration accuracy."
                ),
            }

        return {"status": "success", "cm": round(vertical_cm, 1), "m": round(vertical_m, 3)}
    except Exception as e:
        monitoring.capture(e)
        return {"status": "error", "message": str(e)}
