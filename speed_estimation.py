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


def _corroborated_peak_speed_px_s(positions: dict, fps: float) -> Optional[float]:
    """
    Peak frame-to-frame speed (px/s) from raw positions, rejecting
    tracking-glitch spikes that don't correspond to real, directed
    motion. Verified on real footage this matters: raw (unsmoothed) wrist
    tracking during a fast, motion-blurred swing shows both genuine
    sustained bursts (several consecutive frames moving steadily in one
    direction) AND isolated one-frame glitches where the tracked point
    jumps away then snaps straight back on the very next frame.

    TWO BUGS FIXED HERE, both found from one real reported case (a
    correctly-calibrated setup still reporting ~600km/h — physically
    impossible; fastest recorded deliveries are ~161km/h):

    1. The corroboration test itself was wrong. A jump-then-snap-back
       glitch produces TWO large, adjacent frame-to-frame speeds of
       similar magnitude — the "out" hop and the "back" hop. The old
       check only asked "is a neighboring hop's speed comparably large?"
       — and two hops of a single glitch are, by construction, always
       comparable in magnitude to EACH OTHER. So the exact glitch shape
       this function's docstring describes was never actually caught.
       Fixed with a physically-grounded test: pair each hop with an
       adjacent one and compare NET displacement (straight-line, start
       to end of the pair) against PATH length (sum of the two hops'
       individual magnitudes). Real directed motion covers roughly as
       much net ground as its path length suggests; an out-and-back
       glitch's net displacement collapses toward zero while its path
       length stays large — THAT mismatch is what actually identifies a
       "snapped straight back" glitch. Verified directly: constructing
       exactly this glitch shape, the old check kept it, this one
       rejects it and correctly recovers the genuine (much slower)
       background motion instead.

    2. When NOTHING in the window is corroborated (a window that's pure
       noise from end to end — exactly the case this function exists to
       catch), the old code fell back to `max(speeds.values())`: the
       single WORST, least-trustworthy reading in the entire window,
       with zero filtering. That's backwards — total corroboration
       failure is the strongest possible signal the data is unreliable,
       so this returns None (a real "no reliable reading here" signal
       the caller acts on) instead of silently handing back the exact
       kind of glitch this check was built to reject.
    """
    frames = sorted(positions.keys())
    hops = {}
    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i + 1]
        if f1 - f0 != 1:
            continue
        x0, y0, _ = positions[f0]
        x1, y1, _ = positions[f1]
        hops[f0] = (x1 - x0, y1 - y0)

    if not hops:
        return None

    speeds = {k: math.hypot(*v) * fps for k, v in hops.items()}
    CORROBORATION_FRACTION = 0.5

    def _is_directed_pair(k_a: int, k_b: int) -> bool:
        """k_a, k_b are adjacent hop-start frames with k_b == k_a + 1,
        together covering a contiguous 3-frame span."""
        dxa, dya = hops[k_a]
        dxb, dyb = hops[k_b]
        path = math.hypot(dxa, dya) + math.hypot(dxb, dyb)
        if path == 0:
            return True
        net = math.hypot(dxa + dxb, dya + dyb)
        return net >= CORROBORATION_FRACTION * path

    corroborated = []
    for k in hops:
        neighbors = [k2 for k2 in (k - 1, k + 1) if k2 in hops]
        is_corroborated = any(
            speeds[k2] >= CORROBORATION_FRACTION * speeds[k] and _is_directed_pair(*sorted((k, k2)))
            for k2 in neighbors
        )
        if is_corroborated:
            corroborated.append(speeds[k])

    return max(corroborated) if corroborated else None


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
