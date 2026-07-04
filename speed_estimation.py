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


def compute_release_arm_speed(df: pd.DataFrame, events: dict, fps: float,
                               frame_width: int, frame_height: int,
                               meters_per_pixel: Optional[float]) -> dict:
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

        bowling_arm = _select_bowling_arm(df, br_idx, fps, frame_width, frame_height)
        x, y = _wrist_pixel_series(df, bowling_arm, frame_width, frame_height)

        window = max(3, int(fps * 0.08))  # ~80ms either side of release
        lo = max(0, br_idx - window)
        hi = min(len(df) - 1, br_idx + window)
        if hi - lo < 2:
            return {"status": "error", "message": "Insufficient frames around BR for a velocity estimate."}

        dt_frame = 1.0 / fps
        frame_speeds_px_s = []
        for i in range(lo, hi):
            d = math.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
            frame_speeds_px_s.append(d / dt_frame)

        peak_px_per_s = max(frame_speeds_px_s)
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
            "window_seconds": round((hi - lo) * dt_frame, 4),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
