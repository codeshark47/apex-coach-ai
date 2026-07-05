"""
camera_angle_detection.py

Detects whether footage was shot side-on vs. front-on/rear-on, using real
projection geometry rather than asking the coach to self-report (which they
can forget, or get wrong).

THE GEOMETRY (not fabricated, this is basic projection):
  - Side-on: the camera looks ACROSS the bowler's shoulder line, so the
    left and right shoulders sit almost on top of each other in the 2D
    image (small horizontal separation).
  - Front-on or rear-on: the camera looks ALONG the bowler's direction of
    travel, so the full shoulder width is visible face-on (large
    horizontal separation).

This ratio (shoulder pixel width / body pixel height) is genuinely
computable and meaningfully different between these cases. What it CANNOT
do is distinguish front-on from rear-on — both look nearly identical in
2D joint positions, since the actual difference is facial orientation, and
the landmark data has no "which way is the face pointing" signal (no
per-landmark visibility/confidence is captured by main.py's extraction).
That distinction is left as an honest open question for the coach to
answer, rather than guessed.

The specific ratio threshold below (0.12) is an engineering choice based
on typical human shoulder-width-to-height proportions (~0.20-0.28 when
fully visible), not a cited biomechanics constant — documented as such so
it isn't mistaken for validated science.
"""

import math
from dataclasses import dataclass

import pandas as pd

SIDE_ON_MAX_RATIO = 0.12       # below this: confidently side-on
FRONT_OR_REAR_MIN_RATIO = 0.16  # above this: confidently front-on or rear-on
# Between the two: genuinely ambiguous, reported as "uncertain" rather than
# forced into one bucket.


@dataclass(frozen=True)
class AngleEstimate:
    angle: str            # "side_on" | "front_or_rear" | "uncertain" | "unavailable"
    ratio: float           # shoulder_width_px / body_height_px, or None
    confidence_note: str


def estimate_camera_angle(df: pd.DataFrame, reference_frame_idx: int,
                           frame_width: int, frame_height: int) -> AngleEstimate:
    """
    reference_frame_idx: a frame where the bowler should be reasonably
    upright (e.g. BFC or FFC) — using the run mid-point avoids picking a
    frame with an extreme, non-representative body pose.
    """
    required = ["LEFT_SHOULDER_x", "RIGHT_SHOULDER_x", "NOSE_y",
                "LEFT_ANKLE_y", "RIGHT_ANKLE_y"]
    if not all(c in df.columns for c in required):
        return AngleEstimate("unavailable", None, "Required landmark columns not found.")

    if reference_frame_idx < 0 or reference_frame_idx >= len(df):
        reference_frame_idx = len(df) // 2

    row = df.iloc[reference_frame_idx]
    if any(pd.isna(row[c]) for c in required):
        # try a small neighborhood before giving up
        lo = max(0, reference_frame_idx - 5)
        hi = min(len(df), reference_frame_idx + 5)
        window = df.iloc[lo:hi].dropna(subset=required)
        if window.empty:
            return AngleEstimate("unavailable", None,
                                  "Landmarks missing at and around the reference frame.")
        row = window.iloc[len(window) // 2]

    shoulder_width_norm = abs(float(row["LEFT_SHOULDER_x"]) - float(row["RIGHT_SHOULDER_x"]))
    ankle_y_avg = (float(row["LEFT_ANKLE_y"]) + float(row["RIGHT_ANKLE_y"])) / 2
    body_height_norm = abs(ankle_y_avg - float(row["NOSE_y"]))

    if body_height_norm < 1e-4:
        return AngleEstimate("unavailable", None, "Body height span too small to compute ratio.")

    # Convert to pixels for both dimensions so aspect-ratio differences
    # between width/height don't distort the ratio.
    shoulder_width_px = shoulder_width_norm * frame_width
    body_height_px = body_height_norm * frame_height
    ratio = round(shoulder_width_px / body_height_px, 4)

    if ratio <= SIDE_ON_MAX_RATIO:
        return AngleEstimate("side_on", ratio,
                              "Shoulders nearly edge-on to camera — consistent with side-on filming.")
    elif ratio >= FRONT_OR_REAR_MIN_RATIO:
        return AngleEstimate("front_or_rear", ratio,
                              "Full shoulder width visible — consistent with front-on or rear-on "
                              "filming. Cannot distinguish which from geometry alone.")
    else:
        return AngleEstimate("uncertain", ratio,
                              "Shoulder-width ratio is in an ambiguous range — could be a partial/"
                              "oblique angle. Please confirm the filming angle manually.")
