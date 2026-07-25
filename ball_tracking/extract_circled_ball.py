"""
extract_circled_ball.py

Phase 2: turns a manually-annotated video (coach draws a bright circle
marker around the ball in CapCut/any editor, frame by frame or across a
slow-mo stretch) into real ground-truth labels — the ball's actual pixel
position, automatically extracted, for every frame the marker appears.

WHY THIS EXISTS: verified this session that classical background-
subtraction detection alone is unreliable (12.5% real hit rate across 8
clips) — the honest path forward needs real labeled data to validate or
train against, and hand-labeling every frame by clicking is slow. A
bright, high-contrast circle marker (something the ball itself never is
— no cricket ball is a uniform saturated red/green ring) is a genuinely
easy, reliable thing to detect with simple color thresholding, which
means the coach's annotation work becomes labels automatically instead
of needing a second manual data-entry pass.

HONEST STATUS: detecting a bright, deliberately-drawn marker is a much
easier problem than detecting the ball itself, and this has NOT been
tested against a real annotated clip yet (none exist as of writing this).
Marker color defaults assume a red circle (CapCut's default annotation
color in many templates) — this WILL need adjusting once a real
annotated clip is available to check against, the same "verify before
trusting" discipline as everything else in this codebase.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerDetection:
    frame_index: int
    center_x_px: float
    center_y_px: float
    radius_px: float


# Default assumes a RED circle/ring marker (CapCut's common default
# annotation color). HSV red wraps around 0/180, so two ranges are
# combined. Adjust these — and re-verify — once a real annotated clip
# exists; do not assume these are correct for a different marker color
# without checking.
_RED_HSV_RANGES = [
    (np.array([0, 100, 80]), np.array([10, 255, 255])),
    (np.array([170, 100, 80]), np.array([180, 255, 255])),
]


def extract_marker_positions(
    video_path: str,
    hsv_ranges: Optional[List[tuple]] = None,
    min_radius_px: float = 3.0,
    max_radius_px: float = 40.0,
    debug_output_path: Optional[str] = None,
) -> dict:
    """
    Scans every frame for a colored circle marker matching hsv_ranges,
    returns the marker's center + radius per frame it's found in.

    This does NOT distinguish "a marker the coach drew" from "something
    else in the scene that happens to be the same color" — if red
    stumps, a red logo, or red clothing are in frame, they will also
    match unless hsv_ranges is narrowed or the scene is checked to be
    clear of that color first. Inspect debug_output_path before trusting
    results on any new clip, same as the classical ball detector.

    Returns {"status": "success", "fps": ..., "total_frames": ...,
    "frames_with_marker": int, "positions_by_frame": {frame_idx:
    MarkerDetection}} or {"status": "error", "message": ...}.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "message": f"Video not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": f"Could not open video: {video_path}"}

    ranges = hsv_ranges if hsv_ranges is not None else _RED_HSV_RANGES
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if debug_output_path:
        writer = cv2.VideoWriter(
            debug_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
        )

    positions_by_frame = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            if min_radius_px <= radius <= max_radius_px:
                area = cv2.contourArea(c)
                if best is None or area > best[3]:
                    best = (cx, cy, radius, area)

        if best is not None:
            cx, cy, radius, _ = best
            positions_by_frame[idx] = MarkerDetection(
                frame_index=idx, center_x_px=cx, center_y_px=cy, radius_px=radius
            )

        if writer is not None:
            annotated = frame.copy()
            if best is not None:
                cv2.circle(annotated, (int(best[0]), int(best[1])), int(best[2]), (0, 255, 0), 2)
            writer.write(annotated)

        idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    return {
        "status": "success",
        "fps": fps,
        "total_frames": idx,
        "frames_with_marker": len(positions_by_frame),
        "positions_by_frame": positions_by_frame,
    }
