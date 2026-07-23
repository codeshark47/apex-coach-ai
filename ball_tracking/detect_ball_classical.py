"""
detect_ball_classical.py

Phase 2, Step 1: classical (non-ML) ball-candidate detection for a FIXED,
unmoving camera — per the project brief, background subtraction is the
right first thing to try before assuming a trained detector is needed at
all. A fixed camera means the pitch/background barely changes frame to
frame, so anything that moves against it (the ball, but also players,
birds, passing traffic in the background) can be found by comparing each
frame to a learned background model — no per-video tuning required for
the background itself.

HONEST STATUS — read before trusting any output from this file:
This has NOT been validated against real cricket footage yet. It is a
reasonable, standard first approach (MOG2 background subtraction +
shape/size filtering), not a proven one. Every function below returns
CANDIDATES — plausible "this might be the ball" detections per frame,
usually several per frame (a bird, a player's cap, a shadow can all pass
the same shape/size filter) — not a resolved single-ball track. Turning
candidates into a real per-frame ball position requires temporal
tracking (a real trajectory moves smoothly frame to frame; false
positives usually don't) — that's the next step once real footage shows
whether candidates are even being found reliably.

Do not wire this into any user-facing feature or claim a detection rate
to the user until it's actually been run against real match/nets footage
and the results inspected frame by frame, the same discipline every fix
in the core biomechanics pipeline has already been held to.
"""

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class BallCandidate:
    frame_index: int
    x_px: float
    y_px: float
    radius_px: float
    area_px: float
    circularity: float  # 1.0 = perfect circle; used to reject non-ball blobs


def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter * perimeter))


def detect_ball_candidates(
    video_path: str,
    min_radius_px: float = 2.0,
    max_radius_px: float = 20.0,
    min_circularity: float = 0.55,
    history: int = 120,
    var_threshold: float = 40.0,
    roi: Optional[tuple] = None,
    debug_output_path: Optional[str] = None,
) -> dict:
    """
    Runs MOG2 background subtraction over the whole clip and returns
    per-frame candidate blobs that pass a size + circularity filter.

    Assumes a genuinely FIXED camera (tripod) — if the camera pans/
    shakes, background subtraction will flag the whole frame as
    "moving" and this will produce garbage. That assumption should be
    verified against real footage, not asserted.

    min_radius_px / max_radius_px: real-world ball size in pixels
    depends entirely on camera distance and resolution — these are wide
    starting bounds, not calibrated values. Expect to need per-setup
    tuning once real footage is available, same as calibration.py
    already requires for real-world distance.

    roi: optional (x1, y1, x2, y2) in pixels — restricts detection to
    this region. Verified directly on real footage: with no ROI, the
    dominant source of false positives wasn't the players at all, it was
    wind-moved trees and background clutter far behind the pitch, which
    a fixed camera still picks up as "foreground motion" just like a real
    moving ball. Since the ball physically cannot leave the pitch
    corridor in a fixed behind-the-stumps or side-on shot, restricting
    to that region removes this category of false positive entirely
    instead of trying to filter it out after the fact. Per-camera-setup,
    not a universal constant — must be set per video framing.

    debug_output_path: if given, writes an annotated video with every
    surviving candidate circled — the actual way to evaluate this
    (watch it), not a number to trust blind.

    Returns {"status": "success", "fps": ..., "total_frames": ...,
    "candidates_by_frame": {frame_idx: [BallCandidate, ...]}} or
    {"status": "error", "message": ...}.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "message": f"Video not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": f"Could not open video: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=False
    )

    writer = None
    if debug_output_path:
        writer = cv2.VideoWriter(
            debug_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
        )

    candidates_by_frame = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fg_mask = back_sub.apply(frame)
        # Learning the background takes a running start — early frames
        # before `history` frames have passed are unreliable by MOG2's
        # own design, not a bug here.
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        if roi is not None:
            x1, y1, x2, y2 = roi
            roi_mask = np.zeros_like(fg_mask)
            roi_mask[y1:y2, x1:x2] = 255
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_candidates = []
        for c in contours:
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            if radius < min_radius_px or radius > max_radius_px:
                continue
            circ = _circularity(c)
            if circ < min_circularity:
                continue
            frame_candidates.append(BallCandidate(
                frame_index=idx, x_px=cx, y_px=cy, radius_px=radius,
                area_px=cv2.contourArea(c), circularity=circ
            ))

        if frame_candidates:
            candidates_by_frame[idx] = frame_candidates

        if writer is not None:
            annotated = frame.copy()
            for cand in frame_candidates:
                cv2.circle(annotated, (int(cand.x_px), int(cand.y_px)),
                           int(cand.radius_px), (0, 255, 255), 2)
            writer.write(annotated)

        idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    return {
        "status": "success",
        "fps": fps,
        "total_frames": idx,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frames_with_candidates": len(candidates_by_frame),
        "candidates_by_frame": candidates_by_frame,
    }
