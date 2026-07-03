"""
calibration.py

Establishes a real-world pixel-to-meter scale for a fixed camera setup,
from two points the coach clicks on a reference frame plus a known
real-world distance between them.

This module deliberately contains NO assumed stride length, no average
arm-swing radius, and no default distance constant. If calibration hasn't
been performed, speed_estimation.py will not produce a speed number — it
will report "not available" rather than guess.

Typical reference distances a coach can use (any known distance works):
  - Stump width (leg to off stump):        22.86 cm  = 0.2286 m
  - Popping crease to popping crease:      20.12 m
  - A placed marker of known length (tape measure, cone spacing, etc.)

Calibration is done ONCE per fixed camera position and can be reused for
every video shot from that same spot — it does not need to be redone per
delivery.
"""

import math
from dataclasses import dataclass
from typing import Optional

import cv2


@dataclass(frozen=True)
class Calibration:
    meters_per_pixel: float
    reference_label: str
    reference_distance_m: float
    point_a_px: tuple
    point_b_px: tuple

    def to_dict(self) -> dict:
        return {
            "meters_per_pixel": self.meters_per_pixel,
            "reference_label": self.reference_label,
            "reference_distance_m": self.reference_distance_m,
            "point_a_px": list(self.point_a_px),
            "point_b_px": list(self.point_b_px),
        }

    @staticmethod
    def from_dict(d: dict) -> "Calibration":
        return Calibration(
            meters_per_pixel=float(d["meters_per_pixel"]),
            reference_label=d.get("reference_label", "custom"),
            reference_distance_m=float(d["reference_distance_m"]),
            point_a_px=tuple(d["point_a_px"]),
            point_b_px=tuple(d["point_b_px"]),
        )


def extract_reference_frame(video_path: str, frame_index: int = 0):
    """
    Pulls a single frame from the video for the coach to click on.
    Returns an RGB numpy array, or None if the frame can't be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = max(0, min(frame_index, max(total - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def compute_scale(point_a_px, point_b_px, real_world_distance_m: float,
                   reference_label: str = "custom") -> Calibration:
    """
    point_a_px, point_b_px: (x, y) pixel coordinates clicked by the coach
    real_world_distance_m: the TRUE distance between those two points, in meters

    Raises ValueError on degenerate input rather than returning a bogus scale.
    """
    if real_world_distance_m is None or real_world_distance_m <= 0:
        raise ValueError("real_world_distance_m must be a positive number.")

    dist_px = math.dist(point_a_px, point_b_px)
    if dist_px < 2:
        raise ValueError(
            "The two calibration points are too close together in pixels "
            "to give a reliable scale. Pick two points further apart."
        )

    meters_per_pixel = real_world_distance_m / dist_px

    return Calibration(
        meters_per_pixel=meters_per_pixel,
        reference_label=reference_label,
        reference_distance_m=real_world_distance_m,
        point_a_px=tuple(point_a_px),
        point_b_px=tuple(point_b_px),
    )
