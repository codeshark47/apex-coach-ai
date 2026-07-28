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
import os
from dataclasses import dataclass

import cv2
import streamlit as st


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


def _file_signature(video_path: str) -> float:
    """
    Modification time of the file, used only to key the caches below —
    NOT for any real logic. Several upload flows in this app reuse a
    fixed or filename-derived path across genuinely different uploads
    (calibration's temp file is always "calibration_ref.mp4" regardless
    of what was uploaded; seed reference paths are built from just the
    original filename, which two different recordings can share). A
    cache keyed on path alone would silently serve a stale frame from
    the PREVIOUS file at that same path. Including mtime means a
    freshly-written file (even at an identical path) gets a fresh cache
    entry, while re-requesting the same untouched file still hits cache.
    """
    try:
        return os.path.getmtime(video_path)
    except OSError:
        return 0.0


def get_frame_count(video_path: str) -> int:
    return _get_frame_count_cached(video_path, _file_signature(video_path))


@st.cache_data(show_spinner=False)
def _get_frame_count_cached(video_path: str, file_mtime: float) -> int:
    # BUG FIX: this parameter must NOT start with an underscore — Streamlit
    # treats a leading underscore on a cached function's parameter as "do
    # not hash this, it's unhashable" and silently EXCLUDES it from the
    # cache key entirely. It was named _file_mtime originally (to signal
    # "not real logic, just a cache key"), which meant the whole point of
    # this parameter — busting the cache when a reused path's file changes
    # — was being silently ignored. Verified directly: two genuinely
    # different videos written to the same path still returned the exact
    # same cached frame, byte-for-byte, until this rename.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def extract_reference_frame(video_path: str, frame_index: int = 0):
    """
    Pulls a single frame from the video for the coach to click on.
    Returns an RGB numpy array, or None if the frame can't be read.

    CACHED (see _extract_reference_frame_cached) — BUG FIX found from
    real coach feedback ("the confirmation expanders keep reloading,
    it's annoying"): every widget interaction ANYWHERE on a Streamlit
    page reruns the entire script top to bottom, and with up to 4-5
    separate reference-frame expanders open at once in Dual Camera mode,
    EVERY one of them was re-decoding its frame from the source video on
    EVERY rerun — including reruns triggered by a totally unrelated
    widget (moving a zoom slider in a different expander, typing in a
    different frame-number box). Seeking to an arbitrary frame in a real
    video (especially anything past the first keyframe) is genuinely
    slow — that repeated, unnecessary decoding on every single click is
    what read as the whole page "reloading." Caching means only a frame
    that's actually new gets decoded; everything else returns instantly.
    """
    return _extract_reference_frame_cached(video_path, frame_index, _file_signature(video_path))


# BUG FIX: this cache had no eviction policy at all — Streamlit Cloud runs
# ONE shared process for every visitor (not one per coach), so every
# distinct frame anyone ever scrubbed to, on any video, since the last
# reboot stayed resident forever as a multi-MB decoded array. Confirmed
# directly: the live app hit Streamlit Cloud's "gone over its resource
# limits" memory cap during ordinary testing. max_entries/ttl bound the
# worst case (~30 frames x a few MB each) while still keeping the whole
# point of this cache — instant re-clicks within one coach's own session.
@st.cache_data(show_spinner=False, max_entries=30, ttl=1800)
def _extract_reference_frame_cached(video_path: str, frame_index: int, file_mtime: float):
    # file_mtime must NOT start with an underscore — see the identical note
    # in _get_frame_count_cached above for why.
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
