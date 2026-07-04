"""
run_up_analysis.py

Analyzes the bowler's RUN-UP (all frames before Back Foot Contact) — a phase
that main.py already captures full pose landmarks for, but which Phase 1/2
never used. This module reads that existing landmark data; it does not
require any change to main.py, kinematics.py, or orchestrator.py.

Three things this produces, each with an explicit honesty boundary:

1. Stride/foot-contact detection (BFC_DETECTION)
   Fully objective: a foot-contact is a local peak in that foot's vertical
   pixel position (ground contact = foot at its lowest point in frame,
   which is the largest y-value since image y increases downward). No
   assumed stride length or cadence — purely geometric peak detection.

2. Rhythm consistency
   Reported as the coefficient of variation (CV = std/mean) of the
   intervals between consecutive foot contacts. This is NOT converted into
   a "good/bad" tier — there is no universal validated cutoff for ideal
   run-up rhythm (it depends on bowler height, pace vs. spin, run-up
   length). It's a comparable number: lower CV means more consistent
   pacing, and the real value is in comparing it across a bowler's own
   sessions over time, not against an invented absolute threshold.

3. Heel-strike vs. forefoot-strike classification
   At each detected contact frame, compares the HEEL landmark's vertical
   position against the FOOT_INDEX (toe) landmark's position for that
   foot. Whichever is lower (larger y, closer to ground) touched down
   first/is bearing more weight at contact. A small tolerance band
   ("midfoot") exists because exact equality is noise-sensitive — that
   band width is an engineering choice for stability, not a clinical
   measurement, and is documented as such below.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

# Engineering choice, not clinical data: how close heel/toe height must be
# (as a fraction of that foot's own vertical range during the run-up) to
# call a contact "midfoot" rather than a clear heel or forefoot strike.
MIDFOOT_TOLERANCE_FRACTION = 0.15


def _get_series(df: pd.DataFrame, col: str, frame_width: int, frame_height: int, is_x: bool):
    if col not in df.columns:
        return None
    scale = frame_width if is_x else frame_height
    return df[col].interpolate(method="linear").bfill().ffill().values * scale


def _detect_contacts_for_foot(y_series: np.ndarray, fps: float,
                               min_separation_s: float = 0.18) -> list:
    """
    Local-maxima peak detection on a foot's vertical (y) pixel position.
    No scipy dependency — implemented directly so it doesn't add a new
    requirement.

    A candidate must be the max of its window AND strictly higher than
    BOTH window edges — this is what rejects flat/near-flat baseline
    stretches (where every point ties for "the max" and would otherwise
    all falsely register as contacts).

    min_separation_s: minimum time between two contacts of the SAME foot.
    0.18s is a generous lower bound for a sprint-cadence run-up step cycle
    (fast bowlers' run-up cadence is well under this); it only prevents
    counting sensor/tracking noise as a double-contact, not a claim about
    real stride timing.
    """
    n = len(y_series)
    if n < 5:
        return []

    window = max(2, int(fps * 0.05))
    min_sep_frames = max(1, int(fps * min_separation_s))

    # Minimum prominence: peak must rise above its window edges by at
    # least this fraction of the foot's overall vertical range in this
    # clip — rejects sensor noise, not a claim about biomechanics.
    y_range = y_series.max() - y_series.min()
    min_prominence = max(y_range * 0.15, 1e-6)

    candidates = []
    i = window
    while i < n - window:
        segment = y_series[i - window: i + window + 1]
        left_edge, right_edge = y_series[i - window], y_series[i + window]
        is_peak = (
            y_series[i] == segment.max()
            and y_series[i] > left_edge + min_prominence
            and y_series[i] > right_edge + min_prominence
        )
        if is_peak:
            candidates.append(i)
        i += 1

    # Merge candidates that are within min_sep_frames of each other,
    # keeping the true (highest) peak of each cluster.
    contacts = []
    for c in candidates:
        if contacts and (c - contacts[-1]) < min_sep_frames:
            if y_series[c] > y_series[contacts[-1]]:
                contacts[-1] = c
        else:
            contacts.append(c)

    return contacts


def detect_run_up_strides(df: pd.DataFrame, bfc_frame_idx: int, fps: float,
                           frame_width: int, frame_height: int) -> dict:
    """
    Detects foot-contact events in the run-up window (frames 0..bfc_frame_idx).
    Returns real, measured contact frames — no assumed stride count or length.
    """
    if bfc_frame_idx < 10:
        return {
            "status": "insufficient_runup",
            "message": (
                f"Only {bfc_frame_idx} frames before BFC — too short a clip to "
                f"analyze a run-up (bowler likely started very close to the crease "
                f"in this recording, or the clip was trimmed before the run-up)."
            ),
        }

    run_up_df = df.iloc[:bfc_frame_idx].reset_index(drop=True)

    all_contacts = []
    for foot in ("LEFT", "RIGHT"):
        y = _get_series(run_up_df, f"{foot}_HEEL_y", frame_width, frame_height, is_x=False)
        if y is None:
            y = _get_series(run_up_df, f"{foot}_ANKLE_y", frame_width, frame_height, is_x=False)
        if y is None:
            continue
        contacts = _detect_contacts_for_foot(y, fps)
        for c in contacts:
            all_contacts.append({"frame": c, "foot": foot})

    if not all_contacts:
        return {
            "status": "error",
            "message": "No heel/ankle landmark columns found for either foot in landmark data.",
        }

    all_contacts.sort(key=lambda c: c["frame"])

    intervals_s = []
    for a, b in zip(all_contacts[:-1], all_contacts[1:]):
        intervals_s.append(round((b["frame"] - a["frame"]) / fps, 4))

    rhythm_cv = None
    if len(intervals_s) >= 2:
        mean_i = sum(intervals_s) / len(intervals_s)
        if mean_i > 0:
            variance = sum((x - mean_i) ** 2 for x in intervals_s) / len(intervals_s)
            std_i = math.sqrt(variance)
            rhythm_cv = round(std_i / mean_i, 4)

    return {
        "status": "success",
        "stride_count": len(all_contacts),
        "contacts": all_contacts,          # [{"frame": int, "foot": "LEFT"/"RIGHT"}, ...]
        "step_intervals_seconds": intervals_s,
        "rhythm_consistency_cv": rhythm_cv,  # lower = more consistent; compare across sessions, no absolute cutoff
        "run_up_frame_span": bfc_frame_idx,
        "run_up_duration_seconds": round(bfc_frame_idx / fps, 4),
    }


def classify_strike_patterns(df: pd.DataFrame, contacts: list,
                              frame_width: int, frame_height: int) -> list:
    """
    For each detected contact, classifies heel-strike / midfoot / forefoot
    by comparing HEEL vs FOOT_INDEX vertical position at that frame.
    Returns the same list of contacts, each annotated with "strike_pattern".
    If heel/foot_index columns aren't available, pattern is "unknown" for
    that contact — never guessed.
    """
    annotated = []
    for c in contacts:
        foot, frame = c["foot"], c["frame"]
        heel_col = f"{foot}_HEEL_y"
        toe_col = f"{foot}_FOOT_INDEX_y"

        if heel_col not in df.columns or toe_col not in df.columns or frame >= len(df):
            annotated.append({**c, "strike_pattern": "unknown"})
            continue

        heel_y = df[heel_col].iloc[frame]
        toe_y = df[toe_col].iloc[frame]
        if pd.isna(heel_y) or pd.isna(toe_y):
            annotated.append({**c, "strike_pattern": "unknown"})
            continue

        heel_y_px = heel_y * frame_height
        toe_y_px = toe_y * frame_height

        # Normalize the tolerance band by this foot's own heel-to-toe span
        # across the whole run-up rather than a fixed pixel count, so it
        # scales correctly regardless of how close/far the camera is.
        heel_series = df[heel_col].dropna() * frame_height
        toe_series = df[toe_col].dropna() * frame_height
        foot_span = abs(heel_series.mean() - toe_series.mean()) or 1.0
        tolerance_px = foot_span * MIDFOOT_TOLERANCE_FRACTION

        diff = heel_y_px - toe_y_px  # positive => heel lower (closer to ground) => heel-strike
        if abs(diff) <= tolerance_px:
            pattern = "midfoot"
        elif diff > 0:
            pattern = "heel"
        else:
            pattern = "forefoot"

        annotated.append({**c, "strike_pattern": pattern})

    return annotated


def summarize_strike_patterns(annotated_contacts: list) -> dict:
    counts = {"heel": 0, "forefoot": 0, "midfoot": 0, "unknown": 0}
    for c in annotated_contacts:
        counts[c.get("strike_pattern", "unknown")] += 1
    return counts
