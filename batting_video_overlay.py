"""
batting_video_overlay.py

Skeleton-overlay video generation for batting analysis — deliberately a
SEPARATE, simpler file rather than trying to extend video_overlay.py's
render_annotated_video, which the initial codebase exploration for this
feature found to be ~1000+ lines tightly coupled to bowling-specific
event keys (BFC/FFC/BR) and a live hero-metric chart tuned specifically
for knee-bracing/hip-shoulder-separation. Reuses the genuinely generic
pieces (the plausibility gates that keep a mistracked limb from being
drawn) via direct import rather than duplicating that logic.

Deliberately SIMPLER than the bowling overlay: skeleton + a plain event
label at STANCE/BACKLIFT/CONTACT/FOLLOW-THROUGH, no live per-frame
hero-metric chart bar. The metrics themselves (and the coaching report)
are the actual analysis value; this video is a supporting visual, not
where the core value lives — building an equally elaborate live chart
for batting is a reasonable future polish item, not a blocker for a
correct first version.
"""

import cv2
import pandas as pd

from video_overlay import (
    torso_shape_is_plausible,
    body_size_is_plausible,
    implausible_arm_nodes,
    _torso_height,
)

_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"), ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"), ("LEFT_HEEL", "LEFT_FOOT_INDEX"), ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"), ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"), ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
    ("NOSE", "LEFT_SHOULDER"), ("NOSE", "RIGHT_SHOULDER"),
]
_JOINT_NODES = ["LEFT_KNEE", "RIGHT_KNEE", "LEFT_HIP", "RIGHT_HIP",
                "LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE",
                "LEFT_SHOULDER", "RIGHT_SHOULDER", "NOSE"]

BONE_SHADOW = (25, 25, 25)
BONE_CORE = (123, 101, 44)
JOINT_OUTLINE = (146, 151, 187)
JOINT_FILL = (253, 254, 253)

_EVENT_LABELS = {
    "STANCE": "STANCE",
    "BACKLIFT": "BACKLIFT",
    "CONTACT": "CONTACT",
    "FOLLOW_THROUGH": "FOLLOW-THROUGH",
}


def render_batting_annotated_video(video_path: str, output_path: str,
                                    df: pd.DataFrame, events: dict,
                                    slow_motion_factor: float = 4.0,
                                    batting_hand: str = "left"):
    """
    Generates a skeleton-overlay video for a batting clip using mp4v
    codec, same slow-motion approach as the bowling overlay (all frames
    kept, output FPS divided by slow_motion_factor for true slow motion).

    events: the dict from batting_events.detect_batting_events — keys are
    frame INDICES into df (0-based row position, matching main.py's
    "frame" column), not events-in-time-order keys like bowling's
    BFC/FFC/BR naming implies phases; here STANCE/BACKLIFT/CONTACT/
    FOLLOW_THROUGH are just labels drawn on the matching frame.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    output_fps = max(1, int(round(source_fps / slow_motion_factor)))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))

    df = df.copy()
    landmark_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    # Same time-based gap-fill limit as the bowling overlay, same
    # reasoning: a genuinely long tracking loss should stay real NaN
    # (skipped gracefully below) rather than get padded into a fabricated,
    # frozen position that draws as a stray disconnected limb.
    draw_gap_fill_limit = max(1, int(round(source_fps * 0.1)))
    df[landmark_cols] = df[landmark_cols].interpolate(
        method="linear", limit=draw_gap_fill_limit, limit_direction="both", limit_area="inside"
    )

    frame_to_event = {}
    for key, frame_idx in events.items():
        label = _EVENT_LABELS.get(key)
        if label is not None and isinstance(frame_idx, (int, float)):
            frame_to_event[int(frame_idx)] = label

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rows = df[df["frame"] == f_idx]
        if not rows.empty:
            row = rows.iloc[0]
            # NOTE: torso_h here is in NORMALIZED (0-1-ish) units, matching
            # the raw landmark columns — body_size_is_plausible expects a
            # PIXEL height, hence the *height conversion only for that call.
            # implausible_arm_nodes internally compares against normalized
            # bone lengths too, so it's given the normalized torso_h as-is.
            torso_h = _torso_height(row)
            estimated_body_height_px = torso_h * 2.2 * height if torso_h > 0 else 0

            if torso_h > 0 and torso_shape_is_plausible(row) and body_size_is_plausible(estimated_body_height_px):
                implausible_nodes = implausible_arm_nodes(row, torso_h)

                for a, b in _CONNECTIONS:
                    xa, ya = row.get(f"{a}_x"), row.get(f"{a}_y")
                    xb, yb = row.get(f"{b}_x"), row.get(f"{b}_y")
                    if any(pd.isna(v) for v in (xa, ya, xb, yb)):
                        continue
                    if a in implausible_nodes or b in implausible_nodes:
                        continue
                    pa = (int(float(xa) * width), int(float(ya) * height))
                    pb = (int(float(xb) * width), int(float(yb) * height))
                    cv2.line(frame, pa, pb, BONE_SHADOW, 6, cv2.LINE_AA)
                    cv2.line(frame, pa, pb, BONE_CORE, 3, cv2.LINE_AA)

                for node in _JOINT_NODES:
                    if node in implausible_nodes:
                        continue
                    x, y = row.get(f"{node}_x"), row.get(f"{node}_y")
                    if pd.isna(x) or pd.isna(y):
                        continue
                    p = (int(float(x) * width), int(float(y) * height))
                    cv2.circle(frame, p, 6, JOINT_OUTLINE, -1, cv2.LINE_AA)
                    cv2.circle(frame, p, 4, JOINT_FILL, -1, cv2.LINE_AA)

        label = frame_to_event.get(f_idx)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame, (10, 10), (10 + tw + 20, 10 + th + 20), (25, 25, 25), -1)
            cv2.putText(frame, label, (20, 10 + th + 8), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (253, 254, 253), 2, cv2.LINE_AA)

        out.write(frame)
        f_idx += 1
        if f_idx >= total_frames:
            break

    cap.release()
    out.release()
