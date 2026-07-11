import cv2
import pandas as pd
import numpy as np
import os
import urllib.request
from typing import Optional


LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "LEFT_MOUTH_OUTER", "RIGHT_MOUTH_OUTER", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

_LEFT_HIP_IDX = 23
_RIGHT_HIP_IDX = 24

# ============================================================
# SINGLE-TARGET DETECTION WITH CONTINUITY FILTERING
# ============================================================
# Replaces an earlier multi-person "warm-up and lock" heuristic that
# assumed whoever moved most in the first ~1.5s was the bowler. That
# approach caused two confirmed real failures on actual footage:
#   1. Locking onto a different real person (e.g. a fielder walking back
#      to position) instead of the bowler.
#   2. Locking onto a spurious low-confidence detection from background
#      clutter (floodlight poles, buildings) and following that "ghost"
#      for the entire clip, producing a skeleton that floats away from
#      the real player.
# None of the actual footage tested so far genuinely needs multi-person
# disambiguation — it's a single subject on an open field. So instead of
# a fragile heuristic solving a problem that doesn't exist, this uses:
#   - NUM_POSES = 1: MediaPipe returns only its single highest-confidence
#     detection per frame, removing the "pick the wrong candidate" failure
#     mode entirely.
#   - A raised confidence floor, so borderline background false-positives
#     don't get reported as a detection at all.
#   - Frame-to-frame continuity/outlier rejection: if the one detection
#     MediaPipe does return has "teleported" further than a person could
#     plausibly move in a single frame, it's treated as a missed frame
#     (NaN) rather than trusted. The existing short-gap interpolation
#     then bridges it smoothly, instead of drawing a floating ghost.
#
# MAX_PLAUSIBLE_JUMP is a normalized-coordinate threshold (fraction of
# frame width/height) and may need retuning if footage framing changes
# drastically (e.g. a much tighter or wider crop on the bowler).
NUM_POSES = 1
MAX_PLAUSIBLE_JUMP = 0.15
MIN_DETECTION_CONFIDENCE = 0.6
MIN_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6


def _hip_centroid(landmarks) -> Optional[tuple]:
    """Midpoint of left/right hip landmarks — a stable, central body point
    that moves smoothly with the person, less noisy than a limb extremity."""
    try:
        lh, rh = landmarks[_LEFT_HIP_IDX], landmarks[_RIGHT_HIP_IDX]
        return ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)
    except (IndexError, AttributeError):
        return None


def _dist(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def extract_video_landmarks(video_path: str, output_csv_path: str) -> dict:
    """
    Headless Perception Layer. Processes every frame sequentially without
    destructive cropping to ensure zero data loss.

    Uses single-target detection with continuity filtering (see module
    docstring) to reject implausible frame-to-frame jumps caused by
    background false-positive detections.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error_message": f"Input video file not found: {video_path}"}

    model_dir = "models"
    model_path = os.path.join(model_dir, "pose_landmarker_full.task")
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(model_path):
        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        try:
            urllib.request.urlretrieve(model_url, model_path)
        except Exception as e:
            return {"status": "error", "error_message": f"Failed to download model file: {str(e)}"}

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        return {"status": "error", "error_message": "MediaPipe Tasks API framework binding is missing."}

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    columns = ["frame"]
    for name in LANDMARK_NAMES:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    dataset_rows = []
    frame_number = 0
    last_known_pos = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = landmarker.detect(mp_image_frame)
        poses = detection_result.pose_landmarks or []

        row = [frame_number]
        if poses:
            chosen_landmarks = poses[0]
            c = _hip_centroid(chosen_landmarks)
            jumped_too_far = (
                c is not None and last_known_pos is not None
                and _dist(c, last_known_pos) > MAX_PLAUSIBLE_JUMP
            )
            if c is not None and not jumped_too_far:
                for landmark in chosen_landmarks:
                    row.extend([landmark.x, landmark.y, landmark.z])
                last_known_pos = c
            else:
                # Either no valid centroid, or an implausible teleport —
                # treat as a missed detection rather than trusting it.
                row.extend([np.nan] * (33 * 3))
        else:
            row.extend([np.nan] * (33 * 3))

        dataset_rows.append(row)
        frame_number += 1

    cap.release()
    landmarker.close()

    output_df = pd.DataFrame(dataset_rows, columns=columns)
    output_df = output_df.sort_values("frame").reset_index(drop=True)

    # --- GAP-FILL SMOOTHING ---
    # Bridges short dropouts (net occlusion, motion blur, or a frame
    # rejected by the continuity filter above). Gaps longer than 8 frames
    # (~0.25s at 30fps) are left as NaN rather than fabricating an
    # extended fake trajectory.
    landmark_cols = [c for c in output_df.columns if c != "frame"]
    output_df[landmark_cols] = output_df[landmark_cols].interpolate(
        method="linear", limit=8, limit_area="inside"
    )
    # Light smoothing pass to remove residual per-frame jitter.
    output_df[landmark_cols] = output_df[landmark_cols].rolling(
        window=3, center=True, min_periods=1
    ).mean()

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)

    return {
        "status": "success",
        "total_frames_processed": frame_number,
        "fps": fps,
        "output_file": output_csv_path
    }


if __name__ == "__main__":
    print("=== STARTING KINEMATIC EXTRACTION STATE ===")
    extraction_state = extract_video_landmarks("input/input_video.mp4", "output/landmarks.csv")
    print(extraction_state)
