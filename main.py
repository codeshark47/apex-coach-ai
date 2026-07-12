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
# ROOT CAUSE FIX: MediaPipe was running in IMAGE mode, not VIDEO mode
# ============================================================
# The previous version called `landmarker.detect(frame)` and never set
# `running_mode` in PoseLandmarkerOptions — which DEFAULTS TO IMAGE MODE.
# Verified directly against the installed mediapipe package's real API
# signature, not assumed.
#
# IMAGE mode treats every single frame as a completely independent,
# unrelated picture — MediaPipe's own internal temporal tracker (which
# keeps a skeleton smoothly locked onto the same body across a sequence
# of frames) never engages at all. This explains the reported symptom
# precisely: the skeleton visibly wobbling/drifting even with the bowler
# alone in frame, since each frame was re-detected from scratch with zero
# continuity assist.
#
# Confirming evidence: `min_tracking_confidence` was already being set in
# the old code, but that parameter ONLY has any effect in VIDEO/LIVE_STREAM
# mode — in IMAGE mode it is silently ignored. Its presence strongly
# suggests VIDEO mode was the original intent, just never actually enabled.
#
# FIX: running_mode=VIDEO + detect_for_video(image, timestamp_ms), with a
# strictly increasing per-frame timestamp derived from fps. This lets
# MediaPipe's own tracker do the heavy lifting for temporal stability,
# which is a fundamentally better signal than any amount of post-hoc
# smoothing/interpolation applied after the fact.
#
# The frame-to-frame continuity/outlier-rejection layer below is KEPT as a
# secondary safety net specifically for a different failure mode: VIDEO
# mode improves stability of tracking an already-locked subject, but does
# not guarantee MediaPipe will never re-evaluate and pick a different
# person if one becomes more prominent in the frame. That's a distinct
# risk from mode-related jitter, and still needs its own guard.
NUM_POSES = 1
MIN_DETECTION_CONFIDENCE = 0.6
MIN_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

JUMP_HISTORY_LEN = 5
ADAPTIVE_MULTIPLIER = 3.0
HARD_CAP_JUMP = 0.45  # absolute safety ceiling regardless of recent motion


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

    Uses MediaPipe's VIDEO running mode (detect_for_video) so the model's
    own temporal tracker keeps the skeleton locked onto the same subject
    across frames, plus a continuity/outlier-rejection safety net for the
    separate risk of a different person being picked mid-clip.
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
        running_mode=vision.RunningMode.VIDEO,  # THE FIX — was defaulting to IMAGE
        output_segmentation_masks=False,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    ms_per_frame = 1000.0 / fps

    columns = ["frame"]
    for name in LANDMARK_NAMES:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    import collections
    dataset_rows = []
    frame_number = 0
    last_known_pos = None
    jump_history = collections.deque(maxlen=JUMP_HISTORY_LEN)
    last_timestamp_ms = -1  # VIDEO mode requires strictly increasing timestamps

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Strictly increasing timestamp, required by VIDEO mode. Guards
        # against any rounding producing a duplicate/decreasing value on
        # variable-framerate source files.
        timestamp_ms = int(round(frame_number * ms_per_frame))
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        detection_result = landmarker.detect_for_video(mp_image_frame, timestamp_ms)
        poses = detection_result.pose_landmarks or []

        row = [frame_number]
        if poses:
            chosen_landmarks = poses[0]
            c = _hip_centroid(chosen_landmarks)
            accept = c is not None
            if accept and last_known_pos is not None:
                jump = _dist(c, last_known_pos)
                if jump > HARD_CAP_JUMP:
                    accept = False
                elif len(jump_history) >= 2:
                    baseline = sorted(jump_history)[len(jump_history) // 2]  # median
                    if jump > max(baseline * ADAPTIVE_MULTIPLIER, 0.08):
                        accept = False
                if accept:
                    jump_history.append(jump)
            if accept:
                for landmark in chosen_landmarks:
                    row.extend([landmark.x, landmark.y, landmark.z])
                last_known_pos = c
            else:
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
        method="linear", limit=8, limit_direction="both"
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