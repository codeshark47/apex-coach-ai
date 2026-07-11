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

NUM_POSES = 5
WARMUP_SECONDS = 1.5
MAX_MATCH_DIST = 0.3


def _hip_centroid(landmarks) -> Optional[tuple]:
    try:
        lh, rh = landmarks[_LEFT_HIP_IDX], landmarks[_RIGHT_HIP_IDX]
        return ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)
    except (IndexError, AttributeError):
        return None


def _dist(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _greedy_match(track_positions: dict, detections: list, max_dist: float) -> dict:
    pairs = []
    for tid, tpos in track_positions.items():
        for di, dpos in enumerate(detections):
            d = _dist(tpos, dpos)
            if d <= max_dist:
                pairs.append((d, tid, di))
    pairs.sort(key=lambda p: p[0])

    assigned = {}
    used_tracks, used_dets = set(), set()
    for d, tid, di in pairs:
        if tid in used_tracks or di in used_dets:
            continue
        assigned[tid] = di
        used_tracks.add(tid)
        used_dets.add(di)
    return assigned


def _nearest_detection_idx(target_pos: tuple, detections: list) -> Optional[int]:
    if not detections:
        return None
    best_idx, best_dist = None, None
    for i, dpos in enumerate(detections):
        d = _dist(target_pos, dpos)
        if best_dist is None or d < best_dist:
            best_dist, best_idx = d, i
    return best_idx


def extract_video_landmarks(video_path: str, output_csv_path: str) -> dict:
    """
    Headless Perception Layer. Processes every frame sequentially without
    destructive cropping to ensure zero data loss.
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
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    columns = ["frame"]
    for name in LANDMARK_NAMES:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    warmup_frame_count = max(10, int(fps * WARMUP_SECONDS))

    dataset_rows = []
    frame_number = 0

    identity_locked = False
    warmup_buffer = []
    tracks = {}
    next_track_id = 0
    last_known_pos = None

    def _lock_identity():
        nonlocal last_known_pos
        if tracks:
            target_tid = max(tracks, key=lambda k: tracks[k]["movement"])
            target_history = tracks[target_tid]["history"]
            last_known_pos = tracks[target_tid]["pos"]
        else:
            target_history = {}
            last_known_pos = None

        for buf_frame_number, buf_poses in warmup_buffer:
            pose_idx = target_history.get(buf_frame_number)
            row = [buf_frame_number]
            if pose_idx is not None and pose_idx < len(buf_poses):
                for landmark in buf_poses[pose_idx]:
                    row.extend([landmark.x, landmark.y, landmark.z])
            else:
                row.extend([np.nan] * (33 * 3))
            dataset_rows.append(row)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = landmarker.detect(mp_image_frame)
        poses = detection_result.pose_landmarks or []

        if not identity_locked and frame_number < warmup_frame_count:
            warmup_buffer.append((frame_number, poses))
            centroids = [_hip_centroid(p) for p in poses]
            valid = [(i, c) for i, c in enumerate(centroids) if c is not None]

            track_positions = {tid: t["pos"] for tid, t in tracks.items()}
            assigned = _greedy_match(track_positions, [c for _, c in valid], MAX_MATCH_DIST)
            used_valid_idx = set(assigned.values())
            for tid, vi in assigned.items():
                pose_i, c = valid[vi]
                moved = _dist(tracks[tid]["pos"], c)
                tracks[tid]["movement"] += moved
                tracks[tid]["pos"] = c
                tracks[tid]["history"][frame_number] = pose_i
            for vi, (pose_i, c) in enumerate(valid):
                if vi not in used_valid_idx:
                    tracks[next_track_id] = {"pos": c, "movement": 0.0,
                                              "history": {frame_number: pose_i}}
                    next_track_id += 1

            frame_number += 1
            if frame_number >= warmup_frame_count:
                identity_locked = True
                _lock_identity()
            continue

        if not identity_locked:
            identity_locked = True
            _lock_identity()

        row = [frame_number]
        if poses:
            centroids = [_hip_centroid(p) for p in poses]
            valid = [(i, c) for i, c in enumerate(centroids) if c is not None]
            if last_known_pos is not None and valid:
                nearest_vi = _nearest_detection_idx(last_known_pos, [c for _, c in valid])
                pose_i, c = valid[nearest_vi]
                chosen_landmarks = poses[pose_i]
                last_known_pos = c
            else:
                chosen_landmarks = poses[0]
                c = _hip_centroid(chosen_landmarks)
                if c is not None:
                    last_known_pos = c
            for landmark in chosen_landmarks:
                row.extend([landmark.x, landmark.y, landmark.z])
        else:
            row.extend([np.nan] * (33 * 3))

        dataset_rows.append(row)
        frame_number += 1

    cap.release()
    landmarker.close()

    output_df = pd.DataFrame(dataset_rows, columns=columns)
    output_df = output_df.sort_values("frame").reset_index(drop=True)

    # --- GAP-FILL SMOOTHING ---
    # MediaPipe can lose detection for a handful of consecutive frames when
    # the bowler passes behind a net, sightscreen, or another player. Left
    # as NaN, this makes the skeleton overlay flicker on/off in the
    # annotated video. Short gaps (<= 8 frames, ~0.25s at 30fps) are
    # linearly interpolated; longer gaps are deliberately left as NaN
    # rather than fabricating a long stretch of invented pose data.
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
