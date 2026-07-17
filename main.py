import cv2
import pandas as pd
import numpy as np
import os
import urllib.request

# ============================================================
# REVERTED to a simple baseline after 9 commits of identity-tracking
# heuristics (multi-person warm-up/lock, ankle-visibility gating, body-
# proportion checks, movement-buffered lock-on, multi-candidate position+
# appearance matching) repeatedly failed on real footage — each fix solved
# one edge case while introducing or leaving another. Confirmed against
# user testing: the pre-heuristics version (this one) tracked reliably.
#
# KEPT from that whole effort: MediaPipe's VIDEO running mode with
# detect_for_video() + strictly increasing timestamps. This is a real,
# verified fix (confirmed against MediaPipe's actual API) — the earlier
# baseline ran in IMAGE mode, analyzing every frame independently with
# zero temporal continuity, which is a genuine bug, not a heuristic guess.
# VIDEO mode lets MediaPipe's own internal tracker do the continuity work,
# which is a fundamentally better signal than any of the custom heuristics
# attempted afterward.
#
# ALSO KEPT: brief interpolation across short gaps (<=5 frames) for
# genuine momentary tracking dropout (net occlusion, motion blur) — this
# is separate from and was not the cause of the identity-switching bugs.
#
# If identity confusion (skeleton switching to a different real person)
# resurfaces, the correct next step is NOT another automatic heuristic —
# repeated attempts at that have not held up. The credible next step is a
# one-time manual seed (coach clicks the bowler in a reference frame),
# discussed and explicitly deferred for now.
# ============================================================

LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "LEFT_MOUTH_OUTER", "RIGHT_MOUTH_OUTER", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]


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
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
        # LOWERED from 0.5: a bowler still distant/small early in the
        # run-up often doesn't clear a 0.5 confidence threshold, so the
        # skeleton doesn't appear until he's closer/larger in frame later
        # in the clip. This is a detection-confidence issue, NOT an
        # identity-switching issue (confirmed: no other person in frame).
        # Lower threshold trades a little more sensitivity to background
        # false positives for earlier detection of a genuine, distant
        # bowler — acceptable here since there's no second person to be
        # confused with.
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.4
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    ms_per_frame = 1000.0 / fps

    columns = ["frame"]
    for name in LANDMARK_NAMES:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    dataset_rows = []
    frame_number = 0
    last_timestamp_ms = -1

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(round(frame_number * ms_per_frame))
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        detection_result = landmarker.detect_for_video(mp_image_frame, timestamp_ms)
        row = [frame_number]

        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            first_person_landmarks = detection_result.pose_landmarks[0]
            for landmark in first_person_landmarks:
                row.extend([landmark.x, landmark.y, landmark.z])
        else:
            row.extend([np.nan] * (33 * 3))

        dataset_rows.append(row)
        frame_number += 1

    cap.release()
    landmarker.close()

    output_df = pd.DataFrame(dataset_rows, columns=columns)

    # Brief gap-fill for genuine short occlusion (net, motion blur) — NOT
    # related to the identity-switching bugs, kept separately since it was
    # a real, narrow improvement on its own.
    landmark_cols = [c for c in output_df.columns if c != "frame"]
    output_df[landmark_cols] = output_df[landmark_cols].interpolate(
        method="linear", limit=5, limit_direction="both"
    )

    # LIGHT SMOOTHING: was present in an earlier working version, removed
    # by accident when the identity-tracking heuristics (a completely
    # separate, unrelated feature) were reverted from this same file.
    # This is intentionally NOT part of "who to track" logic — it only
    # smooths the already-selected trajectory, so it carries none of the
    # identity-switching risk from last night's heuristics. Fixes the
    # skeleton looking "loose"/jittery even when correctly locked onto
    # the bowler the whole time.
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
