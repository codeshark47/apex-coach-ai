import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

def extract_video_landmarks(video_path: str, output_csv_path: str) -> dict:
    """
    Headless Perception Layer. Extracts MediaPipe pose landmarks from video frames.
    Guarantees structural data continuity by filling undetected frames with NaN values.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error_message": f"Input video file not found: {video_path}"}

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    columns = ["frame"]
    for landmark in mp_pose.PoseLandmark:
        columns.extend([f"{landmark.name}_x", f"{landmark.name}_y", f"{landmark.name}_z"])

    dataset_rows = []
    frame_number = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape

        left = int(w * 0.2)
        right = int(w * 0.8)
        cropped = frame[:, left:right]
        cropped_w = right - left

        frame_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        row = [frame_number]

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                global_x = (landmark.x * cropped_w + left) / w
                row.extend([global_x, landmark.y, landmark.z])
        else:
            row.extend([np.nan] * (33 * 3))

        dataset_rows.append(row)
        frame_number += 1

    cap.release()
    pose.close()

    output_df = pd.DataFrame(dataset_rows, columns=columns)
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
