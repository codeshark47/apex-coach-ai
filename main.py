import cv2
import pandas as pd
import numpy as np
import os
import urllib.request
import collections
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
_LEFT_SHOULDER_IDX = 11
_RIGHT_SHOULDER_IDX = 12
_LEFT_ANKLE_IDX = 27
_RIGHT_ANKLE_IDX = 28

# ============================================================
# MULTI-CANDIDATE TRACKING WITH POSITION + APPEARANCE MATCHING
# ============================================================
# HISTORY: an earlier version used NUM_POSES=1, trusting MediaPipe's own
# single top detection each frame. This removed our ability to ever
# disambiguate — if MediaPipe's internal choice silently switched to a
# different real person (e.g. the bowler and a background walker crossing
# paths), we had no second candidate to compare against and no way to
# catch it. Confirmed against real footage: this caused the skeleton to
# drift from the bowler onto an unrelated person mid-clip.
#
# FIX: request a small pool of candidate detections per frame (NUM_POSES)
# and choose among them ourselves using two independent signals:
#   1. Position continuity — how close each candidate is to where we last
#      confirmed the bowler was.
#   2. Appearance similarity — a color-histogram comparison of the region
#      around each candidate against a running appearance profile built
#      from previously-confirmed frames. A different real person crossing
#      paths will usually look visibly different (clothing/skin tone)
#      even at a similar position — this is the signal that position
#      alone cannot provide.
# Appearance is used as a tiebreaker/preference (a scoring bonus), not a
# hard veto — this avoids new false rejections from lighting changes or
# motion blur, while still meaningfully discouraging a switch to a
# different-looking person when positions are close.
#
# INITIAL LOCK-ON: rather than trusting a single frame's detection (which
# can be a MediaPipe hallucination from a partial view, e.g. just a hand
# entering frame), we require several consecutive frames of a plausible,
# continuously-moving candidate before accepting it as the real subject.
NUM_POSES = 3
MIN_DETECTION_CONFIDENCE = 0.6
MIN_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

JUMP_HISTORY_LEN = 5
ADAPTIVE_MULTIPLIER = 3.0
HARD_CAP_JUMP = 0.45
MAX_CONSECUTIVE_REJECTIONS = 3

BUFFER_REQUIRED = 4          # consecutive plausible+continuous frames needed to confirm initial lock
MAX_BUFFER_STEP = 0.12       # max plausible frame-to-frame movement while still buffering (walking pace)
MIN_BUFFER_TOTAL_MOVEMENT = 0.035  # minimum cumulative movement across the buffer to rule out a static false lock (background clutter, a standing bystander)
APPEARANCE_HISTORY_LEN = 5
APPEARANCE_TIEBREAK_WEIGHT = 0.08  # small scoring bonus for appearance match; not a hard gate


def _hip_centroid(landmarks) -> Optional[tuple]:
    try:
        lh, rh = landmarks[_LEFT_HIP_IDX], landmarks[_RIGHT_HIP_IDX]
        return ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)
    except (IndexError, AttributeError):
        return None


def _dist(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _is_plausible_person(landmarks) -> bool:
    """
    Guards against MediaPipe hallucinating a full 33-point guess from a
    partial view (e.g. just a hand/arm at the frame edge). Requires
    visible ankles AND a body shape that's clearly taller than it is
    wide — a real standing/moving person, not a compressed guess.
    """
    try:
        l_ankle_vis = float(landmarks[_LEFT_ANKLE_IDX].visibility)
        r_ankle_vis = float(landmarks[_RIGHT_ANKLE_IDX].visibility)
        if max(l_ankle_vis, r_ankle_vis) < 0.5:
            return False
        l_sh = np.array([float(landmarks[_LEFT_SHOULDER_IDX].x), float(landmarks[_LEFT_SHOULDER_IDX].y)])
        r_sh = np.array([float(landmarks[_RIGHT_SHOULDER_IDX].x), float(landmarks[_RIGHT_SHOULDER_IDX].y)])
        l_an = np.array([float(landmarks[_LEFT_ANKLE_IDX].x), float(landmarks[_LEFT_ANKLE_IDX].y)])
        r_an = np.array([float(landmarks[_RIGHT_ANKLE_IDX].x), float(landmarks[_RIGHT_ANKLE_IDX].y)])
        shoulder_width = _dist(tuple(l_sh), tuple(r_sh))
        mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
        mid_an = ((l_an[0] + r_an[0]) / 2, (l_an[1] + r_an[1]) / 2)
        body_span = _dist(mid_sh, mid_an)
        return shoulder_width > 0 and body_span >= shoulder_width * 1.3
    except (IndexError, AttributeError, TypeError, ValueError):
        return False


def _bbox_from_landmarks(landmarks, width, height, margin=0.04):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = max(0.0, min(xs) - margin), min(1.0, max(xs) + margin)
    min_y, max_y = max(0.0, min(ys) - margin), min(1.0, max(ys) + margin)
    x1, y1 = int(min_x * width), int(min_y * height)
    x2, y2 = int(max_x * width), int(max_y * height)
    return x1, y1, x2, y2


def _compute_histogram(frame_bgr, bbox):
    """HSV color histogram of the region around a candidate — a cheap,
    lighting-tolerant appearance fingerprint (clothing/skin tone)."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist
    except Exception:
        return None


def _hist_similarity(hist, profile_deque) -> float:
    """Best-case similarity against recent accepted appearances (max over
    the last few, so one oddly-lit frame in the profile doesn't
    permanently drag comparisons down)."""
    if hist is None or not profile_deque:
        return 0.0
    best = 0.0
    for past_hist in profile_deque:
        try:
            sim = cv2.compareHist(hist, past_hist, cv2.HISTCMP_CORRELATION)
            best = max(best, sim)
        except Exception:
            continue
    return best


def extract_video_landmarks(video_path: str, output_csv_path: str) -> dict:
    """
    Headless Perception Layer. Processes every frame sequentially without
    destructive cropping to ensure zero data loss.

    Multi-candidate tracking: evaluates several detections per frame
    (rather than blindly trusting MediaPipe's single top pick) using
    position continuity + appearance matching to stay locked on the same
    real person. See module docstring for full rationale.
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
        num_poses=NUM_POSES,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    ms_per_frame = 1000.0 / fps

    columns = ["frame"]
    for name in LANDMARK_NAMES:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    dataset_rows = []
    frame_number = 0
    last_timestamp_ms = -1

    locked = False
    lock_buffer = []  # list of (frame_number, landmarks, centroid, hist)

    last_known_pos = None
    jump_history = collections.deque(maxlen=JUMP_HISTORY_LEN)
    consecutive_rejections = 0
    appearance_profile = collections.deque(maxlen=APPEARANCE_HISTORY_LEN)

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
        poses = detection_result.pose_landmarks or []

        # Build candidate list: only plausible person-shaped detections
        candidates = []
        for landmarks in poses:
            if not _is_plausible_person(landmarks):
                continue
            centroid = _hip_centroid(landmarks)
            if centroid is None:
                continue
            bbox = _bbox_from_landmarks(landmarks, width, height)
            hist = _compute_histogram(frame, bbox)
            candidates.append({"landmarks": landmarks, "centroid": centroid, "hist": hist, "bbox": bbox})

        row = [frame_number]

        if not locked:
            # --- BUFFERING PHASE: require sustained, continuous evidence
            # before trusting this as the real subject ---
            chosen = None
            if not lock_buffer:
                if candidates:
                    # Prefer the largest/most prominent figure as the initial
                    # seed rather than whichever detection MediaPipe happens
                    # to list first — more likely to be the real subject than
                    # an arbitrary background candidate.
                    chosen = max(candidates, key=lambda c: (c["bbox"][3] - c["bbox"][1]))
            else:
                prev_centroid = lock_buffer[-1]["centroid"]
                best, best_dist = None, None
                for c in candidates:
                    d = _dist(c["centroid"], prev_centroid)
                    if d <= MAX_BUFFER_STEP and (best_dist is None or d < best_dist):
                        best, best_dist = c, d
                chosen = best

            if chosen is not None:
                lock_buffer.append(chosen)
            else:
                lock_buffer = candidates[:1] if candidates else []

            if len(lock_buffer) >= BUFFER_REQUIRED:
                # Require genuine sustained MOVEMENT before confirming — a
                # static false-positive detection (background clutter, a
                # standing bystander) can trivially satisfy "consistent
                # position frame-to-frame" without ever being a person
                # walking/running into frame. Without this, a static false
                # lock could form BEFORE the bowler ever appears, then get
                # force-switched onto the real bowler later via the
                # rejection escape valve — the "skeleton waits, then jumps
                # onto him" failure.
                total_movement = sum(
                    _dist(lock_buffer[i]["centroid"], lock_buffer[i + 1]["centroid"])
                    for i in range(len(lock_buffer) - 1)
                )
                if total_movement < MIN_BUFFER_TOTAL_MOVEMENT:
                    lock_buffer.pop(0)
                    row.extend([np.nan] * (33 * 3))
                    dataset_rows.append(row)
                    frame_number += 1
                    continue

                # Confirmed — backfill the buffered frames' data
                locked = True
                start_frame = frame_number - len(lock_buffer) + 1
                for i, buf_item in enumerate(lock_buffer):
                    buf_row = [start_frame + i]
                    for landmark in buf_item["landmarks"]:
                        buf_row.extend([landmark.x, landmark.y, landmark.z])
                    dataset_rows.append(buf_row)
                    if buf_item["hist"] is not None:
                        appearance_profile.append(buf_item["hist"])
                last_known_pos = lock_buffer[-1]["centroid"]
                lock_buffer = []
                frame_number += 1
                continue
            else:
                row.extend([np.nan] * (33 * 3))
                dataset_rows.append(row)
                frame_number += 1
                continue

        # --- LOCKED PHASE: choose best candidate via position + appearance ---
        geo_plausible = [c for c in candidates if _dist(c["centroid"], last_known_pos) <= HARD_CAP_JUMP]

        accept = False
        chosen_candidate = None

        if geo_plausible:
            def _score(c):
                jump = _dist(c["centroid"], last_known_pos)
                sim = _hist_similarity(c["hist"], appearance_profile)
                return jump - (APPEARANCE_TIEBREAK_WEIGHT * sim)

            chosen_candidate = min(geo_plausible, key=_score)
            jump = _dist(chosen_candidate["centroid"], last_known_pos)

            if len(jump_history) >= 2:
                baseline = sorted(jump_history)[len(jump_history) // 2]
                if jump > max(baseline * ADAPTIVE_MULTIPLIER, 0.08):
                    accept = False
                else:
                    accept = True
            else:
                accept = True

            if not accept and consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
                accept = True
                jump_history.clear()

            if accept:
                jump_history.append(jump)
                consecutive_rejections = 0
            else:
                consecutive_rejections += 1
        else:
            # No candidate within HARD_CAP_JUMP — reject, but allow the
            # sustained-motion escape valve using the closest candidate
            # overall (ignoring the cap) if rejections pile up.
            consecutive_rejections += 1
            if consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS and candidates:
                chosen_candidate = min(candidates, key=lambda c: _dist(c["centroid"], last_known_pos))
                accept = True
                jump_history.clear()
                consecutive_rejections = 0

        if accept and chosen_candidate is not None:
            for landmark in chosen_candidate["landmarks"]:
                row.extend([landmark.x, landmark.y, landmark.z])
            last_known_pos = chosen_candidate["centroid"]
            if chosen_candidate["hist"] is not None:
                appearance_profile.append(chosen_candidate["hist"])
        else:
            row.extend([np.nan] * (33 * 3))

        dataset_rows.append(row)
        frame_number += 1

    cap.release()
    landmarker.close()

    output_df = pd.DataFrame(dataset_rows, columns=columns)
    output_df = output_df.sort_values("frame").reset_index(drop=True)

    landmark_cols = [c for c in output_df.columns if c != "frame"]
    output_df[landmark_cols] = output_df[landmark_cols].interpolate(
        method="linear", limit=8, limit_direction="both"
    )
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